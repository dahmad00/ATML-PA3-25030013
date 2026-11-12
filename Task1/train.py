import argparse
import time
import math
import os
import gc

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from config import OUT, STEPS, SAVE_STEPS, METHODS, MODEL, SEED
from utils import seed, load_model, append_jsonl, save_json, rng_state, set_rng, unique_params
from models import make_adapter


def run(method, resume=False):
    seed()
    out = OUT / method
    out.mkdir(parents=True, exist_ok=True)

    full = method in ('full', 'l2sp')
    model, tok = load_model(fp32=full)
    model.config.use_cache = False

    if not full:
        model = make_adapter(model, method)
        model = model.cuda()

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False}
    )

    params = [p for p in unique_params(model) if p.requires_grad]
    ref = None

    if method == 'l2sp':
        ref = [p.detach().cpu().clone() for p in params]

    groups = [
        {
            'params': [p for p in params if p.ndim >= 2],
            'weight_decay': .01
        },
        {
            'params': [p for p in params if p.ndim < 2],
            'weight_decay': 0
        }
    ]

    optimizer = torch.optim.AdamW(
        groups,
        lr=METHODS[method],
        betas=(.9, .999),
        eps=1e-8
    )

    scheduler = get_cosine_schedule_with_warmup(optimizer, 38, STEPS)
    scaler = torch.amp.GradScaler('cuda')

    ds = load_from_disk(str(OUT / 'train_data'))
    start = 0
    ck = out / 'recovery.pt'

    if resume:
        state = torch.load(ck, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        set_rng(state['rng'])
        start = state['step']
        del state

    save_json(
        out / 'config.json',
        {
            'method': method,
            'model': MODEL,
            'seed': SEED,
            'learning_rate': METHODS[method],
            'microbatch': 2,
            'accumulation': 16,
            'updates': STEPS,
            'warmup': 38,
            'max_length': 2048,
            'trainable': sum(p.numel() for p in params),
            'weight_decay': .01,
            'l2sp_lambda': .001 if ref is not None else 0,
            'precision': 'fp16 AMP, fp32 trainable parameters'
        }
    )

    model.train()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()

    for step in tqdm(range(start, STEPS), desc=method):
        optimizer.zero_grad(set_to_none=True)

        batches = [
            ds[step * 32 + j * 2:step * 32 + j * 2 + 2]
            for j in range(16)
        ]

        count = sum(
            sum(
                sum(v != -100 for v in labels[1:])
                for labels in batch['labels']
            )
            for batch in batches
        )

        total_sft = 0.

        for batch in batches:
            length = max(map(len, batch['input_ids']))

            ids = torch.tensor(
                [
                    x + [tok.pad_token_id] * (length - len(x))
                    for x in batch['input_ids']
                ],
                device='cuda'
            )

            labels = torch.tensor(
                [
                    x + [-100] * (length - len(x))
                    for x in batch['labels']
                ],
                device='cuda'
            )

            mask = torch.tensor(
                [
                    [1] * len(x) + [0] * (length - len(x))
                    for x in batch['input_ids']
                ],
                device='cuda'
            )

            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(
                    input_ids=ids,
                    attention_mask=mask
                ).logits

                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                    reduction='sum'
                ) / count

            total_sft += loss.detach().item()
            scaler.scale(loss).backward()

        penalty = 0.
        scaler.unscale_(optimizer)

        if ref is not None:
            for p, r in zip(params, ref):
                difference = p.detach() - r.to(p.device)
                penalty += .5 * difference.square().sum().item()

                if p.grad is not None:
                    p.grad.add_(difference, alpha=.001)

        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        lr = optimizer.param_groups[0]['lr']
        before = scaler.get_scale()

        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < before

        if not skipped:
            scheduler.step()

        append_jsonl(
            out / 'train.jsonl',
            {
                'step': step + 1,
                'examples_seen': (step + 1) * 32,
                'sft_loss': total_sft,
                'penalty': penalty,
                'weighted_penalty': .001 * penalty,
                'total_loss': total_sft + .001 * penalty,
                'lr': lr,
                'grad_norm': norm.item(),
                'skipped': skipped,
                'elapsed_seconds': time.perf_counter() - begin,
                'peak_gpu_gb': torch.cuda.max_memory_allocated() / 1e9
            }
        )

        if step + 1 in SAVE_STEPS:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'rng': rng_state(),
                'step': step + 1,
                'examples_seen': (step + 1) * 32
            }

            tmp = out / 'recovery.tmp'
            torch.save(state, tmp)
            os.replace(tmp, ck)
            del state

            save_json(
                out / 'checkpoint.json',
                {
                    'step': step + 1,
                    'bytes': ck.stat().st_size
                }
            )

            if method == 'full' and step + 1 == 625:
                save_json(
                    out / 'midpoint_pending.json',
                    {
                        'checkpoint': str(ck),
                        'instructions': 'Run evaluate.py full --stage midpoint; qualitative.py full --stage midpoint; diagnostics.py full --stage midpoint, then resume training.'
                    }
                )

                save_json(
                    out / 'resources_partial.json',
                    {
                        'elapsed_seconds': time.perf_counter() - begin,
                        'peak_gpu_gb': torch.cuda.max_memory_allocated() / 1e9
                    }
                )

                print(
                    'Midpoint reached. Run midpoint evaluation, qualitative and diagnostics, then resume with --resume.'
                )
                return

    save_json(
        out / 'resources.json',
        {
            'train_seconds_this_session': time.perf_counter() - begin,
            'peak_gpu_gb': torch.cuda.max_memory_allocated() / 1e9,
            'trainable_parameters': sum(p.numel() for p in params),
            'checkpoint_bytes': ck.stat().st_size,
            'reference_bytes': (
                sum(p.numel() * p.element_size() for p in ref)
                if ref else 0
            )
        }
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('method', choices=METHODS)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    run(a.method, a.resume)