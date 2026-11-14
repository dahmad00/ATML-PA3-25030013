import argparse
import math
import random
import time
from collections import Counter
from importlib.metadata import version

import numpy as np
import torch
from tqdm import tqdm

import config as C
from data import Cache, Sampler
from evaluate import development
from models import VLM, restore, weights
from utils import batches, read, save


def lr_factor(i, total):
    warm = math.ceil(.05 * total)
    if i < warm:
        return (i + 1) / warm
    progress = min(1., (i + 1 - warm) / (total - warm))
    return .1 + .45 * (1 + math.cos(math.pi * progress))


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())


def set_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state(state[3])


def train(stage):
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    ck = torch.load(C.CKPT, map_location="cpu", weights_only=False) if C.CKPT.exists() else None
    prev = ck["stage"] if ck else 0
    if stage < prev or (stage == prev and ck["step"] == C.STEPS[stage-1]):
        print(f"Stage {stage} already completed.")
        return
    if stage > prev + 1 or (prev and stage > prev and ck["step"] != C.STEPS[prev-1]):
        raise ValueError("Complete the preceding stage first.")
    data, cache = read(C.MANIFEST), Cache()
    model = VLM()
    if stage >= 2:
        model.enable_lora()
    if ck:
        restore(model, ck["weights"])
    else:
        rows = random.Random(C.SEED).sample(data["train"]["caption"], 32)
        save(C.OUT / "calibration.json", model.calibrate(rows, cache))
    sampler = Sampler(data["train"], stage)
    rates = C.LR[stage-1]
    groups = [dict(params=model.connector.parameters(), lr=rates[0])]
    if stage >= 2:
        groups.append(dict(params=[p for p in model.lm.parameters() if p.requires_grad], lr=rates[1]))
    total = C.STEPS[stage-1]
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: lr_factor(i, total))
    scaler = torch.cuda.amp.GradScaler()
    history = ck["history"] if ck else []
    dev = ck["dev"] if ck else []
    step, elapsed, peak = 0, 0., 0
    if prev == stage:
        step, elapsed, peak = ck["step"], ck["elapsed"], ck["peak"]
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        sampler.rng.setstate(ck["sampler"])
        set_rng(ck["rng"])
    elif ck:
        set_rng(ck["rng"])
    del ck
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert all(n.startswith("connector.") or (stage >= 2 and "lora_" in n) for n in trainable)
    assert all(p.dtype == torch.float32 for p in trainable.values())
    assert all(p.dtype == torch.float16 for p in model.lm.parameters() if not p.requires_grad)
    train_cfg = dict(stage=stage, seed=C.SEED, model=C.LM, revision=model.lm.config._commit_hash,
                     gpu=torch.cuda.get_device_name(), updates=total, peak_lrs=rates,
                     batch=C.BATCH, accumulation=C.ACCUM, weight_decay=.01, clip=1., warmup=.05,
                     minimum_lr_fraction=.1, trainable={n:p.numel() for n,p in trainable.items()},
                     trainable_total=sum(p.numel() for p in trainable.values()),
                     versions={k:version(k) for k in ("torch","transformers","peft","numpy")})
    save(C.OUT / f"stage{stage}_config.json", train_cfg)
    start = time.perf_counter()
    if not dev:
        dev.append(dict(update=0, **development(model, cache, data["dev"], 0)))
    offset = sum(C.STEPS[:stage-1])
    bar = tqdm(range(step, total), initial=step, total=total, desc=f"Stage {stage}")
    for i in bar:
        model.train()
        batch = sampler.batch(i, C.BATCH * C.ACCUM)
        while True:
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.
            for micro in batches(batch, C.BATCH):
                feat = cache.get([r["image"] for r in micro])
                loss = model.losses(micro, feat, termination=True).mean() / C.ACCUM
                scaler.scale(loss).backward()
                loss_sum += loss.detach().item()
            scaler.unscale_(opt)
            grad = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.)
            scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= scale:
                break
            if scaler.get_scale() < 1e-8:
                raise FloatingPointError("Persistent nonfinite gradients")
        if i == 0:
            assert model.connector[-1].weight.grad.abs().sum().item() > 0
        step, global_step = i+1, offset+i+1
        history.append(dict(update=global_step, stage=stage, loss=loss_sum, grad_norm=grad.item(),
            lrs=sched.get_last_lr(), counts=dict(Counter(r["kind"] for r in batch))))
        sched.step()
        bar.set_postfix(loss=f"{loss_sum:.3f}")
        if global_step % 500 == 0 or step == total:
            dev.append(dict(update=global_step, **development(model, cache, data["dev"], global_step)))
        if global_step % 250 == 0 or step == total:
            seconds = elapsed + time.perf_counter() - start
            peak = max(peak, torch.cuda.max_memory_allocated())
            state = dict(stage=stage, step=step, weights=weights(model), optimizer=opt.state_dict(),
                         scheduler=sched.state_dict(), scaler=scaler.state_dict(), sampler=sampler.rng.getstate(),
                         rng=rng_state(), history=history, dev=dev, elapsed=seconds, peak=peak)
            tmp = C.CKPT.with_suffix(".tmp")
            torch.save(state, tmp)
            del state
            save(C.OUT / "training.json", history)
            save(C.OUT / "development.json", dev)
            save(C.OUT / f"stage{stage}_resources.json", dict(seconds=seconds, peak_bytes=peak, updates=step))
            tmp.replace(C.CKPT)
    bar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=(1, 2, 3))
    train(parser.parse_args().stage)
