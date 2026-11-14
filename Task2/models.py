import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import LM

SHORT = "\nAnswer with a short answer."


class VLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(LM)
        self.lm = AutoModelForCausalLM.from_pretrained(LM, torch_dtype=torch.float16).cuda()
        self.lm.requires_grad_(False)
        self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.connector = nn.Sequential(nn.Linear(768, 1024), nn.GELU(), nn.Linear(1024, 1024)).cuda()
        template = self.tok.apply_chat_template([{"role": "user", "content": "{QUESTION}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        before, after = template.split("{QUESTION}")
        self.before, self.after = self.ids(before), self.ids(after)
        self.end = self.tok.convert_tokens_to_ids("<|im_end|>")

    def ids(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def embed(self, ids):
        return self.lm.get_input_embeddings()(torch.tensor(ids, device="cuda", dtype=torch.long))

    def enable_lora(self):
        config = LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, bias="none",
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                            task_type="CAUSAL_LM")
        self.lm = get_peft_model(self.lm, config)
        for p in self.lm.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    @torch.no_grad()
    def calibrate(self, rows, cache):
        text = self.embed([t for r in rows for t in self.ids(r["target"])[:64]])
        visual = self.connector(cache.get([r["image"] for r in rows]).float())
        a, b = (v.float().square().mean().sqrt().item() for v in (text, visual))
        factor = a / b
        self.connector[-1].weight.mul_(factor)
        self.connector[-1].bias.mul_(factor)
        return dict(rule="Scale final linear weight and bias by caption-token RMS / visual RMS",
                    qids=[r["qid"] for r in rows], text_rms=a, visual_rms=b, factor=factor)

    def inputs(self, rows, features, targets):
        visual = self.connector(features.float()).half() if features is not None else None
        seqs, labels = [], []
        for j, r in enumerate(rows):
            caption = r["kind"] == "caption"
            q = self.ids(r["question"])[:80] + ([] if caption else self.ids(SHORT))
            parts = [self.embed(self.before)]
            if visual is not None:
                parts.append(visual[j])
            parts.append(self.embed(q + self.after))
            prefix = torch.cat(parts)
            label = torch.full((len(prefix),), -100, device="cuda")
            if targets:
                ans = self.ids(r["target"])[:64 if caption else 16] + [self.end]
                prefix = torch.cat([prefix, self.embed(ans)])
                label = torch.cat([label, torch.tensor(ans, device="cuda")])
            seqs.append(prefix)
            labels.append(label)
        def pad(items, value=0):
            if targets:
                return pad_sequence(items, batch_first=True, padding_value=value)
            reversed_items = [item.flip([0]) for item in items]
            return pad_sequence(reversed_items, batch_first=True, padding_value=value).flip([1])

        mask = pad([torch.ones(len(s), device="cuda", dtype=torch.long) for s in seqs])
        return pad(seqs), mask, pad(labels, -100)

    def losses(self, rows, features, termination):
        with torch.autocast("cuda", dtype=torch.float16):
            x, mask, labels = self.inputs(rows, features, True)
            base = self.lm.get_base_model() if hasattr(self.lm, "peft_config") else self.lm
            hidden = base.model(inputs_embeds=x, attention_mask=mask,
                                position_ids=mask.cumsum(1).sub(1).clamp_min(0), use_cache=False).last_hidden_state
            labels = labels[:, 1:]
            keep = labels.ne(-100)
            if not termination:
                keep &= labels.ne(self.end)
            logits = base.lm_head(hidden[:, :-1][keep])
            loss = F.cross_entropy(logits.float(), labels[keep], reduction="none")
            tokens = torch.zeros_like(labels, dtype=torch.float).masked_scatter(keep, loss)
            return tokens.sum(1) / keep.sum(1)

    @torch.inference_mode()
    def generate(self, rows, features):
        with torch.autocast("cuda", dtype=torch.float16):
            x, mask, _ = self.inputs(rows, features, False)
            base = self.lm.get_base_model() if hasattr(self.lm, "peft_config") else self.lm
            ids = base.generate(inputs_embeds=x, attention_mask=mask, max_new_tokens=16,
                                   do_sample=False, use_cache=True, eos_token_id=self.end,
                                   pad_token_id=self.tok.pad_token_id)
        return self.tok.batch_decode(ids, skip_special_tokens=True)


def weights(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def restore(model, state):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in state.items():
            params[n].copy_(v)
