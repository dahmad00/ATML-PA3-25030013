from importlib.metadata import version
from time import perf_counter

import pandas as pd
from tqdm import tqdm

from config import LIVE, MODEL, REV, REVIEW, SAMPLING, SEED, SOLVE, SPLITS, SYSTEM
from data import pool
from scoring import MISSING, boxed, equivalent, groups, normalize, votes
from utils import bootstrap, load, save


def generate(model, tokenizer, messages, limit, seed):
    import torch
    from transformers import set_seed

    set_seed(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    n = inputs.input_ids.shape[1]
    with torch.inference_mode():
        tokens = model.generate(**inputs, **SAMPLING, max_new_tokens=limit,
                                pad_token_id=tokenizer.eos_token_id)[0, n:].tolist()
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    torch.cuda.synchronize()
    seconds = perf_counter() - start
    eos = model.generation_config.eos_token_id
    terminated = tokens[-1] in (eos if isinstance(eos, list) else [eos])
    return dict(prompt=prompt, messages=messages, text=text, seed=seed, limit=limit,
                raw=boxed(text), input_tokens=n, output_tokens=len(tokens),
                retained_tokens=len(tokens)-int(terminated), capped=len(tokens)==limit,
                truncated=len(tokens)==limit and not terminated, seconds=seconds,
                peak_mib=torch.cuda.max_memory_allocated()/2**20)


def strategies(q, calls, gold):
    def answer(key):
        return normalize(calls[key]["raw"])

    def correct(a):
        return a != MISSING and equivalent(gold, a, True)

    parallel = ["p1", "p2", "p3"]
    pick, _ = votes(groups([answer(k) for k in parallel], True), [1]*3)
    initial = correct(answer("p1"))
    for strategy, keys, selected in [("long", ["long"], "long"), ("parallel", parallel, parallel[pick]),
                                     ("refine", ["p1", "revision"], "revision")]:
        a = answer(selected)
        selected_calls = [calls[k] for k in keys]
        row = dict(q=q, strategy=strategy, selected=selected, answer=a, correct=correct(a),
                   initial_correct=initial, extracted=a != MISSING, calls=len(keys),
                   extracted_calls=sum(c["raw"] is not None for c in selected_calls),
                   truncated_calls=sum(c["truncated"] for c in selected_calls),
                   selected_truncated=calls[selected]["truncated"],
                   peak_mib=max(c["peak_mib"] for c in selected_calls))
        row.update({k: sum(c[k] for c in selected_calls)
                    for k in ("input_tokens", "output_tokens", "retained_tokens", "seconds")})
        yield row


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ids = load(SPLITS / "math.json")["live"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, torch_dtype=torch.float16).to("cuda").eval()
    manifest = dict(model=MODEL, revision=REV, sampling=SAMPLING, seed=SEED, dtype="float16",
                    gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                    transformers=version("transformers"), cuda=torch.version.cuda)
    run = LIVE / "settings.json"
    if run.exists() and load(run) != manifest:
        raise ValueError("Live run settings changed. Move the previous live results before restarting.")
    save(run, manifest)
    results = []
    for i, q in enumerate(tqdm(ids, desc="Live MATH")):
        problem = pool("math", q)
        path = LIVE / "calls" / (q.replace(":", "_") + ".json")
        calls = load(path) if path.exists() else {}
        base = [dict(role="system", content=SYSTEM), dict(role="user", content=problem["question"]+"\n\n"+SOLVE)]
        for j, (k, cap) in enumerate([("long",3072), ("p1",1024), ("p2",1024), ("p3",1024), ("revision",2048)]):
            if k in calls:
                continue
            messages = base
            if k == "revision":
                messages = base + [dict(role="assistant", content=calls["p1"]["text"]), dict(role="user", content=REVIEW)]
            calls[k] = generate(model, tokenizer, messages, cap, SEED + 5*i + j)
            save(path, calls)
        results.extend(strategies(q, calls, problem["gold"]))
    frame = pd.DataFrame(results)
    frame.to_parquet(LIVE / "questions.parquet", index=False)
    grouped = frame.groupby("strategy")
    summary = grouped[["correct","extracted","selected_truncated","input_tokens","output_tokens",
                       "retained_tokens","seconds","peak_mib"]].mean()
    summary["call_extraction_rate"] = grouped.extracted_calls.sum() / grouped.calls.sum()
    summary["call_truncation_rate"] = grouped.truncated_calls.sum() / grouped.calls.sum()
    summary.to_csv(LIVE / "summary.csv")
    revisions = frame[frame.strategy.eq("refine")]
    pd.crosstab(revisions.initial_correct, revisions.correct).reindex(
        index=[False,True], columns=[False,True], fill_value=0).to_csv(LIVE / "transitions.csv")
    acc = frame.pivot(index="q", columns="strategy", values="correct").astype(float)
    pd.DataFrame([dict(method=a, baseline=b, **bootstrap(acc[a]-acc[b]))
                  for a,b in [("refine","parallel"),("refine","long"),("parallel","long")]]).to_csv(
                      LIVE / "paired.csv", index=False)


if __name__ == "__main__":
    main()
