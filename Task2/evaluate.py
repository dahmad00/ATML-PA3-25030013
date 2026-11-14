import csv
import time
from collections import defaultdict
from statistics import mean

import numpy as np
from tqdm import tqdm

from config import BATCH, CKPT, CONTROLS, MANIFEST, OUT, SEED, STEPS
from data import Cache
from utils import batches, normalize, read, save, soft_score


def measure(model, cache, split, final):
    import torch
    with torch.inference_mode():
        return _measure(model, cache, split, final)


def _measure(model, cache, split, final):
    import torch
    model.eval()
    records = []
    for suite, items in split.items():
        caption = suite == "caption"
        rows = items if caption else [r for pair in items for r in pair]
        partners = {} if caption else {r["qid"]: p[1-j] for p in items for j, r in enumerate(p)}
        controls = CONTROLS if final else (("correct",) if caption else ("correct", "swap", "none"))
        for control in controls:
            for batch in tqdm(batches(rows, BATCH), total=(len(rows)+BATCH-1)//BATCH,
                              desc=f"{suite} {control}", leave=False):
                ids = [partners[r["qid"]]["image"] if control == "swap" else
                       r["mismatch"] if control == "mismatch" else r["image"] for r in batch]
                features = None if control == "none" else cache.get(ids)
                nll = model.losses(batch, features, termination=caption).tolist()
                if final:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    outputs = model.generate(batch, features)
                    torch.cuda.synchronize()
                    seconds = (time.perf_counter() - start) / len(batch)
                for j, r in enumerate(batch):
                    row = dict(r, control=control, suite=suite, nll=nll[j],
                               input_image=None if features is None else ids[j])
                    if final:
                        pred = normalize(outputs[j])
                        exact = bool(pred) and pred == normalize(r["target"])
                        partner = partners[r["qid"]]
                        row.update(generation=outputs[j], prediction=pred, exact=exact,
                                   accuracy=(soft_score(pred, r["refs"]) if pred and suite == "natural" else float(exact)),
                                   partner_target=partner["target"], follows_partner=bool(pred) and pred == normalize(partner["target"]),
                                   seconds=seconds)
                    records.append(row)
    return records


def development(model, cache, split, step):
    rows = measure(model, cache, split, False)
    save(OUT / "development" / f"{step:04}.json", rows)
    groups = defaultdict(list)
    for r in rows:
        groups[f"{r['suite']}_{r['control']}"].append(r["nll"])
    metrics = {k: mean(v) for k, v in groups.items()}
    for suite in ("grounding", "natural"):
        metrics[suite + "_penalty"] = metrics[suite + "_swap"] - metrics[suite + "_correct"]
    return metrics


def pair_metrics(pair):
    a, b = pair
    return dict(accuracy=mean(r["accuracy"] for r in pair), pair_accuracy=float(a["exact"] and b["exact"]),
                identical=float(a["prediction"] == b["prediction"]),
                empty=mean(not r["prediction"] for r in pair), nll=mean(r["nll"] for r in pair),
                seconds=mean(r["seconds"] for r in pair), follows_partner=mean(r["follows_partner"] for r in pair))


def table(rows):
    result = []
    for suite in sorted({r["suite"] for r in rows}):
        selected = [r for r in rows if r["suite"] == suite]
        original = {r["qid"]:r["prediction"] for r in selected if r["control"] == "correct"}
        for kind in ["all", *sorted({r["kind"] for r in selected})]:
            grouped = defaultdict(lambda: defaultdict(list))
            for r in selected:
                if kind == "all" or kind == r["kind"]:
                    grouped[r["control"]][r["pair"]].append(r)
            ids = sorted(grouped["correct"])
            ix = np.random.default_rng(SEED).integers(len(ids), size=(4000, len(ids)))
            values = {}
            def record(control, metric, v):
                lo, hi = np.percentile(v[ix].mean(1), [2.5, 97.5])
                result.append(dict(suite=suite, kind=kind, control=control, metric=metric,
                                   pairs=len(ids), value=v.mean(), low95=lo, high95=hi))
            for control in CONTROLS:
                per_pair = [pair_metrics(grouped[control][i]) for i in ids]
                values[control] = {k: np.array([p[k] for p in per_pair]) for k in per_pair[0]}
                for k, v in values[control].items():
                    record(control, k, v)
                if control == "swap":
                    for follows in (True, False):
                        rates = [mean(r["prediction"] != original[r["qid"]] and r["follows_partner"] == follows
                                      for r in grouped[control][i]) for i in ids]
                        record(control, "changed_to_partner" if follows else "changed_to_noncanonical", np.array(rates))
            for k, sign in (("accuracy", 1), ("nll", -1)):
                record("correct_advantage", k, sign*(values["correct"][k]-values["swap"][k]))
    with (OUT / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, result[0])
        writer.writeheader()
        writer.writerows(result)


def main():
    import torch
    from models import VLM, restore
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    if ck["stage"] != 3 or ck["step"] != STEPS[-1]:
        raise ValueError("Final evaluation requires the completed stage 3 checkpoint.")
    model = VLM()
    model.enable_lora()
    restore(model, ck["weights"])
    start = time.perf_counter()
    rows = measure(model, Cache(), read(MANIFEST)["test"], True)
    save(OUT / "predictions.json", rows)
    table(rows)
    save(OUT / "evaluation_resources.json", dict(seconds=time.perf_counter()-start,
         generation_seconds=sum(r["seconds"] for r in rows), peak_bytes=torch.cuda.max_memory_allocated()))


if __name__ == "__main__":
    main()
