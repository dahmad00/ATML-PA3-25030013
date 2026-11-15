import argparse
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import ADAPT, BUDGETS, FOLDERS, GAMMAS, OFF, SPLITS, TAUS
from data import pool
from scoring import MISSING, features, groups, rwc, votes
from utils import bootstrap, load, save


def prefix(rows, math):
    labels = groups([r["a"] for r in rows], math)
    n = len(rows)
    sc, vote_margin = votes(labels, [1] * n)
    scores, _ = features(rows)
    picks = {"FirstAns": 0, "SC": sc, "shortest": min(range(n), key=lambda i: rows[i]["n"])}
    quality = "certainty" if math else "visconf"
    for name, values in scores.items():
        picks[name] = int(np.argmax(values))
        if name != "logp":
            winner, margin = rwc(labels, values)
            picks[name + "_RWC"] = winner
            if name == quality:
                quality_margin, same = margin, labels[sc] == labels[winner]
    stats = dict(unique=vote_margin > 0, fraction=labels.count(labels[sc]) / n,
                 valid=labels[sc] != MISSING, same=same, margin=quality_margin,
                 distinct=len(set(labels)), oracle=any(r["correct"] for r in rows))
    return picks, stats


def fixed(ds, split):
    ids = load(SPLITS / f"{ds}.json")[split]
    orders = load(SPLITS / f"{ds}_orders.json")
    records, traces = [], []
    for q in tqdm(ids, desc=f"Fixed {ds}/{split}"):
        rows = pool(ds, q)["rows"]
        full_fraction = sum(r["correct"] for r in rows) / len(rows)
        for order, permutation in enumerate(orders[q]):
            for n in BUDGETS:
                seen = [rows[i] for i in permutation[:n]]
                picks, stats = prefix(seen, ds == "math")
                cost = sum(r["n"] for r in seen)
                for method, i in picks.items():
                    r = seen[i]
                    first = method == "FirstAns"
                    records.append((q, order, n, method, r["id"], r["a"], r["correct"],
                                    seen[0]["n"] if first else cost, 1 if first else n,
                                    stats["oracle"], stats["distinct"]))
                sc = seen[picks["SC"]]
                traces.append(dict(q=q, order=order, n=n, **stats, tokens=cost, rid=sc["id"],
                                   answer=sc["a"], correct=sc["correct"], full_fraction=full_fraction))
    columns = "q order n method rid answer correct tokens consumed oracle distinct".split()
    frame = pd.DataFrame.from_records(records, columns=columns)
    frame.to_parquet(OFF / f"{ds}_{split}.parquet", index=False)
    pd.DataFrame(traces).to_parquet(OFF / f"{ds}_{split}_prefixes.parquet", index=False)
    return frame


def summarize(ds, frame):
    frame["sampling_failure"] = ~frame.oracle
    frame["selection_failure"] = frame.oracle & ~frame.correct
    metrics = ["correct", "tokens", "consumed", "oracle", "distinct", "sampling_failure", "selection_failure"]
    qmeans = frame.groupby(["q", "n", "method"])[metrics].mean()
    qmeans.to_parquet(OFF / f"{ds}_questions.parquet")
    qmeans.groupby(["n", "method"]).mean().to_csv(OFF / f"{ds}_summary.csv")
    intervals = []
    for n, part in qmeans.reset_index().groupby("n"):
        acc = part.pivot(index="q", columns="method", values="correct")
        for a, b in product(acc, acc):
            if a < b:
                intervals.append(dict(n=n, method=a, baseline=b, **bootstrap(acc[a] - acc[b])))
    pd.DataFrame(intervals).to_csv(OFF / f"{ds}_paired.csv", index=False)


def decide(prefixes, family, tau, gamma):
    eligible = prefixes.n.eq(32)
    if tau is not None:
        stop = prefixes.unique & prefixes.valid & prefixes.fraction.ge(tau)
        if family == "quality":
            stop &= prefixes.same & prefixes.margin.ge(gamma)
        eligible |= prefixes.n.isin([4, 8, 16]) & stop
    return prefixes.loc[eligible].drop_duplicates(["q", "order"])


def tune(ds, dev):
    chosen, grid = {}, []
    baseline = dev.loc[dev.n.eq(32), "correct"].mean()
    for family in ("agreement", "quality"):
        settings = [(None, None), *product(TAUS, GAMMAS if family == "quality" else [0])]
        candidates = []
        for tau, gamma in tqdm(settings, desc=f"Tune {ds}/{family}"):
            stop = decide(dev, family, tau, gamma)
            acc, cost = stop.correct.mean(), stop.tokens.mean()
            grid.append(dict(family=family, tau=tau, gamma=gamma, accuracy=acc, tokens=cost,
                             baseline=baseline, eligible=acc >= baseline - .01 - 1e-12))
            if grid[-1]["eligible"]:
                candidates.append((cost, -acc, tau if tau is not None else 2, gamma or 0, tau, gamma))
        chosen[family] = min(candidates)[-2:]
    pd.DataFrame(grid).to_csv(ADAPT / f"{ds}_dev_grid.csv", index=False)
    save(ADAPT / f"{ds}_thresholds.json", chosen)


def adaptive(ds):
    def traces(split):
        return pd.read_parquet(OFF / f"{ds}_{split}_prefixes.parquet").sort_values(["q", "order", "n"])

    ADAPT.mkdir(exist_ok=True, parents=True)
    tune(ds, traces("dev"))
    choices = load(ADAPT / f"{ds}_thresholds.json")
    held = traces("eval")
    base = held[held.n.eq(32)][["q", "order", "correct"]].rename(columns={"correct": "fixed32_correct"})
    stops = []
    for family, params in choices.items():
        d = decide(held, family, *params).merge(base, on=["q", "order"])
        d["family"] = family
        d["early_error"] = d.n.lt(32) & ~d.correct
        d["recoverable"] = d.early_error & d.fixed32_correct
        stops.append(d)
    result = pd.concat(stops)
    result.to_parquet(ADAPT / f"{ds}_decisions.parquet", index=False)
    metrics = ["correct", "tokens", "n", "early_error", "recoverable"]
    qmeans = result.groupby(["family", "q"])[metrics].mean()
    qmeans.groupby("family").mean().to_csv(ADAPT / f"{ds}_summary.csv")
    counts = result.groupby(["family", "n"]).size().unstack(fill_value=0).reindex(columns=[4,8,16,32], fill_value=0)
    counts.to_csv(ADAPT / f"{ds}_stop_counts.csv")
    comparisons = []
    for family in choices:
        a = qmeans.loc[family]
        for n, d in held.groupby("n"):
            b = d.groupby("q")[["correct", "tokens"]].mean()
            comparisons.append(dict(family=family, n=n, tokens_delta=(a.tokens-b.tokens).mean(),
                                    **bootstrap(a.correct-b.correct)))
    comparison = pd.DataFrame(comparisons)
    comparison["nearest_cost"] = comparison.tokens_delta.abs().eq(
        comparison.tokens_delta.abs().groupby(comparison.family).transform("min"))
    comparison.to_csv(ADAPT / f"{ds}_paired.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fixed", "adaptive"])
    args = parser.parse_args()
    for ds in FOLDERS:
        if args.stage == "fixed":
            for split in ("dev", "eval"):
                result = fixed(ds, split)
                if split == "eval":
                    summarize(ds, result)
        else:
            adaptive(ds)
