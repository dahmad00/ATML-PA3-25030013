import pandas as pd
from tqdm import tqdm

from config import ADAPT, FIG, FOLDERS, LIVE, OFF, SPLITS
from data import pool
from utils import load, save


def example(ds, q, order, n, method_picks):
    problem = pool(ds, q)
    order_ids = load(SPLITS / f"{ds}_orders.json")[q][order]
    return dict(dataset=ds, q=q, question=problem["question"], gold=problem["gold"], order=order,
                consumed=n, rollout_order=order_ids, selected=method_picks,
                candidates=[{k:r[k] for k in ("id","raw","a","correct","n","logp","certainty","text")}
                            for r in (problem["rows"][i] for i in order_ids)])


def offline(ds):
    frame = pd.read_parquet(OFF / f"{ds}_eval.parquet")
    scores = pd.read_parquet(OFF / f"{ds}_scores.parquet")
    name = "certainty_RWC" if ds == "math" else "visconf_RWC"
    part = frame[frame.n.eq(32) & frame.order.eq(0)].set_index(["q","method"])
    a, b = [part.xs(k, level="method") for k in ["SC", name]]
    out = {}
    for label, mask in [("quality_recovers", ~a.correct & b.correct), ("consensus_recovers", a.correct & ~b.correct)]:
        examples = []
        for q in sorted(a.index[mask])[:1]:
            e = example(ds,q,0,32,{"SC":int(a.loc[q,"rid"]), name:int(b.loc[q,"rid"])})
            for r in e["candidates"]:
                s = scores[(scores.q==q) & (scores.rid==r["id"])].iloc[0]
                r["quality"] = float(s[name.removesuffix("_RWC")])
            examples.append(e)
        out[label] = examples
    stops = pd.read_parquet(ADAPT / f"{ds}_decisions.parquet")
    out["premature"] = []
    for r in stops[stops.recoverable].sort_values(["q","order","family"]).head(1).itertuples():
        fixed = frame[(frame.q==r.q) & (frame.order==r.order) & frame.n.eq(32) & frame.method.eq("SC")].iloc[0]
        e = example(ds,r.q,r.order,r.n,{r.family:int(r.rid),"fixed32_SC":int(fixed.rid)})
        e.update(fraction=r.fraction, margin=r.margin)
        out["premature"].append(e)
    return out


if __name__ == "__main__":
    cases = {ds:offline(ds) for ds in tqdm(FOLDERS, desc="Selection and stopping examples")}
    d = pd.read_parquet(LIVE / "questions.parquet")
    d = d[d.strategy.eq("refine")]
    for initial, final in [(False,True),(True,False),(True,True),(False,False)]:
        examples = []
        for q in d.loc[d.initial_correct.eq(initial) & d.correct.eq(final),"q"].sort_values().head(1):
            p = pool("math", q)
            calls = load(LIVE / "calls" / (q.replace(":","_")+".json"))
            examples.append(dict(q=q, question=p["question"], gold=p["gold"],
                                 initial=calls["p1"], revision=calls["revision"]))
        cases[f"revision_{int(initial)}_to_{int(final)}"] = examples
    save(FIG / "examples.json", cases)
