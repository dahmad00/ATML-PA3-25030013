import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import ADAPT, FIG, FOLDERS, OFF, SPLITS
from data import pool
from scoring import features, mean
from utils import load


def corr(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def signals(ds):
    ids = load(SPLITS / f"{ds}.json")["eval"]
    points, comparisons, traces = [], [], []
    for q in tqdm(ids, desc=f"Diagnostics {ds}"):
        rows = pool(ds, q)["rows"]
        scores, cs = features(rows)
        length = np.array([r["n"] for r in rows])
        correct = np.array([int(r["correct"]) for r in rows])
        scores["length"] = length
        if cs:
            scores["calibrated"] = [mean(c) for c in cs]
        for name, values in scores.items():
            rank = np.argsort(values, kind="stable")
            top, bottom = correct[rank[-8:]].mean(), correct[rank[:8]].mean()
            comparisons.append(dict(q=q, score=name, top=top, bottom=bottom, gap=top-bottom,
                                    r_length=corr(values,length), r_correct=corr(values,correct)))
        for i, r in enumerate(rows):
            point = dict(q=q, rid=r["id"], answer=r["a"], correct=r["correct"], capped=r["capped"],
                         **{k:s[i] for k,s in scores.items()})
            if cs:
                c, v = cs[i], r["v"]
                point.update(r_conf_vision=corr(c,v), covariance=mean((c-mean(c))*(v-mean(v))),
                             fusion_gap=scores["tokenwise"][i]-scores["visconf"][i])
                if q in ids[:3] and i == 0:
                    traces.extend(dict(q=q, rid=i, token=t, confidence=a, vision=b, kl=k)
                                  for t,(a,b,k) in enumerate(zip(c,v,r["kl"])))
            points.append(point)
    df = pd.DataFrame(points)
    df.to_parquet(OFF / f"{ds}_scores.parquet", index=False)
    comp = pd.DataFrame(comparisons)
    comp.to_csv(FIG / f"{ds}_associations_per_question.csv", index=False)
    comp.groupby("score").agg({k:["mean","count"] for k in ["gap","top","bottom","r_length","r_correct"]}).to_csv(
        FIG / f"{ds}_associations.csv")
    if traces:
        qstats = df.groupby("q")[["r_conf_vision","covariance","fusion_gap"]].mean()
        qstats.to_csv(FIG / "vision_per_question.csv")
        qstats.describe().to_csv(FIG / "vision_summary.csv")
        trajectory = pd.DataFrame(traces)
        trajectory.to_csv(FIG / "trajectories.csv", index=False)
        fig, axes = plt.subplots(3, 1, figsize=(9,7), constrained_layout=True)
        for ax,(q,d) in zip(axes, trajectory.groupby("q", sort=False)):
            ax.plot(d.token, d.confidence, label="Calibrated confidence", alpha=.75)
            ax.plot(d.token, d.vision, label="Visual engagement")
            ax.set(title=f"MathVista {q}, rollout 0", ylabel="Score", xlabel="Generated token")
        axes[0].legend()
        fig.savefig(FIG / "trajectories.png", dpi=160)
        plt.close(fig)


def curves(ds):
    fixed = pd.read_csv(OFF / f"{ds}_summary.csv")
    adapt = pd.read_csv(ADAPT / f"{ds}_summary.csv")
    sc = fixed[fixed.method.eq("SC")]
    fig, axes = plt.subplots(1,2,figsize=(13,5), constrained_layout=True)
    for ax, x, label in zip(axes, ["n","tokens"], ["Available candidates","Mean retained tokens consumed"]):
        for name,d in fixed.groupby("method"):
            ax.plot(d[x], d.correct, ".-", label=name)
        ax.plot(sc[x], sc.oracle, "k--", label="Oracle coverage")
        if x == "tokens":
            for r in adapt.itertuples():
                ax.scatter(r.tokens, r.correct, marker="*", s=125, label=r.family)
        ax.set(xlabel=label, ylabel="Accuracy / coverage", title=ds)
    axes[1].legend(bbox_to_anchor=(1.02,1), loc="upper left", fontsize=8)
    fig.savefig(FIG / f"{ds}_accuracy.png", dpi=180)
    plt.close(fig)
    stops = pd.read_parquet(ADAPT / f"{ds}_decisions.parquet")
    stops.groupby(["family","n"]).agg(decisions=("q","size"), agreement=("fraction","mean"),
        full_correct_fraction=("full_fraction","mean"), accuracy=("correct","mean"),
        early_errors=("early_error","sum"), recovered_at_32=("recoverable","sum")).to_csv(FIG / f"{ds}_stops.csv")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True, parents=True)
    for dataset in FOLDERS:
        signals(dataset)
        curves(dataset)
