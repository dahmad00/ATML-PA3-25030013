import argparse
import csv
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import CONTROLS, FEATURES, MANIFEST, OUT, STEPS
from data import examples
from utils import read, save


def check():
    data = read(MANIFEST)
    groups = {"train": {r["image"] for r in examples(data["train"])}}
    for split in ("dev", "test"):
        for suite, items in data[split].items():
            rows = items if suite == "caption" else [r for pair in items for r in pair]
            groups[f"{split}_{suite}"] = {r["image"] for r in rows}
    for a, b in combinations(groups, 2):
        assert groups[a].isdisjoint(groups[b]), f"Image overlap: {a}, {b}"
    ids = read(OUT / "image_ids.json")
    array = np.load(FEATURES, mmap_mode="r")
    assert set(ids) == set.union(*groups.values())
    assert array.shape == (len(ids), 196, 768) and array.dtype == np.float16
    for suite in data["test"].values():
        for pair in suite:
            original = {r["image"] for r in pair}
            assert all(r["mismatch"] not in original for r in pair)
    result = dict(images={k:len(v) for k,v in groups.items()}, feature_shape=list(array.shape),
                  dtype=str(array.dtype), disjoint=True)
    save(OUT / "data_checks.json", result)
    print(result)


def plots():
    hist, dev = read(OUT / "training.json"), read(OUT / "development.json")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    for i in range(0, len(hist), 25):
        window = hist[i:i+25]
        axes[0].plot(window[-1]["update"], np.mean([r["loss"] for r in window]), ".", color="C0")
    x = [r["update"] for r in dev]
    axes[1].plot(x, [r["caption_correct"] for r in dev], ".-")
    for suite in ("grounding", "natural"):
        for c in ("correct", "swap", "none"):
            axes[2].plot(x, [r[f"{suite}_{c}"] for r in dev], ".-", label=f"{suite} {c}")
        axes[3].plot(x, [r[suite+"_penalty"] for r in dev], ".-", label=suite)
    for ax, title in zip(axes, ("Training loss (25-update mean)", "Development caption loss",
                                 "Development answer NLL", "Swap NLL minus correct NLL")):
        ax.set(xlabel="Optimizer update", ylabel="Nats per token", title=title)
        for boundary in np.cumsum(STEPS)[:-1]:
            ax.axvline(boundary, ls="--", color=".65", lw=1)
    axes[2].legend(fontsize=8)
    axes[3].legend()
    fig.savefig(OUT / "learning_curves.png", dpi=170)
    plt.close(fig)
    with (OUT / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axs, suite in tqdm(zip(axes, ("grounding", "natural")), total=2, desc="Final plots"):
        for ax, metric in zip(axs, ("accuracy", "pair_accuracy", "nll")):
            for j, c in enumerate(CONTROLS):
                points = [r for r in rows if (r["suite"],r["metric"],r["control"]) == (suite,metric,c)]
                x = np.arange(len(points))+(j-1.5)*.16
                y, lo, hi = np.array([[float(r[k]) for r in points] for k in ("value","low95","high95")])
                ax.vlines(x, lo, hi, color=f"C{j}")
                ax.scatter(x, y, label=c, s=16, color=f"C{j}")
            ax.set_xticks(np.arange(len(points)), [r["kind"] for r in points], rotation=20)
            ax.set(title=f"{suite} · {metric}", ylabel="Nats per token" if metric=="nll" else "Fraction")
    axes[0,0].legend(fontsize=8)
    fig.savefig(OUT / "final_results.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("check", "plots"))
    {"check": check, "plots": plots}[p.parse_args().action]()
