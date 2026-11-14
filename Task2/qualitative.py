import random
import textwrap
from collections import defaultdict
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from config import MANIFEST, OUT, RAW, SEED
from data import image_name
from utils import read, save


def main():
    rng = random.Random(SEED)
    lookup = {(r["qid"], r["control"]):r for r in read(OUT / "predictions.json")}
    selected = []
    dest = OUT / "qualitative"
    dest.mkdir(exist_ok=True)
    with ZipFile(RAW / "val2014.zip") as z:
        for suite, pairs in read(MANIFEST)["test"].items():
            groups = defaultdict(list)
            for p in pairs:
                groups[p[0]["kind"]].append(p)
            for kind, pool in tqdm(sorted(groups.items()), desc=suite):
                p = rng.choice(sorted(pool, key=lambda p:p[0]["pair"]))
                fig, axs = plt.subplots(1, 2, figsize=(11, 6))
                fig.subplots_adjust(bottom=.29, top=.83)
                fig.suptitle(textwrap.fill(p[0]["question"], 76))
                for ax, r in zip(axs, p):
                    with z.open(image_name("val", r["image"])) as f:
                        ax.imshow(Image.open(f).convert("RGB"))
                    ax.axis("off")
                    correct, swap = (lookup[r["qid"], c] for c in ("correct", "swap"))
                    label = "\n".join(textwrap.fill(f"{k}: {v}", 50) for k,v in (
                        ("Reference", r["target"]), ("Correct image", correct["generation"] or "∅"),
                        ("Partner image", swap["generation"] or "∅")))
                    ax.text(.5, -.03, label, transform=ax.transAxes, va="top", ha="center", fontsize=10)
                    selected.append(dict(suite=suite, pair=r["pair"], qid=r["qid"], image=r["image"],
                                         question=r["question"], refs=r["refs"], correct=correct["generation"],
                                         swap=swap["generation"]))
                fig.savefig(dest / f"{suite}_{kind.replace('/','-')}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
    save(dest / "examples.json", dict(rule="Seed 6304, uniform draw per suite and type from pair-ID order; independent of predictions.",
                                       rows=selected))


if __name__ == "__main__":
    main()
