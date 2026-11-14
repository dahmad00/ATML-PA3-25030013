import argparse
import json
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from tqdm import tqdm

from config import FEATURES, MANIFEST, OUT, RAW, SEED, TYPES, VISION
from utils import batches, normalize, read, save

COCO = "https://s3.amazonaws.com/images.cocodataset.org"
VQA = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa"


def download():
    RAW.mkdir(exist_ok=True)
    urls = [f"{COCO}/annotations/annotations_trainval2014.zip"]
    for season in ("Train", "Val"):
        urls.append(f"{COCO}/zips/{season.lower()}2014.zip")
        for kind in ("Annotations", "Questions", "Complementary_Pairs"):
            urls.append(f"{VQA}/v2_{kind}_{season}_mscoco.zip")
    for url in tqdm(urls, desc="Archives"):
        dest = RAW / Path(url).name
        if dest.exists():
            continue
        temp = dest.with_suffix(".part")
        with urlopen(url, timeout=180) as src, temp.open("wb") as dst:
            with tqdm.wrapattr(dst, "write", total=int(src.headers.get("Content-Length", 0)),
                                 desc=dest.name, unit="B", unit_scale=True) as sink:
                shutil.copyfileobj(src, sink, 1024 * 1024)
        temp.replace(dest)


def metadata(archive, suffix=".json"):
    with ZipFile(RAW / archive) as z:
        name = next(n for n in z.namelist() if n.endswith(suffix) and "__MACOSX" not in n)
        with z.open(name) as f:
            return json.load(f)


def image_name(season, image):
    return f"{season}2014/COCO_{season}2014_{image:012d}.jpg"


def natural(season, targets, rng, used):
    def load(kind):
        return metadata(f"v2_{kind}_{season.title()}_mscoco.zip")

    questions = {r["question_id"]: r["question"] for r in load("Questions")["questions"]}
    annotations = {r["question_id"]: r for r in load("Annotations")["annotations"]}
    pairs = load("Complementary_Pairs")
    rng.shuffle(pairs)
    result = []
    remaining = dict(zip(TYPES, targets))
    for a, b in tqdm(pairs, desc=f"{season} natural pairs", leave=False):
        left, right = annotations[a], annotations[b]
        kind = left["answer_type"]
        ids = [r["image_id"] for r in (left, right)]
        if (not remaining[kind] or right["answer_type"] != kind or ids[0] == ids[1]
                or used.intersection(ids) or questions[a] != questions[b]):
            continue
        rows = []
        for qid, r in ((a, left), (b, right)):
            refs = [x["answer"] for x in r["answers"]]
            ans, count = Counter(map(normalize, refs)).most_common(1)[0]
            if len(refs) != 10 or count < 8 or not ans:
                break
            rows.append(dict(image=r["image_id"], question=questions[a], qid=qid,
                             pair=f"vqa-{a}-{b}", kind=kind, refs=refs,
                             target=next(x for x in refs if normalize(x) == ans)))
        if len(rows) == 2 and normalize(rows[0]["target"]) != normalize(rows[1]["target"]):
            result.append(rows)
            remaining[kind] -= 1
            used.update(ids)
            if not any(remaining.values()):
                break
    if any(remaining.values()):
        raise ValueError(f"Missing natural pairs: {remaining}")
    return result


def captions(season, count, per_image, rng, used):
    rows = metadata("annotations_trainval2014.zip", f"captions_{season}2014.json")
    by_image = defaultdict(list)
    for r in rows["annotations"]:
        by_image[r["image_id"]].append(r)
    ids = rng.sample(sorted(i for i, v in by_image.items() if i not in used and len(v) >= per_image), count)
    used.update(ids)
    return [dict(image=i, qid=r["id"], kind="caption", question="Describe this image.",
                 target=r["caption"]) for i in ids for r in rng.sample(by_image[i], per_image)]


PLURAL = {"person": "people", "mouse": "mice", "sheep": "sheep", "skis": "skis",
          "scissors": "scissors", "sports ball": "sports balls", "knife": "knives",
          "bus": "buses", "sandwich": "sandwiches", "toothbrush": "toothbrushes",
          "bench": "benches", "couch": "couches", "wine glass": "wine glasses",
          "broccoli": "pieces of broccoli"}


def grounding(season, sizes, rng, used, allowed=None):
    source = metadata("annotations_trainval2014.zip", f"instances_{season}2014.json")
    counts = defaultdict(Counter)
    crowds = defaultdict(set)
    for obj in source["annotations"]:
        i, c = obj["image_id"], obj["category_id"]
        counts[i][c] += 1
        if obj["iscrowd"]:
            crowds[c].add(i)
    ids = sorted(allowed if allowed is not None else (r["id"] for r in source["images"]))
    pools = {}
    for cat in source["categories"]:
        c = cat["id"]
        bins = defaultdict(list)
        for i in ids:
            n = counts[i][c]
            if i not in used and (n == 0 or i not in crowds[c]):
                bins[min(5, n)].append(i)
        pools[cat["name"]] = bins
    result = []
    for kind, size in zip(("presence", "counting"), sizes):
        for k in tqdm(range(size), desc=f"{season} {kind}", leave=False):
            while True:
                name = rng.choice(list(pools))
                bins = pools[name]
                positive = [n for n in bins if n and bins[n]]
                small = [n for n in positive if n <= 4]
                if kind == "presence" and bins[0] and positive:
                    a, b = rng.choice(positive), 0
                    q, targets = f"Does this image contain {name}?", ("yes", "no")
                elif kind == "counting" and len(small) >= 2:
                    a, b = rng.sample(small, 2)
                    plural = PLURAL.get(name, name[:-1] + "ies" if name.endswith("y") and name[-2] not in "aeiou" else name + "s")
                    q, targets = f"How many {plural} can you see?", (str(a), str(b))
                else:
                    continue
                images = [rng.choice(bins[n]) for n in (a, b)]
                if allowed is not None or not used.intersection(images):
                    break
            pair = f"{season}-{kind}-{images[0]}-{images[1]}-{k}"
            result.append([dict(image=i, question=q, target=t, refs=[t], kind=kind,
                                pair=pair, qid=f"{pair}-{side}")
                           for side, (i, t) in enumerate(zip(images, targets))])
            if allowed is None:
                used.update(images)
    return result


def prepare():
    rng = random.Random(SEED)
    m = {s: {} for s in ("train", "dev", "test")}
    train, held = set(), set()
    for split, quotas in zip(m, [(640, 256, 640), (12, 8, 12), (36, 24, 36)]):
        m[split]["natural"] = natural("train" if split == "train" else "val", quotas, rng,
                                      train if split == "train" else held)
    m["train"]["caption"] = captions("train", 6000, 3, rng, set())
    m["dev"]["caption"] = captions("val", 96, 1, rng, held)
    ids = {r["image"] for r in m["train"]["caption"]}
    m["train"]["grounding"] = grounding("train", (2500, 2500), rng, set(), ids)
    for split, size in (("dev", 12), ("test", 32)):
        m[split]["grounding"] = grounding("val", (size, size), rng, held)
    for suite in ("grounding", "natural"):
        pairs = m["test"][suite]
        offset = rng.randrange(1, len(pairs))
        for j, pair in enumerate(pairs):
            other = pairs[(j + offset) % len(pairs)]
            for side, row in enumerate(pair):
                row["mismatch"] = other[side]["image"]
    save(MANIFEST, m)


def examples(split):
    return [*split.get("caption", []), *(r for s in ("grounding", "natural") for pair in split[s] for r in pair)]


class Sampler:
    def __init__(self, split, stage):
        self.stage = stage
        self.rng = random.Random(SEED)
        self.caps = split["caption"]
        self.pools = {k: [p for p in split["grounding"] if p[0]["kind"] == k]
                      for k in ("presence", "counting")}
        self.pools["natural"] = split["natural"]
        self.order = []
        if stage == 1:
            for _ in range(2):
                self.order += self.rng.sample(range(len(self.caps)), len(self.caps))

    def batch(self, update, size):
        if self.stage == 1:
            start = update * size
            return [self.caps[i] for i in self.order[start:start + size]]
        weights = (0, .5, .25, .25) if self.stage == 2 else (.7, .1, .1, .1)
        rows = []
        for k in self.rng.choices(("natural", "presence", "counting", "caption"), weights, k=size // 2):
            rows.extend(self.rng.choices(self.caps, k=2) if k == "caption" else self.rng.choice(self.pools[k]))
        return rows


class Cache:
    def __init__(self):
        import numpy as np
        self.array = np.load(FEATURES, mmap_mode="r")
        self.index = {i: n for n, i in enumerate(read(OUT / "image_ids.json"))}

    def get(self, ids):
        import torch
        return torch.from_numpy(self.array[[self.index[i] for i in ids]]).cuda()


def cache():
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, SiglipVisionModel

    start = time.perf_counter()
    m = read(MANIFEST)
    images = {r["image"]: ("train" if s == "train" else "val") for s in m for r in examples(m[s])}
    ids = sorted(images)
    proc = AutoImageProcessor.from_pretrained(VISION, use_fast=False)
    model = SiglipVisionModel.from_pretrained(VISION, torch_dtype=torch.float16).cuda().eval().requires_grad_(False)
    temp = FEATURES.with_suffix(".partial.npy")
    array = np.lib.format.open_memmap(temp, mode="w+", dtype="float16", shape=(len(ids), 196, 768))
    with ZipFile(RAW / "train2014.zip") as train, ZipFile(RAW / "val2014.zip") as val, torch.inference_mode():
        for n, group in enumerate(tqdm(batches(ids, 32), total=(len(ids)+31)//32, desc="Cache SigLIP")):
            imgs = []
            for i in group:
                season = images[i]
                with (train if season == "train" else val).open(image_name(season, i)) as f:
                    imgs.append(Image.open(f).convert("RGB"))
            pixels = proc(images=imgs, return_tensors="pt").pixel_values.cuda().half()
            features = model(pixels).last_hidden_state
            assert features.shape[1:] == (196, 768)
            array[n*32:n*32+len(group)] = features.cpu().numpy()
    del array
    temp.replace(FEATURES)
    save(OUT / "image_ids.json", ids)
    save(OUT / "cache_resources.json", dict(images=len(ids), seconds=time.perf_counter()-start,
         peak_bytes=torch.cuda.max_memory_allocated(), gpu=torch.cuda.get_device_name(), model=VISION,
         revision=model.config._commit_hash))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["download", "prepare", "cache"])
    {"download": download, "prepare": prepare, "cache": cache}[parser.parse_args().action]()
