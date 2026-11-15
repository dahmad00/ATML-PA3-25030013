import pickle
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DATA, FOLDERS, OFF, ORDERS, SPLITS
from scoring import MISSING, equivalent, mean, normalize
from utils import load, save, seeded_rng

KEY = ["dataset", "sample_id", "rollout_id"]
DRIVE = "https://drive.usercontent.google.com/download?export=download&confirm=t&id="
INPUTS = {
    "qwen25-math/generations.parquet": (DRIVE + "1SaWxZ4mpycw2BWZ5G0dahoYZ6Hh8tPWP", "ffdecd753a2c0183b811d5679f3cf15e91bce46d1aacb196e28afb553d7e1c84"),
    "qwen25-math/probability_metrics.parquet": (DRIVE + "1RCYHrTnUKtF7gHLSTe92x6W9ZsCOekfQ", "48a492cd2c93bebfbd12b0b9df6a5783b611103e7a3cfefe010573ff6e200379"),
    "qwen25vl-mathvista/generations.parquet": (DRIVE + "1sedMpWLI_UgPvAQVj7tS3HYVH0ARS1yA", "dac12a7dc14baaa44e6c3ed8f8dfcf9d39b8493462c8a60ea162d0a66dda5435"),
    "qwen25vl-mathvista/probability_metrics.parquet": (DRIVE + "1f2XAu_J5NviYHEhtkRrFbKc7b2us1WiW", "c1c7a9b9724aa4930673c1bb5759d2d82bacc57f4a530a83f533e9d2d43102ac"),
    "qwen25vl-mathvista/attention_metrics.parquet": (DRIVE + "1Nzg2VwE2qFzA10BF1lTxEcBZCePBe51q", "4f6cfce99519808c75f950e53c957795697d8b503a7fd407fcfdd260f06d0058"),
    "testmini.json": ("https://raw.githubusercontent.com/lupantech/MathVista/main/data/testmini.json", "ab0148dca9d401c31cc47f29b3826eec28a76ff155a1c0878d81e8daf0413480"),
}


def ensure_inputs():
    for name, (url, expected) in INPUTS.items():
        path = DATA / name
        if path.is_file() and path.stat().st_size:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            partial = Path(tmp.name)
        try:
            digest = sha256()
            with urlopen(url, timeout=120) as src, partial.open("wb") as out, tqdm(
                desc=name, total=int(src.headers.get("Content-Length", 0)) or None,
                unit="B", unit_scale=True,
            ) as bar:
                while chunk := src.read(1024 * 1024):
                    out.write(chunk)
                    digest.update(chunk)
                    bar.update(len(chunk))
            if digest.hexdigest() != expected:
                raise ValueError(f"Downloaded {name} differs from the assignment release")
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)


def joined(folder, visual):
    def table(name, columns=None):
        return pd.read_parquet(folder / (name + ".parquet"), columns=columns).set_index(KEY, verify_integrity=True)

    tables = [table("generations"), table("probability_metrics", KEY + ["logp", "kl_u_p"])]
    if visual:
        tables.append(table("attention_metrics").rename(columns={"n_generated_text_tokens": "preceding"}))
    if any(len(tables[0].index.symmetric_difference(t.index)) for t in tables[1:]):
        raise ValueError(f"Unmatched rollout keys in {folder}")
    return pd.concat(tables, axis=1).reset_index().sort_values("rollout_id")


def cache_path(ds, qid):
    return OFF / "pools" / ds / (qid.replace(":", "_") + ".pkl")


def pool(ds, qid):
    with cache_path(ds, qid).open("rb") as f:
        return pickle.load(f)


def prepare(ds):
    visual = ds == "mathvista"
    metadata = load(DATA / "testmini.json") if visual else {}
    df = joined(DATA / FOLDERS[ds], visual)
    by_id = df.groupby("sample_id", sort=False)
    ids = sorted(by_id.groups)
    if len(ids) != 1000 or df.dataset.nunique() != 1:
        raise ValueError(f"Expected one dataset and 1,000 questions in {ds}")
    perm = seeded_rng().permutation(ids).tolist()
    split = dict(dev=perm[:200], eval=perm[200:])
    if not visual:
        split["live"] = perm[200:250]
    rng = seeded_rng()
    orders = {}
    for q in tqdm(ids, desc=f"Prepare {ds}"):
        frame = by_id.get_group(q)
        if frame.rollout_id.tolist() != list(range(32)):
            raise ValueError(f"Invalid rollout IDs for {q}")
        orders[q] = [rng.permutation(32).tolist() for _ in range(ORDERS)]
        meta = metadata[q] if visual else None
        first = frame.iloc[0]
        gold = normalize(first.gold_answer, meta)
        rows = []
        for r in frame.itertuples():
            n = int(r.n_generated_text_tokens)
            arrays = [r.generated_token_ids, r.logp, r.kl_u_p]
            if visual:
                arrays.extend([r.img_attn_total, r.prompt_text_attn_total, r.generated_text_attn_total, r.preceding])
            if any(len(a) != n for a in arrays):
                raise ValueError(f"Misaligned token lists: {ds}/{q}/{r.rollout_id}")
            raw = None if pd.isna(r.extracted_answer) else r.extracted_answer
            a = normalize(raw, meta)
            row = dict(id=int(r.rollout_id), n=n, raw=raw, a=a, text=r.generated_text,
                       capped=bool(r.hit_max_new_tokens), correct=a != MISSING and equivalent(gold, a, not visual),
                       logp=mean(r.logp), certainty=mean(r.kl_u_p))
            if visual:
                image = np.asarray(r.img_attn_total)
                total = image + r.prompt_text_attn_total + r.generated_text_attn_total
                if np.any(total <= 0):
                    raise ValueError(f"Undefined visual engagement: {q}/{r.rollout_id}")
                row.update(kl=np.asarray(r.kl_u_p), v=image / total)
            rows.append(row)
        path = cache_path(ds, q)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(dict(question=first.question_text, gold=gold, rows=rows), f, protocol=5)
    save(SPLITS / f"{ds}.json", split)
    save(SPLITS / f"{ds}_orders.json", orders)


if __name__ == "__main__":
    ensure_inputs()
    for dataset in FOLDERS:
        prepare(dataset)
    save(SPLITS / "versions.json", {k: version(k) for k in ("numpy", "pandas", "pyarrow", "math-verify")})
