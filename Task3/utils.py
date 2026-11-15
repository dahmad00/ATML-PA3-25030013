import json
import logging
from pathlib import Path

import numpy as np

from config import BOOT, SEED

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("task3")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    tmp.replace(path)
    log.debug("Saved %s", path)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def seeded_rng():
    return np.random.default_rng(SEED)


def bootstrap(differences):
    d = np.asarray(differences, dtype=float)
    draws = seeded_rng().choice(d, size=(BOOT, len(d)))
    low, high = np.quantile(draws.mean(axis=1), [0.025, 0.975])
    return dict(delta=d.mean(), low=low, high=high)
