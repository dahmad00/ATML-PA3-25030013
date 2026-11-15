from collections import Counter
from functools import lru_cache
import re

import numpy as np
from math_verify import parse, verify

MISSING = "NO_ANSWER"


def boxed(text):
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    start += 7
    depth = 1
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start:i].strip() or None
    return None


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, c in enumerate(a, 1):
        nxt = [i]
        for j, d in enumerate(b, 1):
            nxt.append(min(row[j] + 1, nxt[-1] + 1, row[j-1] + (c != d)))
        row = nxt
    return row[-1]


def normalize(value, meta=None):
    s = "" if value is None else str(value).strip()
    if not s or s == MISSING:
        return MISSING
    if meta is None:
        return s
    if meta["question_type"] == "multi_choice":
        choices = meta["choices"]
        labels = [chr(65 + i) for i in range(len(choices))]
        match = re.search(r"\(([a-zA-Z])\)", s)
        label = (match[1] if match else s).upper()
        if label in labels:
            return label
        return labels[min(range(len(choices)), key=lambda i: distance(s, str(choices[i])))]
    try:
        if meta["answer_type"] == "integer":
            return str(int(float(s)))
        if meta["answer_type"] == "float":
            return str(round(float(s), int(meta["precision"])))
    except (ValueError, OverflowError):
        return MISSING
    return s


@lru_cache(8192)
def parsed(s):
    return parse(r"\boxed{" + s + "}")


@lru_cache(65536)
def equivalent(a, b, math):
    if a == b:
        return True
    return math and MISSING not in (a, b) and bool(verify(parsed(a), parsed(b)))


def groups(answers, math):
    reps, labels = [], []
    for a in answers:
        group = next((b for b in reps if equivalent(b, a, math)), None)
        if group is None:
            reps.append(a)
            group = a
        labels.append(group)
    return labels


def ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def votes(labels, weights):
    tally = Counter()
    for a, w in zip(labels, weights):
        tally[a] += w
    ordered = tally.most_common()
    leader, first = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0
    return labels.index(leader), float((first - second) / sum(weights))


def rwc(labels, values):
    if len(labels) == 1:
        return 0, 1.
    # A common scale cancels from the margin. Doubled ranks keep exact vote ties.
    return votes(labels, 2 * (ranks(values) - 1))


def features(rows):
    scores = {key: np.array([r[key] for r in rows]) for key in ("logp", "certainty")}
    calibrated = []
    if "v" in rows[0]:
        pooled = np.concatenate([r["kl"] for r in rows])
        lo, hi = np.percentile(pooled, [5, 95]) if pooled.size else (0., 0.)
        calibrated = [np.clip((r["kl"] - lo) / max(hi - lo, 1e-8), 0, 1) for r in rows]
        scores["vision"] = np.array([mean(r["v"]) for r in rows])
        scores["visconf"] = scores["vision"] * [mean(c) for c in calibrated]
        scores["tokenwise"] = np.array([mean(c * r["v"]) for c, r in zip(calibrated, rows)])
    return scores, calibrated


def mean(x):
    return float(np.mean(x, dtype=np.float64)) if len(x) else 0.
