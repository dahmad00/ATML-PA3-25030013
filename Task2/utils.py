import json
import re
from functools import lru_cache
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf8")
    temp.replace(path)


def batches(rows, n):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


NUMBERS = dict(zip("none zero one two three four five six seven eight nine ten".split(),
                   "0 0 1 2 3 4 5 6 7 8 9 10".split()))
CONTRACTIONS = {word.replace("'", ""): word for word in """
ain't aren't can't could've couldn't couldn't've didn't doesn't don't hadn't hadn't've
hasn't haven't he'd he'd've he'll he's how'd how'll how's i'd i'd've i'll i'm i've
isn't it'd it'd've it'll it's might've mightn't mightn't've must've mustn't mustn't've
needn't not've oughtn't shan't she'd she'd've she'll she's should've shouldn't
shouldn't've somebody'd somebody'd've somebody'll somebody's someone'd someone'd've
someone'll someone's something'd something'd've something'll something's
that'd that'd've that'll that's there'd there'd've there're there's they'd they'd've
they'll they're they've wasn't we'd we'd've we'll we're we've weren't what'll what're
what's what've when's where'd where's where've who'd who'd've who'll who're who's
who've why's won't would've wouldn't wouldn't've y'all y'all'd y'all'd've y'all're
y'all've you'd you'd've you'll you're you've""".split()}
# Keep ordinary English words that also spell apostrophe-free contractions.
for word in ("well", "were", "ill", "hell", "shell", "shed", "wed", "its"):
    CONTRACTIONS.pop(word, None)


@lru_cache(maxsize=65536)
def normalize(answer):
    text = answer.lower().strip().replace("\n", " ").replace("\t", " ")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"\.(?!\d)", "", text)
    text = re.sub(r'[;:/\[\]"{}()=+\\_\-><@,\x60?!]', " ", text)
    words = (NUMBERS.get(w, w) for w in text.split() if w not in {"a", "an", "the"})
    return " ".join(CONTRACTIONS.get(w, w) for w in words)


def soft_score(pred, refs):
    matches = sum(normalize(a) == pred for a in refs)
    return (matches * min(1, max(0, matches - 1) / 3)
            + (10 - matches) * min(1, matches / 3)) / 10
