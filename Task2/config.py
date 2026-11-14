from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
RAW = ROOT / "raw"
FEATURES = OUT / "features.npy"
MANIFEST = OUT / "splits.json"
CKPT = OUT / "checkpoint.pt"
VISION = "google/siglip-base-patch16-224"
LM = "Qwen/Qwen3-0.6B"
SEED = 6304
STEPS = (2250, 1500, 750)
LR = ((1e-3, 0), (1e-4, 2e-5), (5e-5, 1e-5))
BATCH = 8
ACCUM = 2
TYPES = ("yes/no", "number", "other")
CONTROLS = ("correct", "swap", "mismatch", "none")
