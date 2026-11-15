import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.getenv("TASK3_DATA", ROOT / "ATML-PA3-Data"))
OUT = ROOT / "outputs"
SPLITS, OFF, LIVE, ADAPT, FIG = (OUT / x for x in ("splits", "offline", "live", "adaptive", "figures"))
FOLDERS = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}
SEED, ORDERS, BOOT = 6304, 10, 1000
BUDGETS = (1, 2, 4, 8, 16, 32)
TAUS, GAMMAS = (0.5, 0.625, 0.75, 0.875, 1.0), (0, 0.1, 0.25, 0.5)
MODEL = "Qwen/Qwen2.5-3B-Instruct"
REV = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SYSTEM = "You are a helpful assistant."
SOLVE = r"Think step by step inside <think> </think> tags, then give your final answer inside \boxed{}."
REVIEW = r"Review the solution above. Identify and correct any errors; if it is correct, retain its answer. Give your final answer inside \boxed{}."
SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)
