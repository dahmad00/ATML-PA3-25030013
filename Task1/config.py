from pathlib import Path
SEED=6304
MODEL='Qwen/Qwen3-0.6B'
OUT=Path('outputs')
TARGETS=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
METHODS={'full':2e-5,'l2sp':2e-5,'lora4':2e-4,'lora16':2e-4,'oplora':2e-4}
STEPS=1250
SAVE_STEPS={312,625,937,1250}
TASKS=['gsm8k_cot','hellaswag','boolq','arc_easy']
