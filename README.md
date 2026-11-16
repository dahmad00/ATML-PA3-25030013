# ATML PA3

## Setup

The commands use Bash on Linux. For the full experiments, start with Python 3.10–3.12 and a CUDA-enabled PyTorch environment (PyTorch 2.6+ with matching torchvision). Only live generation needs a GPU in Task 3. Allow about 30 GB of free disk space for Task 2 alone. The preparation scripts need internet access for missing datasets and model weights.

```bash
git clone https://github.com/dahmad00/ATML-PA3-25030013.git
cd ATML-PA3-25030013
export PA3_ROOT="$PWD"
```

Each task has its own requirements. Run the setup at the start of each task section in the same shell. The virtual environments inherit PyTorch from the base installation and keep each task's Transformers version separate.

The clone includes published outputs. **For a fresh reproduction**, run the following block once immediately after cloning to keep those results separate from new logs and cached predictions. Skip this step when inspecting the published results or resuming your own run.

```bash
for task in Task1 Task2 Task3; do
  mv "$PA3_ROOT/$task/outputs" "$PA3_ROOT/$task/published_outputs"
done
```

## Task 1 · Fine-tune and evaluate Qwen3-0.6B

Prepare 40,000 MetaMathQA examples, then score the pretrained baseline on GSM8K, HellaSwag, BoolQ, and ARC-Easy. The qualitative script uses the supplied `first_plot_questions.yaml`.

```bash
cd "$PA3_ROOT/Task1"
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python data.py
python evaluate.py baseline
python qualitative.py baseline
```

Full fine-tuning pauses at update 625. Run all three midpoint analyses **before resuming** because training overwrites the rolling checkpoint.

```bash
python train.py full
python evaluate.py full --stage midpoint
python diagnostics.py full --stage midpoint
python qualitative.py full --stage midpoint
python train.py full --resume
```

Train the remaining methods, score each final checkpoint at update 1,250, and collect the comparison tables.

```bash
for method in l2sp lora4 lora16 oplora; do
  python train.py "$method" || exit 1
done

for method in full l2sp lora4 lora16 oplora; do
  python evaluate.py "$method" && python diagnostics.py "$method" && python qualitative.py "$method" || exit 1
done

python utils.py
deactivate
```

Read `outputs/analysis/benchmark_metrics.csv` for scores and `outputs/analysis/training_curves.csv` for training history. To resume interrupted training from a saved checkpoint, use `python train.py METHOD --resume`, replacing `METHOD` with its command-line name.

## Task 2 · Train and test visual grounding

Download the missing COCO 2014 and VQAv2 archives, prepare the splits, and cache SigLIP patch features. Run the three training stages in order. Each stage hands its connector and adapter weights to the next through `outputs/checkpoint.pt`.

```bash
cd "$PA3_ROOT/Task2"
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python data.py download
python data.py prepare
python data.py cache
python diagnostics.py check

python train.py --stage 1
python train.py --stage 2
python train.py --stage 3

python evaluate.py
python diagnostics.py plots
python qualitative.py
deactivate
```

Rerun the interrupted stage's command to resume. Compare correct, swapped, unrelated, and absent images in `outputs/metrics.csv`. Inspect individual answers in `outputs/predictions.json` and paired images in `outputs/qualitative/`. See the [Task 2 experiment notes](Task2/README.md) for scoring and split definitions.

## Task 3 · Evaluate test-time compute

The offline experiment uses saved rollouts and budgets of 1, 2, 4, 8, 16, and 32 candidates per question. Data preparation reads `ATML-PA3-Data/` inside Task 3, downloads missing inputs, and builds the question pools. To use an existing external copy, set `TASK3_DATA` to the directory containing both rollout folders and `testmini.json`.

```bash
cd "$PA3_ROOT/Task3"
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python data.py
python evaluate.py fixed
python evaluate.py adaptive
python diagnostics.py
```

The live experiment runs Qwen2.5-3B-Instruct on held-out MATH questions. On a CUDA machine, run generation and then collect qualitative examples. If interrupted, rerun generation to reuse completed calls.

```bash
python generate.py
python qualitative.py
deactivate
```

Fixed-budget results are in `outputs/offline/`, stopping policies and their evaluations in `outputs/adaptive/`, and live responses and timings in `outputs/live/`. See the [Task 3 experiment notes](Task3/README.md) for the voting rules, stopping criteria, and cost accounting.
