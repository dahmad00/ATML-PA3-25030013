# Task 3 — test-time compute

Use a Linux RunPod PyTorch template with CUDA and Python 3.10–3.12. All commands below run from the extracted task3 directory. Only live generation needs a GPU. The experiments keep weights fixed. Development tuning selects two stopping policies for each dataset.

## 1. Set up

```bash
cd /workspace
unzip task3.zip
cd task3
python -m pip install -r requirements.txt
export HF_HOME=/workspace/hf-cache
export TASK3_DATA=/workspace/ATML-PA3-Data
```

PyTorch comes from the template. The live script loads a single FP16 Qwen2.5-3B-Instruct snapshot, uses batch size one and writes no additional model checkpoints.

## 2. Get the inputs

Copy the two folders from the [course rollout export](https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23) under $TASK3_DATA, preserving their names. Alternatively run this Bash block on the pod. Skip the five parquet downloads if you already uploaded the files.

```bash
mkdir -p "$TASK3_DATA"/{qwen25-math,qwen25vl-mathvista}
fetch() {
  curl --fail --location --retry 3 "https://drive.usercontent.google.com/download?export=download&confirm=t&id=$2" -o "$TASK3_DATA/$1.part" && mv "$TASK3_DATA/$1.part" "$TASK3_DATA/$1"
}
fetch qwen25-math/generations.parquet 1SaWxZ4mpycw2BWZ5G0dahoYZ6Hh8tPWP
fetch qwen25-math/probability_metrics.parquet 1RCYHrTnUKtF7gHLSTe92x6W9ZsCOekfQ
fetch qwen25vl-mathvista/generations.parquet 1sedMpWLI_UgPvAQVj7tS3HYVH0ARS1yA
fetch qwen25vl-mathvista/probability_metrics.parquet 1f2XAu_J5NviYHEhtkRrFbKc7b2us1WiW
fetch qwen25vl-mathvista/attention_metrics.parquet 1Nzg2VwE2qFzA10BF1lTxEcBZCePBe51q
```

Download the released question metadata into the same root.

```bash
curl -fL --retry 3 https://raw.githubusercontent.com/lupantech/MathVista/main/data/testmini.json -o "$TASK3_DATA/testmini.json"
echo "ab0148dca9d401c31cc47f29b3826eec28a76ff155a1c0878d81e8daf0413480  $TASK3_DATA/testmini.json" | sha256sum -c -
```

## 3. Run each stage

```bash
python data.py
python evaluate.py fixed
python evaluate.py adaptive
python generate.py
python diagnostics.py
python qualitative.py
```

Each line is a separate command. Generation can start any time after data preparation. Re-run it to resume from the last completed model call. Summaries and examples require their upstream stages. In a new shell, restore the two environment variables above. Archive the results separately after the run with the following command.

```bash
tar -czf /workspace/task3-results.tar.gz outputs
```

## Results

| Directory | Contents |
|---|---|
| outputs/splits | Ordered development, held-out and live IDs, ten nested rollout orders per question, scorer and library versions. |
| outputs/offline | Prepared question pools including raw responses, extracted and normalized answers, reference, correctness and retained lengths. Evaluation parquet files record one decision per question, order, budget and method. Prefix files record agreement, quality margin, costs and retrospective oracle labels. Question means, summaries and paired intervals are saved separately. |
| outputs/adaptive | Full development grid and frozen settings, held-out decisions, counts by stopping budget, errors recoverable at 32, paired comparisons against each fixed SC budget, and the nearest observed cost. |
| outputs/live | Per-call rendered prompts, messages, responses, seeds, boxes, lengths, cap flags, seconds and peak allocated GPU memory. Per-question selections, strategy means, paired differences, and revision transition counts. |
| outputs/figures | Accuracy by candidate count and token cost, within-question score tables, MathVista trajectories, adaptive error breakdowns and qualitative examples. |

In parquet outputs, q is the source sample ID, rid is the chosen rollout ID, and order is a zero-based index in the dataset's saved orders file. The first n IDs reconstruct the candidate prefix. FirstAns consumes only its first ID. Its oracle, distinct-group count and failure decomposition describe the available budget for comparison. Missing predictions remain in the pool and can win a vote.

## Analysis conventions

- Splits restart NumPy default_rng(6304) on lexicographically sorted IDs for each dataset. A separate generator with the same seed draws ten permutations per question in sorted ID order. Calibration, grouping, votes, tie-breaking and stopping use only the revealed prefix.
- MATH uses math-verify 0.8.0, with exact-string matches followed by boxed mathematical equivalence. Group representatives are the first equivalent answer in reveal order. Missing answers form one always-incorrect group. MathVista uses metadata-based option letters, edit-distance choice matching, integer conversion and rounded floats. See the [Math-Verify implementation](https://github.com/huggingface/Math-Verify) and [MathVista scorer](https://github.com/lupantech/MathVista/blob/main/evaluation/calculate_score.py).
- VisConf pools retained token KL values in the current prefix and uses linear 5th/95th percentiles. Each token's visual share uses the sum of all three eligible attention masses. Average ranks determine RWC weights. A zero-length rollout has zero scalar features, contributes no calibration tokens and remains in voting and cost accounting.
- Score associations use within-question Pearson correlations. Quartiles contain eight candidates each, ordered by score and then rollout ID. Undefined correlations are excluded from correlation means, with valid counts reported. For vision, correlations and fusion gaps are averaged within each question before dataset summaries.
- Accuracy differences are fractions. Paired 95% intervals use 1,000 question bootstrap draws after averaging the ten orders. Adaptive tuning minimizes mean cost within one percentage point of development SC at 32. Ties use accuracy and smaller thresholds. A null threshold denotes always using 32. The full-pool correct fraction and recovery flags support retrospective analysis.
- Offline cost counts all retained tokens consumed. Live output tokens include emitted stop tokens. Retained tokens exclude termination. A capped call reached its allowance, and a truncated call exhausted it without a stop token. Timing includes prompt formatting, tokenization, device transfer, generation and decoding on one GPU. The refinement strategy pays for its initial call and the full revision prompt. Peak memory is the maximum allocated across a strategy's calls.
- The fixed trajectories use rollout 0 of the first three held-out MathVista IDs. Selection cases use order 0 at budget 32 and the lexicographically first question on which one of SC or quality-RWC succeeds and the other fails, in each direction. Premature stopping uses the first recoverable early error sorted by question, order and family. Revisions use the first question per initial/final correctness combination. An empty example list indicates that no case matched.
