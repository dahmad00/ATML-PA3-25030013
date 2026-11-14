# Task 2 · Visual grounding

Run the commands below in order on a single CUDA GPU with a RunPod PyTorch template (Python 3.10+, PyTorch 2.6+ and matching torchvision). Use a persistent volume for the archives, feature cache and checkpoint. Budget roughly 30 GB of free disk space. The image archives stay compressed. Training reads only the selected FP16 features.

## Commands

From the directory containing the zip:

```bash
unzip task2.zip
cd task2
python -m pip install -r requirements.txt
```

Download the official COCO 2014 and VQAv2 archives, build the subsets, cache patch features, and verify split isolation:

```bash
python data.py download
python data.py prepare
python data.py cache
python diagnostics.py check
```

Run each stage as a separate process. Rerunning the interrupted stage resumes from its latest committed 250-update checkpoint. Feature caching runs in its own process so SigLIP leaves GPU memory before Qwen loads.

```bash
python train.py --stage 1
python train.py --stage 2
python train.py --stage 3
```

Score the completed stage 3 model, then render figures and complementary-image examples:

```bash
python evaluate.py
python diagnostics.py plots
python qualitative.py
```

## Experiment

The main experiment uses 196 visual tokens. The connector is calibrated once on 32 seeded training captions by matching the initial visual embedding RMS to caption-token embedding RMS. Stage 1 visits each caption once per shuffled pass for two passes. Stages 2 and 3 sample complete pairs or two replay captions per draw. The backbone stays frozen. Stage 2 introduces attention LoRA, and stage 3 continues the same adapters. Each stage starts a fresh AdamW optimizer and warmup/cosine schedule.

The grounding training pool contains 2,500 presence and 2,500 counting pairs drawn from the selected caption images. Category-specific crowd annotations are excluded from positive/count candidates. Negative presence examples have zero annotations for that category. The held-out grounding sets are balanced across presence and counting. Natural pairs must have the same answer type, a shared question, distinct normalized targets, and at least eight agreeing references per side. The manifest stores original answers and fixes the unrelated-image mapping before inference.

Development caption loss includes end-of-turn. Development and final VQA reference NLL exclude it. The shared local answer normalizer handles case, whitespace, punctuation, number words, articles and common contractions. Natural accuracy uses the ten-reference leave-one-out rule. Pair accuracy always uses the normalized canonical target on both sides. Empty predictions count as incorrect. The final metrics describe these selected subsets.

## Saved evidence

All paths below are inside `outputs/`.

| File | Contents |
|---|---|
| `splits.json`, `image_ids.json`, `data_checks.json` | Selected images, questions, caption and pair IDs, original targets, ten VQA references, control mapping and isolation checks |
| `features.npy`, `cache_resources.json` | Row-indexed FP16 SigLIP cache, cache time and peak allocated GPU memory |
| `calibration.json`, `stage*_config.json` | Calibration IDs and scale, resolved model revision, package versions, trainable parameter names and counts |
| `training.json` | Per-update example-weighted loss, gradient norm, learning rates and observed mixture counts |
| `development.json`, `development/*.json` | Learning curves and per-example reference losses before training, every 500 updates and at stage endpoints |
| `stage*_resources.json`, `evaluation_resources.json` | Per-stage elapsed time including development evaluation, peak allocated GPU bytes, final evaluation and generation times |
| `predictions.json` | Raw generations, normalized answers, original references, input image IDs, per-example scores, partner agreement and batch-amortized generation time |
| `metrics.csv` | Suite and type summaries with 95% percentile intervals from 4,000 complete-pair resamples. Matched resamples give accuracy advantage (correct minus swap) and NLL advantage (swap minus correct). Rates use fractions. |
| `learning_curves.png`, `final_results.png` | Stage boundaries, development trends and image-condition comparisons |
| `qualitative/` | One seeded pair per suite and question type, images side by side with references and responses, plus the selection manifest |
| `checkpoint.pt` | One rolling checkpoint containing current connector/LoRA, training progress, optimizer, scheduler, scaler, sampler and RNG states |

Stage endpoints use the same checkpoint path. Replacing it preserves historical metrics and frees the preceding weights. Restoring earlier-stage weights after a later save requires rerunning training. A temporary file is used during checkpoint replacement. Frozen pretrained weights are downloaded separately by Hugging Face. Only the latest training checkpoint is retained.

For swap analysis, compare the fractions of all answers that changed to the partner target and that changed to another normalized answer. Inspect the latter against all ten references before interpreting it as unsupported. Annotation omissions and count ambiguities can affect grounding scores. The qualitative sample is chosen independently of predictions.

## Sources

- [COCO dataset](https://cocodataset.org/#download)
- [VQAv2 questions, complementary pairs and annotations](https://visualqa.org/download.html)
- [Official VQA evaluation](https://visualqa.org/evaluation.html)
- [SigLIP model](https://huggingface.co/google/siglip-base-patch16-224)
- [Qwen3-0.6B model and chat template](https://huggingface.co/Qwen/Qwen3-0.6B)
