# BraTS 2024 Glioma Grade Classification - Deep Handover

**Project:** major101 (PATTASWAMY-VISHWAK-YASASHREE/major101)  
**Branch:** pattaswamy-vishwak-yasashree-cuddly-lamp  
**Session:** 5f06e56e-c030-42dc-b55d-81c23c1e782a  
**Dates:** 2026-08-01 to 2026-08-08 (8 days, ~84.5h estimated human effort on this project)  
**AI Credits:** ~41,486 total (across all projects this period)  
**Last Updated:** 2026-08-14

## Repair status — 2026-08-12

The old metrics in this document are historical. The repaired path is
`scripts/train_ultra_light.py` with `src/grade_data.py` and
`src/grade_model.py`: it verifies all 882 preprocessed files, deduplicates the
994-row label CSV to 876 cases, keeps repeated acquisitions subject-disjoint,
uses mmap/crop-first loading, and enforces 2 GiB VRAM / 3 GiB process-RAM
guards.

### Selected repaired result

| Item | Value |
|------|-------|
| Checkpoint | `outputs/training/repaired_candidate/best_checkpoint.pth` |
| Best validation epoch | 8 |
| Validation balanced accuracy | **0.7571** |
| Validation accuracy @ thr 0.46 | 0.6136 |
| Validation AUROC | 0.7675 |
| Locked test balanced accuracy | **0.5853** |
| Locked test accuracy | 0.5114 |
| Locked test AUROC | **0.7121** |
| Locked test F1 / sens / spec | 0.6055 / 0.4648 / 0.7059 |
| Locked test confusion | TN=12, FP=5, FN=38, TP=33 |
| Threshold | 0.46 (validation-only) |
| Eval summary | `outputs/evaluation/repaired_test/summary.json` |

Whole-volume comparison (`outputs/training/whole_volume_candidate/`): best
validation balanced accuracy **0.6579** — weaker than the crop candidate on
validation; kept as a comparison artifact only.

Research visuals are under `outputs/explainability/repaired_validation/` and
`outputs/explainability/repaired_locked_test_final/`. The label remains the
ET-derived `grade_proxy`, so these results are not independent clinical-grade
classification evidence.

The presentation-ready consolidated report is
`research/BraTS_MRI_Grade_Classification_Panel_Report.md`.

CT+MRI is a later, gated research track. BraTS 2024 in this checkout is
MRI-only; no CT loader, fusion model, or CT experiment should be added until
the MRI completion gate in `plan/process-brats-classification-next-steps-1.md`
is satisfied and a separate paired CT+MRI dataset passes its own integrity
checks.

## Final MRI validation status — 2026-08-14

The MRI completion gate is now satisfied. The final development-only protocol
completed five subject-disjoint folds on all 788 development cases, calibrated
the pooled out-of-fold predictions, trained one final checkpoint on the 788
development cases, evaluated the locked 88-case test exactly once, and
generated final qualitative evidence.

### Final artifacts and results

| Artifact or metric | Final value |
|---|---|
| Data report | `outputs/data_quality/preprocessed_data_report.json` — 882/882 valid |
| CV summary | `outputs/cv/full_epoch_baseline_5fold_5ep/summary.json` |
| CV mean best AUROC | **0.7987** across five folds |
| CV mean best balanced accuracy | **0.7592** across five folds |
| Pooled OOF AUROC | **0.7641**, bootstrap 95% CI 0.7261–0.8018 |
| Pooled OOF balanced accuracy | **0.7288**, bootstrap 95% CI 0.6936–0.7631 |
| Calibration | temperature **0.8011**, development threshold **0.53** |
| Calibration change | Brier 0.1973 → 0.1960; ECE 0.2366 → 0.2279 |
| Final checkpoint | `outputs/training/repaired_final/best_checkpoint.pth` |
| Final fit | 788 development cases, five complete balanced epochs, test not read |
| Locked test | 88 cases; evaluated once from the final checkpoint |
| Locked-test accuracy / balanced accuracy | **0.7045 / 0.5261** |
| Locked-test AUROC / AP | **0.7672 / 0.9431** |
| Locked-test F1 / sensitivity / specificity | **0.8169 / 0.8169 / 0.2353** |
| Locked-test confusion | TN=4, FP=13, FN=13, TP=58 |
| Final visuals | `outputs/explainability/repaired_final/` — four cases |
| Completion gate | `outputs/mri_completion_gate.json` |

The final thresholded result remains weak against the majority-HIGH accuracy
baseline and is not clinical-grade evidence. The target is still the
ET-derived `grade_proxy`, not independent pathology or WHO-grade annotation.
The MRI-only gate is complete; independent clinical labels, external
validation, and qualitative explanation review remain research limitations.

### Final verification

- `python -m unittest discover -s tests -v` — **10/10 passed**.
- `python -m compileall -q src scripts tests` — passed.
- Final smoke — three epochs, peak reserved VRAM 1.02 GiB, peak process RAM
  1.35 GiB.
- Full CV and final fit stayed below the 2 GiB VRAM / 3 GiB process-RAM
  guards; no parallel training jobs were used.
- The locked-test evaluator was run once for the final checkpoint. Do not run
  it again for tuning.

CT/MRI remains deferred. The next research step is independent clinical-label
validation or a separately audited segmentation/molecular endpoint, not more
unbounded proxy tuning.

## Improvement controls and blind raw validation — 2026-08-14

The post-gate extension is implemented without reopening the consumed locked
test. `scripts/cross_validate_repaired.py` now supports deterministic
1–8-view validation, AUROC-based bounded early stopping, per-fold run summaries,
VRAM/RAM guards, and `--allow-incomplete` for explicitly incomplete smoke runs.
The one-fold five-view smoke completed with `complete_oof: false`; its metrics
must not be reported as a five-fold estimate.

The separate raw BraTS validation cohort was verified as 188 four-modality cases
with no labels and no case overlap with the supervised data. The frozen final
development checkpoint was run once through
`scripts/infer_raw_validation_stream.py`:

- predictions: `outputs/external_validation/predictions.csv` — 188 rows
- manifest: `outputs/external_validation/validation_manifest.json` — zero failures
- metrics: intentionally absent because the cohort is unlabeled
- persistent processed cache: none created
- peak reserved VRAM / process RAM: 0.221 / 0.484 GiB

The `ensemble_methods_research_brief.pdf` was read once. It supports a future
compute-matched independent probability ensemble with development-only OOF
calibration and diversity/error-correlation checks; it does not justify
weight-averaging or MoE as the first experiment. That follow-up remains pending
and does not alter the locked-test result.

### Multi-view decision

The bounded five-view training comparison completed all five folds and all 788
development cases, but it was intentionally short (`3` epochs maximum,
`16` steps per epoch): pooled OOF AUROC **0.5134** and balanced accuracy
**0.5268**. It is not a candidate model. To isolate inference-time crop
aggregation from under-training, `scripts/evaluate_cv_views.py` re-evaluated
the mature single-view fold checkpoints with five views. That also declined:

| Evaluation | Pooled OOF AUROC | Pooled OOF balanced accuracy |
|---|---:|---:|
| Mature checkpoints, one view | 0.7641 | 0.7288 |
| Same checkpoints, five views | 0.6058 | 0.5962 |

The five-view option remains available for future experiments, but it is not
selected for the final checkpoint or any locked-test evaluation.

### Compute-matched ensemble decision

`scripts/cross_validate_ensemble.py` completed a leakage-safe five-member OOF
pilot: five independent members, fixed folds, two epochs and 16 steps per
epoch, with probability averaging and diversity diagnostics. Its pooled OOF
AUROC was **0.5463** and balanced accuracy **0.5715**. A compute-matched single
member trained for ten epochs and 16 steps reached AUROC **0.5622** and
balanced accuracy **0.5585**. Both are far below the mature single-view
baseline (AUROC **0.7641**, balanced accuracy **0.7288**), so the ensemble is
rejected and no checkpoint or threshold changed.

The member predictions, logs, and summary are under
`outputs/cv/ensemble_5member_2ep_16steps/`. The run covered all 788 development
cases and recorded `locked_test_evaluated: false`.

### Bounded BA75 search runner — implemented, not launched

The requested long-running improvement attempt is now implemented in
`scripts/background_search.py`. It uses the fixed seed-42 88-case test split
only to exclude it, then divides the remaining 788 development cases into a
709-case search pool and a 79-case independent confirmation split. Three
subject-disjoint folds produce pooled search OOF metrics; a configuration is
accepted only if pooled search balanced accuracy and the separately trained
confirmation balanced accuracy are both at least **0.75**. The search is
bounded to 100 deterministic configurations or 30 hours, whichever comes
first, and writes `status.json`, per-attempt summaries, OOF predictions,
confirmation predictions, and `winner.json` without overwriting the selected
checkpoint.

The hidden Windows launcher is
`scripts/start_background_search.ps1`. It runs one CUDA process with
`num_workers=0`, batch size 2, 96³ patches, and 2 GiB VRAM / 3 GiB process-RAM
guards. Start it from the repository root only when desired:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_background_search.ps1
```

Resume the same output directory after interruption with:

```powershell
python scripts/background_search.py --output-dir outputs/search/ba75 --resume
```

Verification completed before handoff:

- Dry run: 709 search / 79 confirmation / 88 locked-test cases; 108 candidate
  configurations; `locked_test_evaluated: false`.
- GPU smoke: `outputs/search/ba75_smoke2/`, one short attempt, complete 709-case
  OOF, balanced accuracy **0.5036**, AUROC **0.4805**, locked test false.
- Contract suite: **12/12 passed**; Python compilation passed.

The 30-hour search was intentionally not auto-started by this implementation
turn. Do not run `scripts/evaluate_repaired.py` or any locked-test evaluator
while the search is active.

## Previous MRI improvement handoff — 2026-08-12

### Executive state

The MRI pipeline is repaired and runnable, but the requested “10–20 better”
target has not been proven. The current label is an ET-derived `grade_proxy`,
not an independent clinical grade annotation, so published results are not
directly comparable unless the dataset, label definition, split, and metric
match. The correct next acceptance gate is a completed subject-disjoint
five-fold development evaluation followed by one final locked-test evaluation.

The last completed locked-test result remains the repaired candidate:

| Metric | Locked test result |
|---|---:|
| Cases | 88 |
| Balanced accuracy | **0.5853** |
| Accuracy | 0.5114 |
| AUROC | **0.7121** |
| F1 | 0.6055 |
| Sensitivity / specificity | 0.4648 / 0.7059 |
| Confusion | TN=12, FP=5, FN=38, TP=33 |

This test result must not be used for tuning. Its predictions are reproducible
and were evaluated only after the earlier candidate was frozen.

### Data and split facts

- `outputs/data_quality/preprocessed_data_report.json` is complete: 882 files
  inspected, zero invalid files, expected shape `(4,182,218,182)`, finite
  float32 data, and no missing labelled files.
- The labels CSV has 994 rows but 876 unique cases after deduplication; there
  are no conflicting duplicate labels. The six unlabelled orphan volumes are
  not used for training.
- The canonical seed-42 split is subject-disjoint: 700 train cases, 88
  validation cases, and 88 locked-test cases. Repeated acquisitions remain in
  one partition.
- The development set for cross-validation is train+validation only: 788 cases.
  The locked 88-case test partition is excluded from every CV run.

### Code now considered canonical

- `src/grade_data.py`: verified case table, subject grouping, stratified split,
  `build_cross_validation_folds`, mmap/crop-first dataset, and balanced sampler.
- `src/grade_model.py`: tiny GroupNorm 3D CNN, corrected per-sample binary loss,
  metrics, and validation-only threshold selection.
- `scripts/train_ultra_light.py`: single-process CUDA trainer with AMP,
  `num_workers=0`, bounded crops, and 2 GiB VRAM / 3 GiB process-RAM guards.
- `scripts/cross_validate_repaired.py`: development-only five-fold runner;
  default `--steps-per-epoch 0` now means a complete balanced epoch, supports
  deterministic multi-view validation and bounded early stopping, and selects
  checkpoints by AUROC while balanced accuracy is still reported at a
  validation-only threshold.
- `scripts/infer_raw_validation_stream.py`: one-case-at-a-time inference for
  the unlabeled raw validation cohort; it never writes processed case arrays.
- `scripts/evaluate_cv_views.py`: development-only re-evaluation of completed
  fold checkpoints under deterministic multi-view crops.
- `scripts/cross_validate_ensemble.py`: fixed-fold independent-member OOF
  ensemble runner with compute-matched controls and diversity diagnostics.
- `scripts/evaluate_repaired.py`: locked-test evaluator; do not run it for
  tuning.
- `scripts/generate_research_visuals.py`: MRI slice, modality, saliency, and
  Grad-CAM evidence generation.

### What was changed and measured in this phase

1. Added a shared `subject_id` helper and deterministic
   `build_cross_validation_folds` with leakage assertions.
2. Added seven fast contract tests covering duplicate labels, conflicting
   labels, subject-disjoint splitting, balanced sampling, loss weighting, and
   multi-crop aggregation. Latest result: **7/7 passed**.
3. Added optional deterministic multi-crop validation aggregation. It is safe
   but is not promoted as a gain: one-fold tests did not consistently improve
   AUROC or thresholded balanced accuracy.
4. Tested `base_channels=16`; it stayed within memory but learned worse in the
   short smoke test, so the default remains 12.
5. Tested and removed a residual-block branch after it produced weaker
   one-fold AUROC. No dead residual option remains in the canonical model.
6. Corrected the CV runner’s short-smoke bookkeeping and changed checkpoint
   selection from validation balanced accuracy to AUROC to reduce threshold
   overfitting.

### Results from completed development experiments

The earlier three-epoch, 64-step-per-epoch five-fold run completed all folds,
but was underpowered: gradient accumulation made it only about 16 optimizer
updates per epoch. It is retained as a baseline artifact, not a final result:

| Run | Mean best balanced accuracy | Mean best AUROC | Test evaluated |
|---|---:|---:|---|
| `outputs/cv/repaired_baseline_3ep/` | 0.6338 | 0.6118 | No |

The corrected full-balanced-epoch run was intentionally interrupted after the
user stopped the turn. Its partial artifact is:
`outputs/cv/full_epoch_baseline_5fold_5ep/`.

| Fold | Epochs completed | Best AUROC | Balanced accuracy at best AUROC |
|---|---:|---:|---:|
| 1 | 5/5 | 0.7635 | 0.7300 |
| 2 | 5/5 | 0.8031 | 0.7472 |
| 3 | 2/5 | 0.7772 | 0.7147 |
| 4 | 0/5 | not run | not run |
| 5 | 0/5 | not run | not run |

There is deliberately no aggregate summary or out-of-fold prediction file
for this interrupted run. Do not average the three partial folds and call it a
five-fold result. The completed fold evidence stayed inside the safety budget:
peak VRAM was about 1.02 GiB on the first epoch and about 0.55–0.56 GiB after
that; process RAM was about 0.14–0.15 GiB in the recorded folds.

### Windows and memory safety

- The long run used one Python process, CUDA AMP, batch size 2, mmap/crop-first
  reads, `num_workers=0`, and no multiprocessing DataLoader workers.
- The active CV process left behind by the interrupted turn was verified and
  stopped. No unattended training process should be assumed to be running.
- Do not launch parallel Python training jobs, increase DataLoader workers, or
  target the entire Python process tree. If stopping a future run, identify the
  exact training PID first and stop only that PID.
- Do not use `--evaluate-test` during experiments. It is reserved for the final
  frozen candidate.

### Next session procedure

1. Add a small fold-selection/resume option before restarting the expensive CV
   run, or run the remaining folds in separate output directories. Avoid
   repeating folds 1–2 unnecessarily.
2. Complete five epochs of full balanced training for folds 3–5 using the
   default tiny model (`base_channels=12`, patch 96, batch 2, AMP,
   `num_workers=0`, `--steps-per-epoch 0`). Produce one aggregate mean/std and
   pooled out-of-fold AUROC, average precision, balanced accuracy, and
   confusion matrix.
3. If the aggregate result is stable, train one final MRI checkpoint on the
   development cases only, select its threshold/calibration using development
   data, then run `scripts/evaluate_repaired.py` once on the locked 88-case
   test split.
4. Regenerate the locked-test visual evidence and update
   `research/BraTS_MRI_Grade_Classification_Panel_Report.md` with the final
   fold table, confidence intervals, best/medium/worst cases, and limitations.
5. Keep CT out of scope until the MRI gate is complete and paired CT data gets
   a separate integrity report.

### Useful commands

```powershell
# Fast contract and syntax checks
python -m unittest discover -s tests -v
python -m compileall -q src scripts

# Continue CV only after adding resume/fold selection, or run a fresh full CV
python scripts/cross_validate_repaired.py --epochs 5 --steps-per-epoch 0 --output-dir outputs/cv/full_epoch_next

# Locked test: final candidate only
python scripts/evaluate_repaired.py --checkpoint <frozen-checkpoint>
```

### Canonical commands

```bash
python scripts/verify_preprocessed_data.py
python scripts/train_ultra_light.py --output-dir outputs/training/repaired_candidate --epochs 10 --steps-per-epoch 64
python scripts/evaluate_repaired.py --checkpoint outputs/training/repaired_candidate/best_checkpoint.pth
python scripts/generate_research_visuals.py --checkpoint outputs/training/repaired_candidate/best_checkpoint.pth --predictions outputs/evaluation/repaired_test/test_predictions.csv
python scripts/inference.py --case-id BraTS-GLI-02720-100
```

---

## Historical, superseded handoff notes

The following notes describe the pre-repair pipeline and old metrics. They are
retained to explain why the repair was necessary. Use the repair status above
and `research/BraTS_MRI_Grade_Classification_Panel_Report.md` for current
results.

## Executive Summary

Built a **working 3D CNN brain tumour classification pipeline** on BraTS 2024 GLI data (glioma grade: Low-grade LGG vs High-grade GBM) that **beats the majority-class baseline** (0.8046) and achieves **val_acc = 0.8506** (epoch 13) with a clean, minimal architecture.

**Key achievement:** Escaped the "model collapse plateau" where 3D CNNs on 4:1 imbalanced medical data predict all-high or all-low. Fixed via research-backed combination: **GroupNorm + Class-Balanced Loss (beta=0.999) + Focal Loss (gamma=1.0) + Kaiming Normal init + BalancedBatchSampler**.

**Current state:** Ultra-lightweight training script (train_ultra_light.py) running 2-epoch smoke test with BalancedBatchSampler + CombinedLoss (CB + Focal 70/30). First epoch still all-low (0.2412), second epoch all-high (0.2362) - oscillation persists but pattern is understood.

---

## Project Structure

```
major101/
data/
  brats_preprocessed/
    train/                    # 1350 .npy files (4x182x218x182, CTN-normalised)
    labels.csv                # 994 rows, 876 unique cases
outputs/
  brats_training_analysis/    # 12 EDA outputs (class balance, volumes, correlations)
  training/
    M1_best.pth               # Baseline M1 (epoch 13, val_acc=0.8506) 
    M1_history.csv            # Training history
    M1_predictions.csv        # Test predictions
    M1_features_best.pth      # Feature-augmented model (epoch 3, val_acc=0.9885)
    M1_features_history.csv
    M1_features_training.log
    M3_best.pth               # M1 + augmentation (val_acc=0.8462)
    M3_history.csv
    ultra_light_best.pth      # Current ultra-light checkpoint
research/
  ml_plateau_research.md      # 13 primary source citations for plateau escape
  (19 other research files from Phase 0)
scripts/
  train_classifier.py         # MAIN: M1 + M3 with all proven fixes
  train_m1_features.py        # Feature-augmented (ET/WT ratio + volumes, dual-pathway)
  train_ultra_light.py        # CURRENT: 3.5M params, 96^3 patches, batch=2
  analyze_brats_training.py   # EDA script
  inference.py                # Single-case inference
  visualize_inference.py      # Grad-CAM, saliency, modality slices
src/
  data.py                     # BraTS3DDataset, make_dataloaders, BalancedBatchSampler
  model.py                    # GradeClassifier3D (M1), Tiny3DCNN (ultra-light)
  losses.py                   # ClassBalancedLoss, FocalLoss, CombinedLoss
FINAL_REPORT.md               # Comprehensive benchmark comparison
```

---

## Data Summary

| Aspect | Detail |
|--------|--------|
| **Dataset** | BraTS 2024 GLI Training (glioma grade classification) |
| **Total cases** | 1350 preprocessed .npy files |
| **Modalities** | 4 channels: T1c, T1n, T2f, T2w (stacked as 4x182x218x182) |
| **Normalization** | CTN (Combined Tissue Normalization) to z-score per modality, then clipped to [-3, 3] and scaled to [-1, 1] |
| **Labels** | labels.csv: case, et, tc, wt, wt_volume, tc_volume, et_volume, grade_proxy |
| **Grade proxy** | 0 = Low-grade (LGG, n=173), 1 = High-grade (GBM, n=703) - **derived from ET presence** |
| **Imbalance** | ~4:1 (703 high : 173 low) - major cause of collapse |
| **Split** | 80/10/10 stratified (train=788, val=88, test=88) |

**Critical data insight:** grade_proxy is **not a BraTS annotation** - it is a clinical proxy: ET > 0 -> high-grade. This means ALL 173 low-grade cases have ET=0 and ALL 703 high-grade have ET>0. Any model using ET volume/features learns the labeling rule, not tumor appearance.

---

## Model Evolution & Key Findings

### M1 - Baseline 3D CNN (GradeClassifier3D)
- **Architecture:** 4x Conv3D blocks (32->64->128->256) + GlobalAvgPool + Linear(1)
- **Params:** ~3.5M
- **Initial result:** val_acc = 0.8046 (majority baseline - collapsed to all-high)
- **Fixed result:** val_acc = 0.8506 (epoch 13) after fixes below

### M1_features - Feature-Augmented (GradeClassifier3DWithFeatures)
- **Added:** 3 scalar features - log(ET_vol), log(WT_vol), ET/WT_ratio
- **Architecture:** Dual-pathway residual fusion (image_feat + projected_scalar_feat)
- **Result:** **100% test accuracy (artifact)** - learns ET presence = grade_proxy
- **Lesson:** Feature model is **not usable for real deployment** (labels derived from ET)

### M3 - M1 + Augmentation (Ablation)
- **Augmentation:** Random flips (3 axes), Gaussian noise (sigma=0.1)
- **Result:** val_acc = 0.8462 (slightly worse than M1 - augmentation didn't help)

### Ultra-Light (Tiny3DCNN) - Current Focus
- **Architecture:** 4x Conv3D (16->32->64->128), 96^3 patches, batch=2, grad_accum=4
- **Params:** 3.5M
- **Target:** 3GB VRAM / 3GB RAM on RTX 2050 (4GB)
- **Sampler:** BalancedBatchSampler (1 low + 1 high per batch)
- **Loss:** CombinedLoss = 0.7xClassBalancedLoss(beta=0.999) + 0.3xFocalLoss(gamma=1.0)
- **2-epoch test:** Epoch 1 all-low (0.2412), Epoch 2 all-high (0.2362) - **oscillation persists**

---

## The Plateau & How We Escaped It (Research-Backed)

### The Problem
Binary classification with 4:1 imbalance -> model predicts all-high (val_acc=0.8046, F1=0). Training dynamics:
- Majority gradient 4x larger than minority
- BCE loss pushes decision boundary to classify everything as high-grade
- Model never sees enough low-grade gradients to learn the boundary

### Research Findings (13 primary sources in ml_plateau_research.md)

| Fix | Source | Why It Works |
|-----|--------|--------------|
| **GroupNorm(num_groups=8)** | Wu & He 2018 (GroupNorm paper) | InstanceNorm with batch=2-4 has noisy per-channel stats (~8000 values). GroupNorm shares stats across channels - stable gradients |
| **Class-Balanced Loss (beta=0.999)** | Cui et al. 2019 (CVPR) | Effective number = (1-beta^n)/(1-beta). Low-grade (n=173) gets ~3.2x weight vs high-grade (n=703) |
| **Focal Loss gamma=1.0 (not 2.0)** | Lin et al. 2017 (ICCV) + binary classification adjustment | gamma=2.0 over-suppresses easy samples for binary; gamma=1.0 balances hard/easy mining |
| **Kaiming Normal init (a=0.01)** | He et al. 2015 (ICCV) | Critical for ReLU + 3D Conv + GroupNorm. Default init kills gradient flow in deep 3D nets |
| **BalancedBatchSampler** | Custom implementation | Guarantees 1:1 class ratio per batch - every batch has minority gradient signal |
| **CombinedLoss (CB 70% + Focal 30%)** | Empirical (mirrors M1 config) | CB provides class reweighting; Focal down-weights easy majority samples |

### What Finally Worked (M1, epoch 13 -> 0.8506)
```python
# In src/model.py GradeClassifier3D:
nn.GroupNorm(num_groups=8, num_channels=...)  # NOT InstanceNorm3d

# In src/losses.py:
ClassBalancedLoss(beta=0.999)  # class_counts from TRAIN split only
FocalLoss(gamma=1.0)
CombinedLoss(cb_weight=0.7, focal_weight=0.3)

# In src/data.py:
BalancedBatchSampler(dataset, batch_size=4)  # 2 low + 2 high per batch

# In train_classifier.py:
kaiming_normal_init(model)  # a=0.01 for LeakyReLU
```

---

## Current Ultra-Light Training Status

### What's Running
```bash
python scripts/train_ultra_light.py --full
```

### Configuration (train_ultra_light.py)
```python
# Data
patch_size = (96, 96, 96)           # Random crops from 182x218x182
batch_size = 2                       # 1 low + 1 high via BalancedBatchSampler
grad_accum = 4                       # Effective batch = 8
num_workers = 0                      # Windows spawn + pandas = deadlock

# Model
Tiny3DCNN: 4 conv blocks (16->32->64->128), GroupNorm(8), GlobalAvgPool, Linear(1)
Params: 3,529K

# Loss
CombinedLoss(
    class_counts=[180, 615],        # TRAIN split only (FIXED: was using full dataset)
    beta=0.999,
    gamma=1.0,
    cb_weight=0.7,
    focal_weight=0.3
)

# Optimizer
AdamW(lr=5e-3, weight_decay=1e-4)
ReduceLROnPlateau(patience=5, factor=0.5)
Early stopping: patience=15

# Hardware
Device: CUDA (RTX 2050 4GB)
Mixed precision: torch.amp.autocast('cuda')
Target: <3GB VRAM, <3GB RAM
```

### 2-Epoch Test Results (Latest Run)
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | F1 | Sens | Spec | Time |
|-------|------------|-----------|----------|---------|-----|------|------|------|
| 1 | 2.2104 | 0.5073 | 0.6603 | **0.2412** | 0.0000 | 0.0000 | 1.0000 | 952s |
| 2 | 0.6717 | 0.5098 | 0.5257 | **0.2362** | 0.0000 | 0.0000 | 0.9792 | 846s |

**Pattern:** Classic oscillation - epoch 1 predicts all-low (spec=1.0, sens=0), epoch 2 predicts all-high (sens=1, spec=0). This is the **exact same pattern** M1 had before BalancedBatchSampler + CombinedLoss fixed it.

### Why Oscillation Persists in Ultra-Light
1. **batch=2 is extremely small** - even with BalancedBatchSampler (1:1), gradient estimate is noisy
2. **96^3 patches** - random crops may not capture discriminative regions consistently
3. **LR=5e-3 may be too high** for such small effective batch (8 with accumulation)
4. **Class counts [180, 615]** computed from train split indices - verified correct now

### Immediate Next Steps for Ultra-Light
```python
# Option A: Increase batch (if VRAM allows)
batch_size = 4  # 2 low + 2 high per batch
grad_accum = 2  # Effective batch = 8

# Option B: Lower LR + warmup
lr = 1e-3
warmup_epochs = 5

# Option C: Use proven M1 config directly (batch=4, full 182^3 volumes)
# This worked -> val_acc=0.8506. Ultra-light is over-constrained.

# Option D: Add label smoothing
BCEWithLogitsLoss(label_smoothing=0.1)  # Prevents overconfident predictions
```

---

## Critical Technical Gotchas (Windows + BraTS)

| Issue | Root Cause | Fix |
|-------|------------|-----|
| **DataLoader deadlock** | Windows `spawn` + pandas DataFrame pickling fails | Convert `labels_df` to `list[tuple]` in `BraTS3DDataset.__init__` |
| **RAM thrashing (3.89s/batch)** | `num_workers=4` duplicates .npy tensors in each worker | `num_workers=0` -> 0.45s/batch (8x speedup) |
| **Checkpoint save fails** | `torch.save` on Windows requires `str(path)`, not `Path` | `torch.save(state, str(path))` |
| **Label columns changed** | `labels.csv` uses `case`/`grade_proxy`, not `case_id`/`label` | Updated all scripts to use correct columns |
| **.npy location** | Files in `data/brats_preprocessed/train/`, not root preprocessed dir | Fixed paths in `build_split_indices` |
| **Aug rotation breaks** | `torch.rot90` swaps H/W - stack fails | Removed rotation; kept flips + noise |
| **Per-case normalization** | Destroys absolute T1ce intensity (can't distinguish large ET vs large non-ET) | Root cause for feature model artifact |

---

## File Reference: Key Scripts

### `scripts/train_classifier.py` - MAIN PRODUCTION SCRIPT
```bash
# Full training (100 epochs, early stop patience=10)
python scripts/train_classifier.py

# Quick test (3 epochs)
python scripts/train_classifier.py --test-only
```
**Contains all proven fixes:** GroupNorm, CB Loss, Focal gamma=1.0, BalancedBatchSampler, Kaiming init, checkpoint str(path), early stopping.

### `scripts/train_ultra_light.py` - CURRENT EXPERIMENTAL
```bash
# 2-epoch smoke test (default)
python scripts/train_ultra_light.py

# Full 100-epoch run
python scripts/train_ultra_light.py --full
```
**Current status:** 2-epoch test completes but shows oscillation. Needs config adjustment before full run.

### `scripts/train_m1_features.py` - FEATURE MODEL (ARTIFACT)
```bash
python scripts/train_m1_features.py
```
**Result:** 100% test accuracy - **do not use for real deployment**. Labels derived from ET presence.

### `scripts/inference.py` - SINGLE CASE INFERENCE
```bash
python scripts/inference.py --case BraTS-GLI-xxxx --checkpoint outputs/training/M1_best.pth
```

### `scripts/visualize_inference.py` - GRAD-CAM + VISUALIZATION
```bash
python scripts/visualize_inference.py --case BraTS-GLI-xxxx --checkpoint outputs/training/M1_best.pth --all
```
Outputs: Grad-CAM overlay, saliency map, modality slices, JSON analysis.

### `src/data.py` - DATA LOADING CORE
- `BraTS3DDataset`: Handles .npy loading, augmentation, label mapping
- `make_dataloaders()`: Creates train/val/test loaders with BalancedBatchSampler
- `build_split_indices()`: 80/10/10 stratified split (returns indices, not datasets)
- **Critical:** `num_workers=0`, list[tuple] conversion, patch cropping for ultra-light

### `src/losses.py` - LOSS FUNCTIONS
- `ClassBalancedLoss(beta=0.999)`: Cui et al. 2019 effective number weighting
- `FocalLoss(gamma=1.0)`: Lin et al. 2017, adjusted for binary
- `CombinedLoss(cb_weight=0.7, focal_weight=0.3)`: Weighted sum

### `src/model.py` - MODEL DEFINITIONS
- `GradeClassifier3D`: M1 baseline (4 conv blocks, GroupNorm, GlobalAvgPool)
- `GradeClassifier3DWithFeatures`: Dual-pathway residual fusion
- `Tiny3DCNN`: Ultra-light (4 conv blocks, 16->32->64->128, 96^3 input)
- `kaiming_normal_init()`: Critical initialization for ReLU + GroupNorm

---

## Checkpoints & Artifacts

| File | Epoch | Val Acc | Notes |
|------|-------|---------|-------|
| `M1_best.pth` | 13 | **0.8506** | **Best working model** - use for inference |
| `M1_history.csv` | 1-13 | - | Full training history |
| `M1_predictions.csv` | 13 | - | Test set predictions |
| `M1_features_best.pth` | 3 | 0.9885 | **ARTIFACT** - 100% test acc (ET=label) |
| `M3_best.pth` | 20 | 0.8462 | M1 + augmentation (no gain) |
| `ultra_light_best.pth` | 2 | 0.2362 | Current ultra-light (oscillating) |

---

## Remaining Work (Prioritized)

### 1. Fix Ultra-Light Oscillation (HIGH)
- [ ] Increase batch to 4 (2 low + 2 high) if VRAM allows
- [ ] Lower LR to 1e-3 with 5-epoch warmup
- [ ] Add label smoothing (0.1) to BCE
- [ ] Or: abandon ultra-light, use proven M1 config (batch=4, full volumes)

### 2. Run Full M1 Training on Train+Val Combined (HIGH)
```bash
# Retrain on 788+88=876 cases for final submission
python scripts/train_classifier.py  # Already configured for this
```

### 3. 5-Fold Cross-Validation (HIGH)
- Reliable test estimate (current test=88 cases is small)
- Stratified by grade_proxy
- Report mean +/- std val_acc

### 4. Train M2 - 3D U-Net Segmenter (MEDIUM)
- Input: 4x182x218x182, Output: 4-class segmentation (WT, TC, ET, background)
- Loss: 0.5xDice + 0.5xCrossEntropy
- Reconstruct seg masks from raw `seg.nii.gz` during training

### 5. Temperature Scaling Calibration (MEDIUM)
- Post-hoc calibration on validation set
- Improves probability reliability for clinical use

### 6. Grad-CAM Explainability (LOW)
- `visualize_inference.py` already has Grad-CAM implementation
- Generate figures for paper

### 7. Clean Up Preprocessed Data (SPACE)
- Preprocessed dataset = 95 GB
- Can delete after final model trained (keep labels.csv + splits)
- Raw BraTS downloads can be re-preprocessed if needed

---

## Commands Quick Reference

```bash
# Environment
conda activate brats-env  # or your env name

# Main training (proven config)
python scripts/train_classifier.py

# Quick test (3 epochs)
python scripts/train_classifier.py --test-only

# Ultra-light experiment
python scripts/train_ultra_light.py           # 2-epoch test
python scripts/train_ultra_light.py --full    # 100 epochs

# Feature model (artifact - do not use)
python scripts/train_m1_features.py

# Inference on single case
python scripts/inference.py --case BraTS-GLI-xxxx --checkpoint outputs/training/M1_best.pth

# Visualization (Grad-CAM, saliency, slices)
python scripts/visualize_inference.py --case BraTS-GLI-xxxx --checkpoint outputs/training/M1_best.pth --all

# EDA (already run - outputs in outputs/brats_training_analysis/)
python scripts/analyze_brats_training.py

# Git
git add -A && git commit -m "msg" && git push origin pattaswamy-vishwak-yasashree-cuddly-lamp
```

---

## Environment & Dependencies

```txt
# requirements.txt (core)
torch>=2.0
torchvision
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
pyyaml
nibabel  # for NIfTI if needed
monai    # optional, for U-Net boilerplate (M2)
```

**Hardware:** RTX 2050 4GB VRAM, 8GB RAM, Windows 11  
**CUDA:** 12.x (PyTorch CUDA 12.1 build)  
**Python:** 3.10+

---

## Known Issues & Warnings

1. **Ultra-light oscillation not resolved** - BalancedBatchSampler + CombinedLoss works at batch=4 (M1) but fails at batch=2. Root cause: gradient noise at micro-batch size.

2. **Feature model is an artifact** - 100% accuracy comes from ET presence = grade_proxy labeling rule. Do not report as real performance.

3. **Test set small (n=88)** - 5-fold CV needed for reliable estimate.

4. **Windows DataLoader** - `num_workers=0` mandatory. No multiprocessing support with pandas.

5. **Checkpoint paths** - Always use `str(path)` in `torch.save()` on Windows.

6. **Class counts for CB Loss** - MUST use train split only ([180, 615]), not full dataset ([228, 766]). Fixed in latest ultra-light.

---

## Research Context (Phase 0)

Phase 0 produced 24 research files in `research/` covering:
- 36 PubMed references across 5 categories
- Dataset inventory (BraTS, IBSR, TCGA-GBM/LGG, CPTAC, etc.)
- Architecture comparisons (ResNet3D, DenseNet3D, V-Net, U-Net, Transformers)
- Multimodal fusion strategies (early/late/hybrid)
- Longitudinal analysis (3D CNN-LSTM, Temporal Transformer)
- Preprocessing pipelines (skull-stripping, registration, ComBat)
- Self-supervised pretraining (MAE for CT)
- Radiomics fusion (PyRadiomics + deep features)
- Explainability (Grad-CAM, SHAP)
- Uncertainty calibration (MC Dropout, temperature scaling)
- Survival analysis (DeepSurv, DeepHit)
- Federated learning (out of scope)

Key decision: **Late fusion + 3D ResNet-18 per modality** for MRI-only (BraTS). CT+MRI fusion deferred (no paired data in BraTS).

---

## Handover Checklist for Next Session

- [ ] Read this HANDOVER.md completely
- [ ] Check current ultra-light training status (is it still running?)
- [ ] Decide: fix ultra-light config OR use proven M1 config for final training
- [ ] Run 5-fold CV on M1 for reliable test estimate
- [ ] Retrain M1 on train+val combined (876 cases)
- [ ] Start M2 (3D U-Net segmentation) if classification is solid
- [ ] Clean up 95 GB preprocessed data if space needed
- [ ] Push final models and results to GitHub

---

## Contact / Context

**Session workspace:** `C:\Users\pvish\copilot-worktrees\major101\pattaswamy-vishwak-yasashree-cuddly-lamp`  
**GitHub repo:** PATTASWAMY-VISHWAK-YASASHREE/major101  
**Branch:** pattaswamy-vishwak-yasashree-cuddly-lamp  
**Session state:** `C:\Users\pvish\.copilot\session-state\5f06e56e-c030-42dc-b55d-81c23c1e782a\`  
**Checkpoints:** 50+ checkpoints in session-state/checkpoints/ (see index.md)

**Key checkpoints to review:**
- `048-brats-3d-cnn-pipeline-complete.md` - M1 0.8506 achieved
- `043-m1-model-collapse-diagnosis-an.md` - Root cause analysis
- `047-fixing-brats-3d-cnn-training-p.md` - Fixes applied
- `044-dataloader-deadlock-and-speed.md` - Windows fixes
- `ml_plateau_research.md` - 13 primary source citations

---

*Generated 2026-08-08 from session 5f06e56e-c030-42dc-b55d-81c23c1e782a*
*Total project effort: ~84.5 human-equivalent hours (whatidid heuristic)*
*Code impact: +26,980 / -1,310 lines across project*
