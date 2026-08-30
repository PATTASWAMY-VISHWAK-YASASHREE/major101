# Code Alignment: Fusion and Multi-Dataset Research versus the Active Implementation

**Project:** `major101`  
**Author:** Manus AI  
**Purpose:** Compare the web-researched integration options in [`21_multimodal_fusion_and_dataset_integration.md`](21_multimodal_fusion_and_dataset_integration.md) with the code that is actually present in this checkout.

## Executive finding

The research brief is directionally compatible with the repository, but most of the proposed fusion capabilities are **not implemented yet**. The current code is a strong MRI-only binary baseline with useful safety infrastructure: binary-label validation, subject-disjoint splitting, memory-mapped four-channel volumes, balanced sampling, validation-only threshold selection, cross-validation folds, out-of-fold predictions, and one-checkpoint locked-test evaluation.[1] [2] [3] [4]

The code has **partial readiness for model ensembling** because the cross-validation runner already saves one checkpoint and validation prediction file per fold and writes out-of-fold predictions. It has **no active support for a second modality, a second dataset, two simultaneous encoders, late fusion, feature fusion, missing-modality masks, calibration fitting, stacking, or knowledge distillation**. Those are new pipeline capabilities, not configuration changes.

## 1. Implementation status matrix

| Researched option | What the current code actually supports | Gap to close | Readiness |
|---|---|---|---|
| Preserve MRI baseline | `MemoryMappedPatchDataset` reads one four-channel `(4, 182, 218, 182)` volume; `TinyGradeClassifier3D` accepts four input channels; the trainer creates one model and one checkpoint | None for the baseline itself | **Implemented** |
| MRI-only multi-crop evaluation | Cross-validation can evaluate deterministic crop views and aggregate predictions by case | Does not combine independent models or modalities | **Partial, already useful** |
| CT-only baseline | No CT path, CT-specific shape/intensity contract, or CT label table exists | Add CT preprocessing, integrity checks, loader, and a CT model with a compatible output interface | **Not implemented** |
| Early MRI+CT fusion | Input shape and model are hard-coded around four MRI channels; no registered CT channel is available | Add paired CT-MRI registry, registration/resampling, CT normalization, and a five-or-more-channel input contract | **Not implemented** |
| Intermediate feature fusion | `TinyGradeClassifier3D.forward()` returns only one logit after one stem, three downsampling blocks, pooling, and a head | Refactor the model into reusable encoders returning features, add a fusion block, modality masks, and a joint head | **Not implemented** |
| Late/logit fusion | `evaluate_repaired.py` reloads one checkpoint and evaluates one probability stream at one stored threshold | Add two checkpoint loaders, probability calibration, validation-only weight fitting, and a fused evaluator | **Not implemented** |
| Fold/seed ensemble | Five-fold code saves fold checkpoints and out-of-fold predictions | Add probability aggregation, calibration, ensemble threshold selection, and locked-test evaluation | **Partial infrastructure only** |
| Stacking | Out-of-fold predictions exist, but no meta-classifier or stacker is fit | Add a development-only stacker trained only on out-of-fold predictions | **Not implemented** |
| Knowledge distillation | No teacher/student model, soft-target loss, or distillation coefficient exists | Add teacher inference, student training, and a separate comparison against the MRI baseline | **Not implemented** |
| Missing-modality learning | Dataset always returns four MRI channels; no presence mask, modality dropout, or CT-only fallback exists | Add modality-presence metadata, masked encoders or modality dropout, and explicit missingness evaluation | **Not implemented** |
| Same-label multi-dataset pooling | Labels contain only `case` and binary `grade_proxy`; the audit tracks files and labels, not dataset/site/scanner domains | Add `dataset_id`, site/scanner fields, source-aware sampling, and leave-one-domain-out evaluation | **Not implemented** |
| Different-label multi-dataset training | The loader requires one binary `grade_proxy` and rejects labels outside `{0,1}` | Use harmonized labels, dataset-specific heads, masked loss, or target fine-tuning; do not merge labels blindly | **Not implemented** |
| Paired/unpaired data | `CaseRecord` has one case ID, one label, and one `.npy` path; there is no MRI-CT correspondence field | Add a paired registry and reject ambiguous joins; use a specialized unpaired method if cases are not paired | **Not implemented** |
| Domain adaptation/generalization | Splits are subject-disjoint but do not use domain metadata; evaluation reports one aggregate metric set | Add domain-aware splits, leave-one-domain-out tests, and per-domain metrics before adaptation methods | **Not implemented** |
| Probability calibration | The trainer plots a reliability curve and chooses a validation threshold; it does not fit a calibration transform | Add temperature scaling, isotonic regression, or another prespecified calibration method on development data only | **Diagnostics only** |

## 2. What is already reusable

### 2.1 Data integrity and leakage controls

The existing data layer validates the binary `grade_proxy`, rejects conflicting duplicate labels, checks the expected four-channel shape and `float32` dtype, memory-maps the volume, and rejects non-finite sampled patches.[1] The split builder derives a subject identifier from repeated acquisition IDs, uses `StratifiedGroupKFold`, and asserts both case and subject disjointness.[1] The data-quality script streams files in chunks and records label counts, missing labelled files, and orphan files.[3]

These controls should be reused rather than rewritten for multimodal work. The main extension is to make the case table richer: add `dataset_id`, `site_id`, `scanner_id`, `mri_path`, `ct_path`, `mri_present`, `ct_present`, and `label_source`. The same split assertions should then operate on the new case table.

### 2.2 Balanced training and validation-only thresholding

`BalancedBatchSampler` creates equal low/high batches and oversamples only inside the training split.[1] The trainer selects a threshold from validation predictions and stores it in the checkpoint; it does not evaluate the locked test unless the explicit flag is supplied.[2] This is a suitable safety pattern for fusion, but the fusion weight, calibration transform, stacker, and final threshold must also be fitted only on development data.

### 2.3 Cross-validation outputs as ensemble inputs

The cross-validation runner trains a separate `TinyGradeClassifier3D` per fold, saves a best checkpoint and validation-prediction CSV per fold, and writes `out_of_fold_predictions.csv` after verifying that each development case appears exactly once.[4] These outputs are the closest existing hook for an ensemble or stacker. They are not an ensemble yet: no code loads multiple fold probabilities, calibrates them jointly, fits a meta-classifier, or sends an ensemble to the locked test.

## 3. Code-to-research comparison by proposed strategy

### 3.1 Calibrated late fusion is the best first code extension

The research recommendation fits the current architecture conceptually because the active model already returns one binary logit and saves a checkpoint threshold. However, the implementation still needs a second modality contract and a new evaluator. A practical first version would load an MRI checkpoint and a CT checkpoint, run both on the same paired validation cases, convert logits to calibrated probabilities, fit a single fusion weight on out-of-fold development predictions, and evaluate the frozen rule once on the locked test.

The required new components are a paired case registry, a CT dataset and integrity audit, a CT model or CT-compatible checkpoint, and `scripts/evaluate_late_fusion.py`. The existing `evaluate()` helper is not enough because it accepts one model and one loader; it must be extended or wrapped to evaluate two synchronized streams and record modality presence.

### 3.2 Intermediate feature fusion requires a model refactor

The research brief proposes `z_MRI`, `z_CT`, a fusion block, modality masks, and one classifier head. The current model does not expose `z_MRI`: its forward method runs the entire stem, block stack, global pooling, and head, returning only a scalar logit.[2] The clean refactor is to split the current class into an encoder that returns the pooled feature vector and a classifier head, then introduce a second encoder and a small fusion module.

This should not be implemented by copying the entire training script. The split, loss, balanced sampler, AMP, memory checks, and validation metrics should remain shared. The new model should be tested with synthetic tensors before any expensive training run.

### 3.3 Ensembling is partly enabled by completed CV infrastructure

The current cross-validation runner is closer to the ensemble research than the main trainer is. It already produces fold-specific checkpoints and out-of-fold predictions, which are the correct data products for a leakage-safe stacker.[4] The missing pieces are probability calibration, prediction aggregation, fold-weight selection, and a locked-test ensemble evaluator. The first ensemble should be a simple mean of calibrated fold probabilities; a learned stacker should be a later ablation.

Repeated crop predictions are not the same as independent model ensemble members. The current code can average deterministic views within a case, but view aggregation reduces spatial-sampling noise; it does not measure model uncertainty in the same way as independently trained folds or seeds.[1] [4]

### 3.4 Two datasets require a metadata and label redesign

The current audit and loader are file-centric. They know the case filename, one binary label, and whether the expected `.npy` file exists. They do not record which dataset, site, scanner, protocol, or label source generated a case.[3] Consequently, the code cannot yet run the research brief’s pooled-versus-leave-one-domain-out comparison.

The minimum dataset redesign is a manifest rather than a second folder. Every row should identify the dataset and domain and should make label provenance explicit. If labels are incompatible, a multi-head or target-fine-tuning design is safer than forcing both datasets into the current `grade_proxy` column. If modalities are unpaired, the manifest must represent them as separate modality-specific examples rather than inventing MRI-CT pairs.

## 4. Prioritized implementation path

| Priority | Code change | Why now | Acceptance test |
|---:|---|---|---|
| 1 | Add a modality/dataset manifest and validator | Every later fusion or multi-dataset method depends on correct pairing, provenance, and domain metadata | Reject duplicate case pairs, invalid shapes, conflicting labels, missing paths, and ambiguous modality joins |
| 2 | Add a CT-only dataset/model baseline | Establishes whether CT contributes signal before fusion complexity is introduced | CT-only training and evaluation on the same paired subject split as MRI |
| 3 | Add late probability fusion | Reuses independent model interfaces and is easiest to audit | MRI-only, CT-only, equal-weight, validation-weighted, and stacker-free late-fusion comparison |
| 4 | Add calibration and ensemble evaluation | Existing code has thresholds and CV predictions but no calibration fit | Out-of-fold calibration metrics and one locked-test ensemble report |
| 5 | Refactor encoders and add intermediate fusion | Tests cross-modal interactions after low-risk baselines are known | Intermediate fusion beats or explains its trade-off versus late fusion under the same split |
| 6 | Add missing-modality training | Necessary only if deployment cases can lack CT or MRI | MRI-only, CT-only, paired, and expected deployment-pattern metrics |
| 7 | Add multi-dataset/domain experiments | Prevents pooled-data gains from being mistaken for generalization | Per-domain metrics and leave-one-dataset-out evaluation |
| 8 | Explore distillation or domain adaptation | Higher complexity and greater risk of hiding data problems | Pre-registered ablation showing benefit beyond the simpler baselines |

## 5. Exact boundary between research and code

| Claim | Supported now? | Evidence |
|---|---|---|
| The project has a four-channel MRI classifier | Yes | `EXPECTED_SHAPE`, `TinyGradeClassifier3D(in_channels=4)`, and active trainer |
| The project can combine two trained models | No | Single-model checkpoint load in the evaluator |
| The project can combine MRI and CT | No | No CT path or paired modality manifest |
| The project can ensemble CV folds | Not yet | Fold checkpoints and OOF predictions exist, but no aggregation/evaluator |
| The project can calibrate probabilities | Not yet | Reliability curve is plotted; no calibration transform is fitted |
| The project can train on two heterogeneous datasets | No | No dataset/site/scanner metadata or domain-aware split |
| The project can handle missing modalities | No | Fixed four-channel input and no presence mask |
| The project has leakage-aware evaluation infrastructure | Yes, for the current MRI task | Subject-disjoint splits, locked-test flag, and CV coverage checks |

The correct wording in project documentation is therefore: **“The repository implements a leakage-aware MRI-only baseline and contains partial cross-validation artifacts that can support future ensemble work. Multimodal fusion, multi-dataset training, missing-modality robustness, and domain adaptation remain unimplemented.”**

## References

[1]: [`src/grade_data.py`: current data contract, splits, patch loader, and balanced sampler](../src/grade_data.py)

[2]: [`src/grade_model.py` and `scripts/train_ultra_light.py`: current model, training, threshold, and checkpoint path](../src/grade_model.py)

[3]: [`scripts/verify_preprocessed_data.py`: current file and label-quality audit](../scripts/verify_preprocessed_data.py)

[4]: [`scripts/cross_validate_repaired.py`: development-only folds, checkpoints, and out-of-fold predictions](../scripts/cross_validate_repaired.py)

[5]: [`scripts/evaluate_repaired.py`: current single-checkpoint locked-test evaluator](../scripts/evaluate_repaired.py)

[6]: [Web-researched fusion and heterogeneous-dataset options](21_multimodal_fusion_and_dataset_integration.md)
