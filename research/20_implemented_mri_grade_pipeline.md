# Implementation-Aligned Research Addendum: MRI-Only Grade-Proxy Classification

**Project:** `major101`  
**Author:** Manus AI  
**Purpose:** Extend the research record so that it describes the code that is currently implemented, rather than the broader CT+MRI, segmentation, or longitudinal system that remains planned.

## 1. Scope and evidence boundary

The active, reproducible path in this checkout is a **four-channel MRI-only binary classifier**. It predicts an ET-derived `grade_proxy` label, represented as a low/high target for the current experiment. The implementation does not currently perform CT+MRI fusion, tumour segmentation, longitudinal change modelling, survival analysis, federated learning, or clinical-grade WHO grading. Those topics remain valid research directions, but they must not be presented as implemented capability.

This addendum separates **retrieved implementation facts** from interpretation. Implementation facts are taken from the source files and local run artefacts listed in the References section. The numerical test result is reported as an experiment result, not as evidence of clinical utility or external validity.

> **Decision-useful interpretation:** The codebase now supports a leakage-aware MRI classification experiment with a locked-test evaluation path. It is a research prototype for a binary grade proxy, not a deployable multimodal diagnostic system.

## 2. What is implemented versus what is planned

| Research capability | Current implementation | Evidence in checkout | Status boundary |
|---|---|---|---|
| MRI image classification | Four-channel 3D CNN over preprocessed MRI volumes | `src/grade_data.py`, `src/grade_model.py`, `scripts/train_ultra_light.py` | Implemented |
| Binary target | ET-derived `grade_proxy`, encoded as low/high | `src/grade_data.py`, `research/BraTS_MRI_Grade_Classification_Panel_Report.md` | Implemented for this experiment; not equivalent to clinical WHO grade |
| Subject-disjoint splitting | Seeded stratified grouping by subject identifier | `src/grade_data.py`, split manifest artefact | Implemented |
| Memory-bounded loading | NumPy memory mapping and crop-first patch loading | `src/grade_data.py`, `scripts/train_ultra_light.py` | Implemented |
| Class balancing | Balanced batch sampler with one half-batch from each binary class | `src/grade_data.py` | Implemented for training only |
| CT+MRI fusion | No active CT input path in the repaired trainer | `research/06_final_report.md` and current scripts | Planned, not implemented |
| Segmentation | No segmentation head, mask loss, or Dice/HD95 training path in the active classifier | `src/grade_model.py` | Planned or separate research topic |
| Longitudinal modelling | No timepoint encoder, registration, or temporal aggregation in the active path | `research/05_longitudinal_analysis.md` | Planned |
| Clinical calibration | Threshold selection and binary metrics exist; formal calibration has not been completed | `scripts/evaluate_repaired.py`, panel report | Partial evaluation support |
| Explainability | Explainability artefacts exist in the repository, but saliency evidence is not a validity proof | `outputs/explainability`, panel report | Research diagnostic only |

The main consequence is that implementation claims should now be anchored to the MRI-only path. The older combined-modality research report remains useful as a design rationale, but its CT-fusion accuracy expectations should not be used to characterize the current model.[1]

## 3. Data contract and label semantics

The active data loader expects one `.npy` volume per case with shape `(4, 182, 218, 182)`. The four channels are the preprocessed MRI modalities **T1, T1ce, T2, and FLAIR**. The loader validates the expected shape and rejects non-finite sampled values before a patch is returned. The volumes are treated as already preprocessed; the active trainer does not silently perform a second full-volume normalization step.[2]

The label table is checked before training. Duplicate rows for the same case are collapsed only when their labels agree; conflicting labels are rejected. The local data-quality record reports 994 CSV rows, 876 unique labelled cases, six unlabelled orphan volumes excluded from training, and zero conflicting duplicate labels. These counts describe the current checkout and should be regenerated if the data directory or label file changes.[3]

The target is a dataset-derived `grade_proxy` rather than an independently adjudicated clinical grade. A high locked-test AUROC therefore means that the model ranks the proxy labels on the held-out cases; it does not establish tumour grade diagnosis, treatment selection value, or superiority to a radiologist.[4]

### 3.1 Patch policy

The `MemoryMappedPatchDataset` API defaults to `64 x 64 x 64` patches, but the active `train_ultra_light.py` and `cross_validate_repaired.py` command-line defaults set `--patch-size` to `96`. The actual patch size is therefore a run-level parameter recorded in the run configuration and checkpoint metadata. Patch selection is crop-first, and the dataset can search candidate crop locations rather than loading every full volume into GPU memory. The command-line `--whole-volume` option is available for a downsampled whole-volume alternative, but it is not the default memory-saving path.[2] This distinction matters when comparing experiments: patch size and whole-volume mode change the spatial evidence available to the classifier.

Training-time perturbation is limited to the augmentation controls exposed by `MemoryMappedPatchDataset`, including optional additive noise and spatial flipping. Validation and locked-test loaders are constructed without training augmentation. The implementation should therefore report the patch policy and augmentation settings with every future result rather than comparing metric values as if all runs used the same input view.[2]

## 4. Split design and leakage control

The implementation derives a subject key from case identifiers so that repeated acquisitions remain in the same partition. The canonical seed-42 split recorded in the current evidence package contains **700 training cases, 88 validation cases, and 88 locked-test cases**. The development pool used for cross-validation is the 788-case train-plus-validation set; the locked test set is not used for model selection.[3] [4]

The training script defaults to validation-only operation and requires an explicit `--evaluate-test` flag before the locked partition is evaluated. This is a useful operational safeguard because it makes accidental test peeking less likely. It is not a substitute for reviewing the split manifest, verifying subject grouping, and preserving the test checkpoint before any threshold or hyperparameter decision is made.[5]

The current evidence supports the following evaluation hierarchy:

| Stage | Cases | Permitted use |
|---|---:|---|
| Training | 700 | Gradient updates and training diagnostics |
| Validation | 88 | Early stopping, checkpoint selection, and development decisions |
| Locked test | 88 | One-time held-out estimate after the candidate is frozen |
| Development cross-validation | 788 | Stability estimate using train plus validation only |

A completed five-fold development evaluation is the next acceptance gate. The repository contains partial cross-validation artefacts, but an interrupted run must not be averaged and described as a completed five-fold result.[4]

## 5. Model and optimization path

The active classifier is `TinyGradeClassifier3D`, a deliberately small 3D CNN designed for a constrained GPU budget. With the trainer default `base_channels=12`, the channel schedule is 12, 24, 48, and 96. Each downsampling block uses a 3D convolution followed by GroupNorm and a nonlinearity; adaptive global average pooling feeds a compact fully connected head with dropout and one output logit.[6]

GroupNorm is an engineering choice for small batches. It avoids relying on batch statistics when the memory budget limits the per-step batch size. The choice should be described as a resource-aware design decision, not as evidence that GroupNorm is clinically or universally superior to BatchNorm.[6]

The trainer exposes AdamW-style learning-rate and weight-decay controls, warm-up, early stopping, mixed precision on CUDA, gradient accumulation, and explicit random seeding. Its defaults are a three-epoch validation smoke test, learning rate `1e-4`, weight decay `1e-4`, three warm-up epochs, patience of 12 validation checks, focal gamma `0.0`, label smoothing `0.0`, dropout `0.25`, and seed `42`. A smoke test is a contract check and should not be described as a converged training run.[5]

The binary loss and metric helpers report threshold-dependent confusion-matrix metrics alongside threshold-free ranking metrics. This is appropriate for the current imbalance-sensitive experiment: accuracy alone can hide poor performance on the minority class, while AUROC does not describe the operating point selected for deployment. Future reports should include the threshold, class counts, confusion matrix, AUROC, average precision, balanced accuracy, sensitivity, specificity, and F1 together.[6] [7]

## 6. Current evidence and calibrated interpretation

The last completed locked-test candidate recorded in the repository was evaluated on 88 unseen cases and reported balanced accuracy **0.5853**, accuracy **0.5114**, AUROC **0.7121**, F1 **0.6055**, sensitivity **0.4648**, and specificity **0.7059**, with confusion counts TN=12, FP=5, FN=38, and TP=33.[4] These values are useful for establishing a reproducible baseline, but they do not justify the older research report's CT+MRI performance expectations.

The result has three immediate implications. First, the model contains a measurable ranking signal because AUROC is higher than chance in this held-out experiment, but ranking performance and thresholded classification performance are not interchangeable. Second, the low sensitivity and the confusion matrix show that the selected operating point misses many positive cases; any future clinical discussion must treat this as a safety limitation rather than hiding it behind the AUROC. Third, the result is tied to the current proxy labels, split, preprocessing, patch policy, and single locked-test cohort. It is not an externally validated clinical estimate.[4]

The repository also reports a data-quality audit of 882 preprocessed files with zero invalid files and the expected four-channel volume shape. That audit supports input integrity for this checkout; it does not verify scanner harmonization, annotation validity, acquisition representativeness, or calibration on a new institution.[3]

## 7. Research coverage that should be extended next

The most relevant next research work is narrower than the original multimodal roadmap. The immediate priority is a completed subject-disjoint five-fold development evaluation, followed by a frozen final MRI checkpoint and one locked-test evaluation. The second priority is probability calibration and selective prediction, because the current report includes thresholded metrics but has not established reliable probabilities. The third priority is independent validation of the `grade_proxy` construction and an explicit comparison against clinically meaningful labels. Only after those gates should CT be introduced as a paired-modality experiment with its own registration, missing-modality, and data-integrity analysis.[4]

| Priority | Research question | Required evidence |
|---|---|---|
| 1 | Is the MRI baseline stable across subject-disjoint folds? | Completed five-fold development metrics and out-of-fold predictions |
| 2 | Are probabilities and thresholds reliable? | Calibration curve, Brier score, calibration error, and prespecified operating point |
| 3 | Does the proxy label measure the intended clinical construct? | Independent label audit and explicit mapping from proxy to clinical grade |
| 4 | Does CT add value beyond MRI? | Paired CT-MRI cohort, modality-ablation study, missing-modality analysis, and locked comparison |
| 5 | Does the model generalize? | External cohort with scanner/site metadata and a frozen preprocessing contract |

## 8. Documentation synchronization rules

Future research files should label claims as **implemented**, **planned**, or **not evaluated**. Any metric table should include the dataset version, label definition, subject split, patch or whole-volume policy, model width, checkpoint-selection rule, threshold-selection rule, and whether the test set was touched. This prevents the earlier broad literature targets from being mistaken for results from the current code.

The existing files `research/06_final_report.md`, `research/16f_preprocessing_pipelines.md`, and `research/cat17_evaluation_metrics_progression.md` should be read together with this addendum. The first supplies the broader multimodal rationale, the second supplies preprocessing research context, and the third supplies evaluation-method background. This file is the implementation anchor for the active MRI-only experiment.

## References

[1]: [Research report: Combined CT + MRI brain tumour classification](06_final_report.md)
[2]: [Active data utilities: `src/grade_data.py`](../src/grade_data.py)
[3]: [Preprocessed-data and label-quality evidence in the handover](../HANDOVER.md)
[4]: [BraTS MRI grade-classification panel report](BraTS_MRI_Grade_Classification_Panel_Report.md)
[5]: [Active trainer: `scripts/train_ultra_light.py`](../scripts/train_ultra_light.py)
[6]: [Active classifier and metric helpers: `src/grade_model.py`](../src/grade_model.py)
[7]: [Locked-test evaluator: `scripts/evaluate_repaired.py`](../scripts/evaluate_repaired.py)


## Related web research for the next extension

The follow-on design research is documented in [`21_multimodal_fusion_and_dataset_integration.md`](21_multimodal_fusion_and_dataset_integration.md). It compares early, intermediate, late, ensemble, distillation, shared-specific, missing-modality, pooled-dataset, domain-adaptation, and domain-generalization strategies against the constraints of this implementation.
