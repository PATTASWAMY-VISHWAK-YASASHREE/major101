# Memory-Bounded 3D CNN Classification of a BraTS 2024 Glioma Grade Proxy: A Data-Audited, Subject-Disjoint Baseline with Locked-Test Evaluation

**Manuscript draft v1 — research-grade. Not a clinical tool. No synthetic data. No multimodal fusion.**

---

## Abstract

**Background.** Deep learning classifiers for brain tumour grading are typically trained on the BraTS benchmark, but published pipelines frequently under-report data-quality issues: duplicate label rows, repeated acquisitions crossing train/test boundaries, full-volume memory footprints, and class-collapse hidden by majority-class accuracy. The BraTS 2024 glioma collection also labels cases via a `grade_proxy` derived from enhancing-tumour (ET) presence rather than an independent WHO-grade annotation — a property that must be stated explicitly rather than buried.

**Methods.** We train a compact 3D CNN (501,289 parameters; GroupNorm; 96³ crop-first loading; mixed precision) on 4-channel preprocessed BraTS 2024 MRI under explicit 2 GiB VRAM / 3 GiB RAM guards on a single consumer GPU. The pipeline verifies all 882 preprocessed volumes, deduplicates 994 label rows to 876 unique cases, groups repeated acquisitions by subject, and enforces subject-disjoint splits. Evaluation uses five-fold subject-disjoint cross-validation over all 788 development cases, development-only temperature calibration (T = 0.801) and threshold selection (0.53), and a single locked-test evaluation of 88 unseen cases.

**Results.** Mean best-fold AUROC was 0.7987; pooled out-of-fold AUROC was 0.7641 (bootstrap 95% CI 0.7261–0.8018). On the locked test, calibrated AUROC was 0.7672 and average precision 0.9431, but balanced accuracy was 0.5261 and accuracy 0.7045 — below the 0.8068 majority-HIGH baseline. The model ranks cases above chance but does not currently form a strong balanced thresholded classifier. Negative results are reported: a five-member ensemble underperformed a single mature model (OOF AUROC 0.5463 vs 0.7641), and five-view crop aggregation degraded mature checkpoints (0.7641 → 0.6058).

**Conclusion.** A rigorously audited, leakage-safe, memory-bounded pipeline yields an honest, modest baseline on an ET-derived proxy label. The ranking signal is real; the thresholded classifier is weak. Because the target is proxy-derived, no claim of independent WHO-grade prediction or clinical utility is made. All results derive from repository artifacts and are reproducible under the stated hardware budget.

---

## 1. Introduction

Gliomas are among the most lethal primary brain tumours, and non-invasive grading from routine MRI directly influences surgical planning and treatment intensity. The BraTS challenge ecosystem provides the field's principal public benchmark of preprocessed, expert-annotated glioma MRI [1,2,13,14]. Deep learning classifiers on BraTS-scale data are therefore common, yet three practical problems recur across the literature:

1. **Data-integrity defects inflate results.** Duplicate rows in label files and repeated acquisitions of the same subject crossing train/test boundaries produce optimistic scores. Record-wise image splits can overestimate diagnostic performance, as shown for brain MRI classification specifically [15].

2. **Full-volume loading assumes memory that many institutions do not have.** Typical four-channel BraTS volumes are ~115 MB as float32; naive pipelines materialise many volumes at once. On commodity hardware (a single laptop-class GPU with 4 GB VRAM and 3 GB process RAM), this fails outright.

3. **Class imbalance hides collapse.** The BraTS 2024 glioma collection is roughly 4:1 imbalanced toward the HIGH class. A model predicting the majority class everywhere achieves ~80% raw accuracy while learning nothing; accuracy alone rewards this failure mode.

This paper documents a pipeline built to survive all three constraints, evaluated on the BraTS 2024 preprocessed glioma data with a binary target derived from the collection's `grade_proxy` field. We treat the label's proxy construction — ET presence, not independent pathology — as a first-class scientific limitation rather than a footnote, and we report all metrics, including those that look unfavourable, alongside measured negative results.

The contribution is deliberately narrow: a reproducible, leakage-safe, resource-bounded baseline with an honest characterisation of what it does and does not demonstrate. It is not a clinical tool, makes no claim of WHO-grade prediction, and does not attempt multimodal CT+MRI fusion (no public paired dataset exists at benchmark scale; the gap is discussed in related work).

---

## 2. Related work

**BraTS benchmarks.** The original BraTS benchmark established the multimodal MRI segmentation paradigm [13]; TCGA-derived collections added expert labels and radiomics [14]; the 2024 challenge focuses on post-treatment glioma segmentation [1]. These datasets define the field's evaluation standards, and their preprocessed format (four MRI channels, skull-stripped, co-registered, resampled) is used here unchanged.

**Segmentation and classification methods.** nnU-Net [3] demonstrated that self-configuring pipelines outperform hand-tuned architectures; U-Net [4], V-Net [5], and ResNet-style [6] backbones are standard. For classification, compact 3D CNNs remain competitive when data and memory are limited; transformer hybrids exist but assume larger compute budgets. A survey of deep learning in medical image analysis situates these methods [7].

**Leakage and evaluation rigour.** Subject-wise versus record-wise splitting materially changes reported performance in brain MRI classification [15]; grouped, stratified splits are the accepted mitigation. Calibration of network probabilities via temperature scaling is standard practice for interpreting outputs [8], as is validation-only threshold selection under class imbalance.

**Imbalance handling.** Focal loss [9] and class-balanced losses [10] address long-tailed training; balanced batch sampling exposes the minority class in every optimisation step. We adopt balanced sampling with plain binary cross-entropy after finding stronger focal modulation unstable at this scale.

**CT+MRI fusion.** Multimodal CT+MRI fusion is an active direction for other pathologies and in institutional datasets, but no public paired CT+MRI brain-tumour benchmark exists at BraTS scale; published fusion results are therefore not reproducible or comparable across groups. This work stays MRI-only accordingly.

---

## 3. Data

### 3.1 Source data

The study uses the BraTS 2024 preprocessed adult glioma collection (four channels: T1ce, T1n, T2-FLAIR, T2w; stored shape (4, 182, 218, 182), float32, ~28.8M values per case). Values arrive preprocessed (z-score normalised, clipped, scaled); the pipeline performs no second full-volume normalisation.

### 3.2 Label construction and its limitation

The binary target is the collection's `grade_proxy`: LOW (0) versus HIGH (1), where the proxy is derived from enhancing-tumour presence rather than independently supplied pathology or WHO-grade annotation. Consequently:

> A model can learn a relationship between MRI appearance and the ET-derived proxy without learning clinically valid glioma grade.

This is the largest scientific limitation of the study and is carried through every result below. A previously built scalar feature model using ET-derived volume features was excluded from all claims because those features reproduce the label-construction rule; its near-perfect historical score is an artefact of label construction, not evidence of grading ability.

### 3.3 Data audit

All 882 preprocessed volumes were inspected before training (Table 1).

**Table 1 — File-level audit**

| Check | Observed |
|---|---|
| Files inspected | 882 |
| Shape (4,182,218,182) | 882/882 pass |
| dtype float32 | 882/882 pass |
| Non-finite volumes | 0 |
| Invalid files | 0 |
| Labelled cases missing a file | 0 |
| Conflicting duplicate labels | 0 |

**Table 2 — Label-table audit**

| Property | Observed |
|---|---|
| Raw CSV rows | 994 |
| Duplicate rows | 236 |
| Unique labelled cases | 876 |
| LOW proxy | 172 |
| HIGH proxy | 704 |
| Unlabelled orphan volumes (excluded) | 6 |

### 3.4 Splits

The canonical split (seed 42) is subject-disjoint: repeated acquisitions (e.g., `-100` and `-101` suffixes of one subject) are grouped by subject prefix and cannot cross partitions. Development data (train + validation, 788 cases across 300+37 subjects) are used for all cross-validation, calibration, and final fitting; the locked 88-case test partition (41 subjects; 17 LOW / 71 HIGH) is excluded from every development decision and evaluated exactly once.

---

## 4. Methods

### 4.1 Architecture

TinyGradeClassifier3D: four input channels; base width 12; convolutional widths 12→24→48→96 with stride-2 downsampling; GroupNorm after every convolution; LeakyReLU (slope 0.01); adaptive global average pooling; linear head with dropout 0.25; one output logit. **501,289 parameters.** GroupNorm is chosen because the physical batch is 2, making batch-statistics normalisation fragile [11]; Kaiming initialisation suits rectifier activations [12].

### 4.2 Memory-bounded loading

Volumes are read via `np.load(..., mmap_mode="r")`; only one 96³ crop per sample is copied into memory. Training crops use random candidate search favouring T1ce-bright voxels and valid-brain coverage; validation/test crops use deterministic candidate anchors. `num_workers=0` avoids worker-process memory duplication. Guards: 2 GiB reserved VRAM, 3 GiB process RAM. Observed peaks: 1.02 GiB VRAM / 0.43–1.35 GiB RAM across smoke and full runs.

### 4.3 Training

Balanced binary batches (one LOW + one HIGH per step), physical batch 2 with gradient accumulation 4 (effective batch 8); AdamW (lr 1e-4, weight decay 1e-4); 3-epoch warmup; binary cross-entropy via a focal-loss interface at gamma=0 (modulation disabled after instability tests); random flips and small input noise only — no synthetic data generation of any kind; CUDA automatic mixed precision; seed 42; five epochs for the final fit; checkpoint selection by validation AUROC.

### 4.4 Evaluation protocol

1. Five-fold subject-disjoint CV over all 788 development cases (every case receives exactly one out-of-fold prediction).
2. Deterministic bootstrap 95% CIs on pooled OOF metrics.
3. Temperature scaling (T = 0.801) fit on development OOF logits only; development threshold 0.53.
4. Final model trained on all 788 development cases; locked 88-case test evaluated once with frozen calibration and threshold. No test-informed tuning of any kind.

A regression test suite covering duplicate labels, conflicting labels, subject-disjoint splitting, balanced sampling, per-sample loss weighting, and multi-crop aggregation passes in full (12/12).

---

## 5. Results

### 5.1 Cross-validation (788 development cases)

**Table 3 — Five-fold subject-disjoint CV**

| Fold | Best AUROC | Balanced acc at best epoch | Val cases |
|---|---|---|---|
| 1 | 0.7635 | 0.7300 | 159 |
| 2 | 0.8031 | 0.7472 | 157 |
| 3 | 0.8618 | 0.7955 | 156 |
| 4 | 0.8257 | 0.8104 | 157 |
| 5 | 0.7392 | 0.7131 | 159 |
| **Mean ± std** | **0.7987 ± 0.0488** | **0.7592 ± 0.0420** | — |

Pooled OOF (threshold 0.52): AUROC 0.7641 (95% CI 0.7261–0.8018), balanced accuracy 0.7288 (95% CI 0.6936–0.7631), average precision 0.9267, sensitivity 0.6382, specificity 0.8194. Majority-class baseline on development: accuracy 0.8033, balanced accuracy 0.5.

### 5.2 Locked test (88 cases, evaluated once)

**Table 4 — Locked-test metrics (calibrated, threshold 0.53)**

| Metric | Value |
|---|---|
| AUROC | **0.7672** |
| Average precision | 0.9431 |
| Accuracy | 0.7045 |
| Majority-HIGH accuracy baseline | 0.8068 |
| Balanced accuracy | 0.5261 |
| Sensitivity | 0.8169 |
| Specificity | 0.2353 |
| F1 / precision | 0.8169 / 0.8169 |
| Confusion | TN=4, FP=13, FN=13, TP=58 |

The model predicted HIGH for 71/88 cases — exactly the test prevalence — yet misclassified 26 cases. Raw accuracy is below the majority baseline; AUROC and balanced accuracy exceed chance and the balanced baseline (0.5). Interpretation: a real ranking signal with a weak operating point.

Calibration (development OOF only): Brier 0.1973 → 0.1960; ECE 0.2366 → 0.2279.

### 5.3 Negative results

- **Compute-matched ensemble (5 members, fixed folds, probability averaging):** pooled OOF AUROC 0.5463, balanced accuracy 0.5715 — far below the single mature model (0.7641 / 0.7288). Ensemble rejected.
- **Five-view inference on mature checkpoints:** OOF AUROC fell 0.7641 → 0.6058; balanced accuracy 0.7288 → 0.5962. Multi-view aggregation rejected at inference time.
- **Ultra-light 2-epoch smoke (batch 2, combined CB+focal loss):** oscillated all-LOW (epoch 1) then all-HIGH (epoch 2) — the class-collapse failure mode that motivated balanced sampling with plain BCE.

### 5.4 External-style unlabeled cohort

A separate raw BraTS validation cohort (188 four-modality cases, no labels, no overlap with supervised data) was inferred once case-by-case with the frozen final checkpoint (peak VRAM 0.221 GiB / RAM 0.484 GiB; zero failures; predictions saved). No metrics are computed because no ground truth exists; no external-validation claim is made.

### 5.5 Within-subject consistency across repeated acquisitions

The development cohort contains substantial repeated-scan structure: 227 of 337 subjects (67%) have ≥2 acquisitions (135 subjects with 2, 29 with 3, 26 with 4, 19 with 5, up to one subject with 10). Because the checkout provides no timestamps, treatment records, or survival endpoints, no longitudinal/progression claim is possible; what can be measured honestly is **prediction consistency within a subject** using the pooled OOF predictions (analysis artifact: `outputs/analysis/within_subject_consistency_report.json`):

**Table 5 — Within-subject consistency (227 subjects with ≥2 acquisitions)**

| Measure | Value |
|---|---|
| Global std of p(HIGH) across all 788 cases | 0.1903 |
| Within-subject std of p(HIGH), mean / median | 0.0553 / 0.0372 |
| Stability ratio (global ÷ within-subject) | **3.4×** |
| Identical predicted label across a subject's scans | 173/227 (76.2%) |
| Identical true label across a subject's scans | 194/227 (85.5%) |
| Subjects with within-subject p(HIGH) spread > 0.30 | 25 |

Two findings follow. First, the 3.4× stability ratio indicates the model tracks subject-level signal rather than per-scan noise: predictions vary far less within a subject than across subjects. Second, the model's 76.2% self-agreement sits **below** the labels' own 85.5% consistency — scan-to-scan variation (and crop sensitivity) produces label flips in roughly one subject in four, and 25 subjects show probability swings larger than 0.30 across their own scans (worst case: 0.378 → 0.956 for the same subject). These discordant subjects are precisely where a single-scan prediction should not be trusted and are retained as an explicit error-analysis target. Suffix-ordered probability trends (descriptive only; suffix order is unverified as temporal) split into 50 increasing, 36 decreasing, 61 mixed, 80 near-flat — no interpretation of these as progression is offered.

### 5.6 Visual evidence

Four test cases were selected by an evidence rule (most-confident correct LOW; a correct HIGH; nearest-threshold ambiguous; most-confident false-LOW): BraTS-GLI-02720-100 (true LOW, p=0.294, correct), BraTS-GLI-02307-101 (true HIGH, p=0.595, correct), BraTS-GLI-02225-101 (true LOW, p=0.466, false HIGH, margin 0.006), BraTS-GLI-02651-100 (true HIGH, p=0.286, confident false-LOW). For each, all four MRI channels are shown in three orthogonal planes with Grad-CAM [16] and input-gradient saliency [17] overlays. The maps demonstrate model behaviour — spatially structured responses — and are explicitly not tumour segmentations or localisation claims; no segmentation-ground-truth comparison was performed. Including the confident failure prevents the visual set from becoming promotional cherry-picking.

---

## 6. Discussion

**What the results support.** The pipeline learned a non-random ranking signal for the ET-derived proxy on an unseen, subject-disjoint partition (AUROC 0.7672; AP 0.9431), reproducible under a 2 GiB VRAM / 3 GiB RAM budget with complete data provenance. The audit, grouped splitting, balanced sampling, calibration, and locked-test protocol solved the specific correctness problems that motivated this work: duplicate rows, subject leakage, memory pressure, class collapse, and threshold selection on test data.

**What they do not support.** The model does not beat the majority baseline on raw accuracy, its specificity at the operating point is low (0.2353), and — decisively — the target is an ET-derived proxy. A better score may reflect better recovery of enhancing-tumour presence rather than better grading. No claim of WHO-grade prediction, clinical utility, or generalisation beyond this dataset is made; the raw 188-case cohort inference is a resource demonstration, not validation.

**Why report the negative results.** The ensemble and multi-view failures bound what helps at this scale: under short-training compute budgets, ensembles of undertrained members are worse than one mature model, and naive crop aggregation can dilute a mature checkpoint's signal. These are measured, reported, and recommended against for anyone reproducing this work on comparable hardware — the study's intended audience.

**Resource constraint as design, not apology.** GroupNorm for batch-2 stability, mmap crop-first loading, balanced sampling, AMP, and gradient accumulation were all forced by the hardware budget and all contributed to a reproducible result. The pipeline is a reference implementation for research settings — including undergraduate colleges — where a single consumer GPU is the only compute available.

---

## 7. Limitations

1. **Label validity:** the target is ET-derived proxy, not independent WHO grade or pathology. This bounds every interpretation above.
2. **Sample size:** the locked test has 88 cases (17 LOW); small counts materially move sensitivity/specificity. No test-set CI is reported because the test was evaluated once.
3. **Single split:** development has full OOF coverage, but the locked test remains one seed-42 estimate — not external validation.
4. **Crop coverage:** the model sees 96³ crops, not full volumes; tumour regions outside selected crops are invisible to each prediction. Section 5.5 quantifies the practical consequence: 25 subjects show >0.30 within-subject probability swings, partly attributable to crop sensitivity.
5. **No temporal semantics:** repeated acquisitions lack timestamps and treatment context; the within-subject analysis of Section 5.5 measures consistency only, and no progression or monitoring inference is drawn.
6. **Calibration scope:** temperature was fit on development OOF only; `p(HIGH)` is not a validated clinical probability.
7. **Explanation validity:** Grad-CAM/saliency are qualitative model-behaviour evidence; no faithfulness or expert-segmentation comparison was performed.
8. **External validity:** no hospital, scanner, protocol, or demographic shift was tested; the model must not be deployed outside this research setting.
9. **Resource evidence:** memory guards prove the tested configuration fits the budget; they do not guarantee constant GPU utilisation or freedom from OS paging.

---

## 8. Conclusion

A compact, memory-safe, subject-disjoint 3D CNN pipeline was built and audited end-to-end on BraTS 2024 preprocessed glioma MRI under a 2 GiB VRAM / 3 GiB RAM budget. Five-fold OOF AUROC is 0.7641 (CI 0.7261–0.8018); the single locked-test evaluation gives AUROC 0.7672 but balanced accuracy 0.5261, below majority-baseline accuracy. The strongest defensible claim:

> A compact, memory-bounded 3D MRI model can learn a non-random ranking signal for this dataset's ET-derived grade proxy, but current evidence is insufficient for independent clinical grading, and the thresholded classifier is weak.

The scientifically most valuable next step is a target with independent clinical provenance (pathology-confirmed grade) and external MRI cohorts — not more proxy tuning, not synthetic data, and not multimodal fusion until a genuinely paired, independently labelled public CT+MRI benchmark exists.

---

## References

[1] de Verdier, M. C. et al. *The 2024 Brain Tumor Segmentation (BraTS) Challenge: Glioma Segmentation on Post-treatment MRI.* arXiv:2405.18368 (2024).

[2] BraTS documentation. https://brats.readthedocs.io/en/stable/

[3] Isensee, F. et al. *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods 18, 203–211 (2021). PMID 33288961.

[4] Ronneberger, O., Fischer, P., Brox, T. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015, LNCS 9351, 234–241.

[5] Milletari, F., Navab, N., Ahmadi, S.-A. *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.* 3DV 2016, 565–571.

[6] He, K., Zhang, X., Ren, S., Sun, J. *Deep Residual Learning for Image Recognition.* CVPR 2016, 770–778.

[7] Litjens, G. et al. *A survey on deep learning in medical image analysis.* Medical Image Analysis 42, 60–88 (2017). PMID 28778026.

[8] Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. *On Calibration of Modern Neural Networks.* ICML 2017.

[9] Lin, T.-Y. et al. *Focal Loss for Dense Object Detection.* ICCV 2017.

[10] Cui, Y. et al. *Class-Balanced Loss Based on Effective Number of Samples.* CVPR 2019.

[11] Wu, Y., He, K. *Group Normalization.* ECCV 2018, 3–19.

[12] He, K. et al. *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.* ICCV 2015.

[13] Menze, B. H. et al. *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS).* IEEE TMI 34(10), 1993–2024 (2015). PMID 25547933.

[14] Bakas, S. et al. *Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features.* Scientific Data 5, 170117 (2018). PMID 28872634.

[15] Yagis, E. et al. *Effect of data leakage in brain MRI classification using 2D convolutional neural networks.* Scientific Reports 11 (2021).

[16] Selvaraju, R. R. et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* ICCV 2017, 618–626.

[17] Simonyan, K., Vedaldi, A., Zisserman, A. *Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps.* arXiv:1312.6034 (2014).

[18] Loshchilov, I., Hutter, F. *Decoupled Weight Decay Regularization.* ICLR 2019.

---

**Source-of-truth artifacts for every number above:**
`outputs/data_quality/preprocessed_data_report.json` · `outputs/cv/full_epoch_baseline_5fold_5ep/summary.json` · `outputs/cv/full_epoch_baseline_5fold_5ep/out_of_fold_predictions.csv` · `outputs/analysis/within_subject_consistency_report.json` (Section 5.5) · `outputs/calibration/repaired/calibration.json` · `outputs/training/repaired_final/{best_checkpoint.pth, history.csv, development_manifest.json}` · `outputs/evaluation/repaired_final/{summary.json, test_predictions.csv}` · `outputs/explainability/repaired_final/visual_evidence_manifest.json` · `outputs/external_validation/{predictions.csv, validation_manifest.json}` · `tests/` (12/12)

**BibTeX:** `research/consolidated_verified_references.bib` (all DOIs/PMIDs verified; fabricated-PMID files excluded)
