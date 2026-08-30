# Specific Aims — Brain Tumour CT+MRI Multimodal Classification

**NIH R21-style draft — 2 pages, exploratory/developmental scope**
**Grounded in actual project state: tier-3 college, no budget, public data only, RTX 2050**

---

## Opening Hook

Brain tumours are among the deadliest cancers, with glioblastoma median survival under 15 months despite maximal therapy. Accurate non-invasive grading from medical imaging directly guides treatment intensity, surgical planning, and clinical trial eligibility. MRI is the clinical gold standard for brain tumour imaging, but in many low- and middle-income countries — where the majority of the world's population lives — CT is the primary or only neuroimaging modality available. MRI scanners are expensive, scarce, and concentrated in urban centres; CT scanners are comparatively ubiquitous.

## Gap

Despite the clinical reality that millions of patients receive brain CT scans without access to MRI, virtually all published deep learning research on brain tumour classification uses MRI alone. The Brain Tumour Segmentation (BraTS) benchmark — the field's primary public dataset — provides MRI only. No public benchmark offers paired CT+MRI for brain tumours. The small number of studies that have explored CT+MRI fusion used proprietary hospital data, making their results neither reproducible nor comparable. The fundamental question — "what does CT add to MRI-only brain tumour classification on publicly available data?" — remains unanswered, not because it isn't important, but because the data and baselines have not been assembled.

## Long-Term Goal

Develop validated, open, multimodal (CT+MRI) deep learning tools for brain tumour classification that are reproducible on public data and accessible to researchers and clinicians in resource-limited settings where CT is the primary neuroimaging modality.

## Objective of This Application

Establish the first reproducible CT+MRI fusion benchmark for brain tumour classification on publicly available data, quantify the marginal value of CT over MRI alone, and release an open-source pipeline designed to run on commodity hardware (≤4GB VRAM, ≤3GB RAM).

## Central Hypothesis

CT provides complementary diagnostic information beyond MRI for brain tumour classification — particularly for calcification detection and bone involvement — and this complementarity translates into measurable accuracy gains when both modalities are available, but the gain is modest (≤5% in the literature) and the small size and limited availability of public paired datasets is the binding constraint on what can be learned.

## Rationale

The applicant has completed: (1) a rigorous MRI-only glioma grade-proxy classification pipeline on BraTS 2024 with subject-disjoint five-fold cross-validation (pooled OOF AUROC 0.7641), locked-test evaluation (AUROC 0.7672 on 88 unseen cases), and full data-quality auditing (882/882 preprocessed files verified); (2) a comprehensive literature review of 117 papers across 19 categories spanning 2013–2026, which confirms that CT+MRI fusion is under-studied for brain tumours and that no public paired benchmark exists; and (3) an engineering pipeline designed for the 4GB VRAM / 3GB RAM constraints of commodity hardware, with memory-mapped loading, crop-first reading, GroupNorm-based 3D CNNs, and balanced batch training. The applicant has identified multiple publicly available paired CT+MRI datasets (IEEE DataPort glioblastoma benchmark with 50 patients, Kaggle multimodal brain tumour images, Zenodo brain metastases cohort) that can serve as the first fusion validation sets.

This application builds directly on that foundation. It does not claim to solve clinical brain tumour grading. It claims to establish the first open, reproducible, resource-constrained multimodal fusion benchmark, and to document honestly what CT adds, what it does not add, and what the data limitations are.

## Specific Aims

**Aim 1: Curate and standardise a public CT+MRI brain tumour dataset with documented preprocessing.**

*Rationale*: The binding constraint on CT+MRI fusion research is the absence of a standardised, publicly available paired dataset with documented preprocessing. Multiple small paired datasets exist (IEEE DataPort 50 GBM patients, Kaggle multimodal images, Zenodo brain metastases) but none has a published, reproducible preprocessing pipeline that makes them directly comparable or usable for fusion benchmarking.

*Approach*: Download and verify each publicly available paired CT+MRI dataset. Develop and document a unified preprocessing pipeline: skull-stripping (where feasible), intensity normalisation per modality, rigid registration of CT to MRI space using mutual information maximisation, resampling to isotropic resolution, and quality-control reporting per case. Produce a data report for each dataset analogous to the BraTS preprocessed data report already generated for this project (882/882 files verified). Where registration fails or data is incomplete, document and exclude — the goal is a clean, verified dataset, not maximum case count.

*Expected outcome*: At least one verified, preprocessed, publicly available paired CT+MRI brain tumour dataset with a documented pipeline that other researchers can re-run. A data-quality report documenting case counts, exclusion reasons, and preprocessing parameters.

---

**Aim 2: Train and evaluate the first reproducible late-fusion 3D CNN on public CT+MRI brain tumour data.**

*Rationale*: The literature consensus (6 of 8 reviewed multimodal studies) supports late fusion — separate encoders per modality with feature concatenation — as the most extensible and well-motivated architecture. The applicant's MRI-only pipeline provides a validated 3D CNN baseline (TinyGradeClassifier3D, 501K parameters, GroupNorm, 96³ patches, runs under 2GB VRAM). Extending this to a dual-encoder late-fusion architecture is a natural, low-risk extension that can be evaluated against the MRI-only baseline on the same cases where both modalities are available.

*Approach*: Implement a late-fusion architecture: separate 3D CNN encoders for CT and MRI (reusing the validated TinyGradeClassifier3D design, one branch per modality), feature concatenation, a small dense fusion head, and binary or 4-class output as appropriate for each dataset's labels. Train and evaluate under the same memory guards (≤2GB VRAM, ≤3GB RAM) using the same training protocol (balanced batches, AMP, crop-first loading). For each dataset where both modalities are available for the same case, compare: (a) CT-only, (b) MRI-only, (c) late-fusion CT+MRI. Report AUROC, balanced accuracy, and per-class metrics with confidence intervals. Where labels are not available (e.g., the raw BraTS validation cohort), report predictions only — no metrics.

*Expected outcome*: The first published comparison of CT-only vs. MRI-only vs. CT+MRI late fusion on public brain tumour data, with honest metrics and documented limitations. Quantification of the marginal value of CT (expected: modest, ≤5% accuracy gain based on literature). A clear statement of what the public data supports and what it does not.

---

**Aim 3: Release an open-source, resource-constrained multimodal pipeline and benchmark as a reproducible research artifact.**

*Rationale*: The value of this work to the broader community depends on its reproducibility and accessibility. A pipeline that requires a 40GB GPU cluster is not accessible to the majority of the world's researchers or to the low-resource clinical settings where CT-only AI would be most valuable. A pipeline that runs on a 4GB GPU, with all code, preprocessing, and evaluation protocols open-sourced and documented, is.

*Approach*: Generalise the existing pipeline code (currently MRI-only) to support optional CT input and fusion mode. Document the full pipeline in a single runnable README: data download, preprocessing, training, evaluation, and visual evidence generation. Include the same contract tests and data-quality auditing developed for the MRI-only pipeline. Package the fusion benchmark — dataset, preprocessing, trained checkpoints, evaluation protocol — as a standalone reproducible artifact. Write a panel-ready report (following the format already established in `BraTS_MRI_Grade_Classification_Panel_Report.md`) that presents methods, results, visual evidence, and limitations honestly.

*Expected outcome*: An open-source repository containing: (1) the generalised CT+MRI pipeline, (2) preprocessed public datasets with data-quality reports, (3) trained checkpoints and evaluation results, (4) a panel report with honest interpretation. All designed to run on commodity hardware. This artifact serves as the first open CT+MRI brain tumour fusion benchmark and a reference implementation for resource-constrained medical AI research.

---

## Payoff

If successful, this project will have produced the first reproducible open benchmark for CT+MRI brain tumour classification, quantified the marginal value of CT over MRI alone on public data, and released an accessible pipeline that enables researchers at any institution — including those without GPU clusters or proprietary hospital data — to contribute to multimodal brain tumour AI. It will also have produced, as a necessary byproduct, a rigorously audited MRI-only baseline with honest performance characterisation that the field can use as a reference point.

The project does not claim to solve brain tumour grading, replace clinical diagnosis, or demonstrate generalisability beyond the public datasets used. It claims to fill a specific, real, and under-addressed gap: the absence of any open, reproducible CT+MRI fusion benchmark for brain tumours, and the concentration of the field's methodological development in settings that exclude most of the world's researchers and patients.
