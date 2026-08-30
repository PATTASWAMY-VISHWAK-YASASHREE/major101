# Project Summary — BraTS Glioma Classification Research

**NSF-style project summary draft**
**Scope: research-grade classification on public datasets only. No synthetic data generation. No clinical deployment claims. No multimodal fusion claims beyond literature review.**

---

## Overview

Brain tumour classification from medical imaging is a well-studied problem in deep learning, yet the field's primary public benchmark — BraTS — provides MRI only, and virtually all published methods use MRI alone. This project documents the state of the field, establishes a rigorously audited MRI-only baseline on BraTS 2024, and characterises — through literature review — the open questions around multimodal (CT+MRI) fusion that remain unanswered due to the absence of public paired data.

The work has three components:

1. **A reproducible MRI-only classifier.** A memory-bounded 3D CNN (GroupNorm, crop-first loading, balanced batches, mixed precision) trained and evaluated on BraTS 2024 glioma data under subject-disjoint five-fold cross-validation with a locked test set. The pipeline is data-audited (882/882 preprocessed files verified, 994 label rows reduced to 876 unique cases, conflicting labels rejected, repeated acquisitions grouped by subject to prevent leakage) and resource-bounded (designed to run under 2GB VRAM / 3GB RAM on commodity hardware). The target is the dataset's `grade_proxy` label, which is derived from enhancing tumour presence — explicitly not an independent WHO grade or pathology annotation.

2. **A literature review and position paper on CT+MRI fusion for brain tumours.** A survey of 117 papers across 19 categories (2013–2026) that documents what fusion methods exist in the literature, what claims are made, and why the absence of a public paired CT+MRI brain tumour dataset makes most fusion claims impossible to reproduce or compare. This is a **survey and position paper**, not an attempt to build a fusion classifier. No synthetic CT generation, no MRI-to-CT translation, no data augmentation beyond standard flips and noise.

3. **An honest characterisation of what the results do and do not show.** The MRI-only model produces a non-random ranking signal on the proxy label (locked-test AUROC 0.7672, pooled OOF AUROC 0.7641) but is a weak thresholded classifier (balanced accuracy 0.5261 on locked test, below majority-baseline accuracy). These results are reported fully, including the metrics that look poor. No clinical deployment claim is made. No claim of independent glioma grading is made. The work is positioned as a reproducible baseline and a literature survey, not as a clinical tool.

The project acknowledges upfront that the label is proxy-derived, that the model is not clinically validated, that external validation has not been performed (the unlabeled raw validation cohort exists but has no ground truth), and that generalisation beyond BraTS 2024 is untested.

---

## Intellectual Merit

This project advances knowledge in two areas, both at the research level:

**1. A reproducible, data-audited, resource-constrained MRI baseline with honest reporting.**

The MRI-only pipeline is documented in sufficient detail that another researcher with the same hardware constraints (single consumer GPU, ~4GB VRAM) can re-run it. The data audit — file-level verification, label deduplication, subject-disjoint splitting, memory-mapped loading, contract tests — is itself a transferable contribution. In a field where leakage, inflated scores, and irreproducible pipelines are common, a clean, audited, honestly-reported result has standalone value even when the headline metric is modest.

The pipeline's design choices (GroupNorm for small-batch 3D training, crop-first loading for memory bounded-ness, balanced batch sampling for class imbalance, validation-only threshold selection, locked-test evaluation performed once) are each documented with their rationale and their alternatives considered. The negative results (ensemble underperformance, multi-view degradation, ultra-light batch oscillation) are reported alongside the positive results.

**2. A survey and position paper on the CT+MRI fusion gap for brain tumours.**

The 117-paper literature review identifies a specific, documented gap: CT+MRI fusion for brain tumour classification is under-explored relative to MRI-only methods, and the studies that do exist use proprietary hospital data that cannot be reproduced or compared. No public benchmark offers paired CT+MRI for brain tumours at the scale of BraTS. The position paper documents what fusion architectures have been proposed (early fusion, late fusion, cross-modal attention, decision-level fusion, radiomics-guided fusion), what accuracy claims are made in the literature, and why — absent a public paired dataset with independent labels — those claims cannot be independently verified. This is a contribution to the field's self-understanding: it tells researchers what is actually known, what is claimed but unverified, and what would be required to move the field forward.

The survey also identifies publicly available datasets that contain both CT and MRI for brain-related imaging (IEEE DataPort glioblastoma benchmark, Kaggle multimodal brain tumour images, Zenodo brain metastases cohort, paired CT-MRI medical imaging datasets) and characterises their limitations relative to BraTS-scale research: small case counts, varying label availability, non-standardised preprocessing, and limited or absent expert annotations.

The project does not attempt to build a CT+MRI fusion classifier. It documents the gap, surveys what exists, and makes the case that a public paired benchmark with independent labels would be the necessary precondition for meaningful fusion research. This is an honest scope statement, not a limitation to hide.

---

## Broader Impacts

**1. Reproducible medical AI for resource-constrained settings.** The pipeline is designed to run on commodity hardware — a single RTX 2050-class GPU with 4GB VRAM and 3GB system RAM — without sacrificing data integrity or evaluation rigour. The documentation, code, and pipeline design provide a reference implementation for researchers at institutions without GPU clusters or large compute budgets. This directly addresses the concentration of medical AI research in well-funded institutions with high-end infrastructure, by demonstrating that rigorous, audited, reproducible work is possible on modest hardware.

**2. Training in rigorous ML practices.** The project provides a concrete example of data auditing (file verification, label deduplication, leakage prevention), contract testing (12/12 passing tests covering duplicate labels, conflicting labels, subject-disjoint splitting, balanced sampling, loss weighting, and multi-crop aggregation), and honest evaluation (locked test evaluated once, threshold selected on validation only, all metrics reported including those that look poor). These are transferable skills that apply to any ML domain, not just medical imaging. Undergraduates who work with this pipeline learn not just model training but data integrity, test design, and the discipline of not overclaiming.

**3. Open science and reproducibility.** All artifacts — the data quality report, the fold manifest, the out-of-fold predictions, the calibration parameters, the visual evidence manifest, the contract tests, the consolidated verified references — are deposited in the repository. The work is designed to be re-run and verified, not just read about. This aligns with the broader movement toward reproducible computational research and provides a concrete example in the medical imaging domain.

**4. Honest science as a practice.** The project explicitly reports results that look weak (balanced accuracy below majority baseline on accuracy, thresholded classifier weaker than the baseline it is compared against) alongside results that look meaningful (AUROC above random, calibrated ranking signal). It explicitly labels the target as a proxy derived from enhancing tumour presence, not as an independent clinical grade. It does not claim clinical deployment readiness. This scope honesty is itself a contribution to a field where overclaiming is a known problem.

---

## Keywords

brain tumour, glioma, BraTS, MRI classification, 3D CNN, deep learning, resource-constrained computing, reproducible research, data auditing, literature survey, medical image analysis, grade proxy, subject-disjoint cross-validation, open-source pipeline
