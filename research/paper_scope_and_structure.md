# Paper Scope — Brain Tumour MRI Classification on BraTS 2024

**One paper. Research-grade. No synthesis. No clinical claims. No CT+MRI fusion. For college credit.**

---

## What the paper claims

1. We built a reproducible, memory-bounded 3D CNN pipeline for glioma classification on BraTS 2024.
2. The pipeline is data-audited: 882 preprocessed files verified, 994 label rows reduced to 876 unique cases with deduplication, subject-disjoint splits enforced, repeated acquisitions kept in one partition.
3. The pipeline runs under commodity hardware constraints: designed for ≤2GB VRAM and ≤3GB system RAM (RTX 2050-class GPU, 4GB VRAM available).
4. Five-fold subject-disjoint cross-validation on all 788 development cases produces pooled OOF AUROC 0.7641 (bootstrap 95% CI 0.7261–0.8018) and mean best fold AUROC 0.7987.
5. A single locked-test evaluation on 88 unseen cases produces AUROC 0.7672, balanced accuracy 0.5261, accuracy 0.7045 (below majority-baseline accuracy of 0.8068).
6. The pipeline includes calibration (temperature 0.8011, ECE 0.2366→0.2279), visual evidence generation (Grad-CAM + saliency for 4 cases), and contract tests (12/12 passing) covering duplicate labels, conflicting labels, subject-disjoint splitting, balanced sampling, and loss weighting.

## What the paper does NOT claim

1. **Not clinical-grade glioma grading.** The target label (`grade_proxy`) is derived from enhancing tumour presence (ET > 0 → HIGH). It is not independently annotated WHO grade or pathology. Any model that recovers ET presence recovers the label-construction rule, not independent tumour grade. This is stated explicitly and prominently.

2. **Not a clinical diagnostic tool.** The model is not validated on independent clinical labels, not tested on external cohorts with ground truth, not evaluated for clinical utility. No claim of deployment readiness is made.

3. **Not a CT+MRI fusion result.** The pipeline is MRI-only. No CT data is used. No fusion claims are made. The project's literature review identifies that CT+MRI fusion for brain tumours is under-studied and that no public paired benchmark exists, but this paper does not attempt to fill that gap — it documents that the gap exists.

4. **Not synthetic data of any kind.** No CT generation from MRI, no MRI-to-CT translation, no GAN or diffusion-based augmentation. The only augmentation in the pipeline is random flips (3 axes) and small Gaussian noise on the input. No synthetic samples are created.

5. **Not a state-of-the-art comparison.** The results are modest (AUROC 0.7672, balanced accuracy 0.5261). The paper does not claim SOTA performance. It claims a reproducible, audited, honestly-reported baseline.

## Paper structure

**1. Introduction**
Brain tumour imaging context. BraTS benchmark. Why reproducible, resource-constrained pipelines matter even when results are modest. Brief mention of the CT+MRI fusion gap as context for why MRI-only work is still relevant (lit)."

**2. Data**
BraTS 2024 preprocessed data (4-channel MRI: T1ce, T1n, T2f, T2w). The grade_proxy label and its ET-derived construction. The data audit: 882 files, expected shape (4,182,218,182), float32, 0 invalid files, 0 non-finite volumes. Label table: 994 rows → 876 unique cases after deduplication, 172 LOW, 704 HIGH, 0 conflicting duplicates, 6 unlabelled orphan volumes excluded. Subject-disjoint split: 700 train, 88 validation, 88 locked test.

**3. Methods**
- TinyGradeClassifier3D: 4 input channels, base_channels=12, conv widths 12→24→48→96, GroupNorm after each conv, LeakyReLU (0.01), adaptive global average pooling, linear head with dropout 0.25, 501,289 parameters.
- Memory-bounded loading: np.load(mmap_mode="r"), crop-first reading (96³), num_workers=0, one crop copied per sample.
- Training: AdamW (lr=1e-4, weight_decay=1e-4), BCE loss (focal gamma=0), balanced binary batches (1 LOW + 1 HIGH per batch), 2 physical batch size, gradient accumulation 4, AMP float16, 3-epoch warmup, 10 epochs, 64 balanced steps/epoch, seed 42.
- Five-fold subject-disjoint CV on 788 development cases (train + validation only, locked test excluded).
- Threshold selection: validation only, maximise balanced accuracy then F1, grid 0.05–0.95. Final threshold 0.53.
- Temperature calibration: fit on development OOF predictions only. Temperature 0.8011.
- Locked test: evaluated once from the final checkpoint. No tuning.

**4. Results**
- 5-fold CV: mean best fold AUROC 0.7987, mean best fold balanced accuracy 0.7592. Pooled OOF: AUROC 0.7641 (CI 0.7261–0.8018), balanced accuracy 0.7288 (CI 0.6936–0.7631).
- Locked test: AUROC 0.7672, balanced accuracy 0.5261, accuracy 0.7045, F1 0.8169, sensitivity 0.8169, specificity 0.2353, average precision 0.9431.
- Confusion matrix (threshold 0.53): TN=4, FP=13, FN=13, TP=58.
- Majority baseline comparison: majority-HIGH predicts 71/88 HIGH, accuracy 0.8068, balanced accuracy 0.5000, AUROC 0.5000. Model beats baseline on AUROC and balanced accuracy, not on accuracy.
- Calibration: Brier 0.1973→0.1960, ECE 0.2366→0.2279.
- Negative results (reported, not pursued): ensemble pilot OOF AUROC 0.5463 vs single member 0.7641 (5-member, 2-epoch, compute-matched). 5-view re-evaluation of mature checkpoints: OOF AUROC dropped from 0.7641 to 0.6058. Ultra-light 2-epoch smoke: epoch 1 all-LOW, epoch 2 all-HIGH oscillation.
- Visual evidence: 4 cases (best correct LOW: BraTS-GLI-02720-100, p=0.2935; correct HIGH: BraTS-GLI-02307-101, p=0.5947; ambiguous: BraTS-GLI-02225-101, p=0.4661, true LOW; worst false-LOW: BraTS-GLI-02651-100, p=0.2861, true HIGH). Grad-CAM + input saliency. Explicitly NOT tumour segmentations.

**5. Discussion**
- The model learns a non-random ranking signal on the ET-derived proxy. The ranking is meaningful (AUROC 0.7672 > 0.5) but the thresholded classifier is weak (balanced accuracy 0.5261, below majority baseline on accuracy).
- The proxy limitation is central: ET > 0 → HIGH means the label encodes a radiological finding (enhancing tumour presence), not an independently verified grade. The model may be learning to detect enhancing tumour, which is a different question from glioma grading.
- Comparison to survey findings: CT+MRI fusion for brain tumours is under-studied in the literature (117 papers reviewed). Most multimodal studies use proprietary hospital data. No public paired benchmark exists at BraTS scale. This paper documents that gap but does not attempt to fill it.
- The resource-constrained angle: the pipeline runs on commodity hardware (2GB VRAM / 3GB RAM guards, RTX 2050-class). This is not a limitation to hide but a design constraint. The design choices (GroupNorm, crop-first loading, balanced batches, AMP) are each motivated by the memory constraint.

**6. Limitations**
- Proxy label derived from ET presence, not independent WHO grade or pathology.
- No clinical validation. No external validation with ground truth (188-case raw BraTS validation cohort has no labels — predictions saved but no metrics computed).
- Single benchmark (BraTS 2024 only). Generalisation unknown.
- Modest AUROC and weak thresholded classifier.
- Visual evidence is model behaviour, not tumour segmentation or clinical localisation.
- The 4-case visual set is a representative sample, not an exhaustive analysis.

**7. Conclusion**
We built a reproducible, memory-bounded 3D CNN pipeline for glioma classification on BraTS 2024 and characterised its performance honestly. The pipeline is data-audited, subject-disjoint, and runs under commodity hardware constraints. The model produces a non-random ranking signal on the ET-derived proxy label (AUROC 0.7672) but is a weak thresholded classifier (balanced accuracy 0.5261). The work is a reference baseline and an honest characterisation, not a clinical tool and not a claim of independent glioma grading. The CT+MRI fusion gap in the literature remains unfilled by this work and is documented as a direction for future research that requires public paired data with independent labels.

## References

Use consolidated_verified_references.bib. Verify each PMID cited. Do not use fabricated PMIDs from cat4/5/8_references.bib.

Key citations:
- Isensee et al. 2021 (nnU-Net, PMID 33288961)
- Bakas et al. 2018 (BraTS 2018, PMID 28872634)
- Menze et al. 2015 (original BraTS, PMID 25547933)
- Wen et al. 2023 (RANO 2.0, PMID 37774317)
- Litjens et al. 2017 (survey, PMID 28778026)
- Selvaraju et al. 2017 (Grad-CAM)
- Wu & He 2018 (GroupNorm)
- He et al. 2015 (Kaiming init / Delving Deep into Rectifiers)
- Loshchilov & Hutter 2019 (AdamW)
- Rong et al. 2020 (3D CNN-LSTM for progression, PMID 32891812)
- Islam et al. 2026 (feature fusion, arXiv:2606.11107)
- Almadhor et al. 2026 (CT-MRI integration, Frontiers)

## Document status

This is the scope statement for the paper. Everything else on the disk that assumes grants, CT+MRI fusion aims, synthetic data, or clinical deployment is now out of scope and should not be acted on.
