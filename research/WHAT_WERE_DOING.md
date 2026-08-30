# What We're Actually Doing

**One paper. For college credit. Research-grade only. No synthetic data. No clinical bullshit. No grant proposals. Done.**

---

## The Paper

**What it is:** A methods + results paper on a 3D CNN classifier for BraTS 2024 glioma data, running on commodity hardware (RTX 2050, 4GB VRAM), with full data auditing and honest results.

**What it is NOT:**
- Not a clinical tool
- Not a diagnostic system
- Not claiming WHO-grade accuracy
- Not using any synthetic/generated data (no CT synthesis, no MRI translation, no GAN/Diffusion augmentation)
- Not claiming CT+MRI fusion results (we don't have paired data)
- Not a grant proposal

**What goes in it:**

1. **Introduction** — brain tumour imaging, BraTS benchmark, why a resource-constrained reproducible pipeline matters (even if results are modest)

2. **Data** — BraTS 2024, the grade_proxy label and what it actually is (ET-derived, NOT WHO grade), the data audit (882 files verified, 994→876 cases, deduplication, subject-disjoint splits)

3. **Methods** — the actual pipeline:
   - TinyGradeClassifier3D (501K params, GroupNorm, 96³ crops)
   - Memory-mapped loading, crop-first reading
   - Balanced batches, AdamW, BCE loss, 10 epochs
   - Subject-disjoint 5-fold CV
   - Locked test evaluated once
   - Temperature calibration on development OOF only

4. **Results** — all of them, including the ones that look bad:
   - 5-fold: mean best AUROC 0.7987, OOF AUROC 0.7641 (CI 0.7261–0.8018)
   - Locked test: AUROC 0.7672, balanced acc 0.5261, accuracy 0.7045 (BELOW majority baseline 0.8068)
   - Confusion matrix: TN=4, FP=13, FN=13, TP=58
   - Calibration: temperature 0.8011, ECE 0.2366→0.2279
   - The negative results: ensemble OOF AUROC 0.5463 (worse than single), 5-view re-evaluation dropped from 0.7641 to 0.6058, ultra-light 2-epoch oscillation

5. **Visual evidence** — 4 cases: best correct LOW, correct HIGH, ambiguous near-threshold, worst confident false-LOW. Grad-CAM + saliency. Explicitly NOT tumour segmentations — just model behaviour visualizations.

6. **Limitations** (this needs to be prominent, not hidden):
   - Label is ET-derived proxy, not independent WHO grade or pathology
   - Model is not clinically validated
   - No external validation with ground truth (188-case raw cohort has no labels)
   - Generalisation beyond BraTS 2024 unknown
   - Modest AUROC on locked test
   - Thresholded classifier weaker than majority baseline on accuracy

7. **Literature context** — cite the verified references. Optionally include a short survey section noting that CT+MRI fusion for brain tumours is under-studied and no public paired benchmark exists (cite the 117-paper review as context, not as a separate paper).

8. **Conclusion** — honest: "We built a reproducible, memory-bounded pipeline and characterised its performance on a proxy label. The ranking signal is non-random but modest. The work is a reference implementation and honest baseline, not a clinical tool."

---

## The Paper Is Not Trying To Be

- MICCAI main conference (won't get in with these numbers)
- Medical Image Analysis (same)
- Anything claiming SOTA or clinical utility
- A CT+MRI fusion paper (no data for it)

## The Paper Could Be

- A college project report / term paper for credit
- A workshop paper if there's a reproducibility/methods track that fits
- A technical report on the department website
- A paper in a lower-barrier venue (Cureus, JMIHI, or similar) IF the college needs a publication for credit
- Just internally submitted to the department — depending on what "college credit" actually means here

---

## What's Already Done (just needs writing up)

- ✅ Data audit complete (882/882 files, 876 cases, subject-disjoint splits)
- ✅ 5-fold CV complete on all 788 development cases
- ✅ Locked test evaluated once (88 cases, all metrics saved)
- ✅ Calibration done (temperature 0.8011)
- ✅ Visual evidence generated (4 cases, figures + JSON)
- ✅ Contract tests passing (12/12)
- ✅ Citations partially cleaned (consolidated_verified_references.bib has the good ones)
- ✅ Literature review done (117 papers, can be cited as context)

## What's Left

1. **Write the paper** — use the BraTS_MRI_Grade_Classification_Panel_Report.md as the base (it's already written in panel-report format, just needs reformatting to paper format and trimming)
2. **Clean the citations** — excise any fabricated PMIDs from what you cite, use consolidated_verified_references.bib
3. **Submit / file for credit** — whatever the college requires

---

## What We're NOT Doing

- ❌ CT+MRI fusion (no paired dataset, not building it)
- ❌ Synthetic CT / MRI-to-CT translation (explicitly banned)
- ❌ Any data augmentation beyond flips + noise (already in the pipeline, that's it)
- ❌ Ensemble or multi-view (measured as negative results, not pursuing)
- ❌ Background search for better hyperparams (BA75 runner exists but not running — proxy tuning doesn't change the fundamental limitation)
- ❌ Any grant proposals (NSF, NIH, DST, SERB, ABTA — all of them are out of scope)
- ❌ Clinical deployment claims
- ❌ Pretending the grade_proxy is WHO grade

---

## One Paper. Honest. For Credit. Done.

That's the scope. Everything else the council and the skill-loaded docs talked about — grants, fusion, benchmarking, broader impacts — is noise relative to what you actually want. Cut it.
