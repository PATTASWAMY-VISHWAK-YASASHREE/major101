# Phase 1 — Write the Paper (for college credit)

**One thing. No grants. No CT+MRI. No synthesis. No deployment bs. Just write it up and submit for credit.**

---

## What exists already (just needs writing)

- `BraTS_MRI_Grade_Classification_Panel_Report.md` — this is basically the paper body already written. It's detailed, honest, has all the numbers, the visual evidence descriptions, the limitations. Needs reformatting to paper structure and trimming.

- `consolidated_verified_references.bib` — the clean citations. Use these. Don't use the old cat4/5/8 ones with fabricated PMIDs.

- All the results are in the repo:
  - `outputs/cv/full_epoch_baseline_5fold_5ep/summary.json` — 5-fold metrics + OOF predictions
  - `outputs/evaluation/repaired_final/summary.json` — locked test metrics
  - `outputs/calibration/repaired/calibration.json` — temperature, ECE, Brier
  - `outputs/explainability/repaired_final/visual_evidence_manifest.json` — 4 cases with figure paths

## Paper structure (from the panel report, reformatted)

1. **Title** — something like "Memory-Bounded 3D CNN Classification of BraTS 2024 Glioma Grade-Proxy: A Data-Audited Baseline with Subject-Disjoint Cross-Validation and Locked-Test Evaluation"

2. **Abstract** — 150-250 words. Problem, method, results (AUROC 0.7672, balanced acc 0.5261), honest limitations.

3. **Introduction** — brain tumour imaging context, why BraTS matters, why resource-constrained reproducible pipelines matter.

4. **Data and Label** — BraTS 2024, the grade_proxy construction (ET > 0 → HIGH), the data audit numbers (882 files, 876 cases, deduplication, subject-disjoint splits). Explicitly state the proxy limitation here, not buried in conclusions.

5. **Methods** — the pipeline:
   - TinyGradeClassifier3D architecture (501K params, GroupNorm, 96³ crops)
   - Memory-mapped loading + crop-first reading (why: RAM/VRAM limits)
   - Balanced batch sampling (1 LOW + 1 HIGH per batch)
   - AdamW, BCE loss (focal gamma=0), 10 epochs, 64 steps/epoch
   - Subject-disjoint 5-fold CV on 788 development cases
   - Threshold selected on validation only (0.53 for final)
   - Temperature calibration on OOF only
   - Locked test evaluated once

6. **Results** — all of them:
   - Table 1: 5-fold CV summary (mean best AUROC 0.7987, OOF AUROC 0.7641 with CI)
   - Table 2: Locked test metrics (AUROC 0.7672, balanced acc 0.5261, accuracy 0.7045, F1 0.8169, sens 0.8169, spec 0.2353)
   - Confusion matrix
   - Comparison to majority baseline
   - Calibration results
   - Negative results: ensemble (OOF AUROC 0.5463 vs single 0.7641), 5-view drop (0.7641→0.6058), ultra-light oscillation

7. **Visual Evidence** — 4 cases described. Grad-CAM + saliency. Explicitly NOT tumour segmentations.

8. **Discussion** — what the results mean and don't mean. The proxy limitation. The resource-constrained angle. Comparison to literature (cite the survey — CT+MRI fusion is under-studied, no public paired benchmark).

9. **Limitations** — this can be its own section or merged into discussion. Must include: proxy label, no independent grade, no external validation, single benchmark, modest AUROC.

10. **Conclusion** — one paragraph. Honest. "We built a reproducible pipeline and characterised its performance. The ranking signal is non-random but modest. This is a reference baseline, not a clinical tool."

11. **References** — use consolidated_verified_references.bib. Check every PMID.

## Target

- **If the college needs a journal publication for credit:** something like Cureus, Journal of Medical Imaging and Health Informatics, or similar lower-barrier venue
- **If the college just needs a report/submission:** format it as a technical report or term paper and submit to the department
- **If there's a workshop that fits:** a reproducibility track or methods workshop at a conference the college recognises

## What's NOT happening

- No CT+MRI fusion (no data)
- No synthetic data of any kind
- No ensemble or multi-view development
- No background hyperparameter search
- No grant proposals of any kind
- No clinical deployment claims
- No pretending grade_proxy is WHO grade
- No "future work" section that commits to anything

## Timeline

- **Day 1-3:** Read the panel report, reformat to paper structure, fill in the gaps
- **Day 4:** Clean citations, verify every reference you actually cite
- **Day 5:** Final formatting, figures, submit/file for credit

Done.
