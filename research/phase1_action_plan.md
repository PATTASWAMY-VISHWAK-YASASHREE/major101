# Phase 1 Action Plan — 8-Week Execution Window

**From the Council's recommendation:** Publish the MRI-only paper + verify one public CT+MRI dataset. Everything else is Phase 2+.

**Grounded in reality:** tier-3 college, no budget, RTX 2050 4GB, public data only.

---

## Week 1-2: Pick and verify a public CT+MRI dataset

**Action:** Download and verify ONE public paired CT+MRI brain tumour dataset. The council recommended IEEE DataPort glioblastoma benchmark (50 patients) as the most purpose-built option. Alternatives: Kaggle Brain Tumor Multimodal, Zenodo brain metastases (185 studies, 40 patients).

**Deliverable:** A one-page data verification report answering:
- How many cases are there really?
- Are both CT and MRI present for the same patient?
- Are there labels (tumour type, grade, segmentation)?
- What's the data format (DICOM, NIfTI, PNG slices)?
- What preprocessing would be needed to use this with the existing pipeline?
- What's missing? (registration, skull-stripping, normalisation, etc.)

**Why this matters:** The project's CT+MRI research track was gated on "no paired dataset exists." That claim is now known to be partially false. Updating it with what actually exists is the highest-leverage action.

---

## Week 2-4: Write the MRI-only paper

**Source material:** `research/BraTS_MRI_Grade_Classification_Panel_Report.md` — this is essentially the paper body already written. The panel report is detailed, honest, and panel-ready.

**Structure:**
1. Title: something like "Memroy-Bounded 3D CNN Classification of BraTS 2024 Glioma Grade-Proxy: A Data-Audited, Subject-Disjoint Baseline with Locked-Test Evaluation"
2. Abstract: problem, method, results (AUROC 0.7672, balanced acc 0.5261 on locked test), limitations (ET-derived proxy, not WHO grade)
3. Introduction: brain tumour grading, why MRI classification matters, why rigorous baselines matter
4. Data: BraTS 2024, the grade_proxy label and its ET-derived construction, the data audit (882 files, 876 cases, deduplication)
5. Methods: the repaired pipeline — GroupNorm 3D CNN, crop-first loading, balanced batches, subject-disjoint 5-fold CV, calibration, locked test
6. Results: the full metric table, confusion matrix, visual evidence cases
7. Limitations: ET-derived proxy, no independent grade labels, no external validation with ground truth, single benchmark
8. Discussion: what the results mean, what they don't mean, the resource-constrained contribution
9. References: use `research/consolidated_verified_references.bib`

**Target venue:** Scientific Reports, PLOS ONE, BMC Medical Imaging, or a MICCAI workshop. Mid-tier — this is honest, methodologically sound work but not SOTA-chasing.

**Key framing:** This is NOT a "we achieved great accuracy" paper. It's a "here's a rigorously audited baseline with honest results and full data provenance" paper. That framing is publishable in mid-tier venues and is actually more valuable to the field than another overclaimed SOTA paper.

---

## Week 4-6: Submit + begin dataset preprocessing exploration

**Actions in parallel:**
- Submit the MRI-only paper
- Based on the Week 1-2 data verification, decide if the public CT+MRI dataset is workable
- If yes: begin writing preprocessing code to bring the public dataset into a format the existing pipeline can consume
- If no: pivot Phase 2 to the "reproducibility/data auditing" paper angle instead

---

## Week 6-8: CT+MRI demo (if dataset is workable) OR paper revision (if rejected)

**Option A — Dataset is workable:**
- Minimal extension: add CT as an optional input channel to the existing pipeline
- Train MRI-only baseline on the public CT+MRI cases (where MRI is available)
- Add CT branch, train late fusion
- Report honestly: CT-only vs MRI-only vs CT+MRI on whatever cases have both modalities
- Expect weak results on small data — the contribution is the pipeline + honest evaluation, not accuracy
- Write a short paper (workshop length) or prepare a benchmark release

**Option B — Dataset is NOT workable (too small, no labels, wrong format):**
- Focus on the reproducibility/data auditing paper angle
- "Rigorous Data Auditing and Leakage Prevention for Brain Tumour MRI Classification: Lessons from BraTS 2024"
- Target a methods/reproducibility venue or a data-focused journal
- The data audit report, contract tests, and leakage prevention protocol are the contribution

**Option C — MRI paper is under review:**
- Respond to any reviewer comments
- Begin drafting the grant proposal (see specific_aims_draft.md) using the submitted paper as preliminary data

---

## What NOT to do in Phase 1

- Do NOT design the full fusion architecture with cross-modal attention and temporal transformers
- Do NOT plan a multi-institutional study
- Do NOT start writing a grant proposal without a submitted paper first
- Do NOT add new research categories to the literature review
- Do NOT chase additional public datasets beyond the one you verify
- Do NOT try to build synthetic CT generation into the pipeline

---

## Success criteria for Phase 1

1. MRI-only paper submitted to a real venue (not just drafted — submitted)
2. One public CT+MRI dataset verified with a written report
3. Go/no-go decision on CT+MRI fusion demo based on actual data, not assumption
4. No new architecture diagrams, no new research categories, no new visions — just execution

---

*This plan is intentionally narrow. The council's strongest signal was that the project has too much documentation and not enough execution. Phase 1 fixes that by picking two concrete deliverables and saying no to everything else for 8 weeks.*
