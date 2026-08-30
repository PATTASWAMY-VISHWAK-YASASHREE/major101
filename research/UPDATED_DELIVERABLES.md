# Updated Deliverables

**Scope lock: one paper for college credit. Research-grade. No synthesis. No CT+MRI fusion. No grants.**

---

## ✅ KEPT — still relevant

- **`research/consolidated_verified_references.bib`** — clean citations. Use these in the paper. Don't use old cat4/5/8 with fabricated PMIDs.
- **`research/BraTS_MRI_Grade_Classification_Panel_Report.md`** — the de facto paper body. 784 lines of detailed, honest results and analysis. Reformat this into paper structure.
- **`research/citation_verification_audit.md`** — document of what was verified and what was fabricated. Useful for the paper's citation integrity note.
- **`outputs/cv/full_epoch_baseline_5fold_5ep/summary.json`** — 5-fold metrics + OOF predictions (source of truth for CV numbers)
- **`outputs/evaluation/repaired_final/summary.json`** — locked test metrics (source of truth for test numbers)
- **`outputs/calibration/repaired/calibration.json`** — temperature, ECE, Brier
- **`outputs/explainability/repaired_final/visual_evidence_manifest.json`** — 4 cases with figure paths

## ✅ REPLACED — updated to match new scope

- **`research/grant_project_summary_draft.md`** → rewritten to: research-grade only, no grants, no CT+MRI fusion claims, survey/position paper framing for the CT+MRI gap, honest limitations throughout
- **`research/grant_specific_aims_draft.md`** → rewritten to: scope-locked to the paper + literature survey, no aims about building fusion classifiers or preprocessing public datasets

## ✅ ADDED — new scope-locked documents

- **`research/WHAT_WERE_DOING.md`** — the no-nonsense scope statement: one paper for college credit, what's in it, what's NOT in it, what's already done, what's left (write it, clean citations, submit)
- **`research/PHASE1_WRITE_THE_PAPER.md`** — the actual execution plan: paper structure day-by-day, target venues, what's NOT happening, timeline (5 days)

## ❌ DISPOSABLE — contradicted by the new scope

These are now out of scope and should not be used:

- **`research/grant_specific_aims_draft.md` (OLD VERSION)** — if it exists before the overwrite, it assumed CT+MRI fusion aims and grant framing
- **`research/phase1_action_plan.md`** — assumed CT+MRI dataset verification and possible fusion demo. Replace with PHASE1_WRITE_THE_PAPER.md
- **`research/council_strategy_report.md`** — contains grant-pursing recommendations and CT+MRI fusion demo suggestions that are now out of scope. Keep for reference but don't act on the grant/fusion parts.
- Any document in `research/bibtex/` with fabricated PMIDs (cat4, cat5, cat8 with unverified PMIDs) — don't cite these

---

## The scope in one sentence:

Write the MRI-only BraTS paper from the existing panel report + results + clean citations, submit it for college credit, and nothing else.
