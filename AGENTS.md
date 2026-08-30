# AGENTS.md — The Brain of This Repo

**Read this file FIRST, completely, before doing anything else in this repository.**
It contains the full project identity, history, settled decisions, methods, hard rules, and current mission. Do not re-derive, re-litigate, or re-research anything already settled here. If something here conflicts with older files, THIS file wins.

---

## 1. IDENTITY — What this project IS

**Longitudinal brain-tumour AI research on public data only, run on consumer hardware by a zero-budget college team.**

Two intertwined deliverables:

1. **Phase 1 (COMPLETE):** MRI-only glioma grade-proxy classifier on BraTS 2024 — audited, leakage-safe, memory-bounded, locked-test evaluated. Paper written: `research/paper/MANUSCRIPT.md`.
2. **Phase 2 (ACTIVE):** Longitudinal tumour analysis on the PROTEAS brain-metastasis dataset (real per-patient timepoints, expert segmentations, paired CT/RT) + an **agentic evidence-report layer** that explains WHY the CNN flagged a region — it structures evidence, it NEVER diagnoses.

The architecture metaphor (decided, embraced, and technically grounded): an **ant colony** — many small, specialized models ("ants") that each see one input view, communicate only through a shared evidence map (the **pheromone field** / blackboard), and vote at the patient-temporal level. It is a blackboard system (Hearsay-II, 1980) patched with modern components; the paper framing is exactly that.

## 2. HARD RULES — never violate, no matter what

1. **NO synthetic data. Ever.** No GANs, no diffusion, no MRI→CT translation, no generative augmentation. Real patient scans only. If a dataset contains synthetic content, it is banned and flagged.
2. **NO clinical claims.** This is research-grade work. No diagnosis, no deployment, no "clinically validated". The agentic layer reports evidence provenance; it has zero diagnostic authority. This rule is a paper-level statement, not a config choice.
3. **The BraTS 2024 `grade_proxy` label is ET-derived, NOT WHO grade.** Every claim built on it must say so. (A historical ET-feature model hit ~99% — it learned the label-construction rule and is excluded from all claims.)
4. **The locked test stays locked.** `scripts/evaluate_repaired.py` was run ONCE for the frozen checkpoint. Never re-run it for tuning. Never evaluate the test split during experiments.
5. **No grant applications, no seed-funding pitches.** Decision from 2026-08-30. Ship papers; funding is out of scope.
6. **Memory guards are non-negotiable:** 4 GB VRAM ceiling, mmap + crop-first (96³) loading, `num_workers=0`, no parallel training jobs. This pipeline must run on consumer hardware.
7. **Two-machine split:** THIS machine (i5, 8 GB) is the command centre — planning, code, analysis. It downloads NOTHING. The Ultra 9 (32 GB, 1 TB) is the workhorse — downloads, training. All heavy artifacts live there and come back only as small JSON/CSV reports.
8. **No parallel subagent storms.** The free-tier LLM provider caps at 8 requests/minute; 6 parallel subagents 429'd themselves to death on 2026-08-30. Max 1-2 concurrent subagents; batch tool calls in the main conversation instead.
9. **Citations:** use `research/consolidated_verified_references.bib` ONLY. The old `research/bibtex/cat4/5/8_references.bib` contain FABRICATED PMIDs (documented in `research/bibtex/citation_verification_audit.md`). Never cite an unverified PMID.
10. **Honest reporting always.** Negative results are reported, never hidden. Phase 1 shipped with accuracy BELOW the majority baseline — stated plainly. Do the same in Phase 2.

## 3. HISTORY — how we got here (compressed)

- **Aug 2026, weeks 1-2:** BraTS 2024 MRI classifier. Escaped class collapse (GroupNorm + balanced sampler + BCE). Repaired 7 correctness bugs: duplicate label rows (994→876 unique cases), subject leakage, full-volume memory pressure, background crops, focal-loss broadcast bug, test contamination, mislabeled Grad-CAM channel.
- **Weeks 3-4:** Five-fold subject-disjoint CV over all 788 dev cases; pooled OOF AUROC 0.7641 (95% CI 0.7261–0.8018); temperature calibration (T=0.801, dev-OOF only); final fit on 788; locked test evaluated ONCE: AUROC 0.7672, balanced acc 0.5261 (below majority-baseline accuracy 0.8068 — reported honestly).
- **Negative results (measured, reported, and treated as settled science):** 5-member clone ensemble OOF AUROC 0.5463 (vs single 0.7641) — clones share errors, zero diversity. Five-view crop aggregation 0.6058 — diluted a mature signal. Root cause both times: correlated errors, NOT "ensembles don't work".
- **Aug 30, 2026:** Strategy council (5 archetypes) deliberated → scope-locked: Phase 1 paper for college credit, no grants. Then the user set the REAL goal: **longitudinal analysis is the whole point** + agentic evidence reports. The claim "no public longitudinal CT+MRI brain-tumour data exists" was re-tested and found FALSE: 6-source data sweep (salvaged from rate-limited subagents + main-thread verification) found PROTEAS.
- **Aug 31, 2026:** Colony architecture v1.1 decided (see §5). Ultra 9 mission kit built and unit-tested (download + zip-level audit scripts). Awaiting data.

## 4. CURRENT STATE (as of 2026-08-31)

**Phase 1: DONE.** Manuscript at `research/paper/MANUSCRIPT.md` — verified against artifacts, §5.5 within-subject consistency included (227/337 subjects have ≥2 acquisitions; predictions 3.4× more stable within-subject; model self-agreement 76.2% vs label self-agreement 85.5%; artifact: `outputs/analysis/within_subject_consistency_report.json`). Ready to format & submit for college credit. Not blocked by Phase 2.

**Phase 2: WAITING ON DATA (the gate).**
- Mission kit ready: `research/data_sweep/ULTRA9_MISSION_KIT/` (README_RUNBOOK.md + download_proteas.py + audit_proteas.py — audit parses NIfTI headers from inside zips with pure stdlib, catches missing-sequence/missing-CT defects; verified with synthetic good/broken patient fixtures).
- Next action: user runs the kit on the Ultra 9 → stage 1 (clinical xlsx + P01–P03, ~700 MB) → audit gate → if pass, full 15.1 GB pull → re-audit → bring `proteas_audit_report.json` back here.
- THEN: build longitudinal pipeline (per-patient timelines, temporal aggregation), fine-tune from the BraTS checkpoint, build the colony, build the agentic report layer.

**Key numbers every agent should know:**

| Metric | Value | Source artifact |
|---|---|---|
| CV mean best AUROC (5 folds) | 0.7987 ± 0.049 | `outputs/cv/full_epoch_baseline_5fold_5ep/summary.json` |
| Pooled OOF AUROC | 0.7641 (CI 0.7261–0.8018) | same |
| Locked test AUROC / balanced acc | 0.7672 / 0.5261 | `outputs/evaluation/repaired_final/summary.json` |
| Calibration | T=0.801; Brier 0.197→0.196; ECE 0.237→0.228 | `outputs/calibration/repaired/calibration.json` |
| Clone-ensemble AUROC | 0.5463 (NEGATIVE result) | HANDOVER.md |
| 5-view AUROC | 0.6058 (NEGATIVE result) | HANDOVER.md |
| Model | TinyGradeClassifier3D, 501,289 params, GroupNorm, 96³ crops | `src/grade_model.py` |
| Contract tests | 12/12 passing | `tests/` |

## 5. THE ARCHITECTURE — Colony v1.1 (decided; do not redesign without measuring)

**Why colony:** ensemble error = mean member error − diversity (Krogh & Vedelsby). A trillion ants beat an elephant ONLY if they attack from different angles. Measured failures above prove clones fail. Therefore:

**Member rules:** every ant is (a) individually competent — fine-tuned from the BraTS 2024 checkpoint (876 cases of prior knowledge), never from scratch on 40 patients — and (b) diverse on at least one axis (input view, crop policy, seed, width). No clones. Boosting BANNED at this n (sequential overfitting); bagging and temporal voting only.

**The ants (specialized by input view):**

| Ant | Input | Tracks |
|---|---|---|
| T1c ant | T1-contrast | enhancing tumour |
| FLAIR ant | FLAIR | oedema / infiltration |
| T2 ant | T2 | oedema + cystic change |
| T1n ant | native T1 | structural drift (resection, mass effect) |
| Time ant | prior ants' outputs over t−1, t−2, … | change itself — the only ant that sees time |

**Communication channels (safest first):**
1. **Vote** — fixed late fusion (median/mean) of member probabilities. No learning, cannot overfit.
2. **Pheromone field** — each ant deposits its Grad-CAM into a shared spatial overlay. Ants never read each other's weights; they read/write the environment (stigmergy — the real-ant mechanism, and the Hearsay-II blackboard). k-of-m overlap = strong trail; dissent = explicit instability flag. Zero training cost. **This is the agentic layer's primary data source.**
3. **Learned gating (MoE/stacking)** — BANNED in v1; a learned gate at n=40 is an overfitting machine (same rule that killed boosting). Unbanned only if leave-patients-out CV beats the fixed vote with honest intervals.

**Temporal superpower:** one model × many timepoints per patient (temporal median/vote). This is the measured-stable axis (3.4× within-subject stability). It is how small-n becomes workable.

**Agentic evidence layer (a.k.a. the blackboard controller):** schema-constrained JSON reports per case-timepoint, quoting ONLY computed artifacts: probability + threshold margin, Grad-CAM↔expert-segmentation overlap (possible because PROTEAS has expert segs), saliency↔Grad-CAM agreement (two methods voting; divergence = instability), temporal delta p(t)−p(t−1) with trend label, crop provenance/coverage, k-of-m member agreement, explicit uncertainty and disclaimers. The agent structures and quotes evidence. It never diagnoses, never generates new evidence, never interprets anatomy without ground truth.

Full detail: `research/RESEARCH_DIRECTION_LONGITUDINAL.md` (§5-6 are the colony spec).

## 6. THE DATA (verified 2026-08-31 from source pages)

| Dataset | Role | Why |
|---|---|---|
| **PROTEAS / RFUds** (Zenodo 17253793, 15.1 GB, CC-BY, no signup) | PRIMARY | 40 brain-met patients × baseline + follow-ups (6wk/3/6/9/12mo) × 4 sequences (t1/t1c/t2/fla, NIfTI, BraTS space — drop-in for our pipeline) × expert segs per timepoint × paired CT+RT dose × clinical xlsx (treatment, KPS, survival). Cite Flouri et al., Sci Data 12:1828 (2025), doi:10.1038/s41597-025-06131-0 |
| **Brain-Tumor-Progression** (TCIA / Kaggle mirror, 3.2 GB) | GLIOMA COMPANION | 20 GBM patients × 2 timepoints (post-CRT, at-progression) × DICOM + tumor masks + perfusion. Progression is literally the label. |
| **BraTS-Reg** (Zenodo 14642405, 4.3 GB) | ALIGNMENT REFERENCE | longitudinal glioma pairs + landmark GT; for registration work, not training |
| **Yale-Brain-Mets-Longitudinal** (TCIA, 43 GB, later only) | SCALE | 1,430 patients, 11,892 studies, NIfTI with dates; no segmentations; only if scale claims needed |

Folder shape to expect (PROTEAS): `P01/BraTS/{baseline,fu1,fu2,...}/{t1,t1c,t2,fla}.nii.gz`, `P01/tumor_segmentation/...`, `P01/P01_CT.nii.gz`, `P01/P01_RTP.nii.gz`, `P01/P01_brain_mask.nii.gz`. Double-RT patients: P04a/P04b etc.

Download commands & every link/size/signup note: `research/data_sweep/DATASET_ACQUISITION_GUIDE.md`. Full sweep table: `research/data_sweep/DATASET_SWEEP_REPORT.md`.

## 7. REPO MAP — where everything lives

### 7a. READ-ORDER for a fresh agent (do this sequence, skip everything else on first pass)

**Pass 1 — absorb context (this order, ~5 minutes):**
1. `AGENTS.md` (this file) — full stop, everything you need is here
2. `research/RESEARCH_DIRECTION_LONGITUDINAL.md` — Phase 2 master plan + colony spec (§5-6)
3. `research/data_sweep/ULTRA9_MISSION_KIT/README_RUNBOOK.md` — the current mission's operating manual

**Pass 2 — only when the task requires it (pull on demand):**
- Writing/touching training or eval code → `HANDOVER.md` (canonical commands + guardrails) + `src/grade_data.py` + `src/grade_model.py`
- Verifying a number before citing it → the artifact in `outputs/` named in §4's table (never trust prose, always open the JSON)
- Formatting/submitting the Phase 1 paper → `research/paper/MANUSCRIPT.md` + `research/PHASE1_WRITE_THE_PAPER.md`
- Downloading/auditing data → `research/data_sweep/DATASET_ACQUISITION_GUIDE.md` + the mission-kit scripts
- Citing anything → `research/consolidated_verified_references.bib` + `research/bibtex/citation_verification_audit.md`

### 7b. IGNORE MAP — see `.agentignore` (repo root)

A literal `.agentignore` file sits at the repo root: path-per-line, `!` = do-read exception, with inline reasons. It quarantines: the fabricated-PMID bibs, the historical Phase-0 survey files, superseded pre-Phase-2 vision docs (06_final_report, fusion plans), the strategy-pivot artifacts (grant drafts, council), pre-repair outputs, binary/checkpoint noise, and the satisfied Phase-1 gate doc. The summary table below explains WHY; the file is the machine-readable version.

| Quarantine class | Examples | Reason |
|---|---|---|
| **POISON** | `research/bibtex/cat4/5/8_references.bib`, `research/cat8_ct_mri_multimodal_fusion.md` | Fabricated PMIDs — never cite |
| **HISTORICAL survey** | `research/cat*.md`, TIMELINE/TRENDMAP/MINDMAP/CROSSREF | Phase-0 lit review, superseded, some cite poison |
| **SUPERSEDED vision** | `research/06_final_report.md`, fusion/dataset docs, 12-week plan | Describe never-implemented CT+MRI fusion / 4-class plans — reading first causes wrong assumptions |
| **SUPERSEDED strategy** | grant drafts, council report, session notes | Grants rejected 2026-08-30; historical only |
| **Pre-repair outputs** | FINAL_REPORT.md, old EDA/cv/training dirs | Current truth = MANUSCRIPT + `outputs/{evaluation,repaired_final,cv/full_epoch_baseline_5fold_5ep,analysis,calibration}` |
| **Binary/cache** | `*.pth`, `*.nii.gz`, `*.zip`, `__pycache__`, `.git`, `.codegraph` | Never load into context |

**Rule of thumb:** if a file isn't in §4's artifact table, §7a's read-order, or §9's mission, a fresh agent does not need it in context on day one.

### 7c. Working files (where active work happens)

```
research/paper/MANUSCRIPT.md            ← Phase 1 paper (done, verified vs artifacts)
research/RESEARCH_DIRECTION_LONGITUDINAL.md  ← Phase 2 master plan + colony spec (§5-6)
research/data_sweep/                    ← sweep report, acquisition guide, mission kit
research/data_sweep/ULTRA9_MISSION_KIT/  ← download_proteas.py + audit_proteas.py + runbook
research/consolidated_verified_references.bib ← the ONLY clean bib
research/bibtex/citation_verification_audit.md ← why the other bibs are poison
research/BraTS_MRI_Grade_Classification_Panel_Report.md ← Phase 1 full evidence report
research/council_strategy_report.md     ← strategy deliberation (historical context)
HANDOVER.md                             ← Phase 1 deep handover (canonical commands)
scripts/                                ← verified pipeline (verify_preprocessed_data, train_ultra_light,
                                           cross_validate_repaired, evaluate_repaired [DO NOT re-run],
                                           generate_research_visuals, infer_raw_validation_stream, ...)
src/grade_data.py, src/grade_model.py   ← canonical data/model code
tests/                                  ← 12/12 contract tests
outputs/                                 ← all source-of-truth JSON/CSV/figures (see §4 table)
outputs/analysis/within_subject_consistency_report.json  ← §5.5 evidence
plan/process-brats-classification-next-steps-1.md ← Phase 1 gate doc (satisfied)
```

## 8. WORKFLOW RULES for any agent (or human) working here

1. **Before proposing anything**, check §2 (hard rules), §5 (architecture decisions), and this section. Most bad ideas — grant pitches, synthetic CT, re-running the locked test, ensembles of clones, learned gating at n=40, Kaggle PNG re-scrapes — are already settled against.
2. **Every number must trace to an artifact.** If you state a metric, name the file it came from. No number from memory.
3. **Verification before scale:** download small → audit → then commit disk/time. The PROTEAS flow (xlsx → P01-P03 → audit gate → full pull) is the template for every future dataset.
4. **Unit-test scripts before shipping them across machines** (the mission-kit audit was tested with synthetic good/broken zips on the i5 before being sent to the Ultra 9 — do the same pattern every time).
5. **Phase 2 evaluation protocol** (already decided): patient-disjoint leave-patients-out CV; report single-model vs colony vs temporal-vote with honest intervals; n=40 means wide intervals; a null result gets reported, not hidden, exactly like Phase 1.
6. **When the environment rate-limits:** fewer, bigger calls in the main thread; never >2 concurrent subagents; salvage cached pages from `C:/Users/pvish/AppData/Local/hermes/cache/web/` and delegation transcripts before re-searching.
7. **Match repo style:** British/Indian spelling in research docs ("tumour"), plain-English honest framing, no marketing language, tables for anything enumerable.

## 9. MISSION BRIEF — what happens next (in order)

1. **User runs the Ultra 9 mission kit** (README_RUNBOOK.md steps 1-5). Output: `proteas_audit_report.json` + clinical-xlsx observations come back to the i5.
2. **Go/no-go here:** if audit passes → Phase 2 build begins. If failures → diagnose per-patient, exclude bad patients, re-audit.
3. **Build (in this order):**
   a. `scripts/audit_longitudinal_dataset.py` — full-dataset timeline table (port of Phase 1 data audit; per-patient timepoints, seg coverage, CT pairing, date sanity)
   b. Timeline construction + preprocessing (reuse crop-first/mmap machinery; PROTEAS is already BraTS-space so t1/t1c/t2/fla channels map 1:1)
   c. Fine-tune the BraTS checkpoint on PROTEAS (patient-disjoint folds) — single model first, as the base ant
   d. Colony v1.1: specialist ants (per-sequence) + temporal vote + pheromone field
   e. Agentic evidence-report generator (schema-constrained JSON; multiple weak evidence streams combined; disclaimers mandatory)
   f. Evaluate: single vs colony vs temporal-vote, leave-patients-out, honest intervals
   g. Paper #2: longitudinal + agentic evidence reporting (research-grade framing, old-concepts-patched-modern positioning: blackboard 1980 + stigmergy + MoE-lineage + measured negative results as guardrails)
4. **Phase 1 paper submission** happens independently whenever the user is ready (5-day plan in `research/PHASE1_WRITE_THE_PAPER.md`).

## 10. USER CONTEXT (who you are working for)

Tier-3 college, India. Zero budget — free tiers only. Two machines: i5/8GB (command centre, this repo's usual host) and Ultra 9/32GB/1TB (heavy work). Prefers direct, honest, plain-language communication; gets frustrated by buzzwords and overclaiming; wants evidence-based diagnosis before fixes; likes ambitious-but-real architecture ideas (the ant colony, the blackboard) grounded in measured facts. The goal is real research output, not résumé theatre. Ifykyk.

---

*This file is the single source of truth for project context. Update it whenever a decision is settled, a phase completes, or the mission changes. Last updated: 2026-08-31.*
