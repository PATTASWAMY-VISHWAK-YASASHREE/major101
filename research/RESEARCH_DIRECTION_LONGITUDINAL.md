# Research Direction v2 — Longitudinal Brain Tumour AI with Agentic Evidence Reports

**Date:** 2026-08-30 · **Supersedes:** the MRI-only-for-credit scope (that paper remains written and shippable — see `paper/MANUSCRIPT.md`)
**Hard rules (unchanged):** research-grade only · public datasets only · zero budget · NO synthetic/GAN/diffusion-generated data of any kind · no clinical deployment claims · agentic AI NEVER diagnoses — it explains provenance of model evidence.

---

## 1. The two-part vision (what the user actually wants)

### Part A — Longitudinal analysis (the whole point)

The single-scan grade-proxy classifier was Phase 1. The research question that actually matters:

> **Can we model tumour evolution over time from repeated scans of the same patient — tracking predicted probability, feature drift, and explanation-map movement across timepoints — and does that temporal signal contain information a single scan cannot?**

This requires data where **the same patient has multiple studies at real timepoints**. BraTS 2024 in our checkout does NOT qualify (repeated acquisitions lack timestamps — we verified this in the within-subject consistency analysis: 227/337 subjects have ≥2 scans but no dates, no treatment context).

**So Phase 2 is data-first.** Six subagent sweeps are currently scanning: TCIA (LTP, ACRIN-6686, HGG-TTC, TCGA repeats), Zenodo/figshare (RFUds brain-mets follow-up, BraTS-Reg), Kaggle, NCI Imaging Data Commons (BigQuery: PatientID × StudyDate), MICCAI challenge data (Vienna pseudoprogression cohort, BraTS-Reg), plus non-tumour longitudinal testbeds (OASIS-3/ADNI) as pipeline-development fallbacks only.

**Acceptance criteria for "longitudinal-compatible" (the bar every dataset must clear):**
1. Same PatientID with ≥2 studies, ideally ≥3 timepoints for enough patients
2. Real acquisition dates (StudyDate DICOM tag or equivalent) — order-only suffixes = WEAK
3. Volumetric format (DICOM or NIfTI) — PNG-slice dumps = weak, flagged, last resort
4. Labels beyond images: segmentation, response/progression, survival, or treatment context — raw scans with no endpoints make temporal analysis uninterpretable
5. Open license, no fees, no impossible DUA (DUA-with-application is acceptable if free and realistic)
6. Zero synthetic content (banned outright)

**Verdict scale:** STRONG (meets all) / WEAK (timepoints or labels compromised) / NO (single-timepoint or synthetic).

### Part B — Agentic AI evidence reports (the explainer, NOT the diagnoser)

The CNN flags tissue and produces probabilities. The agentic layer's ONLY job: **generate structured, provenance-grounded reports answering "why did the model flag THIS region at THIS timepoint?"**

**Architecture (post-hoc, deterministic inputs, no diagnostic authority):**

```
CNN prediction (p(HIGH), calibrated)
  + Grad-CAM map (channel 0 = T1ce, verified)
  + input-gradient saliency
  + within-subject temporal delta (p_t vs p_t-1, drift)
  + crop provenance (which 96³ patch, coverage stats)
        │
        ▼
LLM agent (structured prompt, schema-constrained JSON output)
        │
        ▼
Per-case evidence report:
  { case, timepoint, probability, threshold_margin,
    attention_region: {coordinates, T1ce-bright fraction, brain-coverage},
    temporal: {prev_p, delta, trend_label},
    evidence_quality: {gradcam_saliency_agreement, saturation_flags},
    explicit_disclaimers: [not-a-diagnosis, no-segmentation-ground-truth, ...] }
```

**What the agent is ALLOWED to do:**
- Summarize and structure the numeric evidence the pipeline already computed
- Cross-reference Grad-CAM region vs saliency region (agreement = stronger evidence, divergence = instability flag)
- Describe temporal deltas between a patient's timepoints
- Emit explicit uncertainty and disclaimer sections
- Produce the "provenance report": every claim in the report traces to a computed artifact field

**What the agent is FORBIDDEN from doing (and the paper says so):**
- Making or suggesting a diagnosis, grading decision, or clinical recommendation
- Generating new "evidence" not present in the computed artifacts
- Interpreting anatomy it has no ground truth for (no "this is the tumour" claims — only "this is where the model looked")
- Any output used for clinical decisions

This is the honest version of "agentic AI in medical imaging": an evidence-summarizer and provenance-generator over deterministic model outputs, with schema constraints and traceability, explicitly walled off from diagnostic authority. It's defensible in review and it's genuinely useful (structured per-case JSON evidence beats raw heatmaps for error analysis).

---

## 2. Pipeline sketch for Phase 2 (gated on data)

```
Longitudinal dataset (per sweep results)
  → per-patient timeline construction (StudyDate ordering)
  → same preprocessing as Phase 1 (crop-first, mmap, memory guards)
  → CNN frozen or retrained per new data; per-timepoint predictions
  → temporal feature: p(t), Δp(t), Grad-CAM displacement across t
  → agentic evidence report per case-timepoint pair
  → evaluation: does temporal aggregation beat single-scan?
      (e.g., patient-level vote/mean vs first-scan-only AUROC)
```

**Memory constraint unchanged:** everything must still fit 4GB VRAM / ~3GB RAM. Longitudinal does NOT mean bigger models — it means more passes of the same small model over more timepoints of the same patient.

**Honest fallback ladder if NO strong tumour dataset exists:**
1. WEAK tumour data (order-only timepoints) → temporal analysis as "scan-sequence consistency" not "progression" (like Phase 1's §5.5, expanded)
2. Non-tumour longitudinal testbed (OASIS-3 etc.) → develop and validate the temporal pipeline mechanics, labelled clearly as non-tumour
3. If both fail → the longitudinal paper is not honestly writable; ship Phase 1 paper + the data-gap position paper instead

---

## 3. What happens when the sweep reports land

1. Build a comparison table of every dataset found (STRONG/WEAK/NO, patients, timepoints, format, labels, size, access)
2. Pick the primary dataset (decision rule: STRONG > most patients with ≥3 timepoints + labels; ties broken by DICOM+dates)
3. Verify with the DICOM date-recipe (StudyDate tags) before committing
4. Download via the verified API cookbook (TCIA REST / Zenodo / IDC S5CMD)
5. Re-run the Phase 1 data-audit style verification on the new data (file counts, shapes, dedup, subject grouping) — same rigour, new dataset
6. Then, and only then, build the temporal layer

---

## 4. Deliverables of Phase 2

- `research/data_sweep/` — the six sweep reports + comparison table
- `research/data_sweep/DATASET_DECISION.md` — the chosen dataset + evidence + acceptance-criteria scorecard
- Longitudinal pipeline code (temporal aggregation, per-timepoint evidence)
- Agentic evidence-report generator (schema-constrained, provenance-grounded)
- Paper #2: longitudinal analysis + agentic evidence reporting, research-grade, honest limitations
- Paper #1 (Phase 1 manuscript) remains shippable independently and is NOT blocked by Phase 2

---

## 5. Standing rules carried forward

- No synthetic data — if a dataset is GAN-augmented, it's banned, flagged, and reported as such
- No clinical claims; agentic layer has zero diagnostic authority (this is a paper-level statement, not a config choice)
- ET-derived grade proxy limitations carry over if the new dataset's labels are also proxy-derived (state it again, prominently)
- Memory guards: 4GB VRAM ceiling, mmap crop-first loading stays
- Every number in every report traces to a repository artifact (provenance chain is the agentic layer's whole job)

---

## 6. Ensemble policy — "the ant colony" (decided 2026-08-31)

**Principle.** Many weak-but-DIVERSE models can beat one big model — this is a theorem, not a metaphor (boosting: Schapire 1990; bagging/Random Forests: Breiman 2001). But ensemble error = average individual error − DIVERSITY (Krogh & Vedelsby 1995), and this project already measured what happens when diversity ≈ 0: the 5-member clone ensemble (OOF AUROC 0.5463) and the naive 5-crop vote (0.6058) both degraded below the single mature model (0.7641). Root cause in both cases: correlated errors, not "ensembles don't work."

**Colony membership rules:**
1. Every member must be individually competent — fine-tuned from the BraTS 2024 pretrained base (876 cases of prior knowledge), never trained from scratch on ~40 patients.
2. Every member must differ on at least one axis: data slice (bagging), crop/augmentation policy, seed + architecture width, or input view. No clones.
3. Sequential boosting is BANNED at this sample size (overfitting). Parallel bagging and temporal voting only.
4. The primary colony dimension is TIME: one model × many timepoints per patient (per-patient temporal median/vote). This is the measured-stable axis (3.4× within-subject stability, §5.5 of Phase 1).
5. Member disagreement is not failure — it is the uncertainty signal the agentic layer reports ("k of m members flagged this region").

**Colony v1.1 — specialized ants with a communication layer (refines v1):**

*Each ant has its own AWARENESS — a different input view AND a different job. No generalists, no clones.*

| Ant | Awareness (input view) | Specialization (what it tracks) |
|---|---|---|
| T1c ant | T1-contrast channel | enhancing tumour / blood–brain barrier breakdown |
| FLAIR ant | FLAIR channel | oedema / infiltration |
| T2 ant | T2 channel | oedema + cystic change |
| T1n ant | native T1 | anatomy / structural drift (resection, mass effect) |
| Time ant | p(t) sequence of prior ants' outputs at t−1, t−2… | change — the only ant that sees time |

*Communication channels (weakest → strongest pheromone):*
1. **The vote** — late fusion of member probabilities (median or mean; fixed weights, not learned)
2. **The pheromone field** — members deposit their Grad-CAM maps into a shared spatial overlay; overlap of k ants = strong trail, disagreement = explicit instability flag. Costs no extra training, cannot overfit, and is exactly what the agentic report layer reads ("3 ants converged on this region; FLAIR ant dissents")
3. **Learned gating (deferred, gated)** — a tiny meta-learner over member outputs (Wolpert 1992 stacking / MoE gating). BANNED in v1: at n=40 patients a learned gate is an overfitting machine (same rule that killed boosting). Unbanned only if leave-patients-out CV shows it beats the fixed vote, with intervals.

*Real-ant honesty note:* biological ants succeed via stigmergy — they communicate through the environment (pheromone trail), not by voting in a room. Our pheromone field is the faithful version: ants never see each other's weights, they only read/write the shared evidence map. That's why it's the safest channel at small n and the centerpiece of the agentic report.

**Colony v1 design (gated on PROTEAS audit passing):**
- Base: BraTS 2024 checkpoint → fine-tune on PROTEAS, patient-disjoint folds
- Members: 3–5 fine-tuned variants (different crop policies / seeds / augmentations)
- Aggregation: per-timepoint probabilities → per-patient temporal median → patient-level decision
- Evaluation: leave-patients-out CV; report single vs colony vs temporal-vote with honest confidence intervals (n=40 → wide intervals; a null result will be reported like Phase 1's negative results, not hidden)

**Evidence colony for the agentic report layer** (the ant colony applied to explanation):
Every report aggregates multiple INDEPENDENT weak evidence streams, never a single one:
1. Grad-CAM region overlap with expert segmentation (possible now — PROTEAS has expert segs per timepoint)
2. Saliency-vs-GradCAM agreement (two explanation methods voting; divergence = instability flag)
3. Temporal delta p(t) − p(t−1) with trend label
4. Crop provenance and coverage stats
5. Member-agreement count (k of m colony members flagged)
Two methods agreeing = stronger evidence; disagreement = explicit uncertainty in the report. The agent structures and quotes these — it still never diagnoses.
