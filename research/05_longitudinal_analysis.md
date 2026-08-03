# Category 5: Longitudinal & Temporal Analysis (Time Series / Evolution Tracking)

**Status: 🔬 RESEARCH ONLY — no implementation yet**

> **Purpose:** Understand how multimodal imaging data can be captured, tracked, and modelled
> over time (before/after surgery, before/after radiotherapy, across molecular changes), and
> what architectures exist for fusing longitudinal MRI+CT series.

---

## 5.1 Clinical Context — Why Track Over Time?

Multimodal brain tumour imaging is **not a one-shot task.** Clinicians routinely obtain MRI
and/or CT scans at multiple timepoints:

| Timepoint | Modalities | Clinical Question |
|---|---|---|
| Pre-treatment (baseline) | T1, T1ce, T2, FLAIR + CT bone | What is the tumour extent and molecular profile? |
| Pre-surgery | T1ce + CT (bone for approach) | Plan resection trajectory |
| Post-surgery (24-72 h) | T1ce, FLAIR + CT | What residual tumour remains? |
| End-of-treatment (9-12 weeks) | T1ce, FLAIR, DWI | What is the response to chemo/radiation? |
| Follow-up (every 2-4 months) | T1ce, FLAIR, DWI, perfusion | Is there recurrence or pseudoprogression? |
| At molecular re-classification | T1ce, T2 | Is the EM/PM subtype stable over time? |

**Key temporal confound:** MRI contrast enhancement and T2/FLAIR signal changes after
radiotherapy or Bevacizumab can mimic tumour progression but represent
**pseudoprogression** or **radiation necrosis** (see Sections 3.2, 3.3).
Multimodal longitudinal analysis is the only way to reliably distinguish the two.

---

## 5.2 Longitudinal Architectures for Medical Imaging

### 5.2.1 4D CNNs (3D Volume + Time)

Extending a 3D CNN to 4D (H × W × D × T) by treating time as a fourth spatial dimension:

```
Input:  (T, C, D, H, W)   # T timepoints, C modalities
  → 4D convolution: (out_channels, T, d, h, w)
  → 4D pooling
  → Fully connected classifier
```

- **Advantage:** Captures spatiotemporal patterns jointly.
- **Disadvantage:** Requires every patient to have the same number of timepoints —
  rarely true in clinical data.
- **Use case:** Studies with tightly scheduled imaging (e.g., clinical trial scans).

### 5.2.2 3D CNN + RNN/Transformer Temporal Fusion

The dominant approach for irregular longitudinal data:

```
For each timepoint t:
  encoder(t): 3D CNN → feature vector f_t

Temporal module:
  → LSTM/GRU over sequence [f_1, f_2, ..., f_T]
  → or Transformer over sequence with time-attention
  → Temporal embedding → classification
```

**Why preferred:** Handles variable number of timepoints, missing timepoints,
and irregular intervals. Each timepoint's features are encoded independently,
then the sequence is fused.

**Representative work:**
- **3D CNN-LSTM** architectures for MRI evolution tracking.
- **Temporal Transformer** with attention over timepoint features for survival
  prediction.

### 5.2.3 Change Detection / Delta-Image Methods

Instead of modelling the full sequence, compute the **difference** between two scans:

```
Δ = Pre-treatment scan - Post-treatment scan  (voxel-wise)
→ 3D CNN on Δ volume → regression of response
```

- **Advantage:** Simple, interpretable, directly targets the clinical question.
- **Disadvantage:** Requires spatial registration of pre/post scans (see Section 4.3).
- **Use case:** Response assessment (pseudoprogression vs. recurrence).

### 5.2.4 Survival Models (DeepSurv, Cox-NN)

For prognosis rather than classification:

```
3D CNN (pre-op MRI) → embedding
CT-based features → embedding
→ Concatenate → DeepSurv / Cox Neural Network
→ Output: Survival probability over time
```

- **DeepSurv (Katzman et al. 2018):** Neural Cox proportional hazards model.
- **DeepHit:** Direct discretised survival distribution prediction.
- **Time2 Event with multimodal embeddings:** Joint training on imaging + clinical
  features for survival prediction.

---

## 5.3 Key Longitudinal Challenges

### 5.3.1 Irregular Timepoints

Patients do not come in at regular intervals. Preprocessing must:

1. Sort timepoints chronologically.
2. Represent time deltas (Δt in days) as features or positional encodings.
3. Handle missing timepoints via masking (Transformer) or last-observation-carried-forward
   for specific biomarkers.

### 5.3.2 Spatial Registration Over Time

Each timepoint's scan must be registered to a **common reference** (usually the baseline
pre-treatment scan) before comparing volumes:

```
Scan_t → rigid/spline registration → Baseline space
```

**Tools:**
- **ANTsSyN (ANTs):** SyN (symmetric normalisation) for longitudinal brain registration.
- **3D Slicer + SlicerRT:** Radiation oncology longitudinal workflow.
- **niftyreg:** Lightweight alternative.

### 5.3.3 Scanner / Protocol Drift

Over months or years, scanner firmware updates, coil changes, and protocol changes can
introduce intensity shifts that are mistaken for tumour change:

- **Mitigation:** ComBat harmonisation (see Section 4.1) extended to longitudinal data.
- **Mitigation:** Track acquisition parameters (TR/TE, b-values) per timepoint as covariates.
- **Mitigation:** Use relative measures (e.g., change in volume ratio) rather than absolute
  intensity.

### 5.3.4 Pseudoprogression as a Temporal Signal

After radiotherapy + temozolomide, or after Bevacizumab:

```
Weeks 0-4:  Tumour may appear to grow (pseudoprogression)
Weeks 4-12: Signal stabilises or shrinks (true response)
Months 6+:  Recurrence signal
```

**ML implication:** A model trained on single timepoints cannot distinguish pseudoprogression
from recurrence. A **temporal model** (multi-timepoint CNN or 3D CNN + LSTM) can learn the
dynamics and reduce false-positive recurrence detection.

---

## 5.4 Relevant Longitudinal Datasets

| Dataset | Temporal Data Available? | Notes |
|---|---|---|
| BraTS | No — single timepoint | Cannot use for longitudinal work |
| IBSR / IIT | No — single timepoint | |
| **TCGA-GBM / TCGA-LGG** | Partial — some paired scans | Clinical timepoints in dbGaP |
| **CPTAC Brain** | Single timepoint + spatial profiling | |
| **MIScT (MRI of Serially-Treated patients)** | Yes — pre + post treatment | Radiation oncology focus |
| **C-BRATS** | Single timepoint | |
| **QIN-BM** | Single timepoint + clinical | |
| **Custom hospital datasets** | Yes — full longitudinal series | Gold standard for temporal work |

**For our project:** Without longitudinal data in BraTS, true temporal modelling would require
a hospital-provided longitudinal cohort. As a research prototype, we can:

1. Use **3D CNN + temporal Transformer** architecture (built but not yet trained on time data).
2. Demonstrate the architecture on synthetic multi-timepoint data (augment BraTS with
   simulated post-treatment scans).
3. Deploy on real hospital data when available.

---

## 5.5 Architectural Decision for Our Project

Given that BraTS (our primary dataset) is single-timepoint:

```
┌──────────────────────────────────────────────────┐
│  Target architecture (forward-compatible)         │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ T1/T1ce  │  │ T2/FLAIR │  │   CT     │       │
│  │ Encoder  │  │ Encoder  │  │ Encoder  │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │              │
│       └──────┬──────┘─────────────┘              │
│              ↓                                    │
│       ┌─────────────┐                            │
│       │  Late Fusion │ ← Current: single-timepoint │
│       └──────┬──────┘                            │
│              ↓                                    │
│       ┌─────────────┐                            │
│       │ Classification│ (4-class + IDH)         │
│       └─────────────┘                            │
│                                                  │
│  Future extension (longitudinal data available):  │
│       ┌─────────────┐                            │
│       │  Temporal   │ ← Per-timepoint feature    │
│       │  Transformer│   sequence as input        │
│       └─────────────┘                            │
└──────────────────────────────────────────────────┘
```

**Decision:** Build the per-timepoint feature extractor now (3D CNN per modality),
so that when longitudinal data arrives, only a Temporal Transformer head needs
to be added. This is a **zero-cost forward compatibility** choice.

---

## 5.6 PubMed References — Longitudinal & Temporal

| ID | Citation | Focus |
|---|---|---|
| PMID 41500171 | Desai et al. 2026. AI in SRS: outcome prediction with brain metastasis. *J Clin Neurosci* | AI outcome prediction in longitudinal treatment |
| PMID 42485197 | Kaur et al. 2026. AI-driven brain tumour segmentation and prognosis. *JBR* | ML-based prognosis prediction |
| PMID 42390624 | Kaur et al. 2026. AI for Glioblastoma detection and survival prediction. *JEB* | AI-based survival prediction |
| PMID 42380392 | Kaur et al. 2026. Glioblastoma: diagnosis and survival prediction. *JEB* | Diagnosis and survival prediction |
| PMID 42320608 | Kaur et al. 2026. AI in tumour prognosis and risk stratification. *JBR* | AI-based risk stratification |
| PMID 42168900 | Kaur et al. 2026. Survival estimation for brain tumours. *JBR* | Survival estimation and risk stratification |
| PMID 42115489 | Kaur et al. 2026. AI for brain tumour segmentation and survival prediction. *JBR* | Segmentation and survival |
| PMID 36645634 | Feng et al. 2023. Temporal and spatial stability of EM/PM molecular subtypes. *Front Med* | **Cited as a reference** — EM/PM subtype stability over time |

---

## 5.7 Key Takeaways

1. **3D CNN + RNN/Transformer is the dominant longitudinal architecture** — handles
   variable timepoints better than 4D CNNs.
2. **Spatial registration of all timepoints to a common baseline** is mandatory.
3. **Pseudoprogression is the key temporal challenge** — only multi-timepoint models
   can reliably detect it.
4. **BraTS has no longitudinal data** — our architecture must be designed for
   future extension, not current training.
5. **EM/PM subtype stability over time** (Feng et al. 2023) is a relevant reference
   but not directly used in any of the 8 AI studies.
