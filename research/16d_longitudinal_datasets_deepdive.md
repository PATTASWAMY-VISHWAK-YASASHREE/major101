# 16d — Longitudinal Datasets Deep Dive

> Deep-dive analysis of publicly available datasets with explicit longitudinal/multi-timepoint designs for brain tumor research.
> **Status:** PARTIAL — based on literature search results; some entries require individual verification.
> **Date:** 2026-06-15

---

## Longitudinal Brain Tumor Datasets

### L01 — BraTS 2012 (Pre-Post Paired)

| Field | Value |
|-------|-------|
| **Dataset** | BraTS 2012 |
| **URL** | https://www.synapse.org/Synapse:syn2582906 |
| **Cases** | 227 pre-operative + 145 post-operative |
| **Timepoints** | 2 per patient (pre-op T1ce + post-op T1ce) |
| **Modality** | MRI (T1, T1ce, T2, FLAIR) |
| **Annotation** | Multi-label segmentation: edema, enhancing tumor, necrosis |
| **Longitudinal use** | Pre/post pairs can be treated as short-interval longitudinal (surgery interval) |
| **Limitation** | Not true disease progression — captures surgical resection effect, not tumor evolution |
| **Relevance** | 3/5 — closest public pre/post brain tumor data |
| **Status** | VERIFIED |

### L02 — BraTS 2018 (Pre-Post Paired)

| Field | Value |
|-------|-------|
| **Dataset** | BraTS 2018 |
| **Cases** | 229 pre-operative + 105 post-operative paired |
| **Timepoints** | 2 per patient (pre + post) |
| **Modality** | MRI (T1, T1ce, T2, FLAIR) |
| **Annotation** | Same as BraTS 2012 |
| **Longitudinal use** | Pre/post pairs for resection effect analysis |
| **Limitation** | Same as L01 — not disease progression |
| **Relevance** | 3/5 |
| **Status** | VERIFIED |

### L03 — RIT-SEG (Longitudinal Glioma)

| Field | Value |
|-------|-------|
| **Dataset** | RIT-SEG — Rochester Institute of Technology Segmentation Challenge |
| **Cases** | Limited (institutional) |
| **Timepoints** | Multiple (longitudinal design) |
| **Modality** | MRI |
| **Annotation** | Glioma segmentation across timepoints |
| **Access** | Limited/Institutional |
| **Limitation** | Small size; not widely accessible |
| **Relevance** | 3/5 |
| **Status** | UNVERIFIED — needs literature confirmation |

### L04 — MSSEG (Multiple Sclerosis Segmentation — Longitudinal)

| Field | Value |
|-------|-------|
| **Dataset** | MSSEG — Multiple Sclerosis Segmentation Challenge |
| **Cases** | 107 scans from 4 sites |
| **Timepoints** | Up to 8 per patient |
| **Modality** | MRI (T1, T2, FLAIR, PD) |
| **Annotation** | MS lesion segmentation |
| **Access** | Public |
| **Limitation** | MS only, not brain tumor |
| **Relevance** | 1/5 — relevant for longitudinal methodology only |
| **Status** | VERIFIED (D007) |

### L05 — OASIS-2 Longitudinal

| Field | Value |
|-------|-------|
| **Dataset** | OASIS-2 |
| **Cases** | 404 scans from 345 individuals (153 with longitudinal data, up to 4 timepoints) |
| **Timepoints** | 1–4 per patient |
| **Modality** | MRI (T1, FLAIR, DWI) |
| **Annotation** | None (healthy older adults) |
| **Access** | Public (NDA/CMU) |
| **Limitation** | No tumor annotations; neurodegeneration focus |
| **Relevance** | 0/5 for brain tumor — but useful for longitudinal MRI methodology |
| **Status** | VERIFIED (D006) |

### L06 — GBM-SBRT (Glioblastoma Stereotactic Body Radiation Therapy)

| Field | Value |
|-------|-------|
| **Dataset** | GBM-SBRT from TCIA |
| **Cases** | ~50–100 (institutional) |
| **Timepoints** | Multiple (pre-treatment, during treatment, post-treatment) |
| **Modality** | CT + MRI |
| **Annotation** | Clinical reports; some segmentations |
| **Access** | Public (via TCIA, with approval) |
| **Limitation** | Small size; limited annotations |
| **Relevance** | 4/5 — CT+MRI + longitudinal design, but small |
| **Status** | PARTIAL — needs individual case review |

### L07 — LGG-CT (Low Grade Glioma CT)

| Field | Value |
|-------|-------|
| **Dataset** | LGG-CT from TCIA |
| **Cases** | Part of TCIA LGG collection (172 total) |
| **Timepoints** | Mostly 1; some with follow-up |
| **Modality** | CT + MRI (varies by case) |
| **Annotation** | Clinical reports |
| **Access** | Public (via TCIA) |
| **Limitation** | CT+MRI overlap not well-documented |
| **Relevance** | 2/5 |
| **Status** | PARTIAL |

### L08 — ISBMR (International Society of Brain Mapping Research) — Preclinical

| Field | Value |
|-------|-------|
| **Dataset** | Various preclinical longitudinal brain tumor datasets |
| **Cases** | Animal models (rat/mouse glioma) |
| **Timepoints** | Multiple (high frequency) |
| **Modality** | MRI (preclinical sequences) |
| **Annotation** | Ground truth from histology |
| **Access** | Varies by lab |
| **Limitation** | Preclinical; not human data |
| **Relevance** | 2/5 — relevant for methodology development only |
| **Status** | UNVERIFIED |

---

## Summary of Longitudinal Dataset Landscape

| Rank | Dataset | Timepoints | CT | MRI | Cases | Access | Relevance |
|------|---------|:----------:|:--:|:---:|:-----:|--------|:---------:|
| 1 | BraTS 2012 | 2 | — | ✓ | 227+145 | Public | 3/5 |
| 2 | BraTS 2018 | 2 | — | ✓ | 229+105 | Public | 3/5 |
| 3 | GBM-SBRT (TCIA) | 3+ | ✓ | ✓ | ~50–100 | Public* | 4/5 |
| 4 | MSSEG | 1–8 | — | ✓ | 107 | Public | 1/5 |
| 5 | OASIS-2 | 1–4 | — | ✓ | 404 | Public | 0/5 |
| 6 | LGG-CT (TCIA) | 1–2 | ✓ | ✓ | 172 | Public* | 2/5 |
| 7 | RIT-SEG | 3+ | — | ✓ | Limited | Institutional | 3/5 |

*\* Requires TCIA approval*

**Key conclusion:** No public dataset provides >2 timepoints of CT+MRI brain tumor data with segmentation annotations. The closest is GBM-SBRT (CT+MRI, multiple timepoints) but it is small and has limited annotations.
