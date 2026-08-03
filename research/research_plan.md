# Research Plan — Longitudinal CT+MRI Brain Tumor Progression Monitoring

> **Project:** Major101
> **Date initiated:** 2026-06-15
> **Status:** In progress — partial completion achieved

---

## 1. Project Overview

This project investigates AI-powered monitoring of brain tumor progression using longitudinal CT and/or MRI imaging. The core scientific question is:

> Can deep learning models trained on multi-timepoint imaging data accurately detect and quantify brain tumor progression over time, and how do their results compare to manual radiologist assessment using standard criteria (RANO/PRECIST)?

**Project axes (4 pillars):**
1. **Imaging domain:** Brain tumor (glioma, metastasis, meningioma)
2. **Modality:** CT and/or MRI (dual-modality preferred)
3. **Study design:** Longitudinal (multiple timepoints per patient)
4. **Method:** Deep learning (CNN, transformer, or hybrid architectures)

---

## 2. Scope (19 Research Categories)

The research is organized into 19 categories covering the full landscape:

| # | Category | Status |
|---|----------|--------|
| 1 | Brain tumor imaging basics | Planned |
| 2 | Tumor progression monitoring | Planned |
| 3 | Deep learning in medical imaging | Planned |
| 4 | Multimodal imaging fusion (CT+MRI) | Planned |
| 5 | Longitudinal analysis methods | Planned |
| 6 | AI model architectures for brain tumor | Planned |
| 7 | Registration and segmentation for progression | Planned |
| 8 | RANO/PRECIST criteria and clinical standards | Planned |
| 9 | Multi-center dataset harmonization | Planned |
| 10 | Uncertainty quantification in DL predictions | Planned |
| 11 | Explainable AI for clinical adoption | Planned |
| 12 | Data augmentation for medical imaging | Planned |
| 13 | Transfer learning from natural images | Planned |
| 14 | Real-world validation and clinical deployment | Planned |
| 15 | Ethical considerations and bias | Planned |
| 16 | Datasets and benchmarks | **Partial — 16a, 16b completed** |
| 17 | Evaluation metrics for progression monitoring | Planned |
| 18 | Comparison with manual radiologist assessment | Planned |
| 19 | Direct prior work / base papers | **Partial — 5 papers collected, 3 verified as 4/4 base paper candidates** |

---

## 3. Research Questions

### Primary Questions
1. What public datasets provide longitudinal CT+MRI brain tumor data, and what are their limitations?
2. What deep learning architectures have been proposed for brain tumor progression monitoring?
3. How do AI-based progression assessments compare to manual RANO/PRECIST evaluation?

### Secondary Questions
4. What is the state of multimodal (CT+MRI) fusion for brain tumor analysis?
5. How do registration and segmentation errors propagate to progression measurements?
6. What are the key ethical and clinical deployment barriers?

---

## 4. Methodology

### Phase 1: Dataset Inventory (Categories 16a–16h) ✅ Partial
- **16a_dataset_inventory.md** ✅ Completed — 14 datasets inventoried
- **16b_key_dataset_profiles.md** ✅ Completed — BraTS, TCIA, DeepLesion profiles
- **16c_dataset_comparison_matrix.csv** 📋 Pending
- **16d_longitudinal_datasets_deepdive.md** 📋 Pending
- **16e_preprocessing_pipelines.md** 📋 Pending
- **16f_dataset_gap_analysis.md** 📋 Pending
- **16g_access_licensing_ethics.md** 📋 Pending
- **16h_executive_summary.md** 📋 Pending

### Phase 2: Literature Review (Categories 1–15, 17–18) 📋 Pending
- Systematic PubMed/arXiv searches per category
- Paper-level details with verification
- Citation counts and relevance scores

### Phase 3: Base Paper Selection (Category 19) 📋 In Progress
- **19_direct_prior_work_base_papers.md** ✅ Completed — 11 papers, 3 verified as 4/4 candidates
- Key finding: **No public dataset or prior work combines CT+MRI+longitudinal+DL** — all 4/4 candidates use MRI only

### Phase 4: Synthesis
- Cross-category synthesis document
- Gap analysis: what the project can uniquely contribute
- Recommended approach and limitations

---

## 5. Key Findings (to date)

### Critical Gaps Identified

1. **No public longitudinal CT+MRI brain tumor dataset exists.** BraTS (the primary dataset) is MRI-only. TCIA collections have CT+MRI but are not longitudinal.

2. **All verified 4/4 base paper candidates use MRI only.** The project's CT+MRI scope represents a genuine research gap — no prior work has combined CT and MRI for longitudinal brain tumor monitoring.

3. **Two tracks in the field:** (a) recurrence/outcome prediction from longitudinal scans and (b) temporally consistent segmentation across timepoints. A project doing both would cover new ground.

4. **Recent activity (2026)** suggests this is an actively developing area, with the Mathivanan group leading in longitudinal brain tumor DL.

### Dataset Recommendations
- **Primary:** BraTS 2024 (4,047 cases, mpMRI, largest available)
- **Secondary:** TCIA HGG+LGG+GBM collections (416 cases, CT+MRI available)
- **Validation:** Institutional data required for true longitudinal monitoring
- **Pseudo-longitudinal:** BraTS 2012/2018 pre/post pairs (limited but available)

---

## 6. Recommended Next Steps

1. **Complete Phase 1** — finish remaining 16c–16h files
2. **Phase 2** — conduct targeted literature searches for each remaining category
3. **Phase 3** — finalize base paper selection with full paper-level detail
4. **Phase 4** — write synthesis document and gap analysis
5. **Project proposal** — based on research findings, define the AI model architecture and clinical validation plan

---

## 7. Limitations

- This is a desktop research exercise; no actual data access or model training has been performed
- PubMed API rate limits and web_fetch failures limited the depth of some category searches
- Verification is based on abstract-level confirmation; full-text review of papers is pending
- Dataset licensing terms for institutional datasets were not individually confirmed
