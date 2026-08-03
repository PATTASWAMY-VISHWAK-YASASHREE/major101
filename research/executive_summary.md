# Executive Summary — Longitudinal CT+MRI Brain Tumor Progression Monitoring

> **Project:** Major101
> **Date:** 2026-06-15
> **Status:** Phase 1 Complete — ready for Phase 2

---

## 1. The Problem

Brain tumor progression monitoring requires clinical assessment of multi-timepoint imaging to determine whether a tumor is growing, stable, or responding to treatment. Current clinical standards (RANO/PRECIST criteria) rely on manual radiologist assessment, which is time-consuming, subjective, and inter-reader variable. AI-powered automation could improve accuracy and efficiency — but the project scope must be grounded in what is actually feasible given available data.

---

## 2. Research Scope

The research covered 19 categories across four pillars:
- **Imaging domain:** Brain tumor (glioma, metastasis, meningioma)
- **Modality:** CT and/or MRI (dual-modality)
- **Study design:** Longitudinal (multi-timepoint)
- **Method:** Deep learning (CNN, transformer, hybrid)

---

## 3. Key Findings

### Dataset Landscape (Categories 16a–16h)

| Finding | Detail |
|---------|--------|
| **Primary dataset** | BraTS 2024 — 4,047 training + 100 test cases, mpMRI (T1, T1ce, T2, FLAIR), multi-institutional, publicly available via Synapse |
| **CT+MRI source** | TCIA collections (HGG: 144, LGG: 172, GBM: 100) — only public source of brain tumor CT+MRI data, but CT+MRI overlap is undocumented |
| **Longitudinal source** | **None** — no public dataset provides ≥3 timepoints of brain tumor imaging with segmentation |
| **Pre/post approximation** | BraTS 2012/2018 pre/post pairs (227+145 and 229+105) — closest public approximation but represent surgical resection, not disease progression |

### Critical Gaps

| Gap | Severity | Impact |
|-----|:--------:|--------|
| No longitudinal CT+MRI brain tumor dataset | 🔴 Critical | Cannot train/validate a true longitudinal model on public data |
| No progression annotations (RANO/PRECIST) | 🔴 Critical | Ground truth for progression is unavailable |
| CT+MRI overlap in TCIA undocumented | 🔴 Critical | Cannot quantify how many patients have both modalities |

### Prior Work (Category 19)

11 papers were collected; 3 were verified as 4/4 base paper candidates:

| Rank | Paper | Modality | Longitudinal | DL | Contribution |
|------|-------|:--------:|:------------:|:--:|-------------|
| 1 | Chen et al. (2026) — MambaDiff-Net | MRI | ✓ (pre/post) | ✓ | Recurrence risk prediction (AUC 0.887) |
| 2 | Mathivanan et al. (2026) — STMT | MRI | ✓ (multi-timepoint) | ✓ | Temporally consistent segmentation |
| 3 | Mathivanan et al. (2026) — TSTN | MRI | ✓ (multi-timepoint) | ✓ | Temporal spatial transformer for consistency |

**Key observation:** All 4/4 candidates use MRI only. No prior work combines CT+MRI+longitudinal+DL. The project's scope represents a genuine research gap.

---

## 4. Recommended Approach

| Phase | Activity | Dataset | Timeline |
|-------|----------|---------|----------|
| **Phase 1** | Train baseline models on BraTS 2024 | BraTS 2024 (mpMRI) | 2–4 weeks |
| **Phase 2** | Develop temporal model using BraTS 2012/2018 pre/post pairs | BraTS 2012/2018 | 4–6 weeks |
| **Phase 3** | Validate CT+MRI fusion on TCIA collections | TCIA HGG/LGG/GBM | 4–8 weeks |
| **Phase 4** | Clinical validation on institutional longitudinal data | Institutional partner | 6–12 months |

---

## 5. Project Feasibility

| Criterion | Assessment |
|-----------|------------|
| Data availability | ⚠️ Partial — training data available; validation data requires institutional partnership |
| Prior art | ✅ Strong — multiple recent papers provide architectural templates |
| Technical feasibility | ✅ High — modern DL frameworks (PyTorch, MONAI) support this pipeline |
| Clinical relevance | ✅ High — addresses unmet need for objective progression assessment |
| Novelty | ✅ High — CT+MRI+longitudinal+DL intersection is unexplored |

---

## 6. Deliverables Completed (Phase 1)

| File | Description |
|------|-------------|
| `16a_dataset_inventory.md` | Inventory of 14 public brain tumor imaging datasets |
| `16b_key_dataset_profiles.md` | Deep-dive profiles for BraTS, TCIA, DeepLesion |
| `16d_longitudinal_datasets_deepdive.md` | Analysis of 8 longitudinal/multi-timepoint datasets |
| `16e_dataset_comparison_matrix.csv` | Structured comparison of all datasets |
| `16f_preprocessing_pipelines.md` | Standard preprocessing pipeline documentation |
| `16g_dataset_gap_analysis.md` | Gap analysis against project requirements |
| `16h_access_licensing_ethics.md` | Access, licensing, and ethical considerations |
| `19_direct_prior_work_base_papers.md` | 11 papers from Category 19; 5 with full details |
| `research_plan.md` | Full 19-category research plan and methodology |

---

## 7. Next Steps

1. **Phase 2 (Literature Review)** — Complete targeted searches for remaining 18 categories (estimated: 2–3 weeks)
2. **Phase 3 (Base Paper Selection)** — Finalize 3–5 base papers with full paper-level detail
3. **Phase 4 (Synthesis)** — Cross-category synthesis, gap analysis, and project proposal
4. **Model Development** — Begin BraTS 2024 baseline training in parallel

---

## 8. Limitations

- This research is desktop-based; no actual data access or model training has been performed
- PubMed API rate limits and web_fetch failures limited the depth of some category searches
- Verification is at the abstract level; full-text review of papers is pending
- Institutional data access terms were not individually confirmed
- Categories 1–15, 17–18 remain incomplete
