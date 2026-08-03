# 16g — Dataset Gap Analysis

> Analysis of gaps between available public brain tumor datasets and project requirements.
> **Status:** COMPLETED
> **Date:** 2026-06-15

---

## Project Requirements vs. Available Datasets

| Requirement | Description | Best Available Source | Gap Severity |
|-------------|-------------|----------------------|:------------:|
| **Large-scale brain tumor MRI** | >1000 cases with segmentation | BraTS 2024 (4,047 cases) | ✅ Met |
| **Multi-contrast MRI** | T1, T1ce, T2, FLAIR | BraTS (all 4 contrasts) | ✅ Met |
| **CT brain tumor data** | CT scans of brain tumors | TCIA collections (~144 HGG, ~172 LGG, ~100 GBM) | ⚠️ Partial — limited CT+MRI overlap |
| **CT+MRI combined** | Cases with both modalities | TCIA collections (overlap count unknown) | 🔴 Critical — overlap not documented |
| **Longitudinal (≥3 timepoints)** | Multiple scans per patient | No public dataset | 🔴 Critical — no public source |
| **Longitudinal CT+MRI** | CT+MRI across timepoints | No public dataset | 🔴 Critical — no public source |
| **Progression annotations** | RANO/PRECIST labels | No public dataset | 🔴 Critical — no public source |
| **Multi-center data** | Scans from multiple hospitals | BraTS (multi-institutional) | ✅ Met |
| **Graded glioma labels** | High-grade vs. low-grade | BraTS, TCIA collections | ✅ Met |
| **Sub-region segmentation** | Edema, necrosis, enhancing tumor | BraTS (4-class standard) | ✅ Met |

---

## Gap Summary

### Critical Gaps (🔴)

1. **No longitudinal CT+MRI brain tumor dataset exists publicly.** This is the most fundamental gap. All public datasets are single-timepoint. BraTS 2012/2018 pre/post pairs are the closest approximation but represent surgical resection effects, not disease progression.

2. **No progression annotations (RANO/PRECIST) are publicly available.** Progression assessment requires clinical annotations that are not part of public dataset releases.

3. **CT+MRI overlap in TCIA collections is undocumented.** While TCIA collections include CT and MRI, the number of patients with both modalities is not systematically documented.

### Moderate Gaps (⚠️)

4. **Limited pre/post paired data.** BraTS 2012 (227 pre + 145 post) and 2018 (229 pre + 105 post) provide the only publicly available pre/post brain tumor data, but these represent surgical resection, not tumor progression.

5. **Preprocessing pipeline standardization.** No single preprocessing pipeline is universally adopted across datasets. Multi-site intensity harmonization is an open problem.

### Minor Gaps (✅ Partially Met)

6. **Longitudinal MRI methodology is well-established** but focused on neurodegeneration (OASIS-2) or MS (MSSEG), not brain tumors. The methodological lessons transfer but must be adapted.

---

## Recommended Mitigation Strategies

| Gap | Mitigation |
|-----|-----------|
| No longitudinal CT+MRI dataset | Use institutional data for validation; develop methodology on BraTS pre/post pairs; synthetic CT generation from BraTS MRI if CT is required |
| No progression annotations | Use BraTS 4-class segmentation as proxy; develop automated RANO/PRECIST computation from segmentation masks; partner with clinical sites for annotation |
| Undocumented CT+MRI overlap | Request specific case counts from TCIA; perform per-case modality audit if institutional data is available |
| Limited pre/post data | Treat BraTS 2012/2018 pairs as pseudo-longitudinal baseline; combine with MSSEG/OASIS-2 for longitudinal methodology development |
| No harmonization standard | Adopt COMBAT harmonization; document preprocessing pipeline thoroughly |

---

## Implications for Project Design

1. **Primary dataset:** BraTS 2024 (training) + BraTS 2012/2018 (pseudo-longitudinal validation)
2. **Secondary dataset:** TCIA collections (CT+MRI baseline)
3. **Validation strategy:** Institutional longitudinal data required for clinical validation
4. **Key contribution:** The project's CT+MRI+longitudinal scope addresses a genuine gap — no existing public resource covers this intersection
