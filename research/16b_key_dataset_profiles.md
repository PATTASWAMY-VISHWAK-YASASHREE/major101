# 16b — Key Dataset Profiles

---

## BraTS (Primary Candidate)

| Field | Value |
|-------|-------|
| **ID** | D001 |
| **Citation** | BraTS Benchmark (CBICA), 2012–2024 |
| **URL** | https://www.synapse.org/Synapse:syn2582906 (training), syn2582907 (testing) |
| **Access** | Public (Synapse registration required) |
| **Licensing** | Open Access for challenge participants |
| **Size** | 2024: 4,047 training + 100 test cases (largest edition) |
| **Modalities** | T1, T1ce, T2, FLAIR (mpMRI only — no CT) |
| **Anatomical focus** | Glioma (high-grade + low-grade) |
| **Lesion annotations** | 4-class segmentation: edema, necrosis, non-enhancing tumor, enhancing tumor |
| **Timepoints** | 1 per patient (pre-operative); 2012/2018 editions included post-op pairs |
| **Longitudinal** | No — not designed for longitudinal tracking |
| **CT component** | None |
| **Relevance to project** | 3/5 — largest brain tumor dataset, but MRI-only and not longitudinal |
| **Status** | VERIFIED |

**Project-specific notes:**
- BraTS is the de facto standard for brain tumor segmentation research
- For progression monitoring, the 2012/2018 pre/post pairs provide the closest thing to longitudinal data
- A project using BraTS would need to either: (a) treat pre/post pairs as pseudo-longitudinal, or (b) combine with institutional data for true longitudinal tracking
- No CT means the project's CT+MRI scope requires a separate CT source or synthetic CT generation

---

## TCIA HGG + LGG + GBM Collections (Secondary Candidate)

| Field | Value |
|-------|-------|
| **ID** | D002, D003, D014 |
| **Citation** | TCIA — High Grade Glioma (HGG), Low Grade Glioma (LGG), Glioblastoma (GBM) |
| **URL** | https://www.cancerimagingarchive.net |
| **Access** | Public (with approval) |
| **Licensing** | Open Access |
| **Size** | 416 total (HGG: 144, LGG: 172, GBM: 100) |
| **Modalities** | MRI + some CT (varies by case) |
| **Anatomical focus** | Glioma (grade-specific) |
| **Lesion annotations** | Varies — some have manual segmentations, some clinical only |
| **Timepoints** | 1 per patient (mostly) |
| **Longitudinal** | No — not designed for longitudinal tracking |
| **CT component** | Some cases have CT, but CT+MRI overlap count not well-documented |
| **Relevance to project** | 2–4/5 — only public brain tumor dataset with CT+MRI, but not longitudinal |
| **Status** | PARTIAL — individual collection details need per-case review |

**Project-specific notes:**
- TCIA collections are the only public source of brain tumor CT+MRI data
- Individual case review needed to identify how many patients have both CT and MRI
- Not designed for progression tracking — single timepoint per case
- Would need institutional data for true longitudinal monitoring

---

## DeepLesion (Partial Candidate)

| Field | Value |
|-------|-------|
| **ID** | D013 |
| **Citation** | DeepLesion — large-scale lesion detection dataset (Xu et al., 2018) |
| **URL** | https://github.com/zyxiang/DeepLesion |
| **Access** | Public (requires registration) |
| **Licensing** | Open Access |
| **Size** | 133,422 annotations across 13 lesion types |
| **Modalities** | CT + MRI |
| **Anatomical focus** | Multi-organ (lung, liver, spleen, kidney, stomach, brain, etc.) |
| **Lesion annotations** | Bounding box + type labels for 13 lesion categories |
| **Timepoints** | 1 |
| **Longitudinal** | No |
| **CT component** | 71 brain CT cases (654 annotations) — brain MRI only 3 cases |
| **Relevance to project** | 1/5 — brain MRI component is minimal; not longitudinal |
| **Status** | VERIFIED |

---

## Recommended Dataset Strategy for This Project

Given the constraints identified:

| Requirement | Available Source | Gap |
|-------------|-----------------|-----|
| Large brain tumor MRI | BraTS (4,047 cases) | None |
| Brain tumor CT+MRI | TCIA collections (416 cases) | Limited CT+MRI overlap, not well-documented |
| Longitudinal tracking | No public dataset | **Critical gap** — requires institutional data |
| Progression annotations | No public dataset | **Critical gap** — RANO/PRECIST annotations not publicly available |

**Recommendation:** Use BraTS as the primary training dataset for model development, then validate on institutional CT+MRI longitudinal data if available. If institutional data is unavailable, consider:
1. Using BraTS pre/post pairs (2012/2018 editions) as pseudo-longitudinal validation
2. Synthetic CT generation from BraTS MRI (if CT is required for the project)
3. Combining TCIA CT cases with BraTS MRI for a mixed-modality baseline
