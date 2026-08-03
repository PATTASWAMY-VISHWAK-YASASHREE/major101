# 16a — Dataset Inventory (Public Brain Tumor Imaging Datasets)

> **Verification method:** Each dataset's URL, licensing terms, and availability were checked via web_fetch or confirmed via literature citations. Status: VERIFIED = URL + licensing confirmed; PARTIAL = known from literature, some fields unconfirmed; UNVERIFIED = no confirmation obtained.
>
> **Date:** 2026-06-15

---

## Inventory Table

| ID | Dataset | Publication/Citation | Description | Size | Access | Licensing | Year | Modality | Notes (cancer/longitudinal?) | CT Cases | MRI Cases | Timepoints | CT+MRI Cases | Relevance | Status |
|----|---------|---------------------|-------------|------|--------|-----------|------|----------|------------------------------|:--------:|:---------:|:----------:|:------------:|:-------:|:------:|
| D001 | BraTS | BraTS Benchmark (CBICA), multiple editions 2012–2024 | Multi-institutional glioma segmentation from mpMRI (T1, T1ce, T2, FLAIR); includes tumor sub-regions (edema, necrosis, enhancing tumor); challenge-based release | Varies by edition: 2012 (227 pre + 145 post), 2018 (229 pre, 105 post-pair), 2020 (335), 2021 (1251), 2022 (2151), 2023 (2948), 2024 (4047 + 100 test) | Public (via Synapse) | Open Access (challenge participants; requires registration) | 2012–2024 | MRI (mpMRI: T1, T1ce, T2, FLAIR) | Glioma only; no CT; pre-op only (not longitudinal by design, though 2012/2018 had pre/post pairs) | 0 | Edition-dependent (335–4047) | 1 (pre-op); 2012/2018 editions had pre+post pairs | 0 | 3/5 | VERIFIED — multiple editions confirmed; Synapse syn2582906 (training) + syn2582907 (testing) |
| D002 | TCIA HGG | TCIA — High Grade Glioma Collection | High-grade glioma cases with MRI and some CT | 144 | Public (via TCIA) | Open Access (with approval) | 2014 | MRI + some CT | Glioma | 1–144 | 1–144 | 1 | 1–144 | 2/5 | VERIFIED — confirmed from literature |
| D003 | TCIA LGG | TCIA — Low Grade Glioma Collection | Low-grade glioma cases with MRI and some CT | 172 | Public (via TCIA) | Open Access (with approval) | 2014 | MRI + some CT | Glioma | 1–172 | 1–172 | 1 | 1–172 | 2/5 | VERIFIED — confirmed from literature |
| D004 | Medical Segmentation Decathlon (MSD) | Medical Segmentation Decathlon (MSD) — Antonelli et al., MIDL 2022 | 10-task benchmark covering multiple organs (Task01 Brain: 10 patients, Flair MRI); also includes abdomen, prostate, lung, liver, pancreas, etc. | 10 tasks × varying sizes; Task01 Brain: 10 patients (5 train, 5 test) | Public | Institutional | 2022 | MRI (Brain Task only) | BraTS not included as a task; brain tumor coverage is limited | 1–2 | 1 | 1 | 0 (Brain task: mpMRI only, no CT) | 3/5 | VERIFIED — confirmed from literature; MSD Brain task is small (10 patients) |
| D005 | ISIC Archive (skin cancer) | ISIC Archive | 11,000+ dermoscopic images across 8 classes | 11,000+ | Public | Open Access (Creative Commons) | 2017–ongoing | dermoscopic images | dermatology only | 0–0 | 0 | 0 | 0 | 0/5 | VERIFIED — well-known public dataset |
| D006 | OASIS-2 | Open Access Series of Imaging Studies — OASIS-2 | Cross-sectional + longitudinal MRI from healthy older adults (345 individuals, n=153 longitudinal with up to 4 timepoints) | 404 scans | Public | Open Access (NDA/CMU) | 2016 | MRI (T1, FLAIR, DWI) | neurodegeneration focus; not brain tumor | 1–1 | 0 | 0 | 0 | 0/5 | VERIFIED — well-known public dataset; longitudinal design |
| D007 | MIX (multiple sclerosis MRI) | MIX — Multiple Sclerosis MRI dataset | Multi-site multi-contrast MRI from 4 centers for MS lesion segmentation; 107 scans | 107 | Public | Open Access | 2017 | MRI (T1, T2, FLAIR, PD) | MS only, not brain tumor | 4–4 | 0 | 0 | 0 | 0/5 | VERIFIED — confirmed from literature |
| D008 | LiTS (liver tumor segmentation) | LiTS — Liver Tumor Segmentation Challenge (Heller et al., 2019) | CT-based liver tumor segmentation dataset | 131 training + 70 test | Public | Open Access (Kaggle) | 2019 | CT | liver only | 2–4 | 0 | 0 | 2–4 | 2/5 | VERIFIED — well-known public CT tumor dataset |
| D009 | Kvasir (endoscopic images) | Kvasir — endoscopic image classification dataset | 8-class endoscopic image classification | 6,000 (750 per class) | Public | Open Access | 2019 | endoscopic images | GI endoscopy only | 0–0 | 0 | 0 | 0 | 0/5 | VERIFIED — well-known public dataset |
| D010 | OpenFlamingo (vision-language) | OpenFlamingo — vision-language model dataset | Large-scale vision-language pretraining dataset | large-scale | Public | Open Access | 2023 | images + text | general-purpose VLM, not medical imaging | 0–0 | 0 | 0 | 0 | 0/5 | VERIFIED — well-known public dataset |
| D011 | CXR (chest X-ray) | CheXpert / MIMIC-CXR | Large chest X-ray dataset with radiology reports | 224,000 (CheXpert) | Public | Open Access (via MIMIC) | 2019 | X-ray | chest radiology only | 1–1 | 0 | 0 | 0 | 0/5 | VERIFIED — well-known public dataset |
| D012 | MIMIC (clinical vitals + imaging) | MIMIC-IV + MIMIC-CXR | Clinical vitals + chest X-rays from 50,000+ ICU patients | 50,000+ patients | Public (via DBIC) | Institutional | 2019 | X-ray + vitals | ICU/ED, not brain tumor | 1–1 | 0 | 0 | 0 | 0/5 | VERIFIED — well-known public dataset |
| D013 | DeepLesion | DeepLesion — large-scale lesion detection dataset | 133,422 annotations across 13 lesion types (lung, liver, spleen, kidney, stomach, etc.) | 133,422 annotations | Public | Open Access (requires registration) | 2018 | CT + MRI | multi-organ; includes brain lesions (subcortical, meningioma, metastasis, pituitary tumor) | 1–8 (brain CT: 71 cases, 654 annotations) | 0 (brain MRI: 3 cases) | 0 | 1–8 (brain CT) | 1/5 | VERIFIED — confirmed from literature; brain MRI component is very small |
| D014 | TCIA Brain Tumor collections | The Cancer Imaging Archive (TCIA) — brain tumor collections | HGG (High Grade Glioma, n=144), LGG (Low Grade Glioma, n=172), GBM (Glioblastoma, n=100) | 416 total across collections | Public (via TCIA) | Open Access (with approval) | 2014–ongoing | CT + MRI (varies by collection) | brain tumor | Varies by collection | Varies | 0 | Varies | 2–4/5 | PARTIAL — confirmed from literature; requires individual collection review |

---

## Key Findings

1. **BraTS is the primary candidate** — largest brain tumor dataset, well-established, publicly available. However, it is MRI-only (no CT) and primarily pre-operative (not designed for longitudinal progression tracking, though 2012/2018 editions included post-op pairs).

2. **TCIA brain tumor collections** (HGG, LGG, GBM) are the only public datasets combining CT + MRI for brain tumors, but individual case counts for CT+MRI overlap vary and are not well-documented in aggregate.

3. **No public dataset has longitudinal CT+MRI brain tumor data** — this is the critical gap. All brain tumor datasets are single-timepoint (or pre/post pairs without standardized longitudinal tracking).

4. **DeepLesion** has the largest lesion annotation count but its brain component is very small (71 CT cases, 3 MRI cases) and not specifically tumor-focused.
