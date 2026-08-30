# Data Sweep Report — Longitudinal-Compatible Public Brain Tumour Datasets

**Date:** 2026-08-31 · **Method:** 6 subagent sweeps (5 died to rate limits mid-run but their searches + cached pages were salvaged; findings below verified from actual collection pages, not memory)
**Bar:** same patient ≥2 studies · real dates · DICOM/NIfTI · labels · open license · zero synthetic content

---

## THE TABLE

| # | Dataset | Host | Patients | Timepoints | Format | Dates | Labels | CT? | Size | Access | VERDICT |
|---|---------|------|-----------|------------|--------|-------|--------|-----|------|--------|---------|
| 1 | **PROTEAS (RFUds)** | Zenodo 17253793 | 40 (+5 with 2 RT rounds → ~45 dirs) | baseline + fu1/fu2/… @ 6wk/3mo/6mo/9mo/12mo | DICOM raw + NIfTI BraTS-space (t1/t1c/t2/fla per timepoint!) | ✅ real (clinical xlsx: detection dates, RT start/end) | ✅ 65 mets segmented (ET/edema/necrosis, expert-refined), clinical xlsx (KPS, treatment, survival time), radiomics xlsx | ✅ paired CT+RTP per patient | 15.1 GB | open CC-BY zip download | **STRONG** |
| 2 | **Yale-Brain-Mets-Longitudinal** | TCIA (DOI 10.7937/3yat-e768) | **1,430** | 11,892 studies, ~20 years, pre/post SRS/WBRT/resection | NIfTI (caseID_date-time_sequence) | ✅ real (date in filename + clinical xlsx) | clinical xlsx (age, sex, study date, scanner) — NO segmentations | ❌ MRI only | 43 GB | Aspera download (free, plugin), CC BY 4.0 | **STRONG** (scale) |
| 3 | **Brain-Tumor-Progression** | TCIA | 20 GBM | 2 each (post-CRT ~90d + at progression) | DICOM (T1/T1c/FLAIR/T2/ADC/perfusion + tumor masks) | ✅ real (DICOM StudyDate) | ✅ progression timepoint + binary tumor masks | ❌ | 3.2 GB | NBIA retriever, free | **STRONG** (small but perfect fit: progression is THE label) |
| 4 | **BraTS-Reg 2022** | Zenodo 14642405 | ~52 training pairs (glioma) | baseline pre-op + follow-up (27d–48mo window) | NIfTI, BraTS-preprocessed | ⚠️ order (baseline/follow-up), interval known per pair | ⚠️ landmark annotations (registration GT, not tumour labels) | ❌ | 3.72 GB + 0.53 GB | open zip (CC-BY-ND) | **WEAK-STRONG** (longitudinal yes, tumour labels no) |
| 5 | LUMIERE | figshare/Kaggle | 100+ | follow-up MRI + auto segmentations | NIfTI | ✅ real | auto-segmentations (model-generated — flag: not expert) | ❌ | ~20 GB | open | **WEAK** (auto labels) |
| 6 | Kaggle mirrors (andrewmvd/brain-tumor-progression, murtozalikhon CT&MRI etc.) | Kaggle | 20 / ~3k PNGs | 2 / none | PNG/DICOM mix | ⚠️ | partial | ⚠️ | small | open | **WEAK** (re-scrapes; use originals instead) |

Notable NOs (checked, not viable): TCGA-GBM/LGG (mostly single study/patient), BraTS 2024 (repeated acq. no dates — our Phase 1), brain-met single-timepoint sets (BrainMetShare, BraTS-METS 2023 pre-treatment only — explicitly noted in the PROTEAS paper itself).

---

## RECOMMENDATION — two-dataset strategy

### Primary: PROTEAS/RFUds (Zenodo 17253793)
Why: it's the ONLY dataset that hits EVERY requirement at once —
- Per-patient folders `P01…P40` with `BraTS/baseline, fu1, fu2, …` subfolders, each containing the SAME four sequences (t1, t1c, t2, fla) in NIfTI already in BraTS space — **drop-in compatible with our existing 4-channel pipeline** (identical channel semantics to Phase 1!)
- Real timepoints at clinically predefined intervals (6wk/3/6/9/12mo) — true longitudinal, not order-guessing
- Expert-refined segmentations of 3 subregions at EVERY timepoint → we can compute ET-volume trajectories over time as the longitudinal signal, and validate our CNN's attention against real lesion locations (fixes Phase 1's biggest explanation weakness!)
- Paired CT + RT dose maps per patient (bonus: dose overlay analysis; CT is real, not synthetic)
- Clinical xlsx: treatment, KPS, survival — response/progression context exists
- 15.1 GB, per-patient zips (~170–790 MB each) — we can grab a few patients at a time to fit disk, and verify before full pull
- CC-BY open license, data citation = the Scientific Data paper (Flouri et al. 2025, 10.1038/s41597-025-06131-0)

### Secondary/scale: Yale-Brain-Mets-Longitudinal (TCIA)
- 1,430 patients × 11,892 studies is 2 orders of magnitude bigger for any patient-level generalisation claims
- No segmentations → use only if/when we need scale for a model-level claim, not for the evidence/explanation work
- 43 GB — needs disk planning

### Also grab: Brain-Tumor-Progression (3.2 GB, tiny)
- Perfect progression-specific testbed: 20 GBM patients, post-CRT vs progression scans, tumor masks included
- Complements PROTEAS (mets) with a glioma cohort
- Small enough to download whole and verify in an afternoon

### BraTS-Reg: use as registration/alignment reference data
- Longitudinal pairs + landmark GT: useful if we need to align PROTEAS timepoints or study attention displacement; not a primary training set (no tumour labels)

---

## NEXT STEPS (concrete)

1. Download PROTEAS clinical xlsx FIRST (small) → verify patient table, timepoint structure, label fields before committing 15 GB
2. Download 2–3 patient zips (P01, P02, P03 ~700 MB) → run the Phase-1-style audit: file counts, shapes, dtype, dates, segmentation consistency, timepoint mapping
3. If audit passes → download remaining patients in background batches
4. Also download Brain-Tumor-Progression (3.2 GB) as the glioma-progression companion
5. Build `scripts/audit_longitudinal_dataset.py` (port of verify_preprocessed_data.py: per-patient timeline table, timepoint counts, label coverage, CT/MRI pairing, date sanity)
6. Then the longitudinal pipeline work begins (temporal aggregation, per-timepoint evidence, agentic reports)

## DOWNLOAD COOKBOOK (verified paths)

- PROTEAS clinical xlsx: `https://zenodo.org/records/17253793/files/PROTEAS-Clinical_and_demographic_data.xlsx` (or via `/api/records/17253793/files/<name>/content` — the sweeper hit 403 on browser-style URLs; the API content endpoint or the record page download buttons work)
- PROTEAS per-patient: `https://zenodo.org/records/17253793/files/P01.zip` … P40 (+P04a/b, P07a/b, P17a/b, P20a/b, P23a/b)
- BraTS-Reg: `https://zenodo.org/api/records/14642405/files/BraTSReg_Training_Data.zip/content` (3.72 GB) + Validation (0.53 GB)
- TCIA Brain-Tumor-Progression & Yale: NBIA retriever / Aspera per collection pages (free, no DUA — citation required)
- TCIA REST API: public, no API key needed anymore (v4): `https://api.tcia.net/api/v1/...` endpoints (getSeries, getPatientStudy etc.) — good for verifying studies-per-patient before downloading

## CITATIONS TO ADD TO THE BIB

- Flouri, D. et al. *A longitudinal MRI dataset of brain metastases with tumor segmentations, clinical & radiomic data.* Sci Data 12, 1828 (2025). doi:10.1038/s41597-025-06131-0 — Zenodo 17253793
- Chadha, S. et al. *Yale longitudinal dataset of brain metastases on MRI with associated clinical data (v1).* TCIA (2025). doi:10.7937/3YAT-E768
- Clark, K. et al. TCIA. J Digit Imaging 26, 1045–1057 (2013) + Brain-Tumor-Progression collection citation
- Baheti, B. et al. *The BraTS-Reg Challenge.* arXiv:2112.06979 — Zenodo 14642405

---

**Bottom line: the longitudinal goal is REAL and achievable.** PROTEAS alone gives us 40 patients × baseline+follow-ups × 4 BraTS-standard sequences × expert segmentations × clinical endpoints × paired CT — the exact shape of data the old research docs claimed didn't exist. The hunt succeeded (◕‿◕)★
