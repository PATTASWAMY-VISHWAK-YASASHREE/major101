# Dataset Acquisition Guide — Where & How To Download

**Zero budget. No DUA applications needed. Only ONE signup total (Kaggle, optional route).**
**Rule: download order = verify small first, then commit disk.**

---

## DATASET 1 (PRIMARY): PROTEAS / RFUds — 40 patients, longitudinal, segmented

**What:** 40 brain-metastasis patients · baseline + follow-ups at 6wk/3mo/6mo/9mo/12mo · 4 MRI sequences per timepoint (t1, t1c, t2, fla in NIfTI, BraTS space) · expert segmentations at every timepoint · paired CT + RT dose · clinical xlsx (treatment, KPS, survival)

- **Download page:** https://zenodo.org/records/17253793
- **Direct file pattern:** `https://zenodo.org/records/17253793/files/P01.zip` (replace P01 → P02…P40; double-RT patients have P04a/P04b etc.)
- **Size:** 15.1 GB total · per-patient zips ~140–790 MB · has md5 checksums on page
- **Clinical data (get this FIRST):** `PROTEAS-Clinical_and_demographic_data.xlsx` — download button on the same record page
- **Signup needed:** NO. Zenodo is fully open. Click download, done.
- **License:** CC-BY — free to use, just cite: Flouri et al., Sci Data 12, 1828 (2025), doi:10.1038/s41597-025-06131-0
- **Search keywords to refind it:** "PROTEAS brain metastases Zenodo" · "RFUds radiotherapy follow-up dataset" · "longitudinal MRI brain metastases segmentations Scientific Data"
- **Folder structure to expect:** `P01/BraTS/baseline/{t1,t1c,t2,fla}.nii.gz`, `P01/BraTS/fu1/...`, `P01/tumor_segmentation/...`, `P01/P01_CT.nii.gz`, `P01/P01_RTP.nii.gz`
- **Step-by-step:**
  1. Open the record page → download the clinical xlsx → inspect (patient table, timepoints, survival)
  2. Download P01.zip, P02.zip, P03.zip (~700 MB total)
  3. Unzip → verify structure + shapes + dates match the paper's description
  4. If pass → download the rest in background batches (disk needed: ~15 GB + ~15 GB unzipped, so ~30 GB free)

---

## DATASET 2 (GLIOMA COMPANION): Brain-Tumor-Progression — 20 GBM patients, 2 timepoints each

**What:** 20 glioblastoma patients · 2 MRI exams each (post-chemoradiation ~90d + at progression) · DICOM (T1/T1c/FLAIR/T2/ADC/perfusion) · binary tumor masks · all co-registered

**Route A — Kaggle (EASIEST, needs free signup):**
- **Download page:** https://www.kaggle.com/datasets/andrewmvd/brain-tumor-progression
- **Size:** 3.16 GB, 8,798 files
- **Signup:** YES — free Kaggle account (email + password, 2 minutes)
- **License:** CC BY 3.0 · properly cites its TCIA source (Schmainda & Prah 2018, doi:10.7937/K9/TCIA.2018.15quzvnb)
- **Steps:** create account → open dataset page → Download button. Or via API: Settings → Create New API Token → `kaggle datasets download -d andrewmvd/brain-tumor-progression`

**Route B — TCIA original (no signup, needs Java):**
- **Download page:** https://www.cancerimagingarchive.net/collection/brain-tumor-progression
- **Method:** NBIA Data Retriever (Java app) OR the TCIA REST API (no key required):
  - `curl "https://api.tcia.net/api/v1/getSeries?Collection=Brain-Tumor-Progression"` → JSON list of all series with PatientID, StudyDate, Modality — verify 2 studies/patient BEFORE downloading
- **Signup:** NO (API is keyless; NBIA needs no account)
- **Search keywords:** "TCIA brain tumor progression" · "Schmainda Prah brain tumor progression"

---

## DATASET 3 (SCALE, LATER): Yale-Brain-Mets-Longitudinal — 1,430 patients, 11,892 studies

**What:** the largest public brain-met dataset · ~20 years of scans per patient · 4 sequences · NIfTI (filenames contain dates) · clinical xlsx · NO segmentations · MRI only

- **Download page:** https://www.cancerimagingarchive.net/collection/yale-brain-mets-longitudinal/
- **Size:** 43 GB
- **Signup:** NO account, BUT image download requires the **IBM Aspera Connect browser plugin** (free install) — the clinical xlsx alone is a direct download:
  - `https://www.cancerimagingarchive.net/wp-content/uploads/Yale-Brain-Mets-Longitudinal_ClinicalData_20250605.xlsx` (2.67 MB)
- **License:** CC BY 4.0 · cite Chadha et al. 2025, doi:10.7937/3YAT-E768
- **Search keywords:** "Yale brain metastases longitudinal TCIA"
- **DO THIS LATER** — only if we need patient-scale generalisation. 43 GB + no segmentations = not needed for the evidence-report work.

---

## DATASET 4 (REFERENCE): BraTS-Reg — longitudinal glioma pairs + landmark ground truth

**What:** baseline pre-op + follow-up glioma MRI pairs (27 days–48 months apart) · 4 sequences · landmark annotations for registration · use for alignment work, not training

- **Download:** https://zenodo.org/records/14642405 (open, no signup)
- **Direct links:**
  - `https://zenodo.org/api/records/14642405/files/BraTSReg_Training_Data.zip/content` (3.72 GB)
  - `https://zenodo.org/api/records/14642405/files/BraTSReg_Validation_Data.zip/content` (0.53 GB)
- **License:** CC-BY-ND · cite Baheti et al., arXiv:2112.06979
- **Search keywords:** "BraTS-Reg Zenodo" · "brain tumor sequence registration challenge data"

---

## TCIA REST API CHEAT SHEET (no key, no signup)

```bash
# List every series with patient/study/date for a collection:
curl "https://api.tcia.net/api/v1/getSeries?Collection=Brain-Tumor-Progression"

# Check studies per patient before downloading anything:
curl "https://api.tcia.net/api/v1/getPatientStudy?Collection=Brain-Tumor-Progression"
# → count distinct StudyInstanceUID per PatientID in the JSON; expect 2 each

# Full API docs: https://wiki.cancerimagingarchive.net/display/Public/TCIA+Programmatic+Interface+REST+API+Guide
```

---

## RECOMMENDED DOWNLOAD ORDER

| Step | What | Disk | Signup | Why |
|---|---|---|---|---|
| 1 | PROTEAS clinical xlsx | 2 MB | none | verify structure before any big pull |
| 2 | PROTEAS P01–P03 zips | ~700 MB | none | audit pass/fail gate |
| 3 | Brain-Tumor-Progression (Kaggle or TCIA) | 3.2 GB | Kaggle OR none | glioma companion, tiny |
| 4 | PROTEAS remaining patients | ~15 GB (+15 unzip) | none | the main dataset |
| 5 | BraTS-Reg | 4.3 GB | none | alignment reference |
| 6 | Yale (only if needed) | 43 GB | Aspera plugin | scale experiments later |

**Disk needed for steps 1–5: ~40 GB free.** For everything including Yale: ~110 GB.
