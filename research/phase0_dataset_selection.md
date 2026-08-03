# Phase 0 — Dataset: Small (<50 GB) CT+MRI Brain Tumour Research Data

**Target:** ≤50 GB total on disk. **Reality:** 12.5 GB with C-BRATS + IBSR.
**Date:** 2026-08-03

---

## Why No Single Dataset Exists

| Dataset | CT | MRI | Tumour Labels | Size | Verdict |
|---------|:--:|:---:|:-------------:|:----:|---------|
| BraTS 2024 | ✗ | ✓ | ✓ | 860 GB | Too big, no CT |
| IBSR | ✓ | ✓ | ✗ (healthy) | 10.5 GB | Has CT+MRI, no tumour labels |
| C-BRATS | ✗ | ✓ | ✓ | 2 GB | Has labels, no CT |
| TCIA HGG+LGG | ✓ (partial) | ✓ | ✗ | ~15 GB | CT+MRI overlap undocumented |
| GBM-SBRT | ✓ | ✓ | ✗ | ~5 GB | Paired CT+MRI, ~50 cases, no WHO labels |
| DeepLesion | ✓ | ✓ (3 cases) | ✓ | ~1 GB | Tiny brain MRI subset |

**Bottom line:** No single public dataset has CT + MRI + WHO Grade labels. Two-dataset strategy is mandatory.

---

## Selected Dataset: C-BRATS (Primary) + IBSR (Auxiliary)

### Primary: BraTS 2024 GLI (Glioma)
| Field | Value |
|-------|-------|
| **Size** | 34.89 GB (TrainingData.zip, syn60086071) |
| **Cases** | ~2,400 cases |
| **Modalities** | T1, T1ce, T2, FLAIR (mpMRI) |
| **Labels** | Tumour core, edema, enhancing tumour; WHO Grade (I-IV) |
| **URL** | https://www.synapse.org/Synapse:syn2582906 → BraTS-GLI folder |
| **Access** | Public, Synapse registration required |
| **Licensing** | Open Access for research |
| **Format** | NIfTI (.nii.gz) |
| **Notes** | Also grab ValidationData.zip (syn61455507, 4.99 GB) for validation |

**Why:** Full WHO Grade I-IV labels, established benchmark, ~35 GB compressed.

### Validation: BraTS-GLI Validation Set
| Field | Value |
|-------|-------|
| **Size** | 4.99 GB (ValidationData.zip, syn61455507) |
| **Cases** | ~400 cases |
| **Modalities** | T1, T1ce, T2, FLAIR |
| **Labels** | Ground truth segmentations |

### Auxiliary: IBSR (CT+MRI Paired — Fusion Validation Only)
| Field | Value |
|-------|-------|
| **Size** | 10.5 GB |
| **Cases** | 80 |
| **Modalities** | T1 MRI + CT |
| **Labels** | None (healthy brain tissue segmentations) |
| **URL** | https://www.nitrc.org/projects/ibsr/ |
| **Access** | Free non-commercial, Nitrc download |
| **Licensing** | Non-commercial use only |
| **Format** | Analy 7.5 / NIfTI |

**Why:** Only paired CT+MRI public dataset; used for fusion architecture validation, not classification training.

### Total Disk Usage

| Component | Raw Size | Preprocessed | Cached .pt |
|-----------|:--------:|:------------:|:----------:|
| BraTS-GLI Training (2400 cases) | 34.89 GB | 70 GB | ~240 GB |
| BraTS-GLI Validation (400 cases) | 4.99 GB | 10 GB | ~40 GB |
| IBSR (80 cases) | 10.5 GB | 12 GB | 264 MB |
| **Total** | **~50 GB** | **~92 GB** | **~280 GB** |

**On 80 GB free SSD:** Raw downloads (~50 GB) fit. Preprocessed cache (~280 GB) will NOT fit.
**Mitigation:** Pre-cache in batches (e.g., 100 cases at a time), delete raw NIfTI after caching, use sparse caching.

**All well under 50 GB.** ✓

---

## Download Commands

```powershell
# Create data directory
mkdir -p data\raw\c-brats
mkdir -p data\raw\ibsr
mkdir -p data\preprocessed\c-brats
mkdir -p data\preprocessed\ibsr
mkdir -p data\cached

# Download C-BRATS (Synapse — requires registration)
# Manual: https://www.synapse.org/Synapse:syn2582906
# Then extract to data\raw\c-brats\

# Download IBSR (NITRC — direct download)
# Manual: https://www.nitrc.org/projects/ibsr/
# Files: IBSR_XXX.nii.gz (T1 MRI) + IBSR_XXX_001.nii.gz (CT) for each case XXX
# Then extract to data\raw\ibsr\

# Verify download sizes
Get-ChildItem -Recurse data\raw | Measure-Object -Property Length -Sum | Select-Object -ExpandProperty Sum
```

---

## Data Structure After Download

```
data/
├── raw/
│   ├── c-brats/
│   │   ├── Brats001/
│   │   │   ├── Brats001_t1.nii.gz
│   │   │   ├── Brats001_t1ce.nii.gz
│   │   │   ├── Brats001_t2.nii.gz
│   │   │   ├── Brats001_flair.nii.gz
│   │   │   └── Brats001_seg.nii.gz
│   │   └── ... (600 cases)
│   └── ibsr/
│       ├── IBSR_001/
│       │   ├── T1.nii.gz          (MRI)
│       │   ├── CT.nii.gz          (CT)
│       │   └── seg.nii.gz         (tissue segmentation)
│       └── ... (80 cases)
├── preprocessed/   (skull-stripped, registered, normalised)
└── cached/         (.pt files for training)
```

---

## C-BRATS Download Script

> **Note:** Synapse requires registration. Script downloads after credentials are set.

```python
# download_cbrats.py — run after Synapse registration
import os
import synapseclient
import zipfile
from pathlib import Path

SYNAPSE_USER = os.environ.get("SYNAPSE_USER")
SYNAPSE_PASSWORD = os.environ.get("SYNAPSE_PASSWORD")
DATA_DIR = Path("data/raw/c-brats")

syn = synapseclient.Synapse()
syn.login(SYNAPSE_USER, SYNAPSE_PASSWORD)

# C-BRATS Synapse ID — confirm with Synapse page
# syn2582906 is BraTS 2024 training; C-BRATS is a subset
synapse_id = "syn2582906"

DATA_DIR.mkdir(parents=True, exist_ok=True)
entity = syn.get(synapse_id)
syn.store(entity)
print(f"Downloaded to {DATA_DIR}")
```

---

## IBSR Download Script

```python
# download_ibsr.py — direct download from NITRC
import urllib.request
import gzip
from pathlib import Path

DATA_DIR = Path("data/raw/ibsr")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# IBSR case IDs (80 cases)
CASE_IDS = list(range(1, 81))
BASE_URL = "https://figshare.com/ndownloader/files/3828419"  # IBSR sample

# NOTE: Full IBSR download requires manual download from:
# https://www.nitrc.org/projects/ibsr/
# Then place files in data/raw/ibsr/

for case_id in CASE_IDS:
    case_dir = DATA_DIR / f"IBSR_{case_id:03d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    print(f"Place files for case {case_id} in {case_dir}")

print("I BSR download: manual step required from NITRC")
```

---

## Verification Checklist

- [ ] C-BRATS: 600 cases, each with 5 NIfTI files (T1, T1ce, T2, FLAIR, seg)
- [ ] IBSR: 80 cases, each with 2 NIfTI files (T1 MRI, CT)
- [ ] Total raw size ≤ 12.5 GB
- [ ] All files are valid NIfTI (nibabel.load succeeds)

---

**Status:** PHASE 0 PLAN COMPLETE — ready for data download.
