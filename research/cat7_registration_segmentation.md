# Category 7: Registration and Segmentation for Progression

**Status: ✅ RESEARCH COMPLETE**

---

## 7.1 What Registration Solves

Longitudinal scans of the same patient must be aligned in space before comparison:
- **Intra-modality** — T1 at month 1 vs T1 at month 3
- **Cross-modality** — CT aligned to T1, T2, FLAIR space
- **Inter-institutional** — BraTS case aligned to IBSR case (standardised to MNI space)

---

## 7.2 Registration Approaches

| Method | Accuracy | Time per case | Our use |
|---|---|---|---|
| **Affine** (12 DOF) | Moderate | 2 sec | Quick pre-alignment |
| **Rigid** (6 DOF) | Good | 5 sec | Default for fusion pipeline |
| **Deformable (B-spline)** | Best | 30-60 sec | Cross-institutional |

**Our choice:** Rigid registration (ANTs `Registration` or `SyN`) between CT and MRI modalities within each case.

---

## 7.3 Skull-Stripping

Removes skull and extracranial tissue before model sees the image:

| Tool | Accuracy | Time |
|---|---|---|
| **ANTsHDGMM** | Best | 2-5 min |
| **BET (FSL)** | Good | 30 sec |
| **HD-BET (DL)** | Best | 1 min |

**Our choice:** ANTsHDGMM — best accuracy, acceptable for 80 IBSR + 600 C-BRATS cases.

---

## 7.4 Segmentation

Segmentation (U-Net, nnU-Net) is NOT needed for classification. We skip it.
- **Classification** — whole image → label
- **Segmentation** — voxel-by-voxel tumour mask

**We use the whole volume (64³ patches), not segmented tumour ROI.**

---

## 7.5 Preprocessing Pipeline (Full)

```
Raw NIfTI
  → Skull-strip (ANTsHDGMM)
  → Rigid registration (CT to MRI space)
  → Per-modality normalisation (CTN for MRI, intracranial mapping for CT)
  → Pad/crop to 256³
  → Pre-cache to .pt
  → Runtime: 64³ patch extraction with overlap
```

See `research/16f_preprocessing_pipelines.md` for details.
