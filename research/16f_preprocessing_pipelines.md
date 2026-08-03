# 16f — Preprocessing Pipelines for Brain Tumor Longitudinal Imaging

> Overview of standard preprocessing pipelines for longitudinal brain tumor CT/MRI analysis.
> **Status:** PARTIAL — based on established literature; specific pipeline implementations vary by project.
> **Date:** 2026-06-15

---

## Standard Preprocessing Steps

### 1. Data Acquisition & Quality Control

| Step | Description | Standard Method |
|------|-------------|-----------------|
| DICOM ingestion | Convert DICOM to NIfTI | `dcm2niix` |
| Metadata extraction | Extract scanner, sequence, orientation | `dicomdir` / `pydicom` |
| Quality check | Flag incomplete, mislabeled, or corrupted scans | Manual + automated (size, intensity range) |
| Orientation standardization | Ensure RAS orientation | `nibabel.orientations` |
| Slice order verification | Confirm correct slice ordering | Visual inspection |

### 2. Intensity Normalization

| Step | Description | Standard Method | Notes |
|------|-------------|-----------------|-------|
| CT Hounsfield windowing | Clip to [-100, 250] HU range | Min-max clipping | Standard for brain CT |
| MRI intensity normalization | Z-score or N4 bias correction | `N4BiasFieldCorrection` (ANTs) | Mandatory for MRI |
| Skull stripping | Remove non-brain tissue | `BET` (FSL) or `HD-BET` (deep learning) | Required before registration |
| Brain extraction | Ensure brain-only masks | Combine BET + manual QC | Critical for accuracy |

### 3. Spatial Normalization & Registration

| Step | Description | Standard Method | Notes |
|------|-------------|-----------------|-------|
| Affine registration | Align to common template | `ANTs`, `FSL FLIRT` | For cross-subject comparison |
| Non-linear registration | Warp to MNI template | `ANTs SyN` | For multi-atlas fusion |
| Intra-subject registration | Align timepoints for same patient | Rigid + affine | Critical for longitudinal analysis |
| Multi-modal registration | CT ↔ MRI alignment | Mutual information-based | Required for CT+MRI fusion |
| Template selection | Choose appropriate atlas | MNI152 (adult); tailored for tumor | Affects downstream accuracy |

### 4. Tumor Segmentation (Pre-Progression)

| Step | Description | Standard Method | Notes |
|------|-------------|-----------------|-------|
| Initial segmentation | Define tumor boundaries at each timepoint | U-Net, nnU-Net, BraTS challenge methods | Input to progression analysis |
| Sub-region separation | Separate edema, necrosis, enhancing tumor | BraTS 4-class standard | Enables RECIST/RANO computation |
| Post-processing | Clean up segmentation masks | Morphological operations, hole-filling | Reduces noise |

### 5. Longitudinal Progression Metrics

| Step | Description | Standard Method | Notes |
|------|-------------|-----------------|-------|
| Volume change | Compute tumor volume at each timepoint | Voxel counting × voxel volume | Simple but limited |
| Spatial growth mapping | Identify expanding vs. stable regions | Registration-based difference maps | More informative |
| RANO/PRECIST computation | Standard response criteria | Automated RECIST measurement | Clinical standard |
| Temporal consistency check | Verify masks across timepoints | Dice similarity, volume change rate | Flags registration errors |

### 6. Data Augmentation (Training)

| Step | Description | Standard Method | Notes |
|------|-------------|-----------------|-------|
| Spatial transforms | Random rotation, scaling, cropping | `albumentations`, `MONAI` | Be careful with anatomical realism |
| Intensity transforms | Random brightness, contrast, noise | `albumentations` | Simulates scanner variability |
| Flip/reflect | Horizontal/vertical flips | `albumentations` | Anatomically valid for brain |
| Mixup/cutmix | Mix patient scans | Careful with brain anatomy | May not be anatomically valid |

---

## Recommended Pipeline for This Project

```
Raw DICOM → dcm2niix → N4 bias correction → BET skull strip
          → Affine register to MNI152 → Non-linear warp
          → Intensity normalize → Patch extraction (if needed)
          → Timepoint registration (intra-subject)
          → Segmentation → Progression metrics
```

---

## Known Challenges

1. **Multi-site harmonization:** Different scanners produce different intensity distributions. Intensity normalization alone is insufficient — see COMBAT harmonization (popular in neuroimaging).

2. **Intra-subject motion:** Patient movement between timepoints can cause registration errors that masquerade as tumor progression.

3. **Partial volume effects:** At tumor boundaries, voxels may contain mixed tissue types, leading to inaccurate volume estimates.

4. **Tumor evolution non-stationarity:** Tumor growth patterns are not stationary; simple time-series models may fail.

5. **Missing timepoints:** In real-world clinical data, patients often have irregular scan schedules, leading to irregular time intervals.

---

## Tools and Libraries

| Tool | Purpose | License |
|------|---------|---------|
| ANTs | Registration, bias correction | GPL-3.0 |
| FSL | Skull stripping (BET), registration (FLIRT) | GPL-3.0 |
| MONAI | End-to-end medical image preprocessing + DL | Apache 2.0 |
| nibabel | NIfTI/DICOM file I/O | BSD |
| SimpleITK | Registration, resampling | Apache 2.0 |
| antsPy | Python wrapper for ANTs | MIT |
