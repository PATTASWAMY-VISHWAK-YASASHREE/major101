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


---

## Implementation Alignment: Current MRI-Only Trainer

The generic pipeline above describes a future longitudinal CT/MRI workflow. The currently implemented path is narrower: it consumes preprocessed four-channel MRI `.npy` volumes for binary `grade_proxy` classification. The active loader expects shape `(4, 182, 218, 182)` with channels T1, T1ce, T2, and FLAIR, validates the shape and finite values, and uses memory-mapped loading to keep the full volume out of GPU memory.[1]

The default training route extracts `64 x 64 x 64` patches from the memory-mapped volume. It is crop-first and memory bounded; a separate `--whole-volume` option supports a downsampled full-volume alternative. Training augmentation is restricted to the controls exposed by the dataset implementation, including optional additive noise and spatial flipping. Validation and locked-test data are loaded without training augmentation.[1] This means that future experiment tables must state whether a run used patches or whole-volume input, because the two settings expose different spatial context.

The current implementation does **not** execute the DICOM ingestion, N4 bias correction, registration, segmentation, longitudinal timepoint alignment, or RANO progression stages shown in the recommended pipeline above. Those steps remain research guidance for future modalities and tasks, not evidence of current functionality. The implementation-specific method and evidence boundary are documented in [`20_implemented_mri_grade_pipeline.md`](20_implemented_mri_grade_pipeline.md).

### Current preprocessing contract

| Stage | Implemented contract | Research status |
|---|---|---|
| Input | One preprocessed `.npy` volume per case | Implemented |
| Modalities | T1, T1ce, T2, FLAIR | Implemented |
| Shape validation | `(4, 182, 218, 182)` and finite sampled values | Implemented |
| Memory strategy | NumPy memory mapping and crop-first patch loading | Implemented |
| Default model input | `64 x 64 x 64` patch | Implemented |
| Whole-volume alternative | Optional downsampled whole-volume path | Implemented but not default |
| CT registration/fusion | No active path | Planned |
| Longitudinal registration | No active path | Planned |
| Segmentation masks | No active path in the classifier | Planned or separate task |

[1]: [`src/grade_data.py`](../src/grade_data.py)
[2]: [`scripts/train_ultra_light.py`](../scripts/train_ultra_light.py)
[3]: [`20_implemented_mri_grade_pipeline.md`](20_implemented_mri_grade_pipeline.md)
