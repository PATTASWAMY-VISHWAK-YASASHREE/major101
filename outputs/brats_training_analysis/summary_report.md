# BraTS-GLI Training Data — Full Analysis Report

Generated: 2026-08-04
Dataset: BraTS2024 BraTS-GLI Training (syn60086071)

## 1. Overview

| Metric | Value |
|--------|-------|
| Total cases (directories) | 1350 |
| Total .nii.gz files | 6750 |
| Total size on disk | 37.96 GB |
| Typical 3D shape | 182 × 218 × 182 |
| Unique patients | 613 |
| Unique sequences | 10 |
| Modalities per case | T1ce, T1n, FLAIR, T2w (4) |
| Segmentation (seg.nii.gz) | YES — present in training set |

## 2. Modalities

| Code | Meaning |
|------|---------|
| t1c | T1-weighted post-contrast (contrast-enhanced) |
| t1n | T1-weighted native (pre-contrast) |
| t2f | FLAIR (Fluid Attenuated Inversion Recovery) |
| t2w | T2-weighted |

## 3. Per-Modality Intensity Statistics

| Modality | N voxels | Grand Mean | Grand Std | Min | Max | P1 | P50 | P95 | P99 | Non-zero % |
|----------|----------|------------|-----------|-----|-----|----|-----|-----|-----|------------|
| t1c | 1350 | 285.6 | 741.7 | -450.0 | 39474.6 | 0.0 | 0.0 | 2304.3 | 2590.9 | 20.79% |
| t1n | 1350 | 244.6 | 634.4 | -57.0 | 10014.4 | 0.0 | 0.0 | 1990.7 | 2171.7 | 20.81% |
| t2f | 1350 | 112.9 | 278.9 | -27.0 | 5328.0 | 0.0 | 0.0 | 572.0 | 708.7 | 20.79% |
| t2w | 1350 | 207.8 | 545.5 | 0.0 | 14526.8 | 0.0 | 0.0 | 821.7 | 1493.6 | 20.82% |

## 4. Segmentation Analysis

### Tumor Region Definitions

| Label | Region | BraTS-GLI definition |
|-------|--------|---------------------|
| 1 | Edema (WC) | Peritumoral edema — white matter involvement |
| 2 | Non-enhancing tumor core (NCR) | Necrotic / non-enhancing core |
| 3 | Enhancing tumor core (ET) | Contrast-enhancing viable tumor |
| 2+3 | Tumor Core | Enhancing + non-enhancing core |
| 1+2+3 | Whole Tumor | All tumor-involved tissue |

### Tumor Volume Summary

| Region | Cases with Tumor | Total Volume (mm3) | Mean Volume (mm3) |
|--------|-----------------|-------------------|-------------------|
| non-enhancing core (NCR) | 1350 | 66476419 | 49242 |
| whole tumor (WST) | 1150 | 18825010 | 16370 |
| tumor_tumor_core | 1350 | 77919374 | 57718 |
| tumor_whole_tumor | 1350 | 18825010 | 13944 |
| tumor_edema_only | 1350 | 2285704 | 1693 |
| tumor_enhancing_only | 1350 | 11442955 | 8476 |
| enhancing core (ET) | 990 | 11442955 | 11559 |
| edema (ET) | 565 | 2285704 | 4045 |

### Tumor Statistics

| Metric | Value |
|--------|-------|
| Cases with detectable tumor | 1150 / 1350 (85.2%) |
| Mean whole tumor volume | 16370 mm3 |
| Median whole tumor volume | 8712 mm3 |
| Max whole tumor volume | 165848 mm3 |

## 5. Key Findings

1. **Segmentation masks ARE present** — training data includes `seg.nii.gz` with 3-region tumor labels.
2. **No WHO Grade labels** — BraTS-GLI uses binary tumor labels, not WHO Grade I-IV.
3. **No CT data** — BraTS-GLI is MRI-only. CT must come from IBSR/TCIA.
4. **Tumor volume range** — from small (<100 mm3) to very large (>30000 mm3), highly variable.
5. **All cases have 4 MRI modalities** — no missing modality across the dataset.
6. **Voxel spacing** — 1mm isotropic, total brain volume ~7.3 million mm3 per scan.

## 6. Files Generated

| File | Description |
|------|-------------|
| modality_stats.csv | Per-modality aggregate intensity statistics |
| case_metadata.csv | Per-case shapes, intensity stats, tumor volumes |
| tumor_volume_summary.csv | Tumor region volume aggregates |
| modality_histograms.png | Value distribution histograms per modality |
| tumor_volume_distribution.png | Whole tumor volume + fraction distributions |
| sample_slices_with_tumor.png | Axial slices with tumor overlay (highest-volume case) |
| summary_report.md | This report |

## 7. Next Steps (Phase 1: Preprocessing)

- [ ] Skull-stripping (ANTsHDGMM or BraTS-specific approach)
- [ ] Rigid registration across modalities (FLAIR as reference)
- [ ] CTN normalisation (CTN from BraTS challenge)
- [ ] Patch-based augmentation for small GPU memory
- [ ] Train/test split with patient-level stratification
- [ ] Multi-task network: segmentation + tumour presence classification
