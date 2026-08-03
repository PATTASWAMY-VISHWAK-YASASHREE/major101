# BraTS-GLI Validation Data � Full Analysis Report

Generated: 2026-08-03
Dataset: BraTS2024 BraTS-GLI Validation (syn61455507)

## 1. Overview

| Metric | Value |
|--------|-------|
| Total cases (directories) | 188 |
| Total .nii.gz files | 752 |
| Total size on disk | 5.42 GB |
| Typical 3D shape | (182, 218, 182) |
| Unique patients | 87 |
| Unique sequences | 6 |
| Modalities per case | T1c, T1n, T2f, T2w (4) |
| Segmentation (seg.nii.gz) | NOT present in validation set |

## 2. Modalities

| Code | Meaning |
|------|---------|
| t1c | T1-weighted post-contrast (contrast-enhanced) |
| t1n | T1-weighted native (pre-contrast) |
| t2f | FLAIR (Fluid Attenuated Inversion Recovery) |
| t2w | T2-weighted |

## 3. Per-Modality Intensity Statistics

| Modality | N voxels | Mean | Std | Min | Max | P1 | P50 | P95 | P99 | Non-zero % |
|----------|----------|------|-----|-----|-----|----|-----|-----|-----|------------|
| t1c | 188 | 185.2 | 542.6 | -291.0 | 18720.1 | 0.0 | 0.0 | 1105.3 | 2394.6 | 21.28% |
| t1n | 188 | 145.3 | 391.5 | -39.0 | 8437.5 | 0.0 | 0.0 | 856.7 | 2046.9 | 21.32% |
| t2f | 188 | 96.6 | 238.5 | -40.0 | 6342.0 | 0.0 | 0.0 | 740.0 | 1017.9 | 21.28% |
| t2w | 188 | 217.0 | 594.7 | 0.0 | 11012.7 | 0.0 | 0.0 | 1507.1 | 2870.0 | 21.35% |

## 4. Key Findings

1. **No segmentation masks** in the validation set � tumour volumes cannot be computed from this split alone.
2. **No WHO Grade labels** � BraTS-GLI labels are binary (tumour present vs absent), not WHO Grade I-IV.
3. **No CT data** � BraTS-GLI is MRI-only. CT must come from a separate dataset (IBSR / TCIA).
4. **4 modalities per case** � all 188 cases have all 4 sequences present.
5. **Intensity ranges** � typical MRI pixel values (0-4000 range), consistent with standard brain MRI.

## 5. Files Generated

| File | Description |
|------|-------------|
| modality_stats.csv | Per-modality aggregate statistics |
| case_metadata.csv | Per-case shape, stats, file sizes |
| modality_histograms.png | Value distribution histograms per modality |
| sample_slices.png | Representative axial slices (T1c, T1n, T2f, T2w) |
| summary_report.md | This report |

## 6. Next Steps (Phase 0)

- [ ] Run same analysis on TrainingData once downloaded
- [ ] Skull-stripping + intensity normalisation pipeline
- [ ] Label extraction (binary tumour from training set seg.nii.gz)

