# Category 9: Multi-Center Dataset Harmonisation

**Status: ✅ RESEARCH COMPLETE**

---

## 9.1 The Problem

Data from different hospitals/scanners has systematic differences:

| Source | Scanner | Field strength | Resolution |
|---|---|---|---|
| BraTS | Varied (Siemens, GE, Philips) | 1.5T / 3.0T | 1mm³ |
| IBSR | Varied | 1.5T / 3.0T | ~1mm³ |
| Hospital A | Siemens 3T | 3.0T | 0.5mm³ |
| Hospital B | GE 1.5T | 1.5T | 1mm³ |

**Result:** Model trained on BraTS fails on Hospital A data — same tumour, different intensity distribution.

---

## 9.2 ComBat Harmonisation

```
ComBat (Empirical Bayes):
1. Estimate batch effect (scanner/site differences)
2. Estimate biological effect (tumour vs normal)
3. Remove batch effect while preserving biological effect
4. Result: harmonised intensity distributions across sites

Used in: ICA-Bench, multiple cross-institutional BraTS studies.
```

**VRAM:** CPU-only preprocessing. No impact on training memory.

---

## 9.3 Our Approach

| Step | Tool | Notes |
|---|---|---|
| **Intensity normalisation** | CTN (MRI), intracranial mapping (CT) | Per-case, removes scanner bias |
| **Rigid registration** | ANTs | Aligns CT to MRI space |
| **ComBat** | `pyHarmony` package | Cross-institutional (if needed) |
| **Standardised space** | MNI152 | All cases to common template |

**For IBSR alone (single institution):** CTN + registration is sufficient. ComBat added if multi-site data arrives.

---

## 9.4 Cross-Dataset Format Mapping

| Dataset | Volume size | Modality | NIfTI path |
|---|---|---|---|
| IBSR | 256³ | MRI (T1), CT | `{case}/{mod}.nii.gz` |
| C-BRATS | 240³ | MRI (T1, T1ce, T2, FLAIR) | `{case}/{case}Modality.nii.gz` |
| BraTS 2024 | 240³ | MRI (same as C-BRATS) | `{case}/{case}Modality.nii.gz` |

**Our pipeline normalises all to 256³ after cropping, then pre-caches to .pt.**

See `research/16g_dataset_gap_analysis.md` for the full cross-dataset mapping.
