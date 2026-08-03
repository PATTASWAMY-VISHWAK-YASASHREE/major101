# 📋 Project Recipe — Major101 (Brain Tumor DL)

> **Project:** Major101
> **Branch:** `pattaswamy-vishwak-yasashree-cuddly-lamp`
> **Base branch:** `main`
> **Last updated:** 2026-08-03

---

## What This Project Is

Deep learning pipeline for brain tumor classification using CT+MRI fusion.
All research has been completed — implementation has **not started**.

---

## 📁 Full Project Structure

```
major101/
├── .github/
│   └── WORKSPACE.md              # Workspace context & conventions
│
├── .gitignore
├── README.md                     # Project overview
├── CITATION.md                   # Citation format guide
├── CITATION_GUIDE.md             # Quick citation reference
├── requirements.txt              # Python dependencies
├── train.py                      # Training entry point (placeholder)
│
├── src/                          # Source code (not yet implemented)
│   ├── __init__.py
│   ├── data/                     # Data loading
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   ├── resnet3d.py
│   │   └── fusion.py
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── metrics.py
│       └── logger.py
│
├── configs/
│   ├── config.yaml
│   ├── train.yaml
│   ├── eval.yaml
│   └── dataset.yaml
│
└── research/                     # ⭐ All research lives here
    ├── RECIPE.md                 # ← Research file map (detailed)
    ├── research_plan.md          # Full project plan & methodology
    ├── executive_summary.md      # Condensed findings & gaps
    │
    ├── cat1_brain_tumour_imaging_basics.md
    ├── cat2_tumour_progression_monitoring.md
    ├── cat3_deep_learning_medical_imaging.md
    ├── cat4_pet_mri_fusion.md
    ├── cat5_brain_tumor_segmentation.md
    ├── cat6_ai_architectures.md
    ├── cat7_registration_segmentation.md
    ├── cat8_ct_mri_multimodal_fusion.md
    ├── cat9_multi_center_harmonization.md
    ├── cat10_uncertainty_quantification.md
    ├── cat11_explainable_ai.md
    ├── cat11_explainable_ai_clinical.md
    ├── cat12_data_augmentation.md
    ├── cat13_transfer_learning.md
    ├── cat14_clinical_deployment.md
    ├── cat15_ethical_considerations.md
    ├── cat15_ethical_considerations_bias.md
    ├── cat17_evaluation_metrics_progression.md
    ├── cat18_radiologist_comparison.md
    ├── cat19_direct_prior_work_base_papers.md
    ├── cat19_multimodal_fusion_experimental.md
    │
    ├── 00_research_plan.md
    ├── 01_foundational_cnn_backbones.md
    ├── 02_medical_segmentation_cnn.md
    ├── 03_transformer_hybrid.md
    ├── 04_multimodal_fusion.md
    ├── 05_longitudinal_analysis.md
    ├── 06_final_report.md
    ├── 07_efficient_data_loading.md
    ├── 09_radiomics_fusion.md
    ├── 10_self_supervised.md
    ├── 11_generative_augmentation.md
    ├── 12_explainability.md
    ├── 13_uncertainty_calibration.md
    ├── 14_survival_analysis.md
    ├── 15_federated_learning.md
    ├── 16a_dataset_inventory.md
    ├── 16b_key_dataset_profiles.md
    ├── 16d_longitudinal_datasets_deepdive.md
    ├── 16e_dataset_comparison_matrix.csv
    ├── 16f_preprocessing_pipelines.md
    ├── 16g_dataset_gap_analysis.md
    ├── 16h_access_licensing_ethics.md
    │
    ├── cat10/11/12/15_references.bib  ← ⚠️ Unverified PMIDs
    ├── cat17/18/19_references.bib     ← ✅ Verified citations
    ├── bibtex/cat4/5/8_references.bib ← ❌ Fabricated PMIDs
    ├── bibtex/verified_foundational_references.bib  ← ✅ Safe
    └── bibtex/citation_verification_audit.md        ← Full audit
```

---

## 📖 What's In Each Section

### `src/` — Code (not implemented yet)
| File | Purpose | Status |
|------|---------|--------|
| `src/data/dataset.py` | NIfTI loading, caching, transforms | ❌ Not implemented |
| `src/data/transforms.py` | Preprocessing (skull-strip, CTN, normalize) | ❌ Not implemented |
| `src/models/resnet3d.py` | 3D ResNet-18 backbone | ❌ Not implemented |
| `src/models/fusion.py` | Late-fusion head (dense concat) | ❌ Not implemented |
| `src/utils/metrics.py` | Dice, HD95, macro-F1 | ❌ Not implemented |
| `src/utils/logger.py` | TensorBoard logging | ❌ Not implemented |
| `train.py` | Training loop entry point | ❌ Not implemented |

### `configs/` — Configuration (placeholder YAML files)
| File | Purpose |
|------|---------|
| `config.yaml` | Base config |
| `train.yaml` | Training hyperparameters |
| `eval.yaml` | Evaluation settings |
| `dataset.yaml` | Dataset paths and parameters |

### `research/` — All research output
| Section | Files | Coverage |
|---------|-------|----------|
| **Categories 1–15** | `cat1–cat15_*.md` | MRI/CT basics → ethics |
| **Categories 17–19** | `cat17–cat19_*.md` | Evaluation, radiologist comparison, prior work |
| **Phase 1 complete** | `00–16h_*.md` | Datasets, architectures, preprocessing |
| **Citation files** | `*_references.bib/.cite` | BibTeX + .cite for each category |
| **Citation audit** | `bibtex/citation_verification_audit.md` | Full verification report |

---

## 🔑 Key Decisions (from research)

| Decision | Choice |
|----------|--------|
| **Fusion strategy** | Late fusion — per-modality 3D CNN encoders + dense head |
| **Architecture** | 3D ResNet-18 per modality (CT, T1, T1ce, T2, FLAIR) |
| **Classification** | 4-class (WHO Grades I–IV) |
| **Primary dataset** | IBSR (80 paired MRI+CT cases, 10.5 GB) |
| **Secondary dataset** | C-BRATS (600 MRI-only cases, 2 GB) |
| **Preprocessing** | Skull-strip (ANTsHDGMM) + CTN (MRI) + intracranial mapping (CT) |
| **Hardware constraints** | RTX 2050 4GB → 64³ patches, FP16, batch=2, grad-acc=4 |

---

## ⚠️ Known Issues

1. **cat10/11/12/15 BibTeX PMIDs unverified** — need regeneration from subagent outputs
2. **bibtex/cat4/5/8 PMIDs fabricated** — delete and regenerate
3. **Phase 16c comparison matrix pending** — in `research_plan.md`
4. **cat19 experimental notes** — unverified claims, marked as experimental

---

## 🚀 Next Steps (when ready to implement)

1. Read `research/RECIPE.md` for the research file map
2. Download IBSR + C-BRATS datasets
3. Implement preprocessing pipeline (`src/data/transforms.py`)
4. Implement dataset loader (`src/data/dataset.py`)
5. Build 3D ResNet-18 (`src/models/resnet3d.py`)
6. Build fusion head (`src/models/fusion.py`)
7. Write training loop (`train.py`)
8. Evaluate: MRI-only vs CT+MRI baseline
