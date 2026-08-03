# 🧠 Research Recipe — Longitudinal CT+MRI Brain Tumor Progression Monitoring

> **Project:** Major101 · **Branch:** `pattaswamy-vishwak-yasashree-cuddly-lamp`
> **Last updated:** 2026-08-03 · **File:** `research/RECIPE.md`

---

## How to read this file

This is your navigation map. It tells you **what file covers what**, **what's verified**, **what's missing**, and **what order to read things in**. Start at the top and work down.

---

## 🏗️ Phase 1: Foundations (read first)

| # | File | What it covers | Read when |
|---|------|----------------|-----------|
| 1 | `research_plan.md` | Full project scope, 19 categories, methodology, gaps | **Start here** |
| 2 | `executive_summary.md` | Condensed findings, key gaps, recommended approach | After plan |
| 3 | `cat1_brain_tumour_imaging_basics.md` | MRI/CT physics, sequences (T1/T1ce/T2/FLAIR), radiomics | Foundation |
| 4 | `cat2_tumour_progression_monitoring.md` | RANO/PRECIST criteria, progression definition | Foundation |
| 5 | `cat3_deep_learning_medical_imaging.md` | DL in medical imaging overview | Foundation |

---

## 🧬 Phase 2: Core Methods

| # | File | What it covers |
|---|------|----------------|
| 6 | `cat4_pet_mri_fusion.md` | PET-MRI fusion techniques, PET attenuation correction |
| 7 | `cat5_brain_tumor_segmentation.md` | Segmentation architectures, BraTS pipeline |
| 8 | `cat6_ai_architectures.md` | CNN, transformer, hybrid architectures for brain tumors |
| 9 | `cat7_registration_segmentation.md` | Image registration methods, cross-modal alignment |
| 10 | `cat8_ct_mri_multimodal_fusion.md` | CT+MRI fusion, modality registration, data harmonization |
| 11 | `cat9_multi_center_harmonization.md` | Site variability, domain adaptation, harmonization methods |
| 12 | `cat10_uncertainty_quantification.md` | Bayesian DL, MC dropout, ensemble methods |
| 13 | `cat11_explainable_ai.md` | Grad-CAM, SHAP, LIME, attention maps |
| 14 | `cat11_explainable_ai_clinical.md` | Clinical interpretability, trust calibration |
| 15 | `cat12_data_augmentation.md` | Geometric, intensity, GAN-based augmentation |
| 16 | `cat13_transfer_learning.md` | Pretrained weights, natural→medical transfer |
| 17 | `cat14_clinical_deployment.md` | FDA/CE regulatory, clinical workflow integration |
| 18 | `cat15_ethical_considerations.md` | Bias, fairness, informed consent |
| 19 | `cat15_ethical_considerations_bias.md` | Specific bias analysis in medical DL |

---

## 📊 Phase 3: Datasets (already done)

| File | Contents |
|------|----------|
| `00_research_plan.md` | Phase 1 plan with dataset workflow |
| `01_foundational_cnn_backbones.md` | U-Net, ResNet, DenseNet, ViT — all foundational |
| `02_medical_segmentation_cnn.md` | nnU-Net, V-Net, Attention U-Net, TransUNet |
| `03_transformer_hybrid.md` | Vision Transformers, MAE, UNETR |
| `04_multimodal_fusion.md` | Multimodal fusion architectures |
| `05_longitudinal_analysis.md` | Longitudinal DL methods |
| `06_final_report.md` | Synthesis of all Phase 1 findings |
| `07_efficient_data_loading.md` | Memory-efficient loading, caching |
| `09_radiomics_fusion.md` | Radiomics + DL fusion approaches |
| `10_self_supervised.md` | Self-supervised pretraining |
| `11_generative_augmentation.md` | GAN/Diffusion-based data augmentation |
| `12_explainability.md` | XAI methods for brain tumor DL |
| `13_uncertainty_calibration.md` | Calibration methods |
| `14_survival_analysis.md` | Survival prediction DL |
| `15_federated_learning.md` | Federated learning for multi-institutional data |
| `16a_dataset_inventory.md` | 14 public datasets inventoried ✅ |
| `16b_key_dataset_profiles.md` | BraTS, TCIA, DeepLesion profiles ✅ |
| `16d_longitudinal_datasets_deepdive.md` | Longitudinal dataset deep-dive |
| `16e_dataset_comparison_matrix.csv` | Dataset comparison matrix |
| `16f_preprocessing_pipelines.md` | Preprocessing standard pipelines |
| `16g_dataset_gap_analysis.md` | Gap analysis per dataset |
| `16h_access_licensing_ethics.md` | Access terms, licensing, ethics |

---

## 📋 Phase 4: Evaluation & Clinical (categories 17–18)

| File | What it covers |
|------|----------------|
| `cat17_evaluation_metrics_progression.md` | Dice, HD95, RANO concordance, progression metrics |
| `cat18_radiologist_comparison.md` | AI vs. radiologist agreement studies |

---

## 🔬 Phase 5: Prior Work & Experiments (category 19)

| File | What it covers |
|------|----------------|
| `cat19_direct_prior_work_base_papers.md` | 11 collected papers, 3 verified as 4/4 candidates |
| `cat19_multimodal_fusion_experimental.md` | Experimental multimodal fusion notes (⚠️ unverified claims) |

---

## 📚 Citation Files (BibTeX + .cite)

| File | Status | Safe to use? |
|------|--------|--------------|
| `cat10_references.bib` / `.cite` | ⚠️ **Not verified** | ⚠️ Regenerate |
| `cat11_references.bib` / `.cite` | ⚠️ **Not verified** | ⚠️ Regenerate |
| `cat12_references.bib` / `.cite` | ⚠️ **Not verified** | ⚠️ Regenerate |
| `cat15_references.bib` / `.cite` | ⚠️ **Not verified** | ⚠️ Regenerate |
| `cat17_references.bib` / `.cite` | ✅ **PubMed verified** | ✅ Safe |
| `cat18_references.bib` / `.cite` | ✅ **PubMed verified** | ✅ Safe |
| `cat19_references.bib` / `.cite` | ✅ **PubMed verified** | ✅ Safe |
| `bibtex/cat4_references.bib` / `.cite` | ❌ **Fabricated PMIDs** | ❌ Regenerate |
| `bibtex/cat5_references.bib` / `.cite` | ❌ **Fabricated PMIDs** | ❌ Regenerate |
| `bibtex/cat8_references.bib` / `.cite` | ❌ **Fabricated PMIDs** | ❌ Regenerate |
| `bibtex/verified_foundational_references.bib` | ✅ **OpenAlex verified** | ✅ Safe |

> **Full audit:** See `bibtex/citation_verification_audit.md`

---

## 🗂️ Quick File Map

```
research/
├── RECIPE.md                       ← You are here
├── research_plan.md                ← Full project plan
├── executive_summary.md            ← Condensed findings
│
├── cat1_...cat15_*.md              ← Categories 1–15 (basics through ethics)
├── cat17_*.md                      ← Category 17 (evaluation metrics)
├── cat18_*.md                      ← Category 18 (radiologist comparison)
├── cat19_*.md                      ← Category 19 (prior work / experiments)
│
├── 00_...16h_*.md                  ← Phase 1 completed datasets & methods
│
├── cat10/11/12/15_references.bib   ← Citations for those categories
├── cat17/18/19_references.bib      ← Verified citations
├── cat*_references.cite            ← .cite equivalents
│
└── bibtex/
    ├── cat4/5/8_references.bib     ← ❌ Fabricated PMIDs, do not use
    ├── verified_foundational_references.bib  ← ✅ OpenAlex verified
    └── citation_verification_audit.md        ← Full audit report
```

---

## 🧪 What's Missing (Gap Tracker)

| Gap | Severity | Where it's documented |
|-----|:--------:|----------------------|
| No public longitudinal CT+MRI brain tumor dataset | 🔴 Critical | `executive_summary.md`, `cat19_*.md` |
| No RANO/PRECIST progression annotations in public data | 🔴 Critical | `executive_summary.md` |
| TCIA CT+MRI overlap undocumented | 🔴 Critical | `executive_summary.md` |
| cat10/11/12/15 BibTeX files have unverified PMIDs | 🟡 Important | `citation_verification_audit.md` |
| cat4/5/8 bibtex PMIDs are fabricated | 🟡 Important | `citation_verification_audit.md` |
| Phase 16c (comparison matrix) pending | 🟡 Important | `research_plan.md` |
| cat19 experimental notes (unverified claims) | 🟡 Important | `cat19_multimodal_fusion_experimental.md` |

---

## 📖 Reading Order (Recommended)

1. **`research_plan.md`** — understand scope
2. **`executive_summary.md`** — get the tl;dr
3. **`cat1_brain_tumour_imaging_basics.md`** — MRI/CT foundations
4. **`cat2_tumour_progression_monitoring.md`** — what you're measuring
5. **`01_foundational_cnn_backbones.md`** — model foundations
6. **`02_medical_segmentation_cnn.md`** — segmentation architectures
7. **`cat16a/16b_dataset_*.md`** — what data you can actually use
8. **`cat17_evaluation_metrics_progression.md`** — how to measure success
9. **`cat18_radiologist_comparison.md`** — AI vs. clinical standard
10. **`cat19_direct_prior_work_base_papers.md`** — what's already been done

---

## 🔑 Key Takeaway

> **No public dataset or prior work combines CT+MRI+longitudinal+DL for brain tumor progression.** Every verified paper uses MRI only. This project's scope is genuinely novel — but it means the data problem must be solved before the model problem.
