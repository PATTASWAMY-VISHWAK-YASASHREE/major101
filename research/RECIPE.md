# 🧠 Research Recipe — CT+MRI Multimodal Brain Tumour Fusion

> **Project:** Major101 · **Global Goal:** CT+MRI fusion for brain tumour detection, grading, and survival prediction
> **Last updated:** 2026-08-03

---

## Global Goal

Build a longitudinal volumetric DL pipeline that fuses MRI and CT brain scans for
tumour classification, grading, and survival prediction — addressing the gap
where no mature CT+MRI fusion benchmark exists for brain tumours (BraTS is
MRI-only).

## Research Status Summary

| Area | Status | Papers | Directly useful? |
|------|--------|--------|------------------|
| MRI segmentation | ✅ Mature | 5 | Indirect (segmentation for feature extraction) |
| MRI classification | ✅ Mature | 10 | Indirect (baseline to beat) |
| CT+MRI fusion | ⚠️ Experimental | 8 | **YES — core focus** |
| MRI-to-CT translation | 🔬 Prototype | 1 | **YES — fallback when CT missing** |
| PET+MRI fusion | ✅ Mature | 6 | No — different modalities, different physics |
| Federated learning | ✅ Mature | 5 | Peripheral — only if multi-site data |
| Self-supervised pretrain | ✅ Mature | 7 | Yes — pretrain on MRI volumes |
| Explainability | ✅ Mature | 7 | Peripheral — needed for clinical deployment |
| Uncertainty | ✅ Mature | 5 | Peripheral — calibration for clinical use |
| Survival analysis | ⚠️ MRI only | 5 | **YES — survival from CT+MRI is unexplored** |
| Data augmentation | ✅ Mature | 7 | Yes — for scarce CT+MRI data |
| Transfer learning | ✅ Mature | 7 | Yes — pretrain on MRI-only data |

> **Skip:** PET+MRI fusion, ethics/legal docs — not relevant to CT+MRI goal.
> **Use selectively:** Explainability, uncertainty — needed when deploying but not core.

---

## Phase 1: Foundations (read first)

| # | File | Read when | Keep? |
|---|------|-----------|-------|
| 1 | `research_plan.md` | Start here | ✅ |
| 2 | `executive_summary.md` | After plan | ✅ |
| 3 | `cat1_brain_tumour_imaging_basics.md` | MRI/CT physics | ✅ |
| 4 | `cat2_tumour_progression_monitoring.md` | RANO, progression, pseudoprogression | ✅ **CORE** |
| 5 | `cat3_deep_learning_medical_imaging.md` | DL overview | ✅ |

## Phase 2: Core Methods (CT+MRI focused)

| # | File | What it covers | Keep? |
|---|------|----------------|-------|
| 6 | `cat4_pet_mri_fusion.md` | PET-MRI fusion techniques | ❌ Skip (wrong modality) |
| 7 | `cat5_brain_tumor_segmentation.md` | Segmentation (nnU-Net, TransUNet) | ✅ Use (MRI segmentation pipeline) |
| 8 | `cat6_ai_architectures.md` | CNN, ViT, TransUNet, hybrid | ✅ Use (fusion backbone candidates) |
| 9 | `cat7_registration_segmentation.md` | Cross-modal registration | ✅ Use (CT↔MRI alignment) |
| 10 | `cat8_ct_mri_multimodal_fusion.md` | **CT+MRI fusion — core** | ✅ **CORE** |
| 11 | `cat9_multi_center_harmonization.md` | Site variability, domain adaptation | ✅ Use (CT scanner variability) |
| 12 | `cat10_uncertainty_quantification.md` | Bayesian DL, MC dropout | ✅ Use (calibration) |
| 13 | `cat11_explainable_ai.md` | Grad-CAM, SHAP | ✅ Use (interpretability) |
| 14 | `cat12_data_augmentation.md` | Geometric, intensity, GAN | ✅ Use (scarce data) |
| 15 | `cat13_transfer_learning.md` | Pretrained weights | ✅ Use (MRI→CT transfer) |
| 16 | `05_longitudinal_analysis.md` | Temporal Transformer, CNN-LSTM | ✅ **CORE** (longitudinal tracking) |
| 17 | `14_survival_analysis.md` | DeepSurv, DeepHit, Cox NN | ✅ **CORE** (survival from CT+MRI) |
| 18 | `cat14_clinical_deployment.md` | Regulatory, clinical workflow | ✅ Use (AI-Radiologist integration) |

## Phase 3: Datasets & Data

| File | Contents | Keep? |
|------|----------|-------|
| `cat16a_dataset_inventory.md` | 14 public datasets | ✅ Use (find CT+MRI overlap) |
| `cat16b_key_dataset_profiles.md` | BraTS, TCIA, DeepLesion | ✅ Use |
| `cat16g_dataset_gap_analysis.md` | Gap per dataset | ✅ Use |

> **Critical gap:** No dataset provides paired CT+MRI brain tumour data.
> IBSR has CT+MRI but no tumours. BraTS has tumours but no CT.

## Phase 4: Evaluation & Clinical

| File | What it covers | Keep? |
|------|----------------|-------|
| `cat17_evaluation_metrics_progression.md` | Dice, HD95, RANO | ✅ Use |
| `cat18_radiologist_comparison.md` | AI vs radiologist | ✅ Use (clinical validation) |

## Phase 5: Experiments (CT+MRI focused)

| File | Contents | Keep? |
|------|----------|-------|
| `cat19_multimodal_fusion_experimental.md` | CT+MRI fusion experiments | ✅ **CORE** |
| `cat19_direct_prior_work_base_papers.md` | Prior work candidates | ✅ Use |

> ⚠️ cat19 experimental notes contain unverified claims — treat as experiments,
> not verified science. Benchmark before claiming results.

---

## BibTeX / Citation Files — Use Status

| File | Status | Safe to use? |
|------|--------|--------------|
| `bibtex/cat8_references.bib` / `.cite` | CT+MRI fusion | ✅ PubMed verified |
| `bibtex/cat19_references.bib` / `.cite` | Experimental fusion | ✅ PubMed verified |
| `bibtex/cat5_references.bib` / `.cite` | Segmentation | ❌ Fabricated PMIDs — regenerate |
| `bibtex/cat4_references.bib` / `.cite` | PET+MRI | ❌ Fabricated PMIDs — skip |
| `bibtex/verified_foundational_references.bib` | Foundational papers | ✅ OpenAlex verified |

> **Skip citation files for:** PET+MRI (cat4), cat10/11/12/15 (unverified).
> **Safe to cite:** cat5 (regenerate), cat8, cat19, foundational.

---

## 🗺️ File Map (CT+MRI focused)

```
research/
├── RECIPE.md                       ← You are here
├── CROSSREF.md                     ← Cross-reference index
├── research_plan.md                ← Full project plan
├── executive_summary.md            ← Condensed findings
├── IMPLEMENTATION_SPEC.md          ← Theory-to-code bridge
│
├── 🟢 DIRECTLY RELEVANT
│   ├── cat8_ct_mri_multimodal_fusion.md        ← CORE: fusion methods
│   ├── cat9_multi_center_harmonization.md       ← CT scanner variability
│   ├── cat19_multimodal_fusion_experimental.md  ← Experimental CT+MRI
│   ├── cat19_direct_prior_work_base_papers.md   ← Prior work candidates
│   ├── cat5_brain_tumor_segmentation.md         ← MRI segmentation
│   ├── cat6_ai_architectures.md                 ← Fusion backbones
│   ├── cat7_registration_segmentation.md        ← Cross-modal registration
│   ├── cat13_transfer_learning.md               ← Pretrain on MRI
│   ├── cat12_data_augmentation.md               ← Scarce data augmentation
│   ├── cat10_uncertainty_quantification.md      ← Calibration
│   ├── cat11_explainable_ai.md                  ← Interpretability
│   ├── cat17_evaluation_metrics_progression.md  ← Evaluation
│   ├── cat18_radiologist_comparison.md          ← Clinical validation
│   ├── cat2_tumour_progression_monitoring.md    ← RANO, progression, AI assistant
│   ├── 05_longitudinal_analysis.md              ← Temporal Transformer, CNN-LSTM
│   ├── 14_survival_analysis.md                  ← DeepSurv, DeepHit, Cox NN
│   ├── cat14_clinical_deployment.md             ← AI-Radiologist workflow
│   └── 16d_longitudinal_datasets_deepdive.md    ← Longitudinal data sources
│
├── 🟡 INDIRECTLY RELEVANT
│   ├── cat1_brain_tumour_imaging_basics.md      ← MRI/CT physics
│   ├── cat3_deep_learning_medical_imaging.md    ← DL overview
│   └── cat15_ethical_considerations.md          ← Ethics (when deploying)
│
├── 🔴 SKIP (not relevant to CT+MRI goal)
│   ├── cat4_pet_mri_fusion.md                   ← PET, wrong modality
│   └── cat15_ethical_considerations_bias.md     ← Ethics sub-note
│
├── Phase 1 datasets (00_–16h_*.md)
└── bibtex/                                      ← Citations (check status above)
```

---

## 🔑 The Gap (Research Problem Statement)

> **No public dataset or prior work combines CT+MRI+longitudinal+DL for brain
> tumour classification, progression tracking, or survival prediction.** Every
> verified paper uses MRI only. The 8 CT+MRI fusion papers in our bibliography
> either address different pathologies (MICE review), use synthetic or limited
> data, or are early experimental prototypes (Gong 2025, Chen 2026, Islam 2026).
>
> **This project's three research contributions:**
> 1. First systematic DL pipeline for CT+MRI brain tumour fusion
> 2. AI-assisted RANO progression monitoring (automated, not manual)
> 3. Survival prediction from fused CT+MRI data (unexplored)
> — all validated against MRI-only baselines.