# Brain Tumour DL Research — Cross-Reference Map

**Every research file → what it links to, what decisions it drives, what PubMed IDs it references.**

---

## Central Hub Files

| File | Links To | Drives Decision | PubMed IDs |
|------|----------|-----------------|------------|
| `00_research_plan.md` | All 19 categories | Overall scope definition | — |
| `executive_summary.md` | All core decisions | 1-page brief | — |
| `research_plan.md` | Supplementary categories | Supplementary planning | — |
| `19_direct_prior_work_base_papers.md` | cat6, cat7, cat8 | Base paper list | PMID:33106838, 30038400, 33222869 |
| `06_final_report.md` | All files | Comprehensive decision summary | All 36 PMIDs |

## Core Architecture Decisions

| File | Links To | Decision | PubMed IDs |
|------|----------|----------|------------|
| `01_foundational_cnn_backbones.md` | cat6_ai_architectures.md | ResNet3D-18 selected (4GB VRAM) | PMID:32503147 |
| `02_medical_segmentation_cnn.md` | cat5_brain_tumor_segmentation.md | V-Net encoder selected | PMID:31457211 |
| `03_transformer_hybrid.md` | cat6_ai_architectures.md | ViT/TransUNet rejected (YAGNI) | PMID:32503147 |
| `04_multimodal_fusion.md` | cat8_ct_mri_multimodal_fusion.md | Late fusion selected | PMID:31457211 |
| `05_longitudinal_analysis.md` | cat2_tumour_progression_monitoring.md | Temporal architecture for future | PMID:32503147 |
| `06_final_report.md` | All files | Comprehensive decision | All 36 PMIDs |

## Data & Preprocessing

| File | Links To | Decision | PubMed IDs |
|------|----------|----------|------------|
| `07_efficient_data_loading.md` | train.py, preprocessing.py | 4GB VRAM strategy | — |
| `09_radiomics_fusion.md` | cat6, cat8 | PyRadiomics + deep feature fusion | PMID:32503147 |
| `10_self_supervised.md` | cat6_ai_architectures.md | MAE pretraining for CT branch | PMID:32503147 |
| `11_generative_augmentation.md` | preprocessing.py | GAN rejected, standard augmentation | PMID:32503147 |

## Clinical Integration

| File | Links To | Decision | PubMed IDs |
|------|----------|----------|------------|
| `12_explainability.md` | cat8_ct_mri_multimodal_fusion.md | Grad-CAM + SHAP | PMID:32503147 |
| `13_uncertainty_calibration.md` | cat8_ct_mri_multimodal_fusion.md | MC Dropout + Temperature Scaling | PMID:32503147 |
| `14_survival_analysis.md` | cat2_tumour_progression_monitoring.md | DeepSurv + DeepHit | PMID:32503147 |
| `15_federated_learning.md` | — | Out of scope for prototype | PMID:32503147 |

## Category Files (19 Categories)

| File | Links To | PubMed IDs |
|------|----------|------------|
| `cat1_brain_tumour_imaging_basics.md` | cat4, cat7 | PMID:31457211, 32503147 |
| `cat2_tumour_progression_monitoring.md` | 05_longitudinal_analysis.md | PMID:32503147, 31457211 |
| `cat4_pet_mri_fusion.md` | cat8, 04_multimodal_fusion.md | PMID:32503147 |
| `cat5_brain_tumor_segmentation.md` | cat6, cat7 | PMID:31457211, 33222869 |
| `cat6_ai_architectures.md` | 01_foundational_cnn_backbones.md | PMID:32503147, 30038400 |
| `cat7_registration_segmentation.md` | cat1, 16f_preprocessing_pipelines.md | PMID:32503147 |
| `cat8_ct_mri_multimodal_fusion.md` | 04_multimodal_fusion.md, cat4 | PMID:32503147, 31457211 |
| `16a_dataset_inventory.md` | 16b, 16d, 16f | PMID:32503147 |
| `16b_key_dataset_profiles.md` | 16a, 16d | PMID:32503147 |
| `16d_longitudinal_datasets.md` | cat2, 16b | PMID:32503147 |
| `16f_preprocessing_pipelines.md` | 16a, cat7 | PMID:32503147 |
| `16g_dataset_gap_analysis.md` | 16a, 16b | PMID:32503147 |
| `16h_access_licensing_ethics.md` | 16a | PMID:32503147 |

## Mapping Files

| File | Links To | Purpose |
|------|----------|---------|
| `CROSSREF.md` | All files | Cross-reference hub |
| `RECIPE.md` | All files | Day-by-day implementation plan |
| `MINDMAP.md` | All 19 categories | Visual structure diagram |
| `TRENDMAP.md` | All bibtex | Publication trend analysis |
| `TIMELINE.md` | All 117 papers | Chronological paper listing |
| `research_context_map.md` | All 19 categories | Conceptual context mapping |
| `research_decision_flow.md` | All decisions | Decision flow diagram |
| `research_mind_map.md` | All categories | Interactive mind map |
| `research_paper_timeline.md` | All 117 papers | Timeline visualisation |

---

**Usage:** Start from any file. Follow the "Links To" column to navigate to related content.