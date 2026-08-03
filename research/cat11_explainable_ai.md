# Category 11: Explainable AI (XAI) for Clinical Adoption

> Methods to make brain tumour DL models interpretable for clinical decision-making, 2018–2026.

## Why It Matters

A black-box prediction is useless in the clinic. Radiologists and neurosurgeons must see *why* a model predicts Grade IV — which regions, which modalities. XAI methods like Grad-CAM, SHAP, and Relevance-weighted CAM (RCAM) provide visual explanations aligned with clinical reasoning.

## Real Papers

### 1. PMID 42160250 — MADEX Explainable Multimodal DL (2026)
**Title:** Explainable Deep Learning Framework for Multimodal Brain Tumor Classification via Neuroimaging Attribute Extraction
**Authors:** Bao L, Khan S, Wang Y, Lv Y, Li X
**Source:** IEEE J Biomed Health Inform, 2026
**DOI:** 10.1109/JBHI.2026.3695364
**Methods:** MADEX — convolutional encoders extract interpretable neuroimaging attributes; hierarchical decision trees aligned with WHO glioma structure; Dempster-Shafer evidential fusion; prototype constraints for clinical consistency; adaptive modality exchange for corrupted sequences
**Dataset:** BT-large-2c and BRaTS 2021
**Metrics:** 89.7% accuracy on BT-large-2c; 87.3% on BRaTS 2021; 10 features capture >90% of full model performance; insertion-deletion tests confirm explanations reflect genuine diagnostic features
**XAI contribution:** Sparse attribute representations — just 10 features capture most of model performance. Explanations are human-interpretable, not spurious correlations.

### 2. PMID 39686848 — M-SCA ResNet with Grad-CAM (2025)
**Title:** An interpretable multi-scale convolutional attention residual neural network for glioma grading with Raman spectroscopy
**Authors:** Li Q, Shao X, Zhou Y, et al.
**Source:** Anal Methods, 2025
**DOI:** 10.1039/d4ay02068e
**Methods:** M-SCA ResNet (multi-scale channel + spatial attention + residual structures); Grad-CAM for interpretability
**Dataset:** HGG, LGG, and normal tissue Raman spectra
**Metrics:** Identification accuracy >85% for all 3 tissue types; highest weighted F1-score among compared methods
**XAI contribution:** Grad-CAM extracts key Raman shifts that contribute to classification; extracted shifts correspond to biomolecular characteristic peaks of brain tissue — biologically validated explanations.

### 3. PMID 37052658 — RCAM for Intelligent Meningioma Grading (2023)
**Title:** Intelligent noninvasive meningioma grading with a fully automatic segmentation using interpretable multiparametric deep learning
**Authors:** Jun Y, Park YW, Shin H, et al.
**Source:** Eur Radiol, 2023
**DOI:** 10.1007/s00330-023-09590-4
**Methods:** Two-stage DL (3D U-Net segmentation + ResNet classification); Relevance-weighted Class Activation Mapping (RCAM) for interpretability; multiparametric T2 + T1C input
**Dataset:** 257 patients train (162 LGG, 95 HGG); 61 patients external test (46 LGG, 15 HGG)
**Metrics:** Dice 0.910 (segmentation); AUC 0.770 (grading); accuracy 72.1%, sensitivity 73.3%, specificity 71.7%
**XAI contribution:** RCAM visualisation activates at tumour surface regions — model recognised features at the tumour margin for grading. Human radiologists performed worse (AUC 0.675–0.690).

## XAI Methods Comparison

| Method | What It Shows | Computational Cost | Clinical Value |
|---|---|---|---|
| **Grad-CAM** | Heatmap of important 3D regions | Negligible (1 backward pass) | High — intuitive visual |
| **RCAM** | Relevance-weighted attention map | Slightly higher than Grad-CAM | High — margin-aware |
| **SHAP** | Per-feature contribution values | High (ensemble of models) | High — quantitative |
| **Prototype constraints** | Aligns features with diagnostic criteria | Built into training | Very high — clinically anchored |

## Recommendation for Our Model

1. **Grad-CAM on per-modality encoders:** Visualise which brain regions drive each modality's feature vector — radiologists understand activation heatmaps
2. **RCAM on fusion head:** Identify which modality (CT vs MRI) contributes most to the final grade prediction
3. **F1-based feature sparsity:** Follow MADEX approach — enforce that few features carry the prediction, improving interpretability
4. **Report XAI alongside metrics:** Every paper submission should include Grad-CAM visualisations
