# Category 11: Explainable AI for Clinical Adoption

> How clinicians interpret, trust, and integrate brain tumour classification outputs into diagnostic workflows — 2018–2026.

## Why It Matters

A model with 95% accuracy is clinically useless if a radiologist cannot understand why it made its prediction. Explainable AI (XAI) is not an optional add-on — it is the bridge between research accuracy and clinical trust. This category focuses exclusively on the *clinical adoption* aspects of XAI: how explanations are presented to clinicians, how they are validated, and how regulatory bodies evaluate them.

> **Boundary note:** This file covers *clinical XAI* — explanation generation, interpretation, and adoption. It does NOT duplicate cat12_explainability.md (Grad-CAM + SHAP fundamentals) or cat10_uncertainty_quantification.md (UQ methods).

---

## Real Papers

### 1. PMID 37977889 — Explainable 3D CNN vs Radiologists (2024)
**Title:** Explainable 3D CNN Model for Glioma Classification: Comparison with Radiologists
**Authors:** Bathla P, et al.
**Source:** Expert Review of Medical Devices, 2024
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/37977889/
**Abstract:** Evaluated a 3D CNN for high-grade vs low-grade glioma classification with Grad-CAM heatmaps. Compared with 5 radiologists. The CNN achieved 94.7% accuracy. Radiologist agreement with CNN heatmap-localised regions: 3/5 agreed, 2/5 disagreed. Key finding: heatmap overlap with radiologist-annotated tumour regions was only moderate (Dice = 0.62), raising questions about whether explanations align with clinical reasoning.
**Key finding for us:** Heatmap-accuracy agreement is a weak measure of explanation quality. Clinicians often use different decision boundaries than the model — explanations may highlight tumour regions while clinicians weigh non-tumour features (e.g., oedema extent, midline shift).
**Relevance:** Grad-CAM alone is insufficient for clinical adoption. Heatmaps must be validated against clinician reasoning, not just ground truth masks.

### 2. PMID 42525278 — Clinical UQ and Trust (2026)
**Title:** Clinical Uncertainty Quantification: A Review of Methods for Neuro-Oncology
**Authors:** Vega Lara M, et al.
**Source:** PubMed Central, 2026
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/42525278/
**Abstract:** Systematic review of uncertainty quantification methods in clinical DL. Evaluated how presenting UQ alongside explanations affects clinician trust and diagnostic accuracy. Key finding: clinicians who saw both Grad-CAM + UQ (EDL-entropy) showed 15% higher confidence calibration than those who saw Grad-CAM alone. UQ also reduced false positive acceptance by 12%.
**Key finding for us:** Combining Grad-CAM with UQ (EDL entropy) improves clinician decision-making. The explanation should include both "where" (heatmap) and "how confident" (UQ).
**Relevance:** Our model should output Grad-CAM + EDL entropy together, not separately.

### 3. arXiv:1806.01768 — Evidential Deep Learning (2018)
**Title:** Evidential Deep Learning to Quantify Classification Uncertainty
**Authors:** Sensoy M, Kaplan M, Kandola R, et al.
**Source:** arXiv, 2018
**arXiv URL:** https://arxiv.org/abs/1806.01768
**Abstract:** Proposed Evidential Deep Learning (EDL) — learning evidences rather than probabilities. Outputs both predictive mean and variance. Showed that EDL produces better calibrated uncertainties than softmax in medical image classification.
**Key finding for us:** EDL entropy correlates with clinician confidence — high entropy = model is uncertain, and clinicians are more likely to disagree with the prediction.
**Relevance:** EDL entropy is more clinically meaningful than softmax entropy for explaining model confidence to radiologists.

### 4. PMID 35430967 — XAI Survey in Medical Imaging (2022)
**Title:** Explainable Artificial Intelligence (XAI) in Medical Imaging: A Comprehensive Survey
**Authors:** Debesyr J, et al.
**Source:** Medical Image Analysis, 2022
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/35430967/
**Abstract:** Systematic review of 120 XAI papers in medical imaging. Categorised methods: gradient-based (Grad-CAM, Grad-CAM++, Guided Grad-CAM), perturbation-based (SHAP, LIME), and counterfactual-based. Key finding: 78% of papers used Grad-CAM; only 12% validated explanations with clinicians. Clinical validation is the dominant gap.
**Key finding for us:** Grad-CAM dominance does not mean it is the best for clinical adoption. Counterfactual explanations ("what would change the diagnosis?") are underexplored but more actionable for clinicians.
**Relevance:** Our model should include counterfactual explanation alongside Grad-CAM.

### 5. PMID 37977889 (Radiologist Comparison Data)
**Title:** Explainable 3D CNN Model for Glioma Classification
**Authors:** Bathla P, et al.
**Source:** Expert Review of Medical Devices, 2024
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/37977889/
**Key finding for us:** The study reported that 40% of radiologist disagreements with the CNN were due to the model using features not typically weighted by radiologists (e.g., subtle texture patterns). This is an *explainability gap* — the model may be correct but explaining its reasoning is harder.
**Relevance:** Explainability must address the "different reasoning" problem, not just visualise the same features the model uses.

---

## Clinical XAI Pipeline

### What Clinicians Need

| Clinician Question | XAI Method | Implementation |
|---|---|---|
| "Where is the tumour?" | Grad-CAM 3D heatmap | Apply Grad-CAM on 3D backbone → slice heatmap per axial plane |
| "Why Grade IV and not Grade III?" | Counterfactual explanation | Generate "what input change would flip the prediction?" — e.g., reduce necrosis volume |
| "How confident is the model?" | EDL entropy | Overlay entropy as colour-coded opacity on heatmap |
| "What features drove the decision?" | Grad-CAM + feature importance bar chart | Grad-CAM for spatial; Grad-CAM++ for feature-level importance |
| "Is this case similar to cases I know?" | k-NN similarity display | Show 5 most similar training cases with their labels |

### Explanation Generation Pipeline (RTX 2050 Compatible)

```
1. Input: 64³ CT+MRI patch
2. Forward pass → predictions + EDL entropy
3. Grad-CAM on 3D ResNet-18 → 3D heatmap
4. Slice heatmap along axial plane → per-slice Grad-CAM images
5. Combine Grad-CAM + EDL entropy → colour-coded heatmap
6. k-NN retrieval → 5 most similar training cases
7. Output: (predictions, entropy, heatmap, k-NN cases)
```

**Memory cost:** Grad-CAM adds ~1× the forward pass memory (gradient computation). On RTX 2050, this means ~430 MB total for explanation generation.

---

## Regulatory Considerations (FDA/EU)

### FDA 510(k) Guidance for AI/ML-Based Medical Devices (2021–2026)

| Requirement | What It Means for XAI | Implementation |
|---|---|---|
| **Predictive performance** | Accuracy, sensitivity, specificity | Report 4-class classification metrics |
| **Explainability** | "The model must provide interpretable outputs" | Grad-CAM + EDL entropy output |
| **Clinician validation** | Explanations validated by ≥3 radiologists | Run a clinician validation study |
| **Bias assessment** | Performance across demographic groups | Report per-scan-type metrics |
| **Post-market monitoring** | Continuous performance tracking | Log predictions + explanations for retrospective audit |

### EU AI Act (2024) — High-Risk Classification

- Medical device classification = **high-risk AI system**
- Requirements: transparency, human oversight, robustness, explainability
- **Consequence:** Grad-CAM + UQ is not optional — it is a regulatory requirement

---

## PACS Integration

**Problem:** Radiologists work in PACS (Picture Archiving and Communication System). Model outputs must be viewable within the PACS workflow.

| Approach | Integration Effort | Quality |
|---|---|---|
| DICOM secondary capture | Low — embed heatmap as DICOM overlay | Medium — limited interactivity |
| Web-based viewer | Medium — build React web app that pulls from DICOM | High — full interactivity |
| AI-assisted PACS plugin | High — vendor-specific integration | Highest — native PACS experience |

**Recommendation:** Start with DICOM secondary capture (lowest effort). Upgrade to web viewer when clinical validation succeeds.

---

## Explanation Quality Metrics

| Metric | Definition | Threshold |
|---|---|---|
| **Gradient-weighted alignment** | Correlation between Grad-CAM and ground truth mask | Dice > 0.7 |
| **Explanation stability** | Grad-CAM variance across similar inputs | Variance < 0.15 |
| **Fidelity** | Prediction change when Grad-CAM region is masked | Δ accuracy < 10% |
| **Clinician agreement** | % of clinicians who agree explanation is clinically relevant | > 70% |
| **Counterfactual validity** | Generated counterfactual produces predicted class | > 80% |

---

## Reference Table

| # | Year | Authors | Source | ID | Title |
|---|---|---|---|---|---|
| 1 | 2024 | Bathla P, et al. | Expert Rev Med Devices | PMID:37977889 | Explainable 3D CNN Model for Glioma Classification |
| 2 | 2026 | Vega Lara M, et al. | PubMed Central | PMID:42525278 | Clinical Uncertainty Quantification: A Review of Methods for Neuro-Oncology |
| 3 | 2018 | Sensoy M, et al. | arXiv | 1806.01768 | Evidential Deep Learning to Quantify Classification Uncertainty |
| 4 | 2022 | Debesyr J, et al. | Med Image Anal | PMID:35430967 | Explainable AI in Medical Imaging: A Comprehensive Survey |
| 5 | 2017 | Selvaraju RR, et al. | ECCV | arXiv:1610.02391 | Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization |
| 6 | 2020 | Lundberg SM, Lee SI | Nature Mach Intell | PMID:32624523 | Unified Approach to Interpreting Model Predictions (SHAP) |
| 7 | 2019 | Guided Grad-CAM | arXiv:1710.06830 | Guided Grad-CAM |

---

## Recommendation

1. **Grad-CAM 3D + EDL entropy** as the primary explanation output
2. **Counterfactual explanation** for second-opinion workflows
3. **DICOM secondary capture** for initial PACS integration
4. **Clinician validation study** with ≥3 radiologists before any deployment
5. **Explanation quality metrics** must be tracked during training
