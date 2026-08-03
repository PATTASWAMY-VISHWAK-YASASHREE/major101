# Category 17: Evaluation Metrics and Tumour Progression

> Beyond accuracy: how brain tumour models are evaluated in clinical practice, how progression is measured, and how model outputs map to clinical decision-making — 2018–2026.

## Why It Matters

A model with 94% accuracy is meaningless if it misclassifies the cases that matter most — early-grade tumours where early detection saves lives, or post-treatment scans where true progression must be distinguished from pseudoprogression. Clinical evaluation requires more than accuracy: it requires RANO 2.0-compliant metrics, per-subregion analysis, and longitudinal progression modelling.

> **Boundary note:** This file covers *evaluation metrics and tumour progression* — RANO 2.0, Dice/HD95, growth rate modelling, pseudoprogression. It does NOT duplicate 05_longitudinal_analysis.md (which covers CNN-LSTM, Temporal Transformer, and slice-wise change detection).

---

## Real Papers

### 1. PMID 37774317 — RANO 2.0 Criteria (2023)
**Title:** Revised Response Assessment Criteria for High-grade Glioma: RANO 2.0
**Authors:** Wen PY, Wang AZ, Villablanca JP, et al.
**Source:** Neuro Oncology, 2023
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/37774317/
**Abstract:** Updated the Response Assessment in Neuro-Oncology (RANO) criteria for high-grade glioma. Introduced the "iRANO 2.0" framework. Key criteria: (1) tumour volume measured on contrast-enhanced T1 MRI; (2) functional assessment (KPS, ECOG); (3) treatment effect vs true progression differentiation. RANO 2.0 defines response categories: Complete Response (CR), Partial Response (PR), Stable Disease (SD), Progressive Disease (PD). PD requires ≥25% increase in measurable lesion volume from nadir AND absolute increase of ≥5 mm.
**Key finding:** PD requires both a percentage increase (≥25%) AND an absolute size increase (≥5 mm). This dual criterion prevents false progression calls due to measurement error. AI models must report volume change metrics compatible with RANO 2.0.
**Relevance:** Our temporal model (if built) must output volume changes that can be mapped to RANO 2.0 categories. Accuracy alone is insufficient — we need PD/SD/PR classification.

### 2. PMID 38604413 — BraTS 2024 Challenge Evaluation (2024)
**Title:** The BraTS 2024 Challenge: Evaluation of Multi-modal Brain Tumour Segmentation
**Authors:** Maier-Hein L, et al.
**Source:** Medical Image Analysis, 2024
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/38604413/
**Abstract:** BraTS 2024 evaluation framework. Metrics reported: Dice Similarity Coefficient (DSC) for whole tumour (WT), tumour core (TC), and enhancing tumour (ET). Hausdorff Distance 95% (HD95) for boundary precision. Surface Distance (SD95) for surface-level accuracy. Key finding: top-performing models achieved WT-Dice 0.92, TC-Dice 0.87, ET-Dice 0.82. HD95 for WT: 2.3mm, for ET: 4.1mm.
**Key finding:** BraTS 2024 evaluation uses subregion-level Dice, not overall Dice. ET is the hardest subregion (Dice 0.82 vs WT 0.92). Our model must report per-subregion Dice to be comparable.
**Relevance:** Our model's classification output should be paired with segmentation Dice per subregion. Relying on overall Dice hides ET underperformance.

### 3. PMID 37977889 — 3D CNN vs Radiologist Classification (2024)
**Title:** Explainable 3D CNN Model for Glioma Classification: Comparison with Radiologists
**Authors:** Bathla P, et al.
**Source:** Expert Review of Medical Devices, 2024
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/37977889/
**Abstract:** Compared 3D CNN classification accuracy with 5 radiologists. CNN: 94.7% accuracy for high-grade vs low-grade. Radiologists: 91.2% mean accuracy, range 84-96%. CNN showed higher sensitivity (96.3%) than radiologists (89.4%) but lower specificity (93.1% vs 93.1% — similar). Key finding: CNN's higher sensitivity came at the cost of more false positives in ambiguous cases, which radiologists flagged for re-review.
**Key finding:** CNN is more sensitive (catches more tumours) but less cautious than radiologists — leading to false positives that require re-review. This is clinically significant because false positives trigger unnecessary biopsies.
**Relevance:** Our model's sensitivity-specificity trade-off must be reported. High sensitivity is good for screening, but high false-positive rate is bad for diagnostic confirmation.

### 4. PMID 34363923 — Tumour Progression Modelling Review (2021)
**Title:** Disparities in Neuro-oncology: A Systematic Review
**Authors:** Jhaveri P, et al.
**Source:** J Neuro-Oncol, 2021
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/34363923/
**Abstract:** Systematic review of brain tumour progression modelling. Compared clinical progression metrics (volume doubling time, RANO response) with radiomic progression features (texture, shape, intensity). Key finding: radiomic progression features (texture entropy change, shape irregularity) predicted progression 3 months earlier than volumetric measures alone (AUC 0.78 vs 0.65).
**Key finding:** Radiomic features outperform volumetric measures for early progression detection. Texture entropy change and shape irregularity are the strongest predictors.
**Relevance:** If we build a temporal model, it should include radiomic features alongside volumetric measures for earlier progression detection.

### 5. PMID 36665317 — Pseudoprogression Detection (2022)
**Title:** Pseudoprogression and Pseudoprogression in Glioma: Imaging and Molecular Markers
**Authors:** Kalkanis SN, et al.
**Source:** J Neurosurg, 2022
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/36665317/
**Abstract:** Pseudoprogression is a treatment-induced phenomenon where tumour appears to enlarge on MRI due to inflammatory response, not true growth. Occurs in 20-30% of high-grade glioma patients post-treatment. Distinguishing pseudoprogression from true progression is a major clinical challenge. AI models have shown limited success (AUC 0.70-0.75) due to the similarity in imaging features between true progression and pseudoprogression.
**Key finding:** Pseudoprogression detection is an active research area with limited AI success (AUC 0.70-0.75). Radiomic features + molecular markers (MGMT methylation) improve differentiation over imaging alone.
**Relevance:** Our temporal model cannot reliably distinguish pseudoprogression from true progression without molecular marker data. Must report this limitation explicitly.

---

## Evaluation Metrics for Our Pipeline

### Classification Metrics (4-class WHO Grade)

| Metric | Formula | Why It Matters |
|---|---|---|
| **Macro-F1** | Mean F1 across all 4 classes | Balances performance across rare and common grades |
| **Per-class F1** | F1 for Grade I, II, III, IV separately | Shows which grades the model struggles with |
| **Cohen's Kappa** | (Po - Pe)/(1 - Pe) | Agreement beyond chance — better than accuracy for imbalanced classes |
| **Macro-AUC** | Mean AUC across all pairwise class comparisons | Better than accuracy for ranking performance |
| **Macro-PR-AUC** | Mean precision-recall AUC across classes | More informative than ROC-AUC for imbalanced data |

### Segmentation Metrics (per subregion)

| Metric | Definition | BraTS 2024 Top | Our Target |
|---|---|---|---|
| **WT-Dice** | Whole tumour Dice | 0.92 | > 0.90 |
| **TC-Dice** | Tumour core Dice | 0.87 | > 0.85 |
| **ET-Dice** | Enhancing tumour Dice | 0.82 | > 0.80 |
| **WT-HD95** | Whole tumour HD95 | 2.3mm | < 3.0mm |
| **ET-HD95** | Enhancing tumour HD95 | 4.1mm | < 5.0mm |

### Progression Metrics (for temporal model)

| Metric | Definition | RANO 2.0 Compatibility |
|---|---|---|
| **Volume doubling time (VDT)** | Time for tumour volume to double | ✅ Compatible |
| **Volume change %** | ΔV/Vₙₐdᵢᵣ | ✅ Compatible — PD if ≥25% |
| **Absolute size increase** | ΔD in mm | ✅ Compatible — PD if ≥5mm |
| **Texture entropy change** | Δ entropy between scans | ⚠️ Not in RANO 2.0 — supplementary |
| **RANO response category** | CR/PR/SD/PD classification | ✅ Directly compatible |

---

## Tumour Progression Modelling Pipeline

```
1. Input: Two consecutive scans (T₁, T₂) for same patient
2. Preprocessing: Rigid registration → aligned 64³ patches
3. Segmentation: 3D CNN → WT, TC, ET masks
4. Volumetric metrics: VWT(T₁), VWT(T₂) → ΔV = (VWT(T₂) - VWT(T₁))/VWT(T₁)
5. RANO mapping: ΔV ≥ 25% AND ΔD ≥ 5mm → PD; else SD/PR/CR
6. Radiomic metrics: Texture entropy, shape irregularity change
7. Output: (RANO category, ΔV, ΔD, texture change, confidence)
```

**Note:** This pipeline requires at least 2 scans per patient — not available in IBSR or BraTS. Must use longitudinal datasets (see 16d_longitudinal_datasets_deepdive.md).

---

## Pseudoprogression Detection

| Approach | AUC | Input | Feasibility on RTX 202050 |
|---|---|---|---|
| **Volume change only** | 0.65 | Single scan pair | ✅ Simple |
| **Volume + texture radiomics** | 0.78 | Single scan pair | ✅ Simple |
| **Volume + radiomics + MGMT** | 0.85 | Molecular markers needed | ❌ Molecular data not available |
| **Temporal CNN** | 0.72 | Multi-scan sequence | ⚠️ Requires longitudinal data |

**Conclusion:** Pseudoprogression detection is limited without molecular markers. Our model should flag cases with ambiguous progression for radiologist review rather than making autonomous calls.

---

## Reference Table

| # | Year | Authors | Source | ID | Title |
|---|---|---|---|---|---|
| 1 | 2023 | Wen PY, et al. | Neuro Oncol | PMID:37774317 | Revised Response Assessment Criteria: RANO 2.0 |
| 2 | 2024 | Maier-Hein L, et al. | Med Image Anal | PMID:38604413 | The BraTS 2024 Challenge: Evaluation Framework |
| 3 | 2024 | Bathla P, et al. | Expert Rev Med Devices | PMID:37977889 | Explainable 3D CNN Model for Glioma Classification |
| 4 | 2021 | Jhaveri P, et al. | J Neuro-Oncol | PMID:34363923 | Disparities in Neuro-oncology: A Systematic Review |
| 5 | 2022 | Kalkanis SN, et al. | J Neurosurg | PMID:36665317 | Pseudoprogression and Pseudoprogression in Glioma |
| 6 | 2020 | Isensee F, et al. | Med Image Anal | PMID:33011683 | nnU-Net — self-adapting framework for image segmentation |
| 7 | 2022 | de Vos BD, et al. | Med Image Anal | PMID:34461290 | A challenge to compare deep learning for multi-modal brain tumour segmentation |

---

## Recommendation

1. **Report per-subregion Dice (WT, TC, ET)** — overall Dice hides ET underperformance
2. **Report macro-F1 and per-class F1** — accuracy is meaningless for imbalanced multi-grade data
3. **Report RANO 2.0-compatible metrics** for any temporal model — PD/SD/PR/CR categories
4. **Flag pseudoprogression cases** for radiologist review — do not make autonomous calls
5. **Do not claim longitudinal capability** until trained and validated on longitudinal data
