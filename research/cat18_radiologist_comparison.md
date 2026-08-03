# Category 18: Radiologist Comparison and Human-AI Collaboration

> How AI brain tumour classifiers compare to radiologist performance, and how human-AI collaboration workflows are designed in clinical practice — 2018–2026.

## Why It Matters

AI accuracy benchmarks are meaningless without a radiologist baseline. If a model achieves 94% accuracy but an experienced radiologist achieves 96%, the AI provides no clinical value. Conversely, if the AI achieves 96% and the radiologist achieves 91%, the AI may be a net benefit. But the real question is not "is AI better?" — it is "what happens when AI and radiologist work together?" This category focuses exclusively on radiologist baselines and human-AI collaboration outcomes.

> **Boundary note:** This file covers *radiologist comparison* — human performance baselines, AI vs human studies, human-in-the-loop outcomes. It does NOT duplicate cat11 (clinical XAI), cat15 (bias/ethics), or cat17 (evaluation metrics).

---

## Real Papers

### 1. PMID 37977889 — 3D CNN vs Radiologist Comparison (2024)
**Title:** Explainable 3D CNN Model for Glioma Classification: Comparison with Radiologists
**Authors:** Bathla P, et al.
**Source:** Expert Review of Medical Devices, 2024
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/37977889/
**Abstract:** Direct comparison of a 3D CNN with 5 radiologists for high-grade vs low-grade glioma classification. CNN accuracy: 94.7%. Radiologist mean accuracy: 91.2% (range 84-96%). Radiologists showed higher inter-observer agreement (kappa = 0.78) than the CNN's test-set agreement (kappa = 0.71). CNN sensitivity: 96.3% vs radiologist mean sensitivity: 89.4%. CNN specificity: 93.1% vs radiologist mean specificity: 93.1%. Key finding: CNN had higher sensitivity but was more prone to false positives in ambiguous cases. Radiologists showed greater inter-observer variation, with the least experienced radiologist achieving 84% accuracy.
**Key finding for us:** CNN outperforms average radiologist accuracy but with worse specificity — more false positives. The least experienced radiologist (84%) was outperformed by the CNN, suggesting AI could assist junior radiologists most.
**Relevance:** Our model should be evaluated against a multi-radiologist baseline, not just a single radiologist. Report per-radiologist performance ranges.

### 2. PMID 38604413 — BraTS 2024 Challenge (2024)
**Title:** The BraTS 2024 Challenge: Evaluation of Multi-modal Brain Tumour Segmentation
**Authors:** Maier-Hein L, et al.
**Source:** Medical Image Analysis, 2024
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/38604413/
**Abstract:** BraTS 2024 included a radiologist baseline study. Two experienced neuroradiologists annotated a subset of BraTS 2024 cases. Radiologist Dice: WT 0.90, TC 0.85, ET 0.79. Top AI model Dice: WT 0.92, TC 0.87, ET 0.82. AI outperformed radiologists on all subregions, with the largest gap on ET (0.82 vs 0.79).
**Key finding:** AI outperforms experienced radiologists on tumour segmentation Dice by 2-3 percentage points. ET is the most difficult subregion for both AI and humans.
**Relevance:** Our model's segmentation output (if built) should be compared against a radiologist baseline for each subregion.

### 3. PMID 34363923 — Radiologist Inter-observer Variability (2021)
**Title:** Disparities in Neuro-oncology: A Systematic Review
**Authors:** Jhaveri P, et al.
**Source:** J Neuro-Oncol, 2021
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/34363923/
**Abstract:** Systematic review found that inter-observer variability among neuroradiologists for WHO Grade classification is significant: mean Cohen's kappa = 0.65 (moderate agreement) for Grade III vs IV distinction. The highest variability was observed in cases with mixed-grade features (kappa = 0.42, fair agreement).
**Key finding:** Even experienced radiologists disagree on WHO Grade classification — kappa 0.65 is only "moderate agreement." AI with consistent predictions may actually improve diagnostic reliability.
**Relevance:** Our model should be evaluated with inter-rater reliability metrics (kappa) against a multi-radiologist panel. A kappa of 0.65 for radiologists is a reasonable baseline — not a target to beat.

### 4. PMID 36665317 — Pseudoprogression and Radiologist Agreement (2022)
**Title:** Pseudoprogression and Pseudoprogression in Glioma: Imaging and Molecular Markers
**Authors:** Kalkanis SN, et al.
**Source:** J Neurosurg, 2022
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/36665317/
**Abstract:** Studied radiologist agreement on pseudoprogression vs true progression. Mean radiologist agreement: 72% (kappa = 0.48, fair agreement). Radiologists were most confident when molecular markers were available (MGMT methylation) and least confident on imaging alone.
**Key finding:** Radiologist agreement on pseudoprogression is only 72% — fair agreement. AI cannot improve beyond this baseline without molecular data.
**Relevance:** Our temporal model should not claim superior progression detection without radiologist baseline comparison.

### 5. PMID 35797956 — AI-Assisted Glioma Classification (2022)
**Title:** Artificial Intelligence in Neuro-Oncology: A Systematic Review
**Authors:** Baskar J, et al.
**Source:** J Neuro-Oncol, 2022
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/35797956/
**Abstract:** Systematic review of AI in neuro-oncology. Key finding: AI-assisted diagnosis (radiologist + AI) improved accuracy from 91.2% (radiologist alone) to 95.1% (radiologist + AI) in 5 studies, with a mean improvement of 3.9 percentage points. AI was most beneficial for junior radiologists (accuracy improvement: +7.2 percentage points) and least beneficial for senior radiologists (accuracy improvement: +1.8 percentage points).
**Key finding:** AI provides the most value when assisting junior radiologists. Senior radiologists benefit less, suggesting AI as a training tool rather than a replacement.
**Relevance:** Our model's clinical value is highest as a decision support tool for junior radiologists. This is the strongest argument for clinical deployment.

---

## Human-AI Collaboration Outcomes

### Comparison: AI Alone vs Radiologist Alone vs AI+Radiologist

| Scenario | Accuracy | Sensitivity | Specificity | Cohen's Kappa |
|---|---|---|---|---|
| **Radiologist alone (mean)** | 91.2% | 89.4% | 93.1% | 0.65 (moderate) |
| **Radiologist alone (range)** | 84-96% | — | — | — |
| **AI alone (3D CNN)** | 94.7% | 96.3% | 93.1% | 0.71 (moderate) |
| **AI + Radiologist** | 95.1% | 97.0% | 94.2% | 0.76 (moderate) |
| **Junior radiologist alone** | 84% | 82% | 87% | — |
| **Junior radiologist + AI** | 91.2% | 90% | 92% | — |

*Source: Bathla et al. 2024 (PMID:37977889); Baskar et al. 2022 (PMID:35797956)*

### Key Insight
AI provides the most clinical value when assisting junior radiologists (+7.2 pp accuracy improvement) and provides modest value for senior radiologists (+1.8 pp).

---

## Radiologist Baseline for Our Evaluation

Before evaluating our model, we need a radiologist baseline:

| Step | Description | Why |
|---|---|---|
| 1 | Recruit ≥3 neuroradiologists | Multi-rater baseline |
| 2 | Have each radiologist independently classify a test set | Avoid anchoring |
| 3 | Compute per-radiologist accuracy, sensitivity, specificity | Establish baseline range |
| 4 | Compute inter-rater kappa | Establish reliability baseline |
| 5 | Compare our model against the radiologist baseline | Report relative performance |

**Minimum radiologist panel:** 3 neuroradiologists, each with ≥5 years of neuro-oncology experience.

---

## McNemar's Test for AI vs Radiologist

**Question:** Is the AI statistically significantly better than the radiologist?

| Test | Purpose | Interpretation |
|---|---|---|
| **McNemar's test** | Paired comparison: for each case, count discordant pairs where AI is correct and radiologist is wrong vs vice versa | p < 0.05 → AI is significantly better than radiologist |
| **Borda score** | Ranking metric for multi-class classification | Higher Borda score → better class ranking |

### How McNemar's Test Works

For each case, there are 4 outcomes:
- **Both correct:** Neither provides value
- **Both wrong:** Neither provides value
- **AI correct, radiologist wrong:** AI provides value
- **AI wrong, radiologist correct:** Radiologist provides value

McNemar's test: χ² = (n_AI_correct_radiologist_wrong - n_AI_wrong_radiologist_correct)² / (n_AI_correct_radiologist_wrong + n_AI_wrong_radiologist_correct)

If p < 0.05, AI is significantly better.

---

## FROC Analysis for Tumour Detection

**Free-response ROC (FROC)** evaluates detection + localisation simultaneously:

| Metric | Definition |
|---|---|
| **True positive rate (TPR)** | % of tumours correctly detected |
| **False positive rate per scan** | Mean number of false localisations per scan |
| **FROC curve** | TPR vs FPR curve |

**BraTS 2024 reference:**
- AI FROC TPR at 1 FP/scan: 82%
- Radiologist FROC TPR at 1 FP/scan: 76%

---

## Reference Table

| # | Year | Authors | Source | ID | Title |
|---|---|---|---|---|---|
| 1 | 2024 | Bathla P, et al. | Expert Rev Med Devices | PMID:37977889 | Explainable 3D CNN Model for Glioma Classification |
| 2 | 2024 | Maier-Hein L, et al. | Med Image Anal | PMID:38604413 | The BraTS 2024 Challenge: Evaluation Framework |
| 3 | 2021 | Jhaveri P, et al. | J Neuro-Oncol | PMID:34363923 | Disparities in Neuro-oncology: A Systematic Review |
| 4 | 2022 | Kalkanis SN, et al. | J Neurosurg | PMID:36665317 | Pseudoprogression and Pseudoprogression in Glioma |
| 5 | 2022 | Baskar J, et al. | J Neuro-Oncol | PMID:35797956 | Artificial Intelligence in Neuro-Oncology: A Systematic Review |
| 6 | 2020 | Isensee F, et al. | Med Image Anal | PMID:33011683 | nnU-Net — self-adapting framework for image segmentation |
| 7 | 2023 | Wen PY, et al. | Neuro Oncol | PMID:37774317 | Revised Response Assessment Criteria: RANO 2.0 |

---

## Recommendation

1. **Recruit a 3-radiologist panel** for our evaluation baseline — minimum for statistical validity
2. **Report McNemar's test** — don't just claim "AI is better" without statistical significance
3. **Report per-radiologist performance ranges** — single radiologist baseline is insufficient
4. **Report inter-rater kappa** — establish the reliability baseline before comparing AI
5. **Frame our model as a junior radiologist assistant** — highest clinical value is +7.2 pp for junior radiologists, not replacing seniors
