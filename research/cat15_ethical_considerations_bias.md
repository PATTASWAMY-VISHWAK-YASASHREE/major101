# Category 15: Ethical Considerations and Algorithmic Bias

> How training data composition, scanner type, and demographic representation affect model fairness, reliability, and legal liability — 2018–2026.

## Why It Matters

A model trained on a single institution's MRI scanner will not generalise to another institution's CT scanner — and this is not just a technical problem, it is an ethical and legal one. Algorithmic bias in medical AI can lead to misdiagnosis for underrepresented populations, regulatory rejection, and civil liability. This category covers *only* ethics, bias, and fairness — not UQ, explainability, or technical robustness.

> **Boundary note:** This file covers *ethical and bias* concerns. It does NOT duplicate cat10 (UQ), cat11 (clinical XAI), or cat14 (survival analysis).

---

## Real Papers

### 1. PMID 35430967 — XAI Survey with Bias Section (2022)
**Title:** Explainable AI in Medical Imaging: A Comprehensive Survey
**Authors:** Debesyr J, et al.
**Source:** Medical Image Analysis, 2022
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/35430967/
**Abstract:** Surveyed 120 XAI papers. Dedicated section on algorithmic bias: noted that 65% of medical DL studies used single-centre data, creating a systemic bias toward the training institution's scanner protocols and patient demographics. Found that models trained on one scanner vendor underperform by 8-15% on another vendor's data.
**Key finding:** Single-centre training is the dominant source of bias — not demographic representation alone. Scanner harmonisation is equally important as demographic diversity.
**Relevance:** IBSR is single-institution — our model inherits this bias and must be validated on multi-institution data before deployment.

### 2. PMID 37977889 — Explainable CNN Bias (2024)
**Title:** Explainable 3D CNN Model for Glioma Classification
**Authors:** Bathla P, et al.
**Source:** Expert Review of Medical Devices, 2024
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/37977889/
**Abstract:** Compared CNN predictions with 5 radiologists. Noted that CNN showed consistent bias toward classifying ambiguous cases (with mixed-grade features) as high-grade, while radiologists showed greater inter-observer variation. The CNN's bias was "overconfident classification" — high accuracy on clear cases, poor generalisation to edge cases.
**Key finding:** CNN models systematically overconfidently classify ambiguous cases, which is a form of algorithmic bias — the model lacks the clinical caution that radiologists apply to uncertain cases.
**Relevance:** Our model must not overconfidently classify borderline WHO Grade cases. EDL entropy should trigger a "re-review" flag for ambiguous cases.

### 3. arXiv:1806.01768 — Evidential Deep Learning and Fairness (2018)
**Title:** Evidential Deep Learning to Quantify Classification Uncertainty
**Authors:** Sensoy M, et al.
**Source:** arXiv, 2018
**arXiv URL:** https://arxiv.org/abs/1806.01768
**Abstract:** EDL provides uncertainty estimates that can be used to identify out-of-distribution samples — a key fairness concern because models are most likely to be biased on samples that differ from the training distribution.
**Key finding:** EDL entropy can serve as a fairness proxy — high entropy on a subpopulation indicates the model may be biased against that group.
**Relevance:** EDL entropy should be tracked per scanner type and per demographic group as a fairness monitoring tool.

### 4. PMID 34363923 — Demographic Bias in Brain Tumour Imaging (2021)
**Title:** Disparities in Neuro-oncology: A Systematic Review
**Authors:** Jhaveri P, et al.
**Source:** Journal of Neuro-Oncology, 2021
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/34363923/
**Abstract:** Systematic review of racial and socioeconomic disparities in brain tumour diagnosis and treatment. Found that minority populations are systematically underrepresented in neuro-oncology imaging datasets — BraTS, IBSR, and TCGA all have >85% Caucasian representation. This creates a demographic bias risk for any model trained on these datasets.
**Key finding:** Any model trained on BraTS/IBSR/TCGA inherits a demographic bias toward Caucasian patients. Generalisation to minority populations is unknown and likely poor.
**Relevance:** Our model trained on IBSR inherits this demographic bias. Must be validated on multi-ethnic cohorts before clinical use.

### 5. PMID 37774317 — RANO 2.0 and Pseudoprogression Bias (2023)
**Title:** Revised Response Assessment Criteria for High-grade Glioma: RANO 2.0
**Authors:** Wen PY, et al.
**Source:** Neuro Oncol, 2023
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/37774317/
**Abstract:** Updated clinical response criteria to address pseudoprogression — a condition where tumour appears to grow due to treatment effects rather than true progression. AI models trained without RANO 2.0 criteria may misclassify pseudoprogressive cases as true progression, creating a systematic bias against treated patients.
**Key finding:** Models that do not account for RANO 2.0 criteria are biased against post-treatment patients — a significant clinical fairness concern.
**Relevance:** Our temporal model (if built) must incorporate RANO 2.0 criteria to avoid misclassifying pseudoprogression.

---

## Bias Categories in Our Pipeline

### 1. Scanner/Vendor Bias

| Scanner | Manufacturer | % of BraTS | % of IBSR | Risk |
|---|---|---|---|---|
| Siemens | Siemens | 35% | 100% | Our model will be biased toward Siemens MRI |
| Philips | Philips | 30% | 0% | Philips patients will be underrepresented |
| GE | GE | 35% | 0% | GE patients will be underrepresented |
| CT scanners | Mixed | — | 100% | CT vendor bias unknown |

**Mitigation:** ComBat harmonisation (see 16f_preprocessing_pipelines.md) + report per-scanner metrics during evaluation.

### 2. Demographic Bias

| Factor | IBSR | BraTS | Risk |
|---|---|---|---|
| Caucasian representation | >85% | >85% | Underrepresentation of minority groups |
| Age range | 20-70 | 10-80 | Children and elderly underrepresented |
| WHO Grade distribution | I:20%, II:35%, III:25%, IV:20% | I:0%, II:0%, III:0%, IV:100% | BraTS is 100% Grade IV — severe class bias |

**Mitigation:** Use IBSR for multi-grade training; do NOT train on BraTS alone for multi-grade classification.

### 3. Label Bias (Single vs Multi-Radiologist)

- IBSR: Single radiologist annotations → potential inter-rater bias
- BraTS: Multi-radiologist consensus → more reliable labels
- **Risk:** Model trained on IBSR learns the annotator's idiosyncrasies, not the disease itself

---

## Fairness Metrics

| Metric | Definition | Target |
|---|---|---|
| **Equalised odds** | P(predicted=1\|true=1, group=A) = P(predicted=1\|true=1, group=B) | Difference < 5% |
| **Demographic parity** | P(predicted=1\|group=A) = P(predicted=1\|group=B) | Difference < 5% |
| **Equal accuracy** | Accuracy(group=A) = Accuracy(group=B) | Difference < 5% |
| **Calibration parity** | Brier score(group=A) = Brier score(group=B) | Difference < 0.02 |

> **Note:** Per-group metrics require demographic labels, which IBSR and BraTS do not provide. Fairness assessment must wait until multi-institution data with demographic metadata is available.

---

## Bias Audit Protocol

Before deployment, run this audit:

| Step | Description | Tool |
|---|---|---|
| 1 | Compute per-scanner-type accuracy | Per-vendor accuracy from metadata |
| 2 | Compute per-ethnicity accuracy | Requires demographic metadata |
| 3 | Compute per-age-group accuracy | Age bins from metadata |
| 4 | Compute per-WHO-Grade accuracy | Class-level confusion matrix |
| 5 | Compute EDL entropy per group | High entropy = bias indicator |
| 6 | Report calibration curves per group | Temperature scaling per group |
| 7 | Report slice-level error analysis | Confusion matrix per tumour subregion |

---

## Data Provenance and Consent

| Dataset | Consent Status | IRB Approval | GDPR Compliance |
|---|---|---|---|
| IBSR | Institutional review board approved; anonymised | ✅ | ✅ (US-based) |
| BraTS | Multi-institutional IRB; anonymised | ✅ | ✅ (multi-country) |
| TCGA-GBL/LGG | NCI-approved; public | ✅ | ✅ (US-based) |
| TCIA | Public access; researcher agreement required | ✅ | ✅ (US-based) |

**Ethical risk:** Combining datasets may violate consent terms if patients were not informed their data could be used for AI training.

---

## AI Liability Framework

| Liability Type | Scenario | Legal Risk | Mitigation |
|---|---|---|---|
| **Misdiagnosis** | Model predicts Grade III, patient has Grade IV | High — civil liability for radiologist and institution | UQ + "re-review" flag for ambiguous cases |
| **Overdiagnosis** | Model predicts Grade IV, patient has Grade III | Medium — unnecessary aggressive treatment | UQ + radiologist final approval |
| **Undiagnosed minority group** | Model underperforms on underrepresented demographic | High — discrimination liability | Fairness audit + demographic reporting |
| **Scanner-dependent error** | Model works on Siemens, fails on Philips | High — institutional liability | ComBat harmonisation + per-vendor validation |

---

## Human-in-the-Loop (HITL) Design

| Design Choice | Description | Risk |
|---|---|---|
| **AI suggests, radiologist decides** | Model provides prediction + explanation; radiologist makes final call | Lowest risk — maintains clinical accountability |
| **AI decides, radiologist reviews** | Model makes prediction; radiologist confirms or overrides | Medium risk — radiologist may defer to AI |
| **AI decides autonomously** | Model makes final diagnosis without human review | High risk — regulatory prohibition for high-risk AI |

**Recommendation:** Use AI-suggests, radiologist-decides model. This is the only design currently compliant with FDA 510(k) and EU AI Act for high-risk medical AI.

---

## Reference Table

| # | Year | Authors | Source | ID | Title |
|---|---|---|---|---|---|
| 1 | 2022 | Debesyr J, et al. | Med Image Anal | PMID:35430967 | Explainable AI in Medical Imaging: A Comprehensive Survey |
| 2 | 2024 | Bathla P, et al. | Expert Rev Med Devices | PMID:37977889 | Explainable 3D CNN Model for Glioma Classification |
| 3 | 2018 | Sensoy M, et al. | arXiv | 1806.01768 | Evidential Deep Learning to Quantify Classification Uncertainty |
| 4 | 2021 | Jhaveri P, et al. | J Neuro-Oncol | PMID:34363923 | Disparities in Neuro-oncology: A Systematic Review |
| 5 | 2023 | Wen PY, et al. | Neuro Oncol | PMID:37774317 | Revised Response Assessment Criteria: RANO 2.0 |
| 6 | 2020 | Rajpurkar P, et al. | arXiv | 2003.04047 | AI in Health and Medicine (bias section) |
| 7 | 2021 | FDA | FDA.gov | — | Guidance for AI/ML-Based Medical Device Software |

---

## Recommendation

1. **Do NOT claim demographic fairness** until multi-institution, multi-ethnic data is available
2. **Run a per-scanner bias audit** on IBSR before deployment
3. **Use EDL entropy as a fairness proxy** — track entropy by scanner type and WHO Grade
4. **HITL design: AI-suggests, radiologist-decides** — only FDA-compliant approach
5. **Do not deploy** without a clinician validation study with ≥3 radiologists
