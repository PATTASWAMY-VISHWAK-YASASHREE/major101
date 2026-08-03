# Category 15: Ethical Considerations and Bias

**Status: ✅ RESEARCH COMPLETE**

---

## 15.1 Sources of Bias in Brain Tumour AI

| Bias source | How it manifests | Mitigation |
|---|---|---|
| **Scanner bias** | Model only trained on Siemens MRI, fails on GE | Multi-scanner datasets, ComBat |
| **Population bias** | BraTS is mostly Caucasian adults | Diverse training data |
| **Age bias** | Pediatric tumours (Grade I) underrepresented | Include pediatric datasets |
| **Class imbalance** | Grade IV dominates, Grade I rare | Weighted loss, stratified sampling |
| **Label noise** | BraTS labels have inter-rater variability (~90% agreement) | Use multi-rater consensus labels |

---

## 15.2 Gender and Age Distribution (BraTS)

| Group | % of cases | Notes |
|---|---|---|
| Male | ~62% | Reflects true epidemiology |
| Female | ~38% | |
| <40 years | ~25% | Pediatric + young adult |
| 40-70 years | ~65% | Peak incidence |
| >70 years | ~10% | Underrepresented |

**Implication:** Model may perform worse on elderly patients. Stratify evaluation by age group.

---

## 15.3 Ethical Requirements for Clinical Use

| Requirement | Status for this project |
|---|---|
| **Informed consent** | N/A — open datasets only |
| **IRB approval** | N/A — research prototype only |
| **Data privacy (GDPR/HIPAA)** | All datasets are de-identified |
| **Model fairness audit** | ⚠️ Must do before clinical use |
| **Human-in-the-loop** | AI is assistive, not diagnostic |

---

## 15.4 Our Ethical Stance

1. **This is an AI assistant, not a diagnostic tool.** Output is "Grade III (78% confidence)" — radiologist makes the final call.
2. **Calibration matters.** Overconfident wrong predictions are dangerous. MC Dropout + temperature scaling required.
3. **Bias audit before deployment.** Per-age-group and per-gender F1 must be reported.
4. **Data provenance.** All training data comes from open datasets with proper licenses.
