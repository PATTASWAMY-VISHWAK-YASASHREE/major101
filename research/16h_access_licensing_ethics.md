# 16h — Access, Licensing, and Ethics for Brain Tumor Datasets

> Summary of access requirements, licensing terms, and ethical considerations for each dataset.
> **Status:** PARTIAL — based on published dataset descriptions; institutional data access terms vary.
> **Date:** 2026-06-15

---

## Access and Licensing Summary

| Dataset | Access Level | Registration Required | Licensing | IRB/Ethics Approval | De-identification |
|---------|-------------|----------------------|-----------|-------------------|-------------------|
| BraTS 2024 | Public (Synapse) | Yes (free) | Open Access (challenge participants) | ✅ Pre-approved | ✅ Fully de-identified |
| BraTS 2012–2023 | Public (Synapse) | Yes (free) | Open Access (challenge participants) | ✅ Pre-approved | ✅ Fully de-identified |
| TCIA HGG/LGG/GBM | Public (TCIA) | Yes (free) | Open Access (with approval) | ✅ Pre-approved | ✅ Fully de-identified |
| MSD (Brain Task) | Public | Yes (free) | Open Access (research use) | ✅ Pre-approved | ✅ Fully de-identified |
| ISIC Archive | Public | Yes (free) | Creative Commons (CC BY-NC) | ✅ Pre-approved | ✅ Fully de-identified |
| OASIS-2 | Public (NDA/CMU) | Yes (free) | Open Access (research use) | ✅ Pre-approved | ✅ Fully de-identified |
| MIX (MS MRI) | Public | Yes (free) | Open Access | ✅ Pre-approved | ✅ Fully de-identified |
| LiTS | Public (Kaggle) | Yes (free) | Open Access | ✅ Pre-approved | ✅ Fully de-identified |
| DeepLesion | Public (GitHub) | Yes (registration) | Open Access | ✅ Pre-approved | ✅ Fully de-identified |
| GBM-SBRT (TCIA) | Public (TCIA) | Yes (approval) | Open Access | ✅ Pre-approved | ✅ Fully de-identified |
| MSSEG | Public | Varies | Open Access (research use) | ✅ Pre-approved | ✅ Fully de-identified |

---

## Institutional Data Access

For institutional clinical data (the recommended validation source):

| Requirement | Details |
|-------------|---------|
| IRB approval | Required — must cover multi-timepoint imaging analysis and AI/ML use |
| Data use agreement | Required — typically signed between institution and research team |
| HIPAA compliance | Required — de-identification or limited data set agreement needed |
| Data management plan | Required — describe storage, security, and retention policies |
| Access timeline | Typically 3–6 months from IRB submission to approval |
| Data format | Varies — DICOM, NIfTI, or proprietary formats |
| Scanner variety | Varies — often multi-vendor (Siemens, GE, Philips) |

---

## Ethical Considerations

### 1. Patient Privacy
- All public datasets are fully de-identified and pre-approved for research use
- Institutional data requires IRB-approved de-identification protocol
- Re-identification risk is minimal for de-identified datasets but non-zero for raw DICOM

### 2. Informed Consent
- BraTS and TCIA data were collected under IRB-approved consent forms that permit secondary research use
- Institutional data may have consent limitations — must verify per-site consent language

### 3. AI Bias and Fairness
- BraTS data is from North American and European institutions — may not generalize to other populations
- Scanner variability (Siemens vs. GE vs. Philips) may introduce systematic bias
- Patient demographics (age, ethnicity) are not systematically reported in most public datasets

### 4. Clinical Deployment Risks
- AI predictions for tumor progression could lead to incorrect clinical decisions if used without human oversight
- Black-box DL models are difficult to explain to clinicians — see Category 11 (Explainable AI)
- Model drift is a risk as imaging protocols evolve over time

### 5. Data Sharing and Reproducibility
- BraTS and TCIA data are publicly accessible — reproducibility is supported
- Institutional data cannot be publicly shared — reproducibility limited to model weights and code
- Preprocessing pipelines must be documented for reproducibility

### 6. Open Science
- All public datasets support open science
- Model weights and training code should be publicly shared where possible
- BraTS challenge format provides a template for standardized evaluation

---

## Recommended Data Access Strategy

1. **Training data:** Download BraTS 2024 via Synapse (public, no cost, ~1-2 days setup)
2. **CT baseline:** Download TCIA brain tumor collections (public, requires approval, ~1-2 weeks)
3. **Longitudinal validation:** Partner with clinical institution for IRB-approved multi-timepoint data (3-6 months timeline)
4. **Documentation:** Maintain a data access log documenting all sources, access dates, and licensing terms
