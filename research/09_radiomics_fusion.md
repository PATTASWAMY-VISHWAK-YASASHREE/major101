# Category 9: Radiomics + Deep Feature Fusion

**Status: 🔬 RESEARCH ONLY — no implementation yet**

> **Purpose:** Understand how handcrafted radiomic features combine with learned
> deep features. Relevant to multimodal fusion decisions.

---

## 9.1 What Is Radiomics?

Radiomics = extraction of large numbers of quantitative features from medical images
using defined algorithms. Features describe:

| Category | Examples |
|---|---|
| **Shape** | Tumour volume, surface area, sphericity |
| **Intensity** | Mean, median, std, skewness, kurtosis |
| **Texture** | GLCM, GLRLM, GLDM, NGTDM (haralick texture features) |
| **Higher-order** | Wavelet, Laplacian-of-Gaussian filtered features |

**Typical feature set:** 100-1000 features per scan.

---

## 9.2 Deep Features vs. Radiomic Features

| | Deep Features | Radiomic Features |
|---|---|---|
| **Source** | Learned from CNN activations | Handcrafted mathematical formulas |
| **Number** | ~128-512 per modality | ~100-1000 per scan |
| **Interpretability** | Low (black box) | High (each feature has meaning) |
| **Data requirement** | Large dataset for training | Small dataset works |
| **Performance** | Generally better with enough data | Can beat deep learning on small data |

---

## 9.3 Fusion Strategies

### Early Fusion (Feature-level)

```
[Deep features] + [Radiomic features] → Concatenate → Classifier
```

**Simple but effective.** Deep features capture visual patterns; radiomics capture
mathematical texture/shapes that deep nets may miss.

**Representative work:** Yang et al. 2023 (PMID 36870427) — GBM prognosis using
radiomics alone from preoperative MRI.

### Late Fusion (Decision-level)

```
Deep classifier → prediction_1
Radiomics classifier → prediction_2
Ensemble → final prediction
```

**Better when the two feature types are complementary** rather than redundant.

**Representative work:** Yang et al. 2026 (PMID 42352847) — multimodal radiomics for
molecular prediction from CT+MRI.

---

## 9.4 When to Use Radiomics

| Scenario | Verdict |
|---|---|
| **Small dataset (< 200 cases)** | ✅ Radiomics + deep features help |
| **Large dataset (> 500 cases)** | Deep features alone usually suffice |
| **Molecular prediction** | ✅ Radiomics features are strong predictors (IDH, 1p/19q) |
| **Pure classification** | Deep features usually dominate |

**For our project (80 IBSR cases):** Radiomics features are a strong complement to
deep features. With only 80 cases, every source of information matters.

---

## 9.5 Recommended Pipeline for Our Project

```
CT scan  ─→ CTN normalisation ─→ 3D ResNet → deep features [64]
                              ─→ Radiomics extraction → radiomic features [128]
                              ↓
MRI scan ─→ CTN normalisation ─→ 3D ResNet → deep features [64]
                              ─→ Radiomics extraction → radiomic features [128]
                              ↓
                    Concatenate all features → Dense classifier → 4-class
```

**Total feature vector:** 64 (CT deep) + 128 (CT radiomics) + 64 (MRI deep) + 128 (MRI radiomics) = 384 features.

**Tool:** **PyRadiomics** (https://pyradiomics.readthedocs.io/) — Python package
that extracts ~1500 features from a NIfTI volume + segmentation mask.

---

## 9.6 Key Takeaway

Radiomics + deep feature fusion is a strong strategy for small datasets like IBSR.
For our project, it is an **optional enhancement** after the baseline deep fusion
model is working. Do it in a second pass if the model accuracy plateaus.
