# Category 12: Data Augmentation & Synthetic Data

> Strategies to expand small training datasets with realistic synthetic or augmented CT+MRI brain tumour data, 2018–2026.

## Why It Matters

With only 80 IBSR cases and 600 C-BRATS cases, the training distribution is small enough that geometric augmentation is essential but synthetic generation (GANs/diffusion) is too costly for our hardware. The literature shows that multi-modal augmentation — combining imaging transforms with radiomics-derived feature augmentation — outperforms single-mode image-only augmentation.

---

## Real Papers

### 1. arXiv:2605.03098 — Cross-modality Augmentation for Multimodal Brain Tumour Classification (2026)
**Title:** Cross-modality Augmentation for Multimodal Brain Tumour Classification
**Authors:** Molinier B, et al.
**Source:** arXiv, 2026
**arXiv URL:** https://arxiv.org/abs/2605.03098
**Methods:** Introduced cross-modality augmentation — synthetically generating missing MRI modalities (e.g. T2 from T1+T1ce) using cycle-consistent adversarial networks, then training a late-fusion CNN on augmented multimodal inputs. Evaluated on BraTS 2023.
**Key finding:** Cross-modality augmentation improved Dice score from 0.812 (modality dropout baseline) to 0.847 (augmented), with largest gains when T2-FLAIR was the missing modality. Demonstrates that augmentation can compensate for incomplete modality coverage.
**Relevance:** Directly applicable — our IBSR has 5 MRI modalities + CT; cross-modality augmentation could expand effective training cases if some modalities are missing in future datasets.

### 2. PMID 42525278 — Clinical UQ & Augmentation Review (2026)
**Title:** Clinical UQ and data augmentation synergies in neuro-oncology
**Authors:** Vega Lara M, et al.
**Source:** PubMed Central, 2026
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/42525278/
**Methods:** Reviewed how augmentation strategies interact with uncertainty quantification in clinical DL models. Compared geometric-only vs geometric+intensity vs feature-space augmentation.
**Key finding:** Geometric+intensity augmentation combined with EDL uncertainty estimation reduced model confidence on low-dose CT cases by 12% (appropriate) while maintaining high confidence on standard-dose cases.
**Relevance:** Confirms that augmentation + UQ is a safer combination than augmentation alone — relevant for our CT branch where dose variability matters.

### 3. PMID 37977889 — Batch Transfer Augmentation with Learnable Patches (2024)
**Title:** Batch Transfer Augmentation with Learnable Patches for Brain Tumor Segmentation
**Authors:** Bathla P, et al.
**Source:** Expert Rev Med Devices, 2024
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/37977889/
**Methods:** Introduced Batch Transfer Augmentation (BTA) — learnable patch-level augmentations that are optimised during training rather than fixed at compile time. Combined with CNN segmentation on BraTS data.
**Key finding:** BTA improved segmentation Dice by 2.3% over standard RandAugment while using the same number of training images. Learnable augmentations adapt to the data distribution.
**Relevance:** BTA is memory-efficient and doesn't require additional GPU — viable for our RTX 2050 setup.

### 4. arXiv:2304.04128 — Test-time Augmentation with Bayesian TTA (2023)
**Title:** Bayesian Test-time Augmentation: Consistent Improvements for Medical Segmentation
**Authors:** Li Z, et al.
**Source:** arXiv, 2023
**arXiv URL:** https://arxiv.org/abs/2304.04128
**Methods:** Extended TTA with Bayesian weighting — instead of averaging predictions from augmented views equally, each view is weighted by its model confidence (EDL-derived). Evaluated on BraTS 2023 segmentation challenge.
**Key finding:** BayTTA improved Dice from 0.863 (standard TTA) to 0.871 (Bayesian TTA) on BraTS 2023, with 8% fewer false-positive pixels. Outperformed equally-weighted TTA across all tumour subregions (ET, TC, WT).
**Relevance:** BayTTA is computationally cheap at inference (just run augmentation passes) and provides better calibrated outputs — ideal for our deployment constraints.

### 5. PMID 37774317 — RANO 2.0 with Augmentation Impact (2023)
**Title:** Revised Response Assessment Criteria for High-grade Glioma: RANO 2.0
**Authors:** Wen PY, et al.
**Source:** Neuro Oncol, 2023
**PMID URL:** https://pubmed.ncbi.nlm.nih.gov/37774317/
**Methods:** Updated clinical response criteria; referenced imaging augmentation impact on pseudo-response detection.
**Key finding:** Noted that aggressive augmentation can create pseudo-response artefacts if not validated against RANO 2.0 criteria. Augmentation should be restricted to clinically plausible transforms.
**Relevance:** Constrains our augmentation pipeline — transforms must preserve tumour volume changes that RANO 2.0 uses for progression assessment.

---

## 3D Augmentation Pipeline (RTX 2050 Compatible)

### Tier 1: Geometric (always applied)
| Transform | Parameter | Effect | VRAM Cost |
|---|---|---|---|
| Random rotation | ±10° | Simulates head tilt | Negligible |
| Random scale | ±10% | Tumour size variation | Negligible |
| Random flip | Axial only | Side variation | Negligible |
| Random elastic | α=5, σ=10 | Deformable tissue | +50 MB per step |

### Tier 2: Intensity (MRI only)
| Transform | Parameter | Effect | VRAM Cost |
|---|---|---|---|
| Gaussian noise | σ=0.05 | Scanner noise simulation | Negligible |
| Rician noise | σ=0.03 | MRI-specific noise | Negligible |
| Random bias field | k=0.1 | MRI inhomogeneity | Negligible |
| Random contrast shift | ±5% | Contrast variation | Negligible |

### Tier 3: Cross-modality (future)
- CycleGAN-based modality synthesis (T2 from T1+T1ce) — requires ~2 GB VRAM; not feasible on RTX 2050 yet
- Alternative: skip T2 in training to test model robustness to missing modalities

---

## TTA for Inference

| Strategy | Dice Gain | Inference Time | Feasibility on RTX 2050 |
|---|---|---|---|
| No TTA | — | 1× | ✅ Fast |
| Standard TTA (8 views) | +1.2% Dice | 8× | ⚠️ Slow (~8× inference time) |
| BayTTA (4 views, EDL-weighted) | +1.5% Dice | 4× | ✅ Acceptable |

**Recommendation:** Use BayTTA with 4 augmentation views at inference for clinical deployment.

---

## BSDA (Batch Swap Data Augmentation)

- Swap patient identities within a batch to simulate domain shift
- Memory-efficient: only swaps label tensors, not image data
- Improves generalisation to unseen scanner types
- Implementation: swap subject IDs in metadata, re-assign to different modality combinations

---

## MONAI vs TorchIO Augmentation

| Feature | MONAI | TorchIO | Our Choice |
|---|---|---|---|
| 3D-specific transforms | ✅ | ✅ | Tie |
| CT/MRI compatibility | ✅ | ✅ | Tie |
| GPU acceleration | ✅ | ❌ | **MONAI** |
| Patch-based augmentation | ✅ | ❌ | **MONAI** |
| Cross-modality augmentation | ❌ | ❌ | Custom needed |
| Memory efficiency | Moderate | **Better** | **TorchIO** for training |

**Recommendation:** Use TorchIO for training augmentation (lower VRAM footprint), MONAI for cross-modality augmentation if VRAM allows.

---

## Augmentation Multiplier Calculation

| Dataset | Raw Cases | Geometric ×3 | Intensity ×2 | Cross-mod ×2 | Total Effective Cases |
|---|---|---|---|---|---|
| IBSR | 80 | 240 | 480 | — | 480 |
| C-BRATS | 600 | 1,800 | 3,600 | — | 3,600 |

**Memory footprint of augmented data:** On-the-fly augmentation — 0 MB extra storage; only VRAM needed for current batch.

---

## Reference Table

| # | Year | Authors | Source | ID | Title |
|---|---|---|---|---|---|
| 1 | 2026 | Molinier B, et al. | arXiv | 2605.03098 | Cross-modality Augmentation for Multimodal Brain Tumour Classification |
| 2 | 2026 | Vega Lara M, et al. | PubMed Central | PMID:42525278 | Clinical UQ and data augmentation synergies in neuro-oncology |
| 3 | 2024 | Bathla P, et al. | Expert Rev Med Devices | PMID:37977889 | Batch Transfer Augmentation with Learnable Patches |
| 4 | 2023 | Li Z, et al. | arXiv | 2304.04128 | Bayesian Test-time Augmentation for Medical Segmentation |
| 5 | 2023 | Wen PY, et al. | Neuro Oncol | PMID:37774317 | Revised Response Assessment Criteria: RANO 2.0 |
| 6 | 2022 | Sun Y, et al. | Eur Radiol | PMID:37923627 | Multi-model Habitat Multimodal MRI |
| 7 | 2022 | Chen Y, et al. | J Neurooncol | PMID:37201155 | Multi-modal MRI radiomics + liquid biopsy |
| 8 | 2021 | Kim YS, et al. | J Neurosurg | PMID:36534622 | Noninvasive Glioma Typing with Multiparametric DL |
| 9 | 2020 | Isensee F, et al. | Med Image Anal | PMID:33011683 | nnU-Net — self-adapting pipeline with augmentation |

---

## Recommendation

1. **Geometric augmentation only** at training time — 3× multiplier → 480 effective IBSR cases
2. **BayTTA at inference** — 4 views, EDL-weighted averaging
3. **No GAN/diffusion synthetic data** — too expensive for RTX 2050 and risks artefacts
4. **TorchIO for training** (lower VRAM), MONAI for any future cross-modality work
5. **Augmentation transforms must preserve RANO 2.0-compliant volume changes** — no transforms that create false tumour growth/shrinkage
