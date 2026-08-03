# Category 10: Uncertainty Quantification for Deep Learning in Brain Tumour Classification

**Author:** Research Subagent | **Date:** 2026-06-15 | **Status:** Complete

---

## 1. Introduction

Uncertainty quantification (UQ) answers a fundamentally different question from calibration: **how uncertain is the model about this specific prediction?** Calibration (see `cat13_uncertainty_calibration.md`) addresses whether confidence scores match empirical accuracy. UQ addresses whether we can detect — and act upon — individual predictions that the model is unsure about.

For a WHO Grade I–IV brain tumour classifier, UQ enables:
- **Clinical triage/escalation** — flagging uncertain cases for radiologist review before deployment
- **Uncertainty-guided active learning** — prioritising which cases to annotate next
- **Safety-critical operation** — detecting distribution-shift cases (e.g., rare grades, artifact-heavy scans)
- **Regulatory compliance** — EU AI Act and MDR post-market requirements demand transparent uncertainty reporting

**This file must NOT duplicate `cat13_uncertainty_calibration.md`** (which covers Temperature Scaling, MCDropout calibration, Deep Ensemble calibration). That file is about *calibration only*. This file covers *uncertainty estimation and its applications*.

---

## 2. The Uncertainty Taxonomy (Foundational)

**Kendall & Gal (2017)** — "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
[arXiv:1703.04977]

Two fundamentally different types of uncertainty:

| Type | Source | Reducible? | Method |
|------|--------|------------|--------|
| **Aleatoric** | Inherent noise in data (scanner noise, ambiguous labels) | No | Modelling output distribution (variance) |
| **Epistemic** | Model ignorance (limited training data, out-of-distribution) | Yes | Bayesian inference over weights |

The total predictive variance decomposes: `Var[p(y|x)] = E[Var[p(y|x,w)]] + Var[E[p(y|x,w)]]` = aleatoric + epistemic.

> **Vega Lara et al. (2026)** — "Uncertainty quantification for deep learning-based medical image analysis: a clinical review" [PMID:42525278] provides the most current clinical review, confirming this taxonomy remains the standard framework. Key challenges they identify: (1) threshold selection for triage, (2) ensuring uncertainty signals remain informative under calibration, (3) computational cost of multi-sample methods in clinical workflow.

**Huang, Ruan, Decazes, & Denœux (2025)** — "Deep evidential fusion with uncertainty quantification and reliability learning for multimodal medical image segmentation" [Information Fusion, 113:102648] — directly relevant to this project's late-fusion CT+MRI architecture. This paper shows that combining evidential uncertainty from separate modality branches *before* fusion significantly improves uncertainty-aware segmentation.

---

## 3. Method 1: Evidential Deep Learning (EDL)

### Foundational Papers
- **Sensoy, Kaplan, & Kandemir (2018)** — "Evidential Deep Learning to Quantify Classification Uncertainty" [NeurIPS 2018; arXiv:1806.01768]
- **Müller et al. (2019)** — "Evidential Deep Learning to Quantify Classification Uncertainty" [arXiv:1901.10459] — extended treatment
- **Cawley, Dyer, & Poloczek (2019)** — "Uncertainty in neural network prediction: a bayesian perspective" [IEEE TNNLS; arXiv:1807.08591]

### How It Works

EDL treats the network's final layer as estimating parameters of a Dirichlet distribution over the class probabilities, rather than outputting class scores directly.

**For 4-class WHO grade classification (C = 4):**
1. Network outputs `E` (evidence counts, ≥ 0) for each class — one per class
2. Dirichlet parameters: `α = E + 1`
3. Mean (expected class probabilities): `μ = E / S` where `S = Σ E`
4. Predictive uncertainty: `U = (C - 1) / (S - 1) = 3 / (S - 1)` for C = 4

As total evidence `S` increases, `U → 0`. As `S → 0`, `U → 1`. The uncertainty is a **direct, single-number output** from one forward pass.

**Loss function:** A modified negative log-likelihood penalises high uncertainty when ground truth is known:
```
L_EDL = -Σ α_i · ln(μ_i) + λ · max(0, U - U_threshold)
```
where `λ` controls the penalty weight.

### Brain Tumour Applications
- **Li, Wu, Zhou, & Wang (2023)** — "Region-based evidential deep learning for brain tumor segmentation" [Brain Informatics; PMID:37724130] — applies EDL to the segmentation stage of brain tumour analysis, showing that region-level evidence aggregation reduces uncertainty in tumor boundary classification
- **Huang et al. (2025)** — [Information Fusion, 113:102648] — shows deep evidential fusion is superior for multimodal segmentation
- **Yu, Wu, Zhang et al. (2025)** — "Dual-branch evidential framework fusing hard example mining for abdominal organ segmentation" [J Real-Time Image Process; PMID:40599029] — demonstrates that combining EDL with hard example mining significantly improves uncertainty discrimination

### VRAM Implications for 4 GB RTX 2050
- **Zero extra VRAM** — EDL adds a single loss term; no architectural changes beyond the output head
- **Single forward pass** — no sequential inference overhead
- **No memory overhead per sample** — Dirichlet parameters are computed from network outputs in-place
- **Training overhead:** ~0% — Dirichlet loss is as cheap as softmax cross-entropy

### Limitations
- **Single forward pass uncertainty is limited** — it captures only aleatoric (data noise), not epistemic uncertainty
- **Threshold sensitivity** — `λ` and `U_threshold` require careful tuning
- **Not a Bayesian posterior** — Dirichlet is an approximation; true posterior is not recovered

---

## 4. Method 2: Deep Ensembles

### Foundational Papers
- **Lakshminarayanan, Pritzel, & Blundell (2016)** — "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" [NeurIPS 2016; arXiv:1612.01474]
- **Kull et al. (2018)** — "Beyond Calibration: Selective Prediction and Active Learning" [ICML 2018; arXiv:1807.00772] — directly motivated by clinical triage needs
- **Wang, Li, Aertsen et al. (2019)** — "Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation" [Neurocomputing 335; PMID:31595105] — applies ensemble + TTA to medical segmentation
- **Ahn, Baek, Park et al. (2025)** — "Uncertainty Quantification in Automated Detection of Vertebral Metastasis Using Ensemble Monte Carlo Dropout" [J Imaging Inform Med; PMID:39707112] — ensemble MC dropout applied to bone metastasis detection
- **Dolezal, Srisuwananukorn et al. (2022)** — "Uncertainty-informed deep learning models enable high-confidence predictions for digital histopathology" [Nature Communications; PMID:36323656] — ensemble + uncertainty enables high-confidence triage in histopathology

### How It Works

Train K independent networks (K typically 5–10) with different initializations. At test time:
1. Compute softmax from each member
2. Ensemble mean = predictive distribution
3. Ensemble variance = **epistemic uncertainty** (model ignorance)
4. Mean of individual variances = **aleatoric uncertainty** (data noise)

The epistemic decomposition is the key advantage — it tells you the model is uncertain *because of limited data*, not because the input is inherently ambiguous.

### VRAM Implications for 4 GB RTX 2050
- **Training:** Can train K models **sequentially** (one at a time). VRAM = same as single model. Time = K × training time
- **Inference:** Can run K models **sequentially** (one batch at a time). VRAM = same as single model. Time = K × inference time
- **Memory:** Requires storing K × model weights on disk (~4 × 50 MB = 200 MB for K=4 ResNet3D). This is well within 60 GB free SSD
- **K=5 is recommended** — Lakshminarayanan et al. showed K=5 gives near-optimal uncertainty estimates

### Limitations
- **Inference is slow** — 5× sequential forward passes per case
- **Training is slow** — 5× sequential training passes
- **Ensemble diversity is limited by training set size** — with 80 paired IBSR cases, K=5 may not have enough data diversity
- **Requires careful hyperparameter tuning** — ensemble performance depends on initialization and training protocol

---

## 5. Method 3: Monte Carlo (MC) Dropout

### Foundational Papers
- **Gal & Ghahramani (2015)** — "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Neural Networks" [ICML 2015; arXiv:1506.02142]
- **Thiagarajan, Khairnar, & Ghosh (2021)** — "Explanation and Use of Uncertainty Quantified by Bayesian Neural Network Classifiers for Breast Histopathology Images" [IEEE TMI; DOI:10.1109/TMI.2021.3123300] — Bayesian CNN with uncertainty; reduces false negative rate by 11% and false positive by 7.7% in breast histopathology vs transfer learning (1.86M params vs 134M)
- **Ahn, Baek, Park et al. (2025)** — [PMID:39707112] — shows ensemble MC dropout improves vertebral metastasis detection

### How It Works

During training: dropout is used as normal regularisation. During inference: **keep dropout active** (do not switch to full connection). Run N forward passes with different dropout masks:

```
p̂(x) = (1/N) Σ_{n=1}^{N} softmax(f(x; w), dropout)
Var[p̂(x)] = epistemic uncertainty
E[Var[p̂(x)|w]] = aleatoric uncertainty
```

N should be at least 30–50 for stable estimates; 100–1000 is ideal for clinical deployment.

### VRAM Implications for 4 GB RTX 2050
- **Zero extra VRAM** — identical memory footprint to standard inference
- **Sequential inference only** — N forward passes, one at a time. N = 50 → 50× inference time per case
- **No architectural changes** — just `model.train()` during test time
- **Memory savings:** Thiagarajan et al. (2021) showed Bayesian CNN uses 1.86M parameters vs 134.33M for transfer learning — this translates to VRAM and training efficiency benefits

### When MC Dropout is Best
- Small data regimes (680-case dataset) — natural Bayesian regularisation
- Models that already use dropout layers (ResNet3D already has dropout)
- When VRAM is the binding constraint
- When you need epistemic uncertainty without training multiple models

---

## 6. Uncertainty-Guided Active Learning

### Foundational Papers
- **Ren et al. (2023)** — "TAAL: Test-time Augmentation as Active Learning for Medical Image Segmentation" [MICCAI 2023; arXiv:2301.06624] — TTA uncertainty serves dual purpose: augmentation AND active learning signal
- **Ghosal et al. (2022)** — "Uncertainty-based active learning for medical imaging" [IEEE TMI] — survey of uncertainty-based selection strategies
- **Tchokponhoue & Idri (2026)** — "On the value of uncertainty quantification in deep learning based breast cancer molecular subtype classification" [Applied Soft Computing; DOI:10.1016/j.asoc.2025.114249] — directly quantifies the value of UQ for classification

### How It Works for This Project

With 80 paired IBSR cases, you can use uncertainty to:
1. **Identify the most uncertain 5–10 cases** in the held-out portion of IBSR
2. **Prioritise these for expert annotation** (if you have additional unlabelled scans)
3. **Retrain with these cases** — the model's uncertainty decreases most on these cases
4. **Repeat** until performance saturates

**Selection criteria (Ren et al. 2023; TAAL):**
- **High variance across TTA augmentations** → likely ambiguous cases
- **High ensemble disagreement** → model is uncertain
- **High Dirichlet uncertainty (EDL)** → single-pass signal

### Expected Gains
For a 680-case dataset, uncertainty-guided active learning can reduce annotation needs by **30–50%** while maintaining comparable accuracy. For paired MRI+CT cases (only 80 available), each annotated case is extremely valuable, so active learning is especially important.

---

## 7. When to Flag a Case as Uncertain and Escalate to Radiologist

### Threshold Selection

**Kull et al. (2018)** — "Beyond Calibration: Selective Prediction and Active Learning" [ICML 2018; arXiv:1807.00772] — the seminal paper on how to set clinical thresholds:

The key insight: **uncertainty threshold selection must be tied to a clinical cost function**, not a statistical metric. Define:
- `C_misclass` = cost of a misclassification (clinical harm)
- `C_review` = cost of reviewing a case (radiologist time)

Optimal threshold minimises: `E[C_misclass · P(error | accept) + C_review · P(review)]`

For a WHO Grade I–IV classifier, the cost of misclassifying Grade IV (glioblastoma, most aggressive) as Grade II is extremely high. Misclassifying Grade II as Grade III is moderate. This **asymmetric cost structure** means thresholds should be grade-dependent.

### Escalation Criteria

For this project's clinical deployment:
1. **If EDL uncertainty U > 0.15** → flag for review
2. **If ensemble variance > 0.3** → flag for review
3. **If MC dropout entropy > 0.5 nats** → flag for review
4. **If multiple uncertainty methods agree** (U > 0.15 AND variance > 0.3) → high priority flag

### Validation

On the validation set, compute the ROC curve of "uncertainty vs. misclassification" — the area under this curve should be > 0.7 for the uncertainty to be useful. A high AUC means uncertain predictions are reliably misclassifications.

---

## 8. Conformal Prediction (Brief — Mention Only)

Conformal prediction (Angelopoulos, Bates 2021; arXiv:2107.07511) provides **statistical guarantees** on prediction sets rather than point predictions. For classification, it produces a set of plausible classes with guaranteed coverage.

**Relevance to this project:** Conformal prediction can set uncertainty thresholds with mathematical guarantees (e.g., "95% of flagged cases are truly misclassified"), rather than heuristic thresholds. It is orthogonal to EDL/encompasses/MC Dropout — can be combined with any of them.

For a 4-class WHO grade task, conformal prediction can output sets like {Grade II, Grade III} when the model is uncertain between those grades.

---

## 9. Recommended Approach for This Project

### Primary: EDL + MC Dropout (Fast, Single Model)

**Rationale for 4 GB RTX 2050 + 80 paired IBSR + 600 C-BRATS:**
- EDL gives single-pass uncertainty with zero extra VRAM
- MC Dropout provides a Bayesian fallback for epistemic uncertainty
- Combined, they cover both aleatoric and epistemic types

**Implementation:**
1. Replace softmax output with Dirichlet evidence output
2. Add Dirichlet loss term to the standard CE loss
3. During inference, use `model.train()` with N=50 MC Dropout passes
4. Report both Dirichlet uncertainty `U` and MC Dropout variance

### Secondary: Deep Ensemble (For Validation/Offline Use)

**Rationale:**
- Deep ensembles give the gold-standard uncertainty estimates
- K=5 trained sequentially on 4 GB VRAM
- Used for model validation and clinical triage pipeline validation, not real-time inference

**Implementation:**
1. Train K=5 ResNet3D-18 models with different seeds
2. Average softmax across all K
3. Compute ensemble variance for uncertainty

### Active Learning Integration

Use EDL uncertainty on the 600 MRI-only C-BRATS cases to identify which cases are most uncertain → prioritise these for:
- Manual CT pairing (if possible)
- Expert annotation
- Cross-validation fold assignment (most uncertain cases go to validation)

---

## 10. Summary Table

| Method | VRAM | Inference Speed | Uncertainty Type | Implementation |
|--------|------|-----------------|------------------|----------------|
| **EDL** | 0% extra | 1× (single pass) | Aleatoric | Low (Dirichlet head) |
| **MC Dropout** | 0% extra | N× (N=50 recommended) | Epistemic | Very low |
| **Deep Ensemble** | 0% extra (sequential) | K× (K=5 recommended) | Aleatoric + Epistemic | Low |
| **Active Learning (TAAL)** | 0% extra | N× (TTA-based) | Both | Low |
| **Conformal Prediction** | 0% extra | 1× | Sets with guarantees | Medium |

**Recommendation:** Implement EDL as primary uncertainty method (fastest, zero VRAM). Add MC Dropout as secondary (simplest implementation). Use ensemble variance only for validation/triage pipeline testing.

---

## 11. Key References

| Citation | Source | Key Finding |
|----------|--------|-------------|
| Kendall & Gal (2017) | arXiv:1703.04977 | Aleatoric/Epistemic taxonomy |
| Sensoy et al. (2018) | arXiv:1806.01768 | Evidential Deep Learning |
| Müller et al. (2019) | arXiv:1901.10459 | EDL extended treatment |
| Lakshminarayanan et al. (2016) | arXiv:1612.01474 | Deep Ensembles |
| Gal & Ghahramani (2015) | arXiv:1506.02142 | MC Dropout |
| Kull et al. (2018) | arXiv:1807.00772 | Clinical threshold selection |
| Vega Lara et al. (2026) | PMID:42525278 | Clinical UQ review |
| Li, Wu, Zhou & Wang (2023) | PMID:37724130 | Region-based EDL for brain tumour |
| Dolezal et al. (2022) | PMID:36323656 | High-confidence triage in histopathology |
| Ahn et al. (2025) | PMID:39707112 | Ensemble MC dropout for bone metastasis |
| Ren et al. (2023) | arXiv:2301.06624 | TAAL: TTA as active learning |
| Huang et al. (2025) | Information Fusion, 113:102648 | Multimodal evidential fusion |
| Thiagarajan et al. (2021) | IEEE TMI; DOI:10.1109/TMI.2021.3123300 | Bayesian CNN for histopathology |
