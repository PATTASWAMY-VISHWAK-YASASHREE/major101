# Category 13: Uncertainty Quantification & Calibration

**Status: 🔬 RESEARCH ONLY — no implementation yet**

> **Purpose:** Understand how to quantify model confidence. Critical for clinical AI
> where wrong predictions have high stakes.

---

## 13.1 Why It Matters

A model that predicts "Grade IV" with 99% confidence is different from one that
predicts it with 55% confidence. Clinicians need to know when the model is unsure.

**Two types of uncertainty:**
- **Aleatoric:** Uncertainty from data noise (sensor noise, artefacts) — irreducible
- **Epistemic:** Uncertainty from model ignorance (not seen during training) — reducible

---

## 13.2 Bayesian Deep Learning (McDNN / Dropout MC)

```
Monte Carlo Dropout (Gal et al. 2017):
1. Keep dropout ON during inference (not just training)
2. Run N forward passes (N=50-100) with different dropout masks
3. Compute mean prediction = model's best guess
4. Compute std dev = epistemic uncertainty

Result: For each test case, you get:
  - Most likely class (mean of N predictions)
  - Uncertainty (std dev across N predictions)
```

**VRAM cost:** N forward passes × VRAM per pass. With N=10, it's 10× the inference cost.
For 64³ patches, this is acceptable.

**Implementation:** `torch.nn.Dropout(p=0.2)` during training AND inference.
Run `mc_dropout_inference(model, input, n_samples=10)`.

---

## 13.3 Deep Ensembles

```
Deep Ensemble (Lakshminarayanan et al. 2017):
1. Train N independent models (N=3-5) with different random initialisations
2. Aggregate predictions: mean = prediction, variance = uncertainty

VRAM: N× the model memory. For N=3, 3× the VRAM.
```

**For 4GB VRAM:** Use N=2 ensembles max. Or use model distillation to compress
the ensemble into a single model.

---

## 13.4 Temperature Scaling (Calibration)

```
Temperature scaling (Guo et al. 2017):
1. After training, fit a single scalar T on validation logits:
   calibrated_logits = raw_logits / T
2. Optimize T to maximise log-likelihood on validation set
3. Apply T at inference time

Result: Calibration curve — model confidence matches actual accuracy.
```

**VRAM:** Negligible — one-time fitting step on validation set.

---

## 13.5 Decision for Our Project

| Technique | Effort | VRAM | When to add |
|---|---|---|---|
| **MC Dropout** | Low | Low (10× inference) | After baseline works |
| **Deep Ensembles** | Medium | High (N× model) | Only if uncertainty is critical |
| **Temperature Scaling** | Very low | Negligible | ✅ Add immediately after training |

**Recommendation:** Add **temperature scaling** and **MC dropout inference** as part of
the evaluation phase. They are cheap to add and provide clinical confidence metrics.
