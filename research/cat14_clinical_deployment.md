# Category 14: Clinical Deployment and Real-World Validation

**Status: ✅ RESEARCH COMPLETE**

---

## 14.1 The Gap: Research → Clinical Use

| Research metric | Clinical requirement |
|---|---|
| Accuracy on BraTS test set | Performance on your hospital's MRI/CT scanner |
| 4-class accuracy | Confidence intervals + per-class sensitivity/specificity |
| Single-timepoint | Works across multiple scan sessions |
| IBSR dataset | Works on any patient's data |

---

## 14.2 Deployment Pipeline

```
Step 1: Baseline works (this project)
  → 4-class accuracy > 70%, macro-F1 > 0.6

Step 2: Cross-validation on IBSR
  → 5-fold CV, report per-class F1 with 95% CI

Step 3: Add calibration
  → Temperature scaling, reliability diagrams

Step 4: Add explainability
  → Grad-CAM attention maps

Step 5: Hospital pilot
  → Test on 50 real hospital CT+MRI cases

Step 6: Regulatory
  → FDA 510(k), CE marking (requires clinical trial data)
```

---

## 14.3 Our Current Position

| Step | Status |
|---|---|
| 1. Baseline works | 🚧 NOT YET — code not written |
| 2. Cross-validation | ⏳ Future |
| 3. Calibration | ⏳ Future (research/13 covers this) |
| 4. Explainability | ⏳ Future (research/12 covers this) |
| 5. Hospital pilot | ⏳ Future |
| 6. Regulatory | ⏳ Future |

**Key reality:** Steps 5-6 require IRB, hospital data sharing agreements, and clinical trial budget. Out of scope for this prototype.

---

## 14.4 What This Prototype Proves

This project is a **proof-of-concept**:
- "Can CT+MRI fusion outperform MRI-only for WHO grade classification?"
- "What is the marginal CT contribution?"

If Step 1 achieves >70% accuracy and CT contributes >3%, the approach is validated for further investment.
