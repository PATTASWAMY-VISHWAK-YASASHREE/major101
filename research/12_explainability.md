# Category 12: Explainability / Interpretability

**Status: 🔬 RESEARCH ONLY — no implementation yet**

> **Purpose:** Understand how to explain model decisions for clinical trust.
> Not a priority for this research prototype, but relevant for any clinical deployment.

---

## 12.1 Why It Matters

A clinician will not trust a black-box AI that says "this is Grade IV glioblastoma"
without being able to see *why*. Explainability tools show which voxels contributed
to the prediction.

---

## 12.2 Main Techniques

### 12.2.1 Grad-CAM (Graduated Class Activation Mapping)

```
Grad-CAM:
1. Get gradient of target class w.r.t. last conv layer
2. Compute importance weights per channel
3. Weighted sum of channel maps → heatmap
4. Overlay heatmap on original volume

Result: 3D heatmap showing which voxels the model focused on.
```

**Implementation:** `torchcam` library (Grad-CAM, Grad-CAM++, Score-CAM)

**VRAM cost:** Negligible — runs on already-computed activations.

**Interpretation:** "The model flagged these tumour-enhancing regions as Grade IV."

### 12.2.2 SHAP (SHapley Additive exPlanations)

```
SHAP:
1. For each feature (voxel), compute how often it changes the prediction
2. Sum over all feature contributions → per-voxel SHAP values
3. Visualise SHAP values as a heatmap

Result: Per-voxel importance scores (positive = increases Grade IV likelihood).
```

**VRAM cost:** Moderate — SHAP requires many model forward passes per test case.
Not practical for 240³ volumes. Use on 64³ patches only.

### 12.2.3 Attention Visualization

```
Attention U-Net attention gates:
1. Extract attention gate outputs at each skip connection
2. Visualise as 3D maps

Result: "The model learned to suppress brain background and focus on tumour."
```

**VRAM cost:** Negligible if Attention U-Net architecture is used.

---

## 12.3 Decision for Our Project

| Technique | When to add | Effort |
|---|---|---|
| **Grad-CAM** | After baseline model works | Low — 20 lines of code |
| **SHAP** | For publication / clinical validation | Medium — takes time per test case |
| **Attention gates** | If using Attention U-Net architecture | Low — built in |

**Add explainability when:** The model is working well and you want to validate
that it is attending to tumour regions, not skull or background artefacts.

**For this prototype:** Add Grad-CAM in a follow-up pass. It is the easiest to add
and provides the most clinical value.
