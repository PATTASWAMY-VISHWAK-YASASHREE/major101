# Category 14: Survival Analysis / Prognosis Prediction

**Status: 🔬 RESEARCH ONLY — no implementation yet**

> **Purpose:** Understand how imaging features predict patient survival. A natural
> extension of tumour classification — from "what grade is this?" to "how long will
> this patient live?"

---

## 14.1 Why It Matters

Clinical value of AI goes beyond classification. Oncologists need prognosis:
- "What is the expected survival for this Grade IV glioblastoma?"
- "Will this patient respond to temozolomide?"
- "What is the recurrence risk after surgery?"

**Classification → Prognosis pipeline:**
```
4-class tumour grade → Survival prediction → Treatment recommendation
```

---

## 14.2 DeepSurv (Katzman et al. 2018)

```
DeepSurv: Neural Cox proportional hazards model
1. 3D CNN encodes MRI volume → embedding
2. Embedding → fully connected layer → linear predictor
3. Cox partial likelihood loss trains the model
4. Output: hazard ratio (relative risk) for each patient

Advantage: Handles censored data (patients who didn't die during study period).
```

**VRAM:** Same as the CNN encoder (~50 MB for ResNet3D-18 + 64³ patch).

**Key insight:** DeepSurv uses the same Cox loss as the classical Cox regression,
but with a neural network as the feature extractor instead of handcrafted features.

---

## 14.3 DeepHit (Lee et al. 2018)

```
DeepHit: Direct discrete-time survival distribution prediction
1. Output: survival probability at each time bin (e.g., P(T > 1mo), P(T > 3mo), ...)
2. Loss: minimises distance between predicted and true survival distributions
3. No proportional hazards assumption (unlike DeepSurv)

Advantage: Does not assume proportional hazards — better for complex clinical data.
```

**VRAM:** Same as DeepSurv.

---

## 14.4 Multimodal Survival Prediction

```
CT features ──→ embedding ──┐
                            ├──→ Cox loss → hazard ratio
MRI features ──→ embedding ─┘
Clinical features (age, KPS) ─→ concat ─┘
```

**Representative work:** Yang et al. 2023 (PMID 36870427) — radiomics-based GBM
prognosis. Kaur et al. 2025 (PMID 40541161) — AI for GBM classification, survival,
and diagnosis.

---

## 14.5 Decision for Our Project

| Extension | Effort | Data needed | Verdict |
|---|---|---|---|
| **DeepSurv on IBSR** | Medium | Survival times (available in IBSR) | ✅ Do this as extension |
| **DeepSurv on TCGA** | Medium | TCGA has survival data | ⚠️ TCGA has no MRI+CT pairs |
| **CT+MRI DeepSurv** | Medium | IBSR paired + survival times | ✅ Strong candidate |

**Add survival prediction when:** The 4-class classifier is working and you want
to extend the clinical relevance of the model. IBSR includes patient survival times
in its metadata — this data is ready to use.
