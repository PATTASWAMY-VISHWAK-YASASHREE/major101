# Category 10: Self-Supervised / Semi-Supervised / Few-Shot Learning

**Status: 🔬 RESEARCH ONLY — no implementation yet**

> **Purpose:** Understand strategies for training when labelled data is scarce.
> Relevant for the CT branch (only 80 labelled IBSR cases).

---

## 10.1 The Problem

| Data | Cases | Labels | Issue |
|---|---|---|---|
| BraTS MRI | 2,000 | 4-class labels | Plenty of data, but MRI-only |
| IBSR paired | 80 | 4-class labels | Too few for robust CT training |

**Core tension:** We need a CT encoder, but we only have 80 labelled CT cases.
Standard supervised learning on 80 cases risks overfitting.

---

## 10.2 Self-Supervised Pretraining (SSL)

**Concept:** Train the CT encoder on unlabelled data using a pretext task, then
fine-tune on the 80 labelled cases.

### 10.2.1 Pretext Tasks for CT

| Pretext task | Description | CT-specific? |
|---|---|---|
| **Rotation prediction** | Rotate volume, classify rotation angle | ✅ Works on CT |
| **Jigsaw puzzles** | Shuffle patches, classify the shuffle | ✅ Works on CT |
| **Contrastive learning (MoCo/SimCLR)** | Find similar views of same volume | ✅ Works on CT |
| **Masked image modelling (MAE)** | Mask random voxels, reconstruct them | ✅ Works on CT |
| **View matching** | Predict if two views are from the same patient | ✅ Works on CT |

**Recommended:** **Masked Autoencoder (MAE)** on CT volumes. Mask 75% of voxels,
train the encoder to reconstruct them. The encoder learns structural priors about
brain anatomy from unlabelled CT data.

### 10.2.2 CT Data Sources for SSL (unlabelled)

We don't need labels for SSL pretraining. Any CT brain scans work:

| Source | Type | Access | Volume |
|---|---|---|---|
| **TCIA CT collections** | Public, multi-study | Open | Hundreds of CTs |
| **IBSR CTs** | Already downloaded | Labelled | 80 cases |
| **Hospital CTs** | Local | Unlabelled | Unknown |

**Strategy:** Pretrain CT encoder on all available unlabelled CTs (TCIA + IBSR) using
MAE, then fine-tune on IBSR labelled cases.

---

## 10.3 Semi-Supervised Learning (Semi-SL)

**Concept:** Use a small set of labelled cases (IBSR, 80) plus a large set of unlabelled
cases (TCIA, hundreds) to train jointly.

### 10.3.1 Common Approaches

| Method | How it works |
|---|---|
| **Pseudo-labeling** | Train on labelled data, predict labels for unlabelled data, add high-confidence predictions to training |
| **Consistency regularization** | Apply different augmentations to the same unlabelled volume, penalise inconsistent predictions |
| **Mean Teacher** | Maintain a slowly-updated teacher model that provides targets for the student |

**Recommended:** **Pseudo-labeling** on the CT branch. Train CT encoder on IBSR labels,
predict pseudo-labels for TCIA CTs, add them to the training pool iteratively.

---

## 10.4 Few-Shot Learning (FSL)

**Concept:** Learn a metric space where classes cluster together, then classify by
nearest-neighbour in that space.

| Method | Description |
|---|---|
| **Prototypical Networks** | Compute a prototype (mean) for each class, classify by nearest prototype |
| **Relation Networks** | Train a relation network to score the similarity between a query and a support set |
| **Meta-learning (MAML)** | Train to adapt to new classes with few examples |

**Verdict for our project:** FSL is too complex for this project. Semi-SL with
pseudo-labeling is simpler and likely more effective with 80 labelled cases.

---

## 10.5 Decision for Our Project

| Phase | Approach | Data |
|---|---|---|
| **Phase 1 (CT encoder)** | SSL pretraining (MAE) + fine-tune | All TCIA CTs (pretrain) → IBSR CTs (fine-tune) |
| **Phase 2 (fusion)** | Standard supervised | IBSR paired MRI+CT (80 cases) |
| **Phase 3 (optional)** | Semi-SL with pseudo-labels | TCIA unlabelled CTs |

**Add SSL when:** The CT encoder shows signs of overfitting on 80 cases.
Skip SSL when: 80 cases with data augmentation already gives stable training.
