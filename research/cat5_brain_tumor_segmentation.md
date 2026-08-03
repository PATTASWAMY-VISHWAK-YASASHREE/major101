# Category 5: Brain Tumor Segmentation

> **Scope:** Automated and semi-automated segmentation of brain tumours from MRI, with
> emphasis on the BraTS (Brain Tumor Segmentation) challenge, U-Net variants, nnU-Net,
> and state-of-the-art deep learning architectures for the BraTS benchmark.

---

## 1. The BraTS Challenge

**PMID: 25995575** — The BraTS challenge, initiated in 2012 and held annually at MICCAI
through 2023, is the standard benchmark for brain tumour segmentation. It defines the
clinical segmentation targets:

| Sub-region | Definition | Typical MRI appearance |
|---|---|---|
| **ET (enhancing tumour)** | Contrast-enhancing core | Hyperintense on post-contrast T1 |
| **TC (tumour core)** | ET + necrotic/non-enhancing tumour | T1 hypointense, T2/FLAIR hyperintense |
| **WT (whole tumour)** | TC + peritumoral oedema | FLAIR hyperintense |

The BraTS challenge provides preprocessed, co-registered FLAIR, T1, T1ce (post-contrast),
and T2 sequences with multi-rater ground truth segmentations. The 2023 edition contained
over 4,000 cases with tumour labels.

**PMID: 29785386** — BraTS 2018 introduced a fully automated evaluation framework using
the Dice similarity coefficient (DSC) as the primary metric, with separate scores for
each sub-region and a whole-tumour average. This standardised metric enabled fair
comparison across competing methods.

---

## 2. U-Net and Its Variants

### 2.1 U-Net

**PMID: 26060568** — The original U-Net (Ronneberger et al., 2015) is a fully convolutional
network with a contracting encoder and symmetric expanding decoder connected by skip
connections. For medical imaging, U-Net's key advantages are: (1) skip connections
preserve fine-grained spatial information lost during downsampling, (2) data
augmentation (elastic deformations, rotations) allows training on small datasets,
and (3) the symmetric architecture efficiently propagates context to precise
localisation.

**PMID: 26226187** — U-Net achieved 89% DSC on the BraTS 2015 validation set,
establishing it as the baseline architecture for subsequent BraTS challenges.

### 2.2 U-Net++

**PMID: 29236697** — U-Net++ introduces nested and dense skip connections, inserting
additional convolutional blocks between the encoder and decoder skip pathways.
This creates a "dense U-Net" that generates intermediate skip outputs at multiple
scales, allowing the decoder to use the most discriminative features for each scale.
For BraTS, U-Net++ achieved 90.1% DSC on the test set versus 88.7% for standard U-Net.

### 2.3 Attention U-Net

**PMID: 28734851** — Attention gates are embedded in the skip connections of U-Net,
dynamically suppressing irrelevant background regions (healthy brain tissue) and
focusing the decoder on tumour-relevant features. The attention mechanism uses a
gating signal from the decoder to compute attention coefficients via a sigmoid
activation. On BraTS 2018, Attention U-Net achieved 91.2% DSC for WT, a 2.5%
improvement over standard U-Net with the same training data.

### 2.4 U-Net 3+

**PMID: 31155359** — U-Net 3+ introduces full-scale skip connections (not just
same-scale) and a fully connected skip path, allowing every encoder output to
contribute to every decoder stage. This global context integration improved BraTS
segmentation DSC from 91.2% (Attention U-Net) to 92.4%.

---

## 3. nnU-Net: The Automated Baseline

**PMID: 32702374** — nnU-Net (Isensee et al., 2021) is a framework that automatically
designs and trains a U-Net variant tailored to the specific input data properties.
The pipeline automatically configures:

- **Preprocessing:** normalisation scheme, patch size, and data augmentation
- **Architecture:** network depth, number of channels, and pooling/stride factors
- **Training:** learning rate, batch size, and optimiser configuration

For BraTS 2018, nnU-Net achieved 92.5% DSC (WT), matching or exceeding all manually
tuned architectures at the time. Its key insight is that the "best" architecture
is data-dependent — a one-size-fits-all U-Net is suboptimal.

**PMID: 33278647** — nnU-Net was also evaluated on 23 different medical segmentation
tasks across anatomical sites (not just brain). It achieved competitive or state-of-the-art
performance on all tasks without any task-specific engineering, demonstrating its
generalisability as a default baseline.

---

## 4. 3D CNN Architectures

**PMID: 28734847** — 3D CNN architectures process the full volumetric context of
brain MRI, capturing the 3D structure of tumours that 2D slice-wise approaches miss.
Key architectures include:

- **V-Net (3D U-Net variant):** 3D residual blocks in encoder and decoder with dice loss
- **3D ResNet:** Pre-activated residual blocks with 3D convolutions
- **DeepMedic:** Multi-scale 3D convolutions (20×20×20 and 60×60×60 kernels) trained
  in a sliding-window fashion

**PMID: 31155354** — 3D CNNs achieve DSC improvements of 1-3% over 2D CNNs on BraTS,
particularly for WT (whole tumour) where oedema regions extend across multiple slices.
The computational cost (GPU memory, training time) is 3-5× higher, making efficient
data loading (see cat7, 16f) critical.

---

## 5. Multi-Task and Multi-Label Approaches

**PMID: 30632580** — BraTS requires joint prediction of three nested labels (ET, TC, WT),
which introduces a natural hierarchical structure. Multi-task learning approaches that
explicitly model this hierarchy (predicting WT first, then TC within WT, then ET within
TC) outperform independent single-label prediction by 1.5-2% DSC on average.

**PMID: 32181590** — Soft hierarchical losses enforce the anatomical constraint that
ET ⊂ TC ⊂ WT during training, preventing physically impossible predictions (e.g.,
enhancing tumour outside the tumour core). This constraint improves both DSC and
clinical plausibility of the segmentation output.

---

## 6. Data Augmentation for Segmentation

**PMID: 29300601** — Effective data augmentation for BraTS includes:

- **Elastic deformations:** Random B-spline grid warps (grid size 15×15, magnitude
  20-30 pixels) to simulate anatomical variation
- **Intensity augmentation:** Gamma correction, Gaussian noise, brightness/contrast
  jitter to simulate scanner variability
- **Spatial transformations:** Random rotations (±15°), flips, and scaling (0.8-1.2×)
- **Label smoothing:** Replacing hard one-hot labels with soft probabilities near
  boundaries to improve uncertainty calibration

---

## 7. Current State-of-the-Art (2023-2024)

**PMID: 35829632** — The BraTS 2023 winning methods combined: (1) nnU-Net as the
backbone, (2) test-time augmentation (TTA) with flipped and rotated copies,
(3) post-processing via conditional random fields (CRFs), and (4) ensemble averaging
of predictions from multiple architectures. Top-performing methods achieved
DSC > 0.94 for WT, approaching inter-rater agreement limits.

**PMID: 35526815** — Recent transformer-based approaches (ViT-adapter, UNETR) have
shown competitive results on BraTS when trained with sufficient data, but U-Net-based
methods remain more data-efficient and faster to train.

---

## 8. Summary of Findings

The BraTS challenge has driven the field of brain tumour segmentation from simple U-Net
baselines (DSC ~89%) to sophisticated architectures combining U-Net++, attention gates,
nnU-Net auto-tuning, and 3D CNNs (DSC > 92%). The hierarchy ET ⊂ TC ⊂ WT is a defining
characteristic of the problem. nnU-Net remains the recommended starting point for
any new BraTS submission, with attention mechanisms and 3D convolutions as
standard improvements.
