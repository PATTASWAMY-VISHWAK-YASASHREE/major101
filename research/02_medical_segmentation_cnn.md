# Category 2: Medical Image Segmentation CNNs (U-Net Family)

**Status: 🔬 RESEARCH ONLY — no implementation yet**

> **Purpose:** Understand the dominant segmentation architectures in medical imaging,
> which inform our classification architecture choices and can be adapted for
> per-modality feature extraction.

---

## 2.1 Why It Matters

Brain tumour segmentation (identifying tumour boundaries in MRI/CT) is the most-studied
task in neuro-oncology AI. While our project is classification-focused, segmentation
architectures share the same encoder design and provide the best-proven architectural
patterns for medical volumetric imaging.

---

## 2.2 U-Net (Ronneberger et al. 2015)

The seminal architecture that started it all:

```
┌───────────────────────────────────────┐
│  U-Net Encoder-Decoder Structure        │
│                                       │
│  Encoder (left):                        │
│  Conv3d → BN → ReLU → MaxPool3d × 4  │
│  Features: 64 → 128 → 256 → 512       │
│                                       │
│  Bottleneck:                            │
│  Conv3d → BN → ReLU (512 channels)    │
│                                       │
│  Decoder (right):                       │
│  UpConv3d → Concat(skip) → Conv3d     │
│  Features: 512 → 256 → 128 → 64       │
│                                       │
│  Output: 64 → N-class convolution     │
└───────────────────────────────────────┘
```

**Key design:** Skip connections from encoder to decoder preserve fine-grained spatial
information that would otherwise be lost during pooling.

**Why relevant to us:** U-Net's encoder path is exactly what we need as a per-modality
feature extractor. We drop the decoder entirely and use the bottleneck features.

**VRAM:** Full U-Net with 240³ input is too large for 4GB VRAM. Using 64³ patches
reduces VRAM to ~60-80 MB for a ResNet-derived U-Net encoder.

---

## 2.3 U-Net++ (Zhou et al. 2018)

```
U-Net++ adds dense skip connections:
  Skip connections go from EVERY encoder layer to EVERY deeper decoder layer.
  → More gradient paths, faster convergence
```

**Advantage:** Better gradient flow than standard U-Net. Useful for small datasets.

**VRAM:** Similar to U-Net, more activations due to additional skip connections.

---

## 2.4 Attention U-Net (Oktay et al. 2018)

```
Attention gate inserted at each skip connection:
  Input: Skip connection tensor
  Output: Attention-weighted skip tensor
  → "Learns to focus on tumour regions, suppress background"
```

**Advantage:** Attention gates act as a soft ROI crop. The network learns which voxels
to propagate through skip connections.

**Why relevant to us:** If we wanted per-modality attention during fusion, Attention
U-Net provides a tested pattern.

---

## 2.5 V-Net (Milletari et al. 2016)

```
V-Net = U-Net adapted for 3D:
  - Uses residual blocks instead of plain convs in encoder/decoder
  - Dice loss instead of cross-entropy (handles class imbalance in segmentation)
  - 3D convolutions throughout
```

**Why relevant:** V-Net is the most-cited 3D medical segmentation architecture.
Many BraTS papers use V-Net or a V-Net variant as their baseline.

**Key contribution:** Dice loss for segmentation, which is directly applicable to
our fusion classifier's output layer (we can use Dice or Focal loss for the 4-class
output).

---

## 2.6 nnU-Net (Isensee et al. 2021)

```
nnU-Net = U-Net with automatic configuration:
  - Automatically determines optimal:
    * Patch size (based on data statistics)
    * Network depth (based on image size)
    * Normalisation strategy
  - Provides a single config that works well across many datasets
```

**Why it matters:** nnU-Net won BraTS 2021-2023 with minimal hyperparameter tuning.
It is the current SOTA baseline for BraTS segmentation.

**Key insight for us:** nnU-Net's automatic patch-size selection is relevant to our
patch-based training strategy. It validates the approach of cropping patches rather
than training on full volumes.

**VRAM:** nnU-Net is designed to fit in GPU memory automatically. On 4GB VRAM, it
will select smaller patch sizes (~64³-96³) by default.

---

## 2.7 Comparison for Our Project

| Architecture | Primary use | VRAM friendly | Adaptation for classification |
|---|---|---|---|
| **U-Net** | Segmentation | ✅ with patches | Use encoder only as feature extractor |
| **U-Net++** | Segmentation | ✅ with patches | Dense skip connections help gradient flow |
| **Attention U-Net** | Segmentation | ✅ with patches | Attention gates useful for fusion |
| **V-Net** | Segmentation | ✅ with patches | Residual blocks, Dice loss design |
| **nnU-Net** | Segmentation | ✅ auto-config | Best starting point; auto-configures for hardware |

**Recommendation:** Use **V-Net's encoder** (ResNet3D-derived) as the per-modality backbone,
and consider **Attention U-Net-style gates** at the fusion head for modality-specific
feature weighting.

---

## 2.8 PubMed References

All segmentation architectures above are foundational medical imaging papers (not
PubMed-indexed as clinical trials). The nnU-Net paper:

- Isenner F, Jaeger PF, Kalmar B, et al. *nnU-Net: a self-configuring method for deep
  learning-based biomedical image segmentation.* Nature Methods. 2021;18(2):203-211.
  DOI: 10.1038/s41592-020-01008-z
