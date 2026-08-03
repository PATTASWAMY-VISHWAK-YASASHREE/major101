# Category 1: Foundational 3D CNN Backbones for Volumetric Classification

**Status: 🔬 RESEARCH ONLY — no implementation yet**

> **Purpose:** Identify which 3D CNN architectures are viable encoders for our per-modality
> branch, given RTX 2050 (4GB VRAM) constraints.

---

## 1.1 Why It Matters

Every modality branch (CT encoder, T1 encoder, T1ce encoder, T2 encoder, FLAIR encoder)
needs a 3D CNN backbone. The choice of backbone determines:
- Parameter count → VRAM usage
- Depth → accuracy vs. trainability
- Feature richness → what the fusion head sees

---

## 1.2 Candidate Architectures

### ResNet3D (He et al. 2016 — adapted from 2D)

```
ResNet3D-18:  5-10M params,  2 residual blocks/stage
ResNet3D-34:  20M params,   3 residual blocks/stage
ResNet3D-50:  25M params,   4 residual blocks/stage
```

**Why ResNet:** The dominant backbone in medical imaging deep learning.
Almost every BraTS paper (see Category 4) uses ResNet3D or a ResNet-derived variant.

**VRAM for ResNet3D-18 with 64³ patch:** ~50 MB total (model + activations + gradients)
✅ Fits comfortably in 4GB VRAM.

**Key design:** Residual blocks with bottleneck structure. For our project, ResNet3D-18
is the right default — small, well-understood, proven on BraTS.

### DenseNet3D (Huang et al. 2017)

```
DenseNet3D-121:  8-12M params, feature reuse across layers
```

**Advantage:** Dense connections mean features propagate from every layer to every deeper
layer → more parameters-efficient than ResNet for the same depth.

**VRAM:** Similar to ResNet3D-18. May converge faster due to gradient flow.

**Use case:** When you want more feature richness per parameter count.

### EfficientNet3D (Tan & Le 2019 — MBConv blocks, compound scaling)

```
EfficientNet-B0 (3D):  5-7M params, MBConv bottleneck blocks
```

**Advantage:** Compound scaling reduces params while maintaining accuracy.
Fewer FLOPs → faster training on 4GB VRAM.

**Trade-off:** MBConv depthwise separable convolutions are less well-tuned for 3D than
standard convs. Requires more careful hyperparameter search.

### ConvNeXt3D (Liu et al. 2022)

```
ConvNeXt-Base (3D):  15-25M params, Vision Transformer–inspired conv net
```

**Advantage:** Pure convolutional architecture that matches ViT accuracy on 2D.
Modern design with large kernels, layer norm, and stochastic depth.

**VRAM:** ConvNeXt-Base is larger than ResNet3D-18. May not fit cleanly in 4GB.

**Use case:** When you want a modern backbone but don't have ViT training expertise.

---

## 1.3 Comparison for Our Project

| Architecture | Params | VRAM (64³ patch) | Accuracy potential | Verdict |
|---|---|---|---|---|
| **ResNet3D-18** | 5-10M | ~50 MB | Medium-High | ✅ Default choice |
| **ResNet3D-34** | ~20M | ~100 MB | High | ⚠️ Feasible but tighter |
| **DenseNet3D-121** | 8-12M | ~60 MB | Medium-High | ✅ Good alternative |
| **EfficientNet-B0** | 5-7M | ~45 MB | Medium | ✅ Fast training option |
| **ConvNeXt-Base** | 15-25M | ~120 MB | High | ❌ Too large for 4GB |

**Recommendation:** Start with **ResNet3D-18 (32 base channels)** as the default backbone.
It is the most-used backbone in BraTS literature, fits easily in 4GB VRAM, and is the
smallest viable starting point.

---

## 1.4 Implementation Notes

- Use `torch.nn.Conv3d` with kernel sizes 3×3×3 and padding 1 for spatial preservation.
- Use `torch.nn.BatchNorm3d` or `torch.nn.InstanceNorm3d` (InstanceNorm is preferred in
  medical imaging because it is contrast-invariant).
- Use `torch.nn.ReLU` or `torch.nn.LeakyReLU(0.01)`.
- Use `torch.nn.MaxPool3d` with kernel 2×2×2 and stride 2 for downsampling.
- Input tensor: `(C, D, H, W)` where C is the number of modalities (1 for per-modality encoders).
- Output tensor: flattened feature vector of size ~128-256 for fusion.

---

## 1.5 PubMed References

No dedicated PubMed references for generic CNN backbones (these are CS papers).
ResNet, DenseNet, EfficientNet, and ConvNeXt are foundational computer vision architectures
adapted to 3D medical imaging by the BraTS community.
