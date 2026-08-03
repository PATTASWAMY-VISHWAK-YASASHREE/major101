# Category 3: Transformer & Hybrid Segmentation Architectures

**Status: 🔬 RESEARCH ONLY — no implementation yet**

> **Purpose:** Document transformer-based architectures in medical imaging. Important
> for understanding what the field is moving toward, though not our immediate target.

---

## 3.1 Why It Matters

Vision Transformers (ViT) have replaced CNNs as the SOTA in 2D computer vision.
Adaptations to 3D medical imaging are actively researched and offer better long-range
spatial modelling than CNNs — relevant for tumour context modelling.

---

## 3.2 Vision Transformer (ViT) Basics

```
ViT architecture:
  1. Patch embedding: Split 3D volume into N non-overlapping patches
  2. Linear projection: Each patch → token embedding
  3. Positional embedding: Added to each token
  4. Transformer encoder: Self-attention over all tokens
  5. Classification token: [CLS] token for classification

3D ViT patch size: 8×8×8 or 16×16×16 (typical)
```

**Advantage over CNN:** Self-attention captures relationships between distant patches
in a single step (global context). CNNs need many layers to cover long distances.

**VRAM concern:** ViT self-attention is O(N²) in sequence length N. For a 64³ volume
split into 8³ patches: N = 512 patches → 512² = 262K attention pairs → ~1 GB VRAM.

**Bottom line:** ViT is viable on 4GB VRAM but with smaller patch sizes than CNNs.

---

## 3.3 TransUNet (Chen et al. 2021)

```
TransUNet = CNN + Transformer:
  - CNN encoder (ResNet) for initial feature extraction
  - Transformer block (4-6 layers) on top of CNN features
  - CNN decoder (U-Net style) for output

Design rationale:
  - CNN captures local features efficiently (cheap compute)
  - Transformer captures long-range dependencies (expensive but powerful)
  - Only the bottleneck goes through the Transformer
```

**Why relevant to us:** TransUNet's hybrid design is the template for how to use
Transformers in medical imaging without VRAM explosion. The CNN encoder does the heavy
lifting; the Transformer is only on the bottleneck.

**VRAM for 64³ patch:** ~80-120 MB (similar to ResNet3D-18)

---

## 3.4 Swin-UNet (Cao et al. 2021)

```
Swin-UNet = Swin Transformer adapted to U-Net:
  - Swin Transformer uses shifted windows → O(N) attention instead of O(N²)
  - Hierarchical: patches start small, windows merge as depth increases
  - Compatible with U-Net skip connections

VRAM advantage: Swin's windowed attention reduces compute by ~50% vs ViT
```

**Why relevant:** Swin-UNet is the most VRAM-efficient transformer architecture for
medical imaging. It can fit 64³ patches in 4GB VRAM with room to spare.

---

## 3.5 UNETR (Hatamizadeh et al. 2021)

```
UNETR = Vision Transformer as backbone:
  - 3D ViT (patch size 16³) as encoder
  - CNN decoder (U-Net style) for output
  - No CNN pre-encoder — pure ViT from raw volume

Training: Pre-trained ViT weights (ImageNet) → fine-tuned on BraTS
```

**VRAM concern:** UNETR's full 3D ViT encoder is large (~86M params for ViT-Base).
It won't fit in 4GB VRAM with standard batch sizes.

**Verdict for us:** ❌ Too large. TransUNet and Swin-UNet are the more practical
transformer options.

---

## 3.6 Perceptual Transformer (PET) / Medical Vision Transformers

```
PET = Lightweight transformer designed for medical imaging:
  - Uses axial attention (1D attention in each axis separately)
  - Much cheaper than full 3D attention
  - Designed for large volumes without downsampling

VRAM: PET fits 240³ volumes in 4GB VRAM with axial attention
```

**Verdict for us:** Axial attention is interesting but adds complexity. Stick with
CNN backbone for now; consider axial attention if accuracy plateaus.

---

## 3.7 Recommendation for Our Project

| Architecture | VRAM (4GB) | Complexity | Accuracy potential | Verdict |
|---|---|---|---|---|
| **ResNet3D-18 (pure CNN)** | ✅ Fits | Low | Medium-High | ✅ Default |
| **TransUNet (CNN+ViT)** | ✅ Fits with 64³ | Medium | High | ⚠️ Future upgrade |
| **Swin-UNet** | ✅ Fits with 64³ | Medium | High | ⚠️ Future upgrade |
| **UNETR (pure ViT)** | ❌ Too large | High | High | ❌ Not viable |
| **PET (axial ViT)** | ⚠️ Borderline | High | Medium | ❌ Over-engineered |

**Decision:** Pure CNN (ResNet3D-18) for the initial model. If accuracy plateaus,
consider TransUNet or Swin-UNet as a fusion-head upgrade (replace the dense fusion
head with a 2-layer Transformer on the per-modality features).

---

## 3.8 Why Not Transformers for This Project (YAGNI)

1. **4GB VRAM constraint:** Transformers are memory-hungry. CNNs fit comfortably.
2. **Small dataset (80 IBSR cases):** Transformers benefit from large training data.
   CNNs train fine on small data with augmentation.
3. **Classification, not segmentation:** The main transformer benefits (long-range
   context for segmentation) are less relevant for 4-class classification.
4. **Late fusion is simple and effective:** Per-modality CNN + dense fusion head
   already achieves 93-96% accuracy in literature.

**Add Transformer when:** Accuracy plateaus below ~94% and you have more training data.
