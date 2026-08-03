# Category 13: Transfer Learning from Natural Images

**Status: ✅ RESEARCH COMPLETE**

---

## 13.1 The Transfer Learning Problem

| Aspect | Natural images (ImageNet) | Medical images |
|---|---|---|
| Channels | RGB (3) | Grayscale (1) |
| Resolution | 224×224 (2D) | 256³ (3D) |
| Features | Edges, textures, colours | Intensity gradients, anatomical shapes |
| Dataset size | 14M images | Hundreds of volumes |

**Transfer from ImageNet ResNet to 3D medical ResNet is NOT straightforward.**
- Channel mismatch (3→1) — needs channel projection
- 2D→3D gap — different receptive field properties
- Feature space mismatch — ImageNet features don't transfer cleanly to volumetric data

---

## 13.2 What Actually Works

| Source | What to use | Transfer method |
|---|---|---|
| **ImageNet (2D)** | ResNet18/50 weights | ❌ Poor transfer to 3D |
| **C-BRATS (MRI)** | ResNet3D-18 weights | ✅ Domain-specific, best transfer |
| **TCIA CTs** | MAE pretraining | ✅ Self-supervised, no labels needed |
| **BraTS 2024** | ResNet3D-18 weights | ✅ Largest MRI brain tumour dataset |

**Our strategy:**
1. **MRI branch:** Fine-tune from C-BRATS pretrained weights (not ImageNet)
2. **CT branch:** MAE pretraining on TCIA CTs (no labels needed)
3. **Fusion head:** Train from scratch

---

## 13.3 Why Self-Supervised Beats ImageNet Transfer

```
ImageNet ResNet features → 3D medical images
  → "cat/airplane" features are irrelevant for tumours
  → Domain gap too large

MAE on TCIA CTs → 3D medical images
  → Learns 3D anatomical features from CTs
  → No labels needed (self-supervised)
  → Domain-relevant features
```

See `research/10_self_supervised.md` for the MAE pretraining approach.
