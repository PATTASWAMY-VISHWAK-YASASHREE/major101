# Category 6: AI Architectures for Brain Tumour Classification

**Status: ✅ RESEARCH COMPLETE**

---

## 6.1 Architecture Taxonomy

```
1D features: Radiomics (handcrafted)
3D CNN only: ResNet3D, DenseNet3D, U-Net encoder
3D CNN + 2D attention: Attention U-Net
Transformer: ViT, Swin-UNet, UNETR
Hybrid: CNN encoder + Transformer fusion
Late fusion: Per-modality CNN → dense fusion → classifier
```

---

## 6.2 Our Selected Architecture

```
                    ┌─── T1   ───┐
                    │             ├── 3D ResNet3D-18 → 512D feature
                    ├── T1ce ────┤
                    │             ├── 3D ResNet3D-18 → 512D feature
                    ├── T2   ────┤
                    │             ├── 3D ResNet3D-18 → 512D feature
                    ├── FLAIR ───┤
                    └─────────────┘
                              │
                    Concatenate features
                              │
                    ┌─────────┴─────────┐
                    │  Dense fusion head │
                    │  (512 → 256 → 4)  │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │   CT ResNet3D-18  │  → 512D feature
                    └─────────┬─────────┘
                              │
                    Concatenate (MRI + CT features)
                              │
                    ┌─────────┴─────────┐
                    │  Dense fusion head │
                    │ (512+512 → 256 → 4)│
                    └─────────┬─────────┘
                              │
                    Softmax → 4 classes
                    (WHO Grade I, II, III, IV)
```

---

## 6.3 Parameter Budget

| Component | Params |
|---|---|
| MRI ResNet3D-18 (×1, shared across 4 sequences) | ~11.7M |
| CT ResNet3D-18 (×1) | ~11.7M |
| Dense fusion head | ~0.5M |
| **Total** | **~24M** |

**On RTX 2050 (4GB):** 24M × 4 bytes = 96 MB model weights. Well within budget.

---

## 6.4 Why Not Transformer?

See `research/03_transformer_hybrid.md`. Transformers need too much compute and memory for 4GB VRAM. YAGNI.
