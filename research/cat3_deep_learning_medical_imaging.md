# Category 3: Deep Learning in Medical Imaging

**Status: ✅ RESEARCH COMPLETE**

---

## 3.1 What Exists (Overview)

| Domain | Typical approach | Our relevance |
|---|---|---|
| **Segmentation** | U-Net, nnU-Net, V-Net | V-Net encoder for per-modality features |
| **Classification** | ResNet3D, DenseNet3D, EfficientNet3D | ResNet3D-18 for 4-class WHO grade |
| **Object detection** | Faster R-CNN 3D, PointPillars | Not applicable |
| **Self-supervised** | MAE, SimCLR | CT branch pretraining on TCIA |
| **Domain adaptation** | CycleGAN, ComBat | Cross-institutional harmonisation |
| **Transformer hybrids** | ViT, UNETR, Swin-UNet | YAGNI for 4GB VRAM |

---

## 3.2 The Medical Imaging Gap

Natural image DL (ImageNet, ResNet) does not directly apply:
- **No colour channels** — grayscale (CT) or intensity-weighted (MRI)
- **No fixed input size** — volumes vary from 64³ to 512³
- **Extremely small datasets** — hundreds vs ImageNet's 14M
- **No labels at scale** — annotating tumours takes hours per case
- **Class imbalance** — Grade IV dominates in clinical datasets

**Consequence:** Transfer from natural images works poorly. We need domain-specific pretraining (TCIA CTs for CT branch, C-BRATS for MRI branch).

---

## 3.3 Our Choices

| Component | Choice | Reason |
|---|---|---|
| Backbone | ResNet3D-18 | Smallest effective 3D backbone |
| Pretraining | MAE on TCIA (CT) + C-BRATS (MRI) | Domain-specific features |
| Fusion | Late fusion | Each modality processed independently first |
| Patch-based training | 64³ patches | Only 4GB VRAM |

See `research/01_foundational_cnn_backbones.md` and `research/02_medical_segmentation_cnn.md` for details.
