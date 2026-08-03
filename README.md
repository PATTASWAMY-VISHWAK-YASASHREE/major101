# Major101 — Brain Tumour Classification via CT+MRI Multimodal Fusion

**Global Goal:** Build a longitudinal volumetric deep learning pipeline that fuses
MRI and CT brain scans for tumour detection, grading, and survival prediction —
addressing a documented gap in the literature where CT+MRI fusion for brain
tumours lags far behind MRI-only methods.

## Why this matters

Our research audit (117 papers, 2013–2026) reveals:

- **2015–2017:** CT+MRI fusion pioneered for *other* pathologies (MICE review,
  early feature fusion) — but **not for brain tumours**
- **2017–2022:** Surge of MRI-only methods (nnU-Net, ViT, TransUNet) dominates
  the literature; multimodal approaches stagnate
- **2023–2026:** Recent sparks — Gong 2025 (late fusion CT+MRI), Yang 2025
  (radiomics+ViT), Chen 2026 (MRI-to-CT translation), Islam 2026 (feature
  fusion) — but **no mature benchmark exists**

**The gap:** BraTS (the primary benchmark) provides MRI only. No public
dataset offers paired CT+MRI for brain tumours. Every fusion method in this
repo must be validated against this limitation — that's the research problem.

## Architecture direction

```
MRI (T1/T1c/T2/FLAIR) ──┐
                         ├──► Fusion Layer ──► Classifier ──► Grade + Survival
CT (enhanced)    ────────┘         │
                                   │
                         Self-Supervised Pretrain (MAE/DINOv2)
                         on available MRI volumes
```

## Quick start

```bash
pip install -r requirements.txt
python train.py                          # uses configs/default.yaml
python train.py --data /path/to/scans    # override data root
python train.py --epochs 200 --batch 8   # CLI overrides
```

## Data format

Place `.nii.gz` volumes in `data/` organised by class:

```
data/
├── train/
│   ├── 0/          # class 0 (e.g. no tumour)
│   │   ├── mri_scan01.nii.gz
│   │   └── ct_scan01.nii.gz            # paired CT if available
│   └── 1/          # class 1 (e.g. low grade)
│       └── mri_scan03.nii.gz
├── val/
└── test/
```

CT volumes are optional. The pipeline supports:
- **MRI-only mode** (fallback, uses BraTS-style data)
- **Paired CT+MRI mode** (fusion active when both modalities present)

## Model

`src/model.py` ships a lightweight 3D ResNet (~few M params).
Fuse modalities via:

| Strategy | When to use |
|----------|-------------|
| **Early fusion** (concat channels) | Simple, fast, but alignment-sensitive |
| **Late fusion** (separate heads, merge logits) | Gong 2025 style, modality-agnostic |
| **Feature fusion** (attention cross-modal) | Islam 2026 style, most powerful |
| **Translation** (MRI→CT via diffusion) | Chen 2026 style, when CT missing |

Swap in `DenseNet3D` or `ConvNeXt3D` from MONAI when validation plateaus.

## Research status

| Aspect | Status | Reference |
|--------|--------|-----------|
| MRI segmentation | ✅ Mature | nnU-Net, TransUNet |
| MRI classification | ✅ Mature | ViT, DenseNet3D |
| CT+MRI fusion | ⚠️ Experimental | Gong 2025, Yang 2025 |
| MRI-to-CT translation | 🔬 Prototype | Chen 2026 |
| Survival from MRI+CT | 🔬 Unexplored | DeepHit (MRI only) |
| Public CT+MRI dataset | ❌ None exists | IBSR (partial, other pathology) |

## Outputs

Checkpoints saved to `outputs/`. Best model is `outputs/best.pt`.