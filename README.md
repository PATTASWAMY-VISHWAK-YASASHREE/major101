# Major101 — Brain Tumour Classification via CT+MRI Multimodal Fusion

**Global Goal:** Build a longitudinal volumetric deep learning pipeline that fuses
MRI and CT brain scans for tumour detection, grading, progression tracking, and
survival prediction — addressing three documented gaps:

1. **No CT+MRI fusion benchmark for brain tumours** (BraTS is MRI-only)
2. **No automated AI-assisted RANO progression monitoring** (manual, error-prone)
3. **No survival prediction from fused CT+MRI data** (DeepHit uses MRI only)

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
                         ├──► Fusion Layer ──► ┌─► Classification ──► Grade
CT (enhanced)    ────────┘                     │
                                              ├─► Longitudinal Track ──► RANO AI
                                              │                         (progression,
                                              │                          pseudoprogression)
                                              └─► Survival Head ──► Overall Survival
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

## Longitudinal AI Assistant

The pipeline tracks tumours over time — not just single scans. Two sub-goals:

**AI-assisted RANO monitoring:**
- Automate RANO 2.0 measurements (enhancing + non-enhancing tumour burden)
- Flag pseudoprogression vs true progression (CT adds bone/hemorrhage info MRI can't)
- Time-series feature extractor (Temporal Transformer, per `05_longitudinal_analysis.md`)

**Survival prediction:**
- DeepHit-style Cox survival head on fused CT+MRI features
- Currently no public data supports this — it's the research frontier

## Outputs

Checkpoints saved to `outputs/`. Best model is `outputs/best.pt`.