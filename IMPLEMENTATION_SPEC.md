# 🔧 Implementation Spec — From Theory to Working Code

> **Goal:** Turn the existing single-channel T1 MRI classifier into a
> CT+MRI multimodal fusion system with longitudinal support.
> Every section maps research findings → exact code changes.

---

## 1. What Exists vs What Needs to Change

| File | Current | Target | Research Source |
|------|---------|--------|-----------------|
| `src/model.py` | Single-channel ResNet3D | Per-modality encoders + fusion head | `cat8_ct_mri_multimodal_fusion.md`, `04_multimodal_fusion.md` |
| `src/data.py` | Single NIfTI, z-score, resize | CT windowing + MRI CTN + skull-strip + multi-channel stacking | `cat1_brain_tumour_imaging_basics.md`, `16f_preprocessing_pipelines.md` |
| `src/train.py` | Acc + F1 only | Add Dice, HD95, per-class metrics | `cat17_evaluation_metrics_progression.md` |
| `train.py` | FP32, no grad-acc | FP16 + gradient accumulation for RTX 2050 | `RECIPE.md` (hardware constraints) |
| `configs/default.yaml` | Single-channel, batch=4 | Multi-channel, batch=2, FP16 | Hardware constraints |
| `src/data.py` | Full-volume 96³ | Patch sampling 64³ for GPU memory | `07_efficient_data_loading.md` |

---

## 2. Architecture Decision (from research)

**Late fusion** — each modality runs through its own 3D CNN encoder,
then features concatenate into a shared dense head.

```
CT     ──→ ResNet3D(encoder) ──┐
T1     ──→ ResNet3D(encoder) ──┤
T1ce   ──→ ResNet3D(encoder) ──┤
T2     ──→ ResNet3D(encoder) ──┤
FLAIR  ──→ ResNet3D(encoder) ──┤
                          ┌────┴────┐
                          │  CONCAT │
                          └────┬────┘
                               │
                          ┌────▼────┐
                          │  Dense  │
                          │  (512→) │
                          └────┬────┘
                               │
                          ┌────▼────┐
                          │  4-way  │
                          │  Softmax │
                          └─────────┘
```

**Research backing:**
- `cat8_ct_mri_multimodal_fusion.md` — Late fusion > early fusion for brain tumors
- `04_multimodal_fusion.md` — Per-modality encoders preserve modality-specific features
- `01_foundational_cnn_backbones.md` — 3D ResNet-18 is the baseline

---

## 3. Step-by-Step Implementation Plan

### Phase A: Preprocessing Pipeline (data.py)

**What research says** (`16f_preprocessing_pipelines.md`, `cat1.md`):

| Step | MRI | CT | Code location |
|------|-----|----|--------------|
| Skull-strip | ANTsHDGMM | N/A | `preprocess.py` |
| CTN | Apply after skull-strip | N/A | `preprocess.py` |
| HU windowing | N/A | [−100, 300] + clip | `preprocess.py` |
| Intracranial mapping | N/A | Skull ROI | `preprocess.py` |
| Resize | Trilinear to 96³ | Trilinear to 96³ | `data.py` |
| Normalize | Z-score (non-zero) | Z-score | `data.py` |

**Files to create:**
```
src/
├── preprocess.py          ← NEW: skull-strip, CTN, HU windowing
├── dataset.py             ← NEW: split from data.py, add modality stacking
├── transforms.py          ← NEW: geometric augmentations (if needed)
```

**`preprocess.py` — function signatures:**
```python
def skull_strip(volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """ANTsHDGMM skull stripping. Returns skull-stripped volume."""

def apply_chn(volume: np.ndarray) -> np.ndarray:
    """White-matter normalized CTN (Cauchy transformation + shift)."""

def hu_window(volume: np.ndarray, lower: float = -100, upper: float = 300) -> np.ndarray:
    """Window-level CT: clip to [lower, upper], then normalize to [0, 1]."""

def preprocess_mri(volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Full MRI pipeline: skull-strip → CTN → resize → z-score."""

def preprocess_ct(volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Full CT pipeline: HU window → resize → z-score."""
```

**`dataset.py` — modality stacking:**
```python
class MultiModalDataset(Dataset):
    """Loads paired CT+MRI scans.

    Expected structure:
        root/{split}/patient_001/
            ├── ct.nii.gz
            ├── t1.nii.gz
            ├── t1ce.nii.gz
            ├── t2.nii.gz
            └── flair.nii.gz

    Returns: x shape (5, 96, 96, 96) — [CT, T1, T1ce, T2, FLAIR]
    """
    MODALITIES = ["ct", "t1", "t1ce", "t2", "flair"]
```

---

### Phase B: Model Architecture (model.py)

**Current:** Single-channel ResNet3D → 4-class linear head

**Target:** Per-modality encoders → concat → shared dense head

**New file: `src/fusion.py`**
```python
class FusionNet(nn.Module):
    """Late-fusion CT+MRI network.

    Args:
        modality_encoders: dict mapping modality name to encoder Module
        shared_head_layers: list of hidden dims for dense head
        num_classes: int
    """

    def __init__(self, modality_encoders: dict[str, nn.Module],
                 shared_head_layers: list[int], num_classes: int):
        super().__init__()
        self.encoders = nn.ModuleDict(modality_encoders)
        self.shared_head = self._build_head(shared_head_layers, num_classes)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        features = torch.cat([
            enc(m["volume"]) for enc, m in self.encoders.items()
        ], dim=1)
        return self.shared_head(features)
```

**`_build_head` — shared dense head:**
```python
def _build_head(layers: list[int], num_classes: int) -> nn.Module:
    """Build shared classification head after concatenation."""
    out = [
        nn.Linear(total_features, layers[0]),
        nn.BatchNorm1d(layers[0]),
        nn.ReLU(),
        nn.Dropout(0.3),
    ]
    for h in layers[1:]:
        out += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
    out.append(nn.Linear(layers[-1], num_classes))
    return nn.Sequential(*out)
```

**Research backing:**
- `cat8_ct_mri_multimodal_fusion.md` — Late fusion architecture
- `04_multimodal_fusion.md` — Feature concatenation before dense layers
- `cat5_brain_tumor_segmentation.md` — Feature-level fusion for tumor classification

---

### Phase C: Training Loop (train.py + src/train.py)

**Add FP16 mixed precision** (RTX 2050 4GB constraint):
```python
scaler = torch.cuda.amp.GradScaler(enabled=True)
with torch.cuda.amp.autocast(enabled=True):
    out = model(x)
    loss = criterion(out, y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Add gradient accumulation** (batch=2, accumulate=4 → effective batch=8):
```python
grad_accum_steps = 4
for step, (x, y) in enumerate(loader):
    out = model(x)
    loss = criterion(out, y) / grad_accum_steps
    scaler.scale(loss).backward()
    if (step + 1) % grad_accum_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

**Add Dice + HD95 metrics** (from `cat17_evaluation_metrics_progression.md`):
```python
from src.metrics import dice_score, hd95_distance

def eval_epoch_with_metrics(model, loader, criterion, device):
    all_dice = []
    all_hd95 = []
    for x, y in loader:
        probs = torch.softmax(model(x), dim=1)
        dice = dice_score(probs, y)
        all_dice.append(dice)
    return np.mean(all_dice)
```

---

### Phase D: Config Changes (configs/default.yaml)

```yaml
model:
  type: fusionnet            # changed from resnet3d
  modality_encoders:
    - ct_resnet3d
    - t1_resnet3d
    - t1ce_resnet3d
    - t2_resnet3d
    - flair_resnet3d
  shared_head_layers: [512, 256]
  num_classes: 4
  dropout: 0.3

data:
  modality_pairs: true       # NEW: multi-modal dataset
  modalities: [ct, t1, t1ce, t2, flair]
  preprocess_mri: true
  preprocess_ct: true
  skull_strip: true
  ctn: true
  hu_window: [-100, 300]

training:
  batch_size: 2
  grad_accum_steps: 4        # NEW: effective batch=8
  fp16: true                 # NEW: mixed precision
  epochs: 200
  lr: 5.0e-5
  weight_decay: 1.0e-4
  warmup_epochs: 10
  patience: 30
```

---

## 4. Tensor Shapes — Exact Reference

| Stage | Shape | Notes |
|-------|-------|-------|
| Raw CT volume | `(D, H, W)` | NIfTI read, HU values |
| After HU windowing | `(D, H, W)` | clipped to [0, 1] |
| After resize | `(96, 96, 96)` | trilinear interpolation |
| After z-score | `(96, 96, 96)` | mean=0, std=1 |
| After stacking 5 modalities | `(5, 96, 96, 96)` | `[CT, T1, T1ce, T2, FLAIR]` |
| After each encoder | `(256,)` | ResNet3D feature vector |
| After concat (5 encoders) | `(1280,)` | `256 × 5` |
| After dense head (512) | `(512,)` | first dense layer |
| After dense head (256) | `(256,)` | second dense layer |
| Final logits | `(4,)` | [no_tumor, grade_I, grade_II, grade_III_IV] |

---

## 5. Class Labels — WHO Grading Mapping

| Class ID | Label | Description | Research Source |
|----------|-------|-------------|-----------------|
| 0 | No tumor / control | Healthy brain | `cat1_brain_tumour_imaging_basics.md` |
| 1 | Grade I (WHO) | Low-grade glioma, slow growth | `cat2_tumour_progression_monitoring.md` |
| 2 | Grade II (WHO) | Low-grade glioma, moderate growth | `cat2_tumour_progression_monitoring.md` |
| 3 | Grade III–IV (WHO) | High-grade glioma (anaplastic GBM) | `cat2_tumour_progression_monitoring.md` |

> **Note:** The current code has 4 classes. Merge Grade III and IV into class 3
> if dataset doesn't separate them clearly. Research in `cat2.md` covers grading criteria.

---

## 6. Dataset Integration (IBSR + C-BRATS)

**Research says** (`16a_dataset_inventory.md`, `16b_key_dataset_profiles.md`):
- IBSR: 80 paired MRI+CT cases, ~10.5 GB, public download
- C-BRATS: 600 MRI-only cases, ~2 GB, public download

**Directory structure after download:**
```
data/
├── raw/
│   ├── IBSR/
│   │   └── case_001/
│   │       ├── ct.nii.gz
│   │       ├── t1.nii.gz
│   │       ├── t1ce.nii.gz
│   │       ├── t2.nii.gz
│   │       └── flair.nii.gz
│   └── C-BRATS/
│       └── case_001/
│           ├── t1.nii.gz
│           ├── t1ce.nii.gz
│           ├── t2.nii.gz
│           └── flair.nii.gz
├── processed/
│   └── train/
│       └── case_001/
│           ├── ct.nii.gz        # skull-stripped, HU-windowed, z-scored, 96³
│           ├── t1.nii.gz        # skull-stripped, CTN, z-scored, 96³
│           ├── t1ce.nii.gz
│           ├── t2.nii.gz
│           └── flair.nii.gz
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

**`src/download.py` — NEW:**
```python
def download_ibsr(output_dir: Path, num_cases: int = 80):
    """Download IBSR MRI+CT pairs from Harvard."""

def download_cbrats(output_dir: Path, num_cases: int = 600):
    """Download C-BRATS MRI-only from Synapse."""
```

**`src/preprocess_all.py` — NEW:**
```python
def preprocess_dataset(raw_dir: Path, output_dir: Path):
    """Process all raw scans to 96³, modality-normalized volumes."""
```

---

## 7. Longitudinal Support (Phase 2 — future)

**Not yet implemented.** From `05_longitudinal_analysis.md` and `cat2_tumour_progression_monitoring.md`:

```python
class LongitudinalDataset(Dataset):
    """Patient with ≥2 timepoints.

    Expected structure:
        root/patient_001/
            ├── timepoint_1/
            │   ├── ct.nii.gz
            │   └── t1.nii.gz
            ├── timepoint_2/
            │   ├── ct.nii.gz
            │   └── t1.nii.gz
            └── outcome.json       # {"progression": 1, "time_to_progression": 12}

    Returns: x shape (T, C, 96, 96, 96) — T timepoints, C modalities
    """
```

---

## 8. File Creation Checklist (for AI agent)

| # | File | Purpose | Research reference |
|---|------|---------|-------------------|
| 1 | `src/preprocess.py` | Skull-strip, CTN, HU windowing | `16f_preprocessing_pipelines.md` |
| 2 | `src/dataset.py` | Multi-modal dataset loader | `cat8_ct_mri_multimodal_fusion.md` |
| 3 | `src/fusion.py` | Late-fusion network | `04_multimodal_fusion.md` |
| 4 | `src/metrics.py` | Dice, HD95, per-class metrics | `cat17_evaluation_metrics_progression.md` |
| 5 | `src/download.py` | IBSR + C-BRATS download scripts | `16a_dataset_inventory.md` |
| 6 | `src/preprocess_all.py` | Batch preprocessing pipeline | `16f_preprocessing_pipelines.md` |
| 7 | `configs/fusion.yaml` | Multimodal config | — |
| 8 | `configs/longitudinal.yaml` | Future longitudinal config | `05_longitudinal_analysis.md` |

---

## 9. Quick Start Commands (after implementation)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets
python src/download.py --ibsr --cbrats --output data/raw

# 3. Preprocess all scans
python src/preprocess_all.py --raw data/raw --output data/processed

# 4. Train single-modality baseline (MRI only)
python train.py --cfg configs/default.yaml

# 5. Train multimodal fusion
python train.py --cfg configs/fusion.yaml

# 6. Evaluate
python train.py --cfg configs/fusion.yaml --data data/processed --eval-only
```
