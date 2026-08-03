# Efficient Data Loading for Large Medical Imaging Datasets on Constrained Hardware

**Status: 🔬 RESEARCH ONLY — no implementation yet**

> **Constraint:** 8GB system RAM, 4GB GPU VRAM (RTX 2050), 60GB free SSD.
> **Goal:** Load and train on 3D volumetric brain imaging datasets that are nominally 100x
> larger than available memory.

---

## 1. The Memory Problem (Quantified)

A single BraTS patient: 4 modalities × 240³ volume × 4 bytes (FP32) = **~2.2 GB**

| What | Memory (FP32) | Memory (FP16) |
|---|---|---|
| One 240³ NIfTI volume | 55 MB | 27 MB |
| One full BraTS patient (4 modalities) | 220 MB | 110 MB |
| Batch of 2 patients | 440 MB | 220 MB |
| IBSR full dataset (80 patients × 3 modalities) | ~1.7 GB | 850 MB |
| C-BRATS full dataset (600 patients × 4 modalities) | ~13 GB | 6.5 GB |

**Hard limits:**
- **GPU VRAM (4 GB):** Leaves ~3.5 GB for model, activations, gradients after kernel overhead
- **System RAM (8 GB):** ~6 GB usable after OS
- **SSD (60 GB free):** Fits IBSR + C-BRATS + models + checkpoints with margin

---

## 2. Strategies by Memory Layer

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE LAYERS                      │
│                                                             │
│  SSD ─→ mmap() ─→ Lazy Load ─→ Patch Crop ─→ Quantize ─→ GPU│
│   60GB   0 RAM    0 RAM      1-2MB      0.5MB     ~2MB     │
│                                                             │
│  ┌─────┐  ┌────────┐  ┌──────────┐  ┌─────────┐  ┌──────┐  │
│  │NiiGZ│  │Memory- │  │Random    │  │FP32→FP16│  │GPU   │  │
│  │(disk)│  │mapped  │  │Patch     │  │(Tensor) │  │Kernel│  │
│  └─────┘  └────────┘  └──────────┘  └─────────┘  └──────┘  │
│                                                             │
│  Total VRAM for one patch step: ~2-4 MB  ✅                │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Strategy 1: Memory-Mapped Lazy Loading (Zero RAM Copy)

**Concept:** The OS page-cache + `numpy.memmap` / NIfTI `mmapped=True` means data never
enters application RAM until the GPU kernel reads it.

```python
import nibabel as nib
import numpy as np

# Memory-map the NIfTI — zero RAM cost until read
vol = nib.load("scan.nii.gz", mmap=True)
data = vol.get_fdata()  # Reads on-demand into pages, not full copy

# Crop a patch directly from the mmap — only the patch goes to RAM
patch = data[64:128, 64:128, 64:128]  # ~1 MB, not 55 MB
```

**How it works:**
1. NIfTI `.gz` is stored on disk
2. `mmap=True` tells NIfTI not to decompress the whole file into RAM
3. Numpy reads pages on demand via OS page cache
4. You crop patches before they ever enter your process RAM
5. When a patch is needed, it's read → cropped → sent to GPU → discarded

**Result:** One patient loaded in RAM = 0 MB. One patch in RAM = ~1 MB.

**Tools:**
- **nibabel + `mmap=True`**: Standard NIfTI lazy loading
- **torch.nn.utils.rnn.pad_sequence**: For uneven patches
- **MONAI's `CacheDataset` with `cache_rate=0`**: Never caches, always streams

---

## 4. Strategy 2: Patch-Based Training (The Only Realistic Option)

Full 240³ volumes don't fit in 4GB VRAM in a batch. Patch-based training solves this.

```python
import numpy as np
import random

def crop_patch(volume: np.ndarray, patch_size: int = 64) -> np.ndarray:
    """Crop a random patch from a 240³ volume."""
    shape = volume.shape
    # Random starting position (ensure full patch fits)
    x = random.randint(0, shape[0] - patch_size)
    y = random.randint(0, shape[1] - patch_size)
    z = random.randint(0, shape[2] - patch_size)
    return volume[x:x+patch_size, y:y+patch_size, z:z+patch_size]

# For CT+MRI fusion — crop both at same coordinates
def crop_ct_mri_patches(ct: np.ndarray, mri: np.ndarray, patch_size: int = 64):
    shape = ct.shape
    x = random.randint(0, shape[0] - patch_size)
    y = random.randint(0, shape[1] - patch_size)
    z = random.randint(0, shape[2] - patch_size)
    return ct[x:x+ps, y:y+ps, z:z+ps], mri[x:x+ps, y:y+ps, z:z+ps]
```

**Patch size vs. VRAM:**

| Patch size | Patch volume | Batch=2 total | VRAM headroom |
|---|---|---|---|
| 32³ | ~1 MB | ~4 MB | ✅ Plenty |
| 64³ | ~8 MB | ~16 MB | ✅ Good |
| 96³ | ~32 MB | ~64 MB | ✅ Fine |
| 128³ | ~85 MB | ~170 MB | ⚠️ Gets tight |
| 240³ (full) | ~55 MB | ~110 MB + model | ❌ Overflows with model |

**Recommendation:** Start at 64³ patches. Graduate to 96³ as you optimize. Never exceed 128³ on 4GB VRAM with a ResNet3D.

**Trade-off:** Patches lose global context (the model never sees the full tumour).
**Mitigation:** Heavy data augmentation (rotations, flips, patches from many positions
per scan) compensates for the lost context.

---

## 5. Strategy 3: Quantization (FP32 → FP16 → INT8)

### 5.1 Mixed Precision Training (FP16)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    x = batch.to(device)

    with autocast(dtype=torch.float16):
        y = model(x)
        loss = criterion(y, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**VRAM savings:** ~50% (FP16 = 2 bytes vs FP32 = 4 bytes per value)

**With FP16, patch VRAM usage:**

| Patch size | FP32 | FP16 |
|---|---|---|
| 64³ | 8 MB | 4 MB |
| 96³ | 32 MB | 16 MB |
| 128³ | 85 MB | 42 MB |
| 240³ full | 55 MB | 27 MB |

**FP16 lets you train 128³ patches in 4GB VRAM.** This is the single most impactful
change for your hardware.

**Caveat:** FP16 can cause numerical instability in rare cases. PyTorch's `GradScaler`
handles this automatically — no manual tuning needed.

### 5.2 INT8 Quantization (Post-Training)

After training in FP16, convert the model to INT8 for inference:

```python
model.eval()
quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.Conv3d}, dtype=torch.qint8
)
```

**Size reduction:**
- FP32 model: ~20 MB
- FP16 model: ~10 MB
- INT8 model: ~5 MB

**Inference speed on RTX 2050:** INT8 uses Tensor Cores → 2-4x faster than FP16.

**For our project:** Train in FP16, deploy as FP16 or INT8. INT8 is mainly for
inference — don't train in INT8, you'll lose accuracy.

---

## 6. Strategy 4: Smart Dataloader (MONAI's Cache-Streaming Hybrid)

The key insight: you don't need the whole dataset in RAM. You need the right **small
subset** in RAM at any given time.

### 6.1 MONAI CacheDataset with cache_rate=0.0 (Full Streaming)

```python
from monai.data import CacheDataset, load_decathlon_datalist

# cache_rate=0.0: never cache anything, always load from disk
# Each volume is loaded on demand, patches cropped on the fly
train_ds = CacheDataset(
    data=train_data,
    transform=train_transforms,  # Includes patch crop
    cache_rate=0.0,              # Zero cache — pure streaming
    num_cached_per_class=0,
    shuffle=False,
)

train_loader = DataLoader(
    train_ds,
    batch_size=2,
    num_workers=4,               # 4 threads loading from disk in parallel
    pin_memory=True,             # Pin pages for fast GPU transfer
    persistent_workers=True,     # Don't kill workers between epochs
)
```

**Memory usage with cache_rate=0.0:**
- In RAM at any time: 1-2 patches (not whole volumes)
- VRAM at any time: model + 1-2 patches + activations
- Total: ~1-2 GB GPU, ~500 MB system RAM for data

### 6.2 Hybrid Caching (cache_rate=0.1) — Best for IBSR

For IBSR (80 cases, ~1.7 GB total), caching a small fraction in RAM is safe:

```python
train_ds = CacheDataset(
    data=train_data,
    transform=train_transforms,
    cache_rate=0.1,              # Cache 10% of data in RAM (~170 MB)
    num_workers=4,
)
```

**Benefit:** Faster epoch starts (cached data loads from RAM, not disk)
**Risk:** None for IBSR — 10% cache is ~170 MB, well within 8GB RAM

### 6.3 Dataloader Memory Breakdown (IBSR, cache_rate=0.1)

```
┌──────────────────────────────────────────────────────────┐
│  RAM usage                                               │
│                                                          │
│  CacheDataset cache:   170 MB   (10% of 1.7 GB dataset)  │
│  Active patches:        32 MB   (4 patches × 8 MB each)  │
│  Worker buffers:        64 MB   (4 workers × 16 MB)      │
│  Total data in RAM:    ~270 MB   ✅ (6 GB free)          │
│                                                          │
│  GPU VRAM usage                                              │
│  Model (ResNet3D-18):   20 MB                             │
│  Activations:           30 MB                             │
│  Gradients:             20 MB                             │
│  Batch (2 patches):     16 MB                             │
│  Total:                 ~86 MB   ✅ (4 GB available)     │
└──────────────────────────────────────────────────────────┘
```

---

## 7. Strategy 5: Data Compression on Disk

### 7.1 NIfTI GZIP vs. Uncompressed

| Format | IBSR size | Read speed | Notes |
|---|---|---|---|
| NIfTI `.nii.gz` | 4-8 GB | ~50 MB/s (decompress) | Default, safe |
| NIfTI `.nii` uncompressed | 12-24 GB | ~200 MB/s | Fast but 3x size |
| `.npz` (numpy compressed) | 3-6 GB | ~100 MB/s | Good for numpy arrays |
| `.pt` (torch saved) | 5-10 GB | ~100 MB/s | Use for preprocessed data |

**Recommendation:** Keep `.nii.gz` as source. Preprocess to `.pt` or `.npz` for
training (faster reads, no decompress on every epoch).

### 7.2 Preprocessing Before Training (Pre-cached Arrays)

```python
# One-time preprocessing: convert NIfTI → preprocessed .pt arrays
import torch

preprocessed_vol = preprocess_nifti(raw_nifti_path)  # skull-strip, normalize, resize
torch.save(preprocessed_vol, "preprocessed/t1_001.pt")

# Training dataloader reads .pt directly (no decompress, no skull-strip on the fly)
vol = torch.load("preprocessed/t1_001.pt", weights_only=True)
```

**Benefit:** Training dataloader is ~5x faster (no decompress, no skull-strip per epoch)
**Cost:** Extra ~2-4 GB for preprocessed `.pt` files

---

## 8. Strategy 6: Gradient Accumulation (Simulate Larger Batch)

Small batch size (1-2) can destabilize training. **Gradient accumulation** lets you
simulate a larger batch without increasing peak VRAM:

```python
ACCUMULATE_STEPS = 4  # Effectively batch_size = 2 × 4 = 8

for step, batch in enumerate(dataloader):
    with autocast(dtype=torch.float16):
        loss = model(batch) / ACCUMULATE_STEPS

    scaler.scale(loss).backward()

    if (step + 1) % ACCUMULATE_STEPS == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

**Result:** Training dynamics of batch size 8, peak VRAM of batch size 2.

---

## 9. Strategy 7: Memory-Aware Data Pipeline Architecture

```python
# src/data/pipeline.py — memory-aware loading strategy

import torch
from torch.utils.data import DataLoader
from monai.data import CacheDataset
from monai.transforms import Compose, RandCropByPosNegLabel, RandRotate90, ToTensord

# ─────────────────────────────────────────────────────────────────
# DATA LOADING STRATEGY FOR 8GB RAM / 4GB VRAM
# ─────────────────────────────────────────────────────────────────

# 1. Patch-based cropping (64³ patches)
patch_transform = Compose([
    RandCropByPosNegLabel(
        label_key="label",
        spatial_size=(64, 64, 64),
        pos=1, neg=1,   # 50/50 positive/negative patches
    ),
    RandRotate90(
        prob=0.5, spatial_axes=(0, 1),   # Random 90° rotations
    ),
])

# 2. CacheDataset: cache_rate=0.1 (safe for IBSR ~80 cases)
def build_dataloader(data_list, cache_rate=0.1, num_workers=4):
    ds = CacheDataset(
        data=data_list,
        transform=patch_transform,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    return DataLoader(
        ds,
        batch_size=2,          # Max for 4GB VRAM with 64³ patches
        num_workers=num_workers,
        pin_memory=True,       # Faster GPU transfer
        persistent_workers=True,
        drop_last=True,        # Avoid partial last batch
    )

# 3. Memory budget check (runs at startup)
def check_memory_budget():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e6
    print(f"GPU VRAM: {gpu_mem:.0f} MB")

    # Estimate: model(20) + activations(30) + grads(20) + batch(16) = ~86 MB
    # Safe headroom: 25% of VRAM for OS/drivers
    safe_headroom = 0.25
    usable = gpu_mem * (1 - safe_headroom)
    est_batch_mem = 86  # MB for 2 × 64³ patches
    max_batch = int(usable / est_batch_mem)
    print(f"Safe max batch size: {max_batch}")

    if max_batch < 2:
        print("⚠️  Reduce patch_size or enable gradient accumulation")
```

---

## 10. Summary: What to Use and When

| Strategy | VRAM saved | Complexity | Use when |
|---|---|---|---|
| **mmap / lazy NIfTI load** | ~50 MB/patient | Low | Always — never decompress full file |
| **Patch-based cropping** | ~50-80% | Low | Always on 4GB VRAM |
| **FP16 mixed precision** | ~50% | Low | Always — zero accuracy cost |
| **cache_rate=0.0 (stream)** | 0 cache RAM | Low | Large datasets (C-BRATS) |
| **cache_rate=0.1 (hybrid)** | ~170 MB cache | Low | Small datasets (IBSR) |
| **Gradient accumulation** | Batch×N VRAM saved | Medium | When batch=1 causes instability |
| **INT8 quantization** | 50% model size | Low | Inference only |
| **Pre-cached .pt files** | 0 on-disk decompress | Medium | After initial preprocessing |
| **Persistent workers** | 0-2 MB | Low | Always |

## Recommended Pipeline for Your Hardware

```
┌───────────────────────────────────────────────────────────────┐
│  RECOMMENDED PIPELINE (8GB RAM / 4GB VRAM / 60GB SSD)        │
│                                                               │
│  Data:  IBSR (80 cases, 4-8 GB) + C-BRATS (600 cases, 2 GB)  │
│                                                               │
│  Preprocessing:                                                │
│    → Pre-cache to .pt files (one-time, ~5 GB output)           │
│    → Done before training starts                                │
│                                                               │
│  Training pipeline:                                           │
│    .pt file ─→ mmap (0 RAM) ─→ 64³ patch (8 MB) ─→ FP16 (4MB)│
│                                    ↓                             │
│                               GPU (86 MB total)               │
│                                                               │
│  Model:  ResNet3D-18 (32 base width) — 5-10M params          │
│                                                               │
│  Batch:  size=2, accumulate=4 → effective batch=8            │
│                                                               │
│  Expected:  epoch time ~20-40 min on RTX 2050                │
└───────────────────────────────────────────────────────────────┘
```

---

## 11. What NOT to Do

| Anti-pattern | Why it fails |
|---|---|
| `cache_rate=1.0` (cache everything) | IBSR fits but C-BRATS won't; cache management overhead |
| Full 240³ volumes in batch | 220 MB × 2 batch = 440 MB, plus model = exceeds 4 GB |
| `num_workers=0` | CPU becomes bottleneck; dataloader stalls |
| Loading .nii.gz on every epoch | ~30 seconds per epoch just decompressing |
| INT8 training | Accuracy drops 2-5% on medical images |
| Loading whole dataset into numpy array | Instantly OOMs at 2 GB dataset size |
