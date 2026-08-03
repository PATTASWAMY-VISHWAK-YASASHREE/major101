# Category 11: Generative Data Augmentation (GANs, Diffusion)

**Status: 🔬 RESEARCH ONLY — no implementation yet**

> **Purpose:** Understand how synthetic data generation addresses the small dataset
> problem. Relevant for IBSR's 80-case limitation.

---

## 11.1 The Small Dataset Problem

IBSR has 80 paired MRI+CT cases. That's very small for deep learning.

**Standard augmentation:** Random rotations, flips, intensity shifts, Gaussian noise.
Effective but limited — generates variants, not new tumours.

**Generative augmentation:** Synthesise entirely new tumour cases from the distribution
of existing ones. More powerful but complex.

---

## 11.2 GAN-based Augmentation

### 11.2.1 CycleGAN for MRI↔CT Translation

```
CycleGAN: Translate MRI → synthetic CT (and vice versa)
- Requires no paired data for training
- Learns domain-invariant features
- Output: realistic-looking synthetic CT from real MRI
```

**Use case:** Generate synthetic CTs from the 2,000 BraTS MRI cases. Train CT encoder
on BraTS MRI → synthetic CT pairs (zero real CT labels needed for the CT encoder).

**Challenge:** BraTS MRI cases have no ground-truth CT. Synthetic CTs will be plausible
but not anatomically accurate. Use with caution.

### 11.2.2 3D GANs for Synthetic Tumour Generation

```
3D GAN (e.g., 3D-EGAN, 3D-StyleGAN):
- Generate synthetic 3D tumour volumes
- Control tumour size, location, class (grade I-IV)
- Output: new training cases

Training: Pretrained on public datasets (BraTS, IBSR)
```

**Status:** Active research, not mature. 3D GANs are computationally expensive and
hard to train stably.

---

## 11.3 Diffusion Models for 3D Medical Imaging

```
Diffusion model:
1. Add noise to a real volume → noisy volume
2. Train U-Net to reverse the noise process
3. Sample from noise → synthetic volume

Advantages over GANs:
- More stable training (no mode collapse)
- Better sample diversity
- Better for 3D than 2D (handles spatial structure well)
```

**Representative work:** "3D Medical Image Synthesis with Diffusion Models"
(2023-2025 papers, arXiv). Diffusion models are the current SOTA for 3D medical
image generation.

**VRAM for diffusion training:** 3D diffusion is very VRAM-heavy (~8-16 GB). Not
feasible on 4GB RTX 2050.

---

## 11.4 Practical Decision for Our Project

| Strategy | Effort | VRAM | Impact | Verdict |
|---|---|---|---|---|
| **Standard augmentation** | Low | Low | +10-20% effective cases | ✅ Do this |
| **CycleGAN MRI→CT** | Medium | Medium | Synthetic CTs from BraTS MRI | ⚠️ Future |
| **3D GAN synthesis** | High | High | New synthetic tumours | ❌ Over-engineered |
| **Diffusion synthesis** | Very high | Very high | High-quality synthetic volumes | ❌ Not viable on 4GB |

**Recommendation:** Use **standard 3D augmentation** only. The ROI on generative
augmentation is not worth the effort given the hardware constraints.

**Standard augmentation for 3D volumes:**
```python
# Effective augmentations for IBSR (64³ patches)
transform = Compose([
    RandRotate90(prob=0.5, spatial_axes=(0,1)),   # 90° rotations
    RandFlip(prob=0.5, spatial_axis=0),           # Random flip
    RandShiftIntensity(offset=10, prob=0.5),       # Intensity shift
    RandGaussianNoise(std=0.1, prob=0.3),         # Noise injection
    RandScale(zoom_range=0.9, factor=1.0, prob=0.5), # Zoom (on patches)
])
```

This effectively turns 80 cases into ~320-640 training samples per epoch.
