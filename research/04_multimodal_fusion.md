# Category 4: Multimodal CT+MRI Fusion for Brain Tumor Analysis

> **Status:** Complete | **Date:** 2026-08-03
> **Sources:** PubMed (32 papers), arXiv (11 papers)

## Summary

Multimodal fusion of CT and MRI for brain tumor analysis is an actively emerging field. The 2025-2026 literature shows a clear convergence on **attention-based cross-modal fusion** as the state-of-the-art approach, with several 2026 papers reporting accuracy in the 92-96% range. Three distinct fusion strategies dominate: (1) **early fusion** (concatenate modalities as multi-channel input), (2) **feature-level fusion** (separate encoders + cross-modal attention), and (3) **late fusion** (separate models + ensemble). Critically, a 2026 paper by Almadhor et al. directly addresses **CT-MRI integration** for brain tumor classification — the closest match to this project's goals. A 2025 paper by Zhang et al. demonstrates that **synthetic CT can be generated from MRI using physics modeling**, offering a path when paired CT scans are unavailable.

## Key Papers

| # | Paper | Authors | Year | Source | URL/DOI | Key Finding | Relevance |
|---|-------|---------|------|--------|---------|-------------|-----------|
| 1 | **Bridging Modalities: a deep learning framework for brain tumor classification via CT-MRI integration and model fusion** | Almadhor et al. | 2026 | Front Comput Neurosci | [DOI: 10.3389/fncom.2026.1798561](https://doi.org/10.3389/fncom.2026.1798561) | Direct CT-MRI integration for brain tumor classification; model fusion of CT-only and MRI-only CNNs | ⭐⭐⭐⭐⭐ |
| 2 | **MANet: a multimodal attention CNN for brain tumor classification** | Seenu et al. | 2026 | Scientific Reports | [DOI: 10.1038/s41598-026-52615-3](https://doi.org/10.1038/s41598-026-52615-3) | Attention-based multimodal CNN; reported 95.7% accuracy on BraTS | ⭐⭐⭐⭐⭐ |
| 3 | **A Hybrid Perception Network for Accurate and Interpretable Brain Tumor Classification Using Multi-Modal Imaging** | Al-Sharari | 2026 | Asian Pac J Cancer Prev | [DOI: 10.31557/APJCP.2026.27.7.2525](https://doi.org/10.31557/APJCP.2026.27.7.2525) | Hybrid perception network; combines multimodal imaging with interpretable classification | ⭐⭐⭐⭐ |
| 4 | **Multi-modality brain tumor segmentation using dual-attention GAN** | Kawahara et al. | 2026 | Rep Pract Oncol Radiother | [DOI: 10.5603/rpor.110813](https://doi.org/10.5603/rpor.110813) | Dual-attention GAN for multi-modality brain tumor segmentation | ⭐⭐⭐⭐ |
| 5 | **Multimodal Fusion at Three Tiers: Physics-Driven Data Generation and VLM Guidance for Brain Tumor Segmentation** | Zhang et al. | 2025 | arXiv:2507.09966v3 | [arXiv](https://arxiv.org/abs/2507.09966v3) | Three-tier fusion: pixel-level synthetic CT generation (physics), feature-level Transformer cross-modal fusion, semantic-level CLIP guidance. Dice 0.866-0.901 on BraTS 2020/2021/2023 | ⭐⭐⭐⭐⭐ |
| 6 | **Multimodal Brain Tumour Classification Using Feature Fusion** | Islam et al. | 2026 | arXiv:2606.11107v2 | [arXiv](https://arxiv.org/abs/2606.11107v2) | Two-branch network (CNN + MLP radiomic); gated fusion achieves 96.13% accuracy. Compares concatenation vs gated vs bidirectional cross-modal attention | ⭐⭐⭐⭐ |
| 7 | **Hybrid 2D CNN-Transformer Approach for Brain Tumor Detection** | Almutairi et al. | 2026 | Front Artif Intell | [DOI: 10.3389/frai.2026.1986849](https://doi.org/10.3389/frai.2026.1986849) | Combines 2D CNN with Transformer for glioma detection using multi-modal data | ⭐⭐⭐ |
| 8 | **A Survey of Deep Learning Techniques in Neuro-Oncology** | Gharehbaghi et al. | 2026 | Neural Comput | [DOI: 10.1162/neco_a_01655](https://doi.org/10.1162/neco_a_01655) | Comprehensive survey covering multimodal brain tumor DL from CNN to transformer architectures | ⭐⭐⭐ |
| 9 | **Radiomics-based AI models in brain tumors: systematic review and meta-analysis** | Reyes et al. | 2026 | Neuroradiology | [DOI: 10.1007/s00234-026-03983-0](https://doi.org/10.1007/s00234-026-03983-0) | Meta-analysis of radiomics-based AI in brain tumors; includes multimodal studies | ⭐⭐⭐ |
| 10 | **Medical Image Fusion: A survey of the state of the art** | James & Dasarathy | 2014 | Information Fusion | [DOI: 10.1016/j.inffus.2013.12.002](https://doi.org/10.1016/j.inffus.2013.12.002) | Foundational survey on medical image fusion methods, modalities, and challenges | ⭐⭐⭐ |
| 11 | **Simultaneous Tri-Modal Medical Image Fusion and Super-Resolution using Conditional Diffusion Model** | Li et al. | 2024 | arXiv:2404.17357v4 | [arXiv](https://arxiv.org/abs/2404.17357v4) | TFS-Diff: tri-modal fusion + super-resolution via diffusion model with channel attention | ⭐⭐⭐ |

## Fusion Strategies

### 1. Early Fusion (Concatenation)
- **How:** CT and MRI stacks are concatenated as multi-channel input (e.g., 8-channel: 4 MRI contrasts + 4 CT phases)
- **Pros:** Simple; single network; end-to-end differentiable
- **Cons:** Modality mismatch; CT intensity range (-1000 to 3000 HU) vs MRI (arbitrary); alignment sensitive
- **Used by:** Almadhor 2026 (hybrid with model fusion)

### 2. Feature-Level Fusion with Cross-Modal Attention
- **How:** Separate encoders (e.g., ResNet3D for CT, U-Net for MRI) produce feature maps; cross-modal attention module (e.g., co-attention, transformer cross-attention) fuses them
- **Pros:** Handles modality differences; each encoder can use modality-specific architecture
- **Cons:** More parameters; harder to train
- **Used by:** MANet (Seenu 2026), Zhang 2025 (Transformer-based cross-modal), Islam 2026 (gated + bidirectional cross-modal attention)

### 3. Late Fusion (Ensemble)
- **How:** Separate models trained on CT-only and MRI-only data; predictions combined via voting, weighted average, or stacking
- **Pros:** Simple to implement; each model is modality-optimized
- **Cons:** No joint learning; inter-modality interactions lost
- **Used by:** Almadhor 2026 (model fusion as second branch)

### 4. Physics-Driven Synthetic CT Generation
- **How:** Use MRI physics models to generate synthetic CT, then process as multi-channel input
- **Pros:** Enables CT+MRI even when only MRI is available; physics-grounded
- **Cons:** Synthetic CT may not capture pathology; extra preprocessing step
- **Used by:** Zhang 2025 (three-tier fusion, pixel-level)

### 5. Diffusion-Based Fusion
- **How:** Conditional diffusion model takes MRI as condition and generates fused CT+MRI representation
- **Pros:** Captures uncertainty; simultaneous fusion + super-resolution
- **Cons:** Computationally expensive; slower inference
- **Used by:** Li et al. 2024 (TFS-Diff)

## State-of-the-Art Results

| Paper | Method | Modality | Dataset | Accuracy | Metric |
|-------|--------|----------|---------|----------|--------|
| Almadhor 2026 | CT-MRI integration + model fusion | CT+MRI | Kaggle Brain Tumor | ~92% | Accuracy |
| Seenu 2026 (MANet) | Multimodal attention CNN | MRI (BraTS) | BraTS | 95.7% | Accuracy |
| Islam 2026 | CNN + radiomic + gated fusion | MRI + radiomics | Custom | 96.13% | Accuracy |
| Zhang 2025 | Three-tier physics+Transformer | MRI + synthetic CT | BraTS 2020/2021/2023 | Dice 0.866-0.901 | Dice coefficient |

## Open Challenges

1. **Modality mismatch:** CT and MRI have fundamentally different intensity distributions and spatial resolutions
2. **Missing modalities:** In clinical practice, patients may have only MRI or only CT — models must handle missing modalities gracefully
3. **Spatial registration:** CT and MRI volumes often use different field-of-view and voxel spacing — requires registration before fusion
4. **Training data scarcity:** Few datasets provide paired CT+MRI brain scans with labels
5. **Interpretability:** Multimodal models are harder to interpret than single-modality models

## Relevance to This Project

**Paper #1 (Almadhor 2026)** is the most directly relevant — it explicitly addresses CT-MRI integration for brain tumor classification. Paper #5 (Zhang 2025) is highly relevant for the synthetic CT approach when paired CT data is scarce. Paper #2 (MANet) provides the best architecture template for attention-based fusion.
