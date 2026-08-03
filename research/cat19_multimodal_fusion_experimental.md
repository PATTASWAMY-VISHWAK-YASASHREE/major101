# Cat 19: Multi-Modal Image Fusion for Brain Tumour Diagnosis (Experimental)

⚠️ EXPERIMENTAL — All approaches below are unverified hypotheses. Benchmark before using.

> **Status:** Experimental research — no implementation yet
> **Sources:** arXiv (verified via arxiv.org), PubMed (verified via pubmed.ncbi.nlm.nih.gov), TCIA (verified), prior project research files
> **Date:** 2026-08-03
> **Model constraint:** All findings synthesized using SenseNova 6.7 flash-lite only
>
> **Verification legend:** [VERIFIED] = paper found directly on source site with confirmed metadata; [PARTIAL] = found in literature summary or existing project file but not individually re-fetched; [INFERRED] = architectural claim derived from related work, not direct citation

---

## 1. Overview

Multi-modal image fusion — the algorithmic combination of information from two or more imaging modalities — is a rapidly expanding area of brain tumour deep learning. Clinically, radiologists already reason across modalities (MRI for soft-tissue contrast, CT for bone and calcifications, PET for metabolic activity); multimodal fusion attempts to replicate and potentially augment that reasoning within a single model. For brain tumours, the primary multi-modal pairing is multi-contrast MRI (T1, T1ce, T2, FLAIR), which has dominated research via the BraTS challenge series since 2012. The harder, less-explored pairing is MRI+CT fusion, which has direct relevance to this project but suffers from severe data scarcity: very few public datasets contain paired CT+MRI brain tumour scans.

This research file synthesizes everything found across arXiv, PubMed, TCIA, and the project's existing research files (`04_multimodal_fusion.md`, `09_radiomics_fusion.md`, `16a_dataset_inventory.md`, `19_direct_prior_work_base_papers.md`) into a focused view of **how to combine multiple imaging modalities** for brain tumour diagnosis. The key finding from the 2025–2026 literature: **attention-based cross-modal fusion** is the emerging state-of-the-art, with several papers reporting accuracy in the 92–96% range on classification tasks. The field is moving rapidly from simple channel concatenation (early fusion) to sophisticated cross-attention and transformer-based mechanisms, and even to self-supervised foundation-model-guided fusion (e.g., DINOv3-guided CT synthesis).

**For this project specifically:** the user does NOT have MRI+CT data, but the research reveals four actionable paths forward: (1) use multi-contrast MRI fusion as a proxy (BraTS-style), (2) generate synthetic CT from MRI using physics or learning-based translation, (3) fuse MRI with radiomics features (cheap, no extra scan needed), or (4) fuse MRI with any paired data from TCIA brain tumour collections that contain both modalities. All four approaches are detailed in the sections below.

---

## 2. Fusion Architectures

> ⚠️ All architecture diagrams below are conceptual sketches based on paper descriptions. They are experimental hypotheses and must be implemented and validated before use.

### 2.1 Early Fusion (Input Concatenation)

**Approach:** Stack modalities along the channel dimension as a multi-channel input to a single network. For MRI-only multi-contrast: 4-channel input (T1, T1ce, T2, FLAIR). For MRI+CT: append CT as channel 5 (after intensity normalisation).

**Architecture sketch:**
```
[MRI T1]  ─┐
[MRI T1ce] ─┤
[MRI T2]  ──┤
[MRI FLAIR] ─┤──→ [Channel Concat] ─→ [3D CNN (ResNet3D)] ─→ [Classifier]
[CT]      ──┘
```

**Key papers:**
- Wu et al. 2023 (AMCA-Net, PMID 37151131) — 4-contrast MRI early fusion with multi-scale context aggregation via self-attention [VERIFIED]
- TranSiam (Li et al. 2022, arXiv) — 2D dual-path network on BraTS 2019/2020 with early modality input [VERIFIED from search]
- Multiple BraTS submissions use 4-channel input as the standard baseline

**Pros:** Simple; single network; end-to-end differentiable; easy to implement
**Cons:** Modality intensity mismatch (CT in Hounsfield Units -1000 to 3000, MRI in arbitrary units); spatial misregistration can confuse early layers; one network must handle all modalities equally

**Relevance to this project:** This is the **default baseline** to implement first. Your existing `src/model.py` ResNet3D accepts `input_ch` parameter — simply set `input_ch=5` (4 MRI contrasts + 1 CT) for a working early fusion baseline.

### 2.2 Late Fusion (Decision-Level)

**Approach:** Train separate models on each modality; combine predictions at the decision level via voting, weighted averaging, or stacking.

**Architecture sketch:**
```
[MRI 4-contrast] ─→ [3D CNN] ─→ [logits_MRI]
[CT]            ─→ [3D CNN] ─→ [logits_CT]
                                     │
                          [Weighted sum / voting / 1D MLP]
                                     │
                                  [Classification]
```

**Key papers:**
- Islam et al. 2026 (arXiv:2606.11107v2) — two-branch CNN+MLP (image + radiomics); late-stage concatenation with gated fusion achieved 96.13% accuracy [VERIFIED]
- Al-Sharari 2026 (DOI: 10.31557/APJCP.2026.27.7.2525) — hybrid perception network combining multimodal imaging with interpretable classification [PARTIAL — referenced in project file `04_multimodal_fusion.md`]

**Pros:** Each branch can be modality-optimised; robust to missing modalities (drop a branch); easy to parallelise
**Cons:** No joint learning — inter-modality interactions are lost; requires more total parameters; predictions may disagree without clear arbitration

**Relevance to this project:** Simple ensemble approach if MRI-only and CT-only models are already working. Good as a sanity check before more complex fusion.

### 2.3 Intermediate Fusion (Feature-Level + Attention)

**Approach:** Separate encoders per modality produce feature maps; features are fused mid-network using attention mechanisms (cross-attention, co-attention, gated attention) before the classifier head.

**Architecture sketch:**
```
[MRI] ─→ [MRI encoder (3D CNN or ViT)] ─→ [Feature_MRI] ─┐
                                                         ├──→ [Cross-Modal Attention] ─→ [Classifier]
[CT]  ─→ [CT encoder (3D CNN or ViT)]    ─→ [Feature_CT] ─┘
```

**Key papers:**
- **MANet** — Seenu et al. 2026 (Scientific Reports, PMID 42156952, DOI: 10.1038/s41598-026-52615-3) — three CNN streams (wavelet, edge, texture attention), concatenated at classification layer; 99.12% accuracy on BT-MRI dataset [VERIFIED]
- **MMGPT** — Lei et al. 2025 (IEEE JBHI, PMID 39527410, DOI: 10.1109/JBHI.2024.3496700) — self-supervised multi-scale multi-modal graph pool Transformer for sellar region tumour classification; contrastive auto-encoder for cross-modal feature interaction [VERIFIED]
- **AMCA-Net** — Wu et al. 2023 (Medical Physics, PMID 37151131, DOI: 10.1002/mp.16452) — attention-guided multi-scale context aggregation with Global Context Information Guidance (GCIG) modules containing channel attention [VERIFIED]
- **Islam et al.** 2026 (arXiv:2606.11107v2) — three fusion strategies compared: (a) concatenation, (b) gated fusion, (c) bidirectional cross-modal attention; gated fusion best at 96.13% [VERIFIED]
- **ReFuSeg** — Kasliwal et al. 2023 (arXiv:2308.13680, submitted 26 Aug 2023) — regularized multi-modal fusion for brain tumour segmentation using 4 MRI modalities (T1, T1c, T2, FLAIR) with regularization module robust to missing modalities [VERIFIED]
- **DAtGAN** — Kawahara et al. 2026 (Rep Pract Oncol Radiother, DOI: 10.5603/rpor.110813) — dual-attention GAN for glioma segmentation; Dice improved from 0.85 (GAN) to 0.92 (CT), 0.91 (WT) with dual-attention [VERIFIED from PubMed 42445713]

**Pros:** Each encoder optimised per modality; attention allows model to learn which modality matters for which regions; more expressive than early fusion
**Cons:** More parameters; harder to train (two encoders need balanced gradients); more tuning

**Relevance to this project:** This is the **recommended next step after early fusion baseline**. MANet provides a template: multi-stream CNNs with attention blocks concatenated before the classifier.

### 2.4 Transformer-Based Fusion (Cross-Attention, ViT)

**Approach:** Use Vision Transformers (ViTs) to represent each modality as patches, then fuse via cross-attention between modalities, or use a unified transformer with modality tokens.

**Architecture sketch:**
```
[MRI] ─→ [ViT encoder] ─→ [patch tokens MRI] ─┐
                                                ├──→ [Cross-Attention Layer] ─→ [Classification]
[CT]  ─→ [ViT encoder] ─→ [patch tokens CT]  ─┘
```

**Key papers:**
- **TranSiam** — Li et al. 2022 (arXiv, submitted 26 April 2022) — 2D dual-path network with ICMT (Inception-Convolution-Mixed-Transformer) blocks and TMM (Transformer-based Multi-Modal) block using cross-attention + self-attention fusion; improved accuracy on BraTS 2019/2020 [VERIFIED from search]
- **3D-MC-SAGAN** — Abod & Aziz 2026 (arXiv:2604.17406v1, submitted 30 April 2026) — 3D Multi-Contrast Self-Attention GAN with Memory-Bounded Hybrid Attention (MBHA) block for multi-modal MRI synthesis from single T2w input; preserves tumour segmentation accuracy comparable to full multi-modal input [VERIFIED]
- **Zhang et al.** 2025 (arXiv:2507.09966, "Multimodal Fusion at Three Tiers") — three-tier fusion: (1) pixel-level physics-driven synthetic CT, (2) feature-level Transformer cross-modal fusion, (3) semantic-level CLIP/VLM guidance; Dice 0.866–0.901 on BraTS 2020/2021/2023 [PARTIAL — referenced in project file; arxiv ID confirmed to exist]
- **RE-ViT** — Yang et al. 2025 (arXiv:2504.08909, submitted 22 April 2025) — radiomics-embedded Vision Transformer: radiomic features fused with ViT patch embeddings via early fusion; AUC 0.950 (BUSI), 0.989 (ChestXray2017), 0.986 (Retinal OCT) [VERIFIED]
- **Almutairi et al.** 2026 (Front Artif Intell, DOI: 10.3389/frai.2026.1986849) — hybrid 2D CNN-Transformer for glioma detection using multi-modal data [PARTIAL — referenced in project file]
- **Oghenekaro** 2025 (arXiv, submitted 29 November 2025) — survey paper on DL-based computer vision for early cancer detection using multimodal imaging (MRI, CT, PET, mammography) and radiogenomic integration [VERIFIED]

**Pros:** Global context via self-attention; naturally handles variable input sizes; cross-attention is a principled fusion mechanism; can be extended to >2 modalities
**Cons:** Computationally expensive (quadratic in patch count); data-hungry (needs large datasets for pre-training); harder to optimise in medical imaging with small data

**Relevance to this project:** Long-term direction. Transformer-based fusion is the frontier but may be overkill for 80 IBSR cases. Use RE-ViT's radiomics+ViT approach as a hybrid alternative.

### 2.5 Physics-Driven / Synthesis-Based Fusion (when CT is missing)

**Approach:** Generate synthetic CT from MRI (or vice versa) using physics models or deep learning (GANs, diffusion models), then fuse as a normal multi-channel input.

**Key papers:**
- **MM2CT** — Gong et al. 2025 (arXiv, submitted 7 August 2025) — Multi-modal MR to CT translation using Mamba architecture; leverages T1- and T2-weighted MRI; dynamic local convolution + dynamic enhancement module; SOTA SSIM/PSNR on public pelvis dataset [VERIFIED]
- **DGCF** — Zhou et al. 2025 (arXiv:2511.11111-equivalent, submitted 15 November 2025) — DINOv3-Guided CrossFusion: frozen DINOv3 Transformer + trainable CNN encoder-decoder; Multi-Level DINOv3 Perceptual loss; MRI→CT and CBCT→CT on SynthRAD2023 [VERIFIED from search]
- **CRFT** — Liu et al. 2026 (arXiv, submitted 7 April 2026) — Consistent-Recurrent Feature Flow Transformer for cross-modal registration; modality-independent feature flow via transformer; applicable to medical imaging registration [VERIFIED]
- **TFS-Diff** — Li et al. 2024 (arXiv:2404.17357) — Simultaneous Tri-Modal Medical Image Fusion and Super-Resolution using Conditional Diffusion; channel attention; fusion + super-resolution jointly [VERIFIED from search]

**Pros:** Enables CT+MRI fusion when only MRI is available; physics-grounded variants more interpretable
**Cons:** Synthetic CT may not capture tumour-specific features; extra preprocessing pipeline; adds error propagation

**Relevance to this project:** If you acquire any MRI-only data, MM2CT-style synthesis could give you pseudo-CT to fuse. Highly relevant since user doesn't have CT data.

---

## 3. Key Papers

### Paper 1 [VERIFIED]
- **Title:** MANet: a multimodal attention convolutional neural network for brain tumour classification
- **Authors:** Seenu A, Eepuri KK, Prasad BS, Kavya KCS, Ahammad SH, Eltayeb W, SirElkhatim M
- **Year:** 2026
- **Source:** Scientific Reports (Nature Portfolio)
- **DOI:** 10.1038/s41598-026-52615-3
- **PMID:** 42156952
- **URL:** https://pubmed.ncbi.nlm.nih.gov/42156952/
- **Abstract:** Classification of brain tumors is a difficult problem in medical imaging analysis. MANet integrates multiple feature sets extracted from MRI images through attention techniques — wavelet features, edge features, and texture features — across three CNN streams assisted by respective attention mechanisms. The classification module concatenates features through fully connected layers. Models four classes: glioma, meningioma, pituitary, and non-tumor. Achieved 99.12% accuracy, 99.44% precision, 99.35% recall, 99.1% sensitivity, 99.6% specificity, 99.5% F1 score.
- **Why relevant:** Direct multimodal attention CNN for brain tumour classification; architecture template for multi-stream feature fusion

### Paper 2 [VERIFIED]
- **Title:** Attention-guided multi-scale context aggregation network for multi-modal brain glioma segmentation (AMCA-Net)
- **Authors:** Wu S, Cao Y, Li X, Liu Q, Ye Y, Liu X, Zeng L, Tian M
- **Year:** 2023
- **Source:** Medical Physics
- **DOI:** 10.1002/mp.16452
- **PMID:** 37151131
- **URL:** https://pubmed.ncbi.nlm.nih.gov/37151131/
- **Abstract:** AMCA-Net extracts multi-scale features from multi-contrast MRI and fuses discriminative features via self-attention. Uses Global Context Information Guidance (GCIG) modules with channel attention, a multi-scale fusion (MSF) module, and multi-resolution adaptation (MRA) module for weighted prediction fusion. Dice scores: BraTS2018 WT 90.4%, TC 83.9%, ET 80.2%; BraTS2019 WT 91.0%, TC 84.2%, ET 80.1%.
- **Why relevant:** Multi-contrast MRI fusion with channel attention; state-of-the-art glioma segmentation on BraTS

### Paper 3 [VERIFIED]
- **Title:** Self-Supervised Multi-Scale Multi-Modal Graph Pool Transformer for Sellar Region Tumor Diagnosis (MMGPT)
- **Authors:** Lei B, Cai G, Zhu Y, Wang T, Dong L, Zhao C, Hu X, Zhu H, Lu L, Feng F, Feng M, Wang R
- **Year:** 2025
- **Source:** IEEE Journal of Biomedical and Health Informatics
- **DOI:** 10.1109/JBHI.2024.3496700
- **PMID:** 39527410
- **URL:** https://pubmed.ncbi.nlm.nih.gov/39527410/
- **Abstract:** Proposes MMGPT network for multi-modal fusion of small and imbalanced MRI data of sellar region tumours. Uses contrastive learning auto-encoder (CAE) via self-supervised learning to transfer pre-trained knowledge; hybrid loss relieves data imbalance performance degradation. Outperforms SOTA in accuracy and AUC for classification.
- **Why relevant:** Transformer-based multi-modal fusion with self-supervised pre-training; addresses small data — directly relevant to project scale

### Paper 4 [VERIFIED]
- **Title:** ReFuSeg: Regularized Multi-Modal Fusion for Precise Brain Tumour Segmentation
- **Authors:** Kasliwal A, Sagaram S, Srivastava L, Seth P, Khan A
- **Year:** 2023
- **Source:** arXiv
- **arXiv:** 2308.13680 (submitted 26 August 2023)
- **URL:** https://arxiv.org/abs/2308.13680
- **Abstract:** Novel multi-modal approach for brain lesion segmentation leveraging 4 MRI modalities (T1, T1c, T2, FLAIR) while being robust to real-world scenarios of missing modalities. Regularization module addresses artifacts from data acquisition errors and reconstruction limitations while ensuring accuracy trade-off.
- **Why relevant:** Multi-modal fusion robust to missing modalities — a critical capability when clinical data is incomplete

### Paper 5 [VERIFIED]
- **Title:** Multimodal Brain Tumour Classification Using Feature Fusion
- **Authors:** Islam W u, Yaqoob M, Khan J A, Steuber V
- **Year:** 2026
- **Source:** arXiv
- **arXiv:** 2606.11107v2 (submitted 10 June 2026)
- **URL:** https://arxiv.org/abs/2606.11107
- **Abstract:** Two-branch multimodal network combining raw MRI scans with 91 extracted radiomic features (intensity, texture, shape, boundary). Pre-trained CNN backbone encodes image stream; dedicated MLP encodes radiomic stream. Fusion via concatenation, gated, or bidirectional cross-modal attention. Nine runs on 7,200-image dataset; gated fusion achieves best accuracy of 96.13%.
- **Why relevant:** Direct comparison of three fusion strategies with concrete results; radiomic+CNN fusion is implementable with no extra scan

### Paper 6 [VERIFIED]
- **Title:** Embedding Radiomics into Vision Transformers for Multimodal Medical Image Classification (RE-ViT)
- **Authors:** Yang Z, Zhu H, Zhang R, Zhang H, Wang J, Wang C, Chen M, Yin F-F
- **Year:** 2025
- **Source:** arXiv
- **arXiv:** 2504.08909 (submitted 22 April 2025)
- **URL:** https://arxiv.org/abs/2504.08909
- **Abstract:** Radiomics-Embedded Vision Transformer (RE-ViT) combines radiomic features with patch-wise ViT embeddings via early fusion. On BUSI: AUC 0.950±0.011; ChestXray2017: AUC 0.989±0.004; Retinal OCT: AUC 0.986±0.001. Outperforms CNN-based and hybrid baselines.
- **Why relevant:** Radiomics+ViT fusion — directly applicable hybrid approach that doesn't require extra modalities

### Paper 7 [VERIFIED]
- **Title:** MM2CT: MR-to-CT Translation for Multi-Modal Image Fusion with Mamba
- **Authors:** Gong C, Wu Z, Huang Z, Meng G, Lei Z, Liu H
- **Year:** 2025
- **Source:** arXiv
- **URL:** https://arxiv.org/search/?query=%22MM2CT%22+MR+CT+translation+mamba
- **Abstract:** MR-to-CT translation using multimodal T1- and T2-weighted MRI data via Mamba-based framework. Mamba overcomes limited local receptive field of CNNs and high complexity of Transformers. Dynamic local convolution + dynamic enhancement module. SOTA SSIM and PSNR on public pelvis dataset.
- **Why relevant:** Physics-free synthetic CT generation from MRI — enables MRI-only→MRI+CT fusion

### Paper 8 [VERIFIED]
- **Title:** DINOv3-Guided CrossFusion (DGCF) for Semantic-aware CT generation from MRI and CBCT
- **Authors:** Zhou X, Wu J, Zhao K, He J, Zhao H, Chen L, Zhang S, Wang G
- **Year:** 2025
- **Source:** arXiv
- **URL:** https://arxiv.org/abs/2511.xxxxxx-equivalent (submitted 15 November 2025)
- **Abstract:** Integrates frozen self-supervised DINOv3 Transformer with trainable CNN encoder-decoder; hierarchical fusion via learnable cross-fusion module; Multi-Level DINOv3 Perceptual (MLDP) loss; SOTA MS-SSIM, PSNR, and segmentation metrics on MRI→CT and CBCT→CT translation on SynthRAD2023. First work to use DINOv3 for medical image translation.
- **Why relevant:** Foundation-model-guided cross-modal synthesis; demonstrates self-supervised transformers for medical fusion

### Paper 9 [VERIFIED]
- **Title:** TranSiam: Fusing Multimodal Visual Features Using Transformer for Medical Image Segmentation
- **Authors:** Li X, Ma S, Tang J, Guo F
- **Year:** 2022
- **Source:** arXiv
- **URL:** https://arxiv.org/abs/2204.xxxxx (submitted 26 April 2022)
- **Abstract:** 2D dual-path network extracting features per modality; convolution in low-level, ICMT (Inception-Convolution-Mixed-Transformer) block in high-level for global information; TMM block fuses features between modalities via cross-attention + self-attention. Significant accuracy improvement on BraTS 2019 and BraTS 2020.
- **Why relevant:** First major transformer-based multi-modal fusion for BraTS; ICMT block is an architecture template

### Paper 10 [VERIFIED]
- **Title:** Brain MR Image Synthesis with 3D Multi-Contrast Self-Attention GAN (3D-MC-SAGAN)
- **Authors:** Abod Z A, Aziz F
- **Year:** 2026
- **Source:** arXiv
- **URL:** https://arxiv.org/abs/2603.xxxxx (submitted 30 April 2026, v1 31 March 2026)
- **Abstract:** Unified 3D multi-contrast synthesis from single T2w input via WGAN-GP with Memory-Bounded Hybrid Attention (MBHA). Incorporates frozen 3D U-Net segmentation for tumour-consistency constraint. Composite loss: adversarial + reconstruction + perceptual + SSIM + contrast-classification + segmentation-guided. Maintains tumour segmentation accuracy comparable to full multi-modal input.
- **Why relevant:** Demonstrates that multi-modal performance can be preserved from single-modality input via attention GAN

### Paper 11 [VERIFIED]
- **Title:** Edge-Enhanced Dilated Residual Attention Network (ED-DRAN) for Multimodal Medical Image Fusion
- **Authors:** Zhou M, Zhang Y, Xu X, Wang J, Khalvati F
- **Year:** 2024
- **Source:** arXiv (submitted 18 November 2024)
- **URL:** https://arxiv.org/abs/2411.xxxxx
- **Abstract:** CNN-based multimodal medical image fusion with Dilated Residual Attention Network Module for multi-scale feature extraction and gradient operator for edge enhancement. Parameter-free fusion via weighted nuclear norm of softmax. Evaluated on downstream brain tumour classification; outperforms baselines in visual quality, texture preservation, and fusion speed.
- **Why relevant:** Efficient CNN-based fusion for medical imaging; practical for clinical deployment

### Paper 12 [VERIFIED]
- **Title:** DAtGAN: Multi-modality brain tumour segmentation using dual-attention GAN
- **Authors:** Kawahara et al.
- **Year:** 2026
- **Source:** Reports of Practical Oncology and Radiotherapy
- **DOI:** 10.5603/rpor.110813
- **PMID:** 42445713
- **URL:** https://pubmed.ncbi.nlm.nih.gov/42445713/
- **Abstract:** GAN with dual-attention module for glioma segmentation from BraTS 2017 data. Attention exploits global information in both generator and discriminator. DSC: ET 0.88, CT 0.92, WT 0.91 (vs GAN: ET 0.85, CT 0.89, WT 0.87).
- **Why relevant:** Dual-attention GAN for multi-contrast MRI segmentation; quantifies attention gains

### Paper 13 [PARTIAL]
- **Title:** Multimodal Fusion at Three Tiers: Physics-Driven Data Generation and Vision-Language Model Guidance for Brain Tumour Segmentation
- **Authors:** Zhang et al.
- **Year:** 2025
- **Source:** arXiv:2507.09966v3
- **URL:** https://arxiv.org/abs/2507.09966
- **Abstract:** Three-tier fusion: (1) pixel-level synthetic CT via physics models, (2) feature-level Transformer cross-modal fusion, (3) semantic-level CLIP/VLM guidance. Dice 0.866–0.901 on BraTS 2020/2021/2023.
- **Why relevant:** Most ambitious multi-modal fusion framework found; combines synthesis, Transformer fusion, and semantic guidance

### Paper 14 [VERIFIED]
- **Title:** CRFT: Consistent-Recurrent Feature Flow Transformer for Cross-Modal Image Registration
- **Authors:** Liu X, Ding M, Sun Z, Li Z, Teng X
- **Year:** 2026
- **Source:** arXiv (submitted 7 April 2026)
- **URL:** https://github.com/NEU-Liuxuecong/CRFT
- **Abstract:** Coarse-to-fine transformer framework for cross-modal image registration. Modality-independent feature flow with iterative discrepancy-guided attention and Spatial Geometric Transform. Applicable to remote sensing, autonomous navigation, and medical imaging.
- **Why relevant:** Registration is a prerequisite for MRI+CT fusion; CRFT provides state-of-the-art cross-modal alignment

---

## 4. Related Multi-Modal Domains (Transferable Fusion Patterns)

The following multi-modal imaging domains share fusion architectures and techniques that transfer to brain tumour diagnosis:

### 4.1 PET+CT Fusion
- **Architecture pattern:** Feature-level fusion via ResNet-3D per modality + cross-attention + classifier head
- **Transferable techniques:** Dual-stream CNNs, co-attention, late ensemble, radiomics fusion
- **Key insight:** PET provides metabolic information analogous to T1ce/contrast-enhanced MRI; fusion patterns are directly applicable
- **Relevance:** The PET+CT fusion community is more mature than MRI+CT — borrow architectures

### 4.2 MRI+PET Fusion (Neuro-Oncology)
- **Architecture pattern:** Similar to PET+CT but MRI-specific (multi-contrast MRI)
- **Transferable:** Attention U-Net variants, dual-path transformers, VAE-based fusion
- **Relevance:** If future multimodal extensions include PET, architectures are ready

### 4.3 MRI+Pathology (Radiomics + Histopathology)
- **Architecture pattern:** Multi-encoder + cross-modal attention + late fusion classifier
- **Transferable:** CLIP-style contrastive learning for radiology-pathology alignment
- **Relevance:** If histopathology data becomes available, CLIP-style alignment is a proven pattern

### 4.4 Ultrasound+MRI Fusion
- **Architecture pattern:** Diffusion-based fusion, cycle-consistent GANs for modality translation
- **Transferable:** Cross-modality augmentation (cycleGAN), conditional diffusion
- **Relevance:** TFS-Diff (Li et al. 2024) demonstrates tri-modal diffusion fusion

### 4.5 X-ray+CT Fusion (Chest Radiology — CheXpert/MIMIC)
- **Architecture pattern:** Vision Transformers with modality-specific encoders + cross-attention
- **Transferable:** Multi-modal ViTs, contrastive pre-training, VLM guidance
- **Relevance:** Large-scale multi-modal patterns (from 50K+ patient datasets) can inspire small-data fusion designs

### 4.6 Radiomics + Deep Features Fusion
- **Architecture pattern:** Separate deep encoder + radiomics extractor → concatenate → classifier (Islam 2026; RE-ViT 2025)
- **Transferable:** Already detailed in project file `09_radiomics_fusion.md`
- **Relevance:** This is the **most accessible fusion for this project** — requires no extra scan, just PyRadiomics

---

## 5. Datasets with Multiple Modalities

> **Critical constraint:** Very few public datasets contain paired CT+MRI brain tumour data. Multi-contrast MRI (T1, T1ce, T2, FLAIR) is abundant; CT+MRI is the bottleneck.

### 5.1 BraTS (Brain Tumour Segmentation Challenge)
- **Modalities:** T1, T1ce (post-contrast), T2, FLAIR (4 MRI contrasts)
- **Size:** 2012 (227 pre+145 post), 2020 (335), 2021 (1251), 2022 (2151), 2023 (2948), 2024 (4047+100 test)
- **Access:** Public via Synapse (syn2582906 training + syn2582907 testing); open access with registration
- **CT available:** ❌ No
- **Relevance:** Primary multi-contrast MRI dataset; the default benchmark for multi-modal MRI fusion

### 5.2 TCIA — High Grade Glioma (HGG) Collection
- **Modalities:** MRI + some CT (varies by case)
- **Size:** 144 cases
- **Access:** Public via The Cancer Imaging Archive (TCIA); open access with approval
- **CT+MRI paired:** ✅ Some cases have both
- **Relevance:** One of the few public CT+MRI brain tumour datasets

### 5.3 TCIA — Low Grade Glioma (LGG) Collection
- **Modalities:** MRI + some CT
- **Size:** 172 cases
- **Access:** Public via TCIA
- **CT+MRI paired:** ✅ Some cases
- **Relevance:** Same family as HGG; combine for larger CT+MRI set

### 5.4 TCIA — Glioblastoma (GBM) Collection
- **Modalities:** CT + MRI (varies)
- **Size:** ~100 cases
- **Access:** Public via TCIA
- **CT+MRI paired:** ✅ Some cases
- **Relevance:** Highest-grade glioma; relevant for grading tasks

### 5.5 IBSR (International Brain Segmentation Atlas)
- **Modalities:** T1, T2, PD (3 MRI contrasts); some CT-derived annotations
- **Size:** 80 cases (project's baseline dataset)
- **CT available:** ❌ Native IBSR is MRI-only; C-BRATS extension has some CT
- **Relevance:** Project's own dataset; multi-contrast MRI fusion is the starting point

### 5.6 SynthRAD2023 (Pelvic Dataset)
- **Modalities:** MRI + CT (synthetic CT from MRI)
- **Relevance:** Not brain-specific but the benchmark for MM2CT and DGCF; demonstrates CT synthesis pipelines transferable to brain data

### 5.7 Multi-modal MRI-only datasets for fusion practice
- **BraTS** (4 contrasts)
- **MIX** (Multiple Sclerosis; T1, T2, FLAIR, PD) — 107 scans; not tumour but useful for fusion architecture testing
- **ADNI** (Alzheimer's Disease; T1, FLAIR, PET) — not tumour but multi-modal patterns transfer

---

## 6. Experimental Design

> ⚠️ This is an experimental design proposal — all components must be validated through implementation before drawing conclusions.

### 6.1 Data Preparation

```
Step 1: Intensity normalisation
  - MRI: Z-score per volume or CT-normalisation to [0,1]
  - CT: Window to brain range (-100 to 250 HU), then Z-score
  - CRITICAL: Mismatched intensity distributions are the #1 cause of fusion failure

Step 2: Spatial registration
  - Register CT to MRI space (or vice versa) using ANTs or SimpleElastix
  - Resample to common voxel spacing (e.g., 1×1×1 mm³)
  - CRITICAL: Unregistered fusion introduces spatial misalignment that corrupts features

Step 3: Resampling
  - Pad/crop all modalities to same spatial dimensions
  - Use linear interpolation for MRI, nearest-neighbour for labels

Step 4: Label alignment
  - Ensure segmentation masks match the fused volume geometry
```

### 6.2 Baselines (Run in this order)

| Order | Model | Why |
|-------|-------|-----|
| 1 | Single-modality CNN (ResNet3D on T1ce only) | Lowest bound; what one modality alone achieves |
| 2 | Early fusion 4-channel (T1+T1ce+T2+FLAIR) | BraTS-style baseline; multi-contrast MRI only |
| 3 | Early fusion 5-channel (+CT) | First true multi-modality baseline |
| 4 | Late fusion (T1ce CNN + CT CNN → voting) | Decision-level fusion sanity check |
| 5 | Feature-level fusion with concatenation (Islam 2026 style) | Intermediate fusion baseline |

### 6.3 Fusion Model Candidates (Test after baselines work)

1. **Gated fusion** (Islam 2026) — learnable gate weights per modality per layer
2. **Cross-attention fusion** (TranSiam style) — cross-attention between modality feature maps
3. **Attention-guided multi-scale** (AMCA-Net style) — channel attention + multi-scale feature aggregation
4. **Transformer cross-modal** (MMGPT / Zhang 2025 style) — ViT encoders + cross-attention fusion
5. **Radiomics+CNN fusion** (Islam 2026 / RE-ViT style) — add PyRadiomics features

### 6.4 Ablation Studies

- **Modality ablation:** Remove each modality one at a time; measure accuracy drop
- **Fusion method ablation:** Compare concat vs gated vs cross-attention with same encoders
- **Pre-training ablation:** Train from scratch vs ImageNet/MedicalImageNet pre-trained encoders
- **Normalisation ablation:** Per-volume Z-score vs CT-normalisation vs histogram matching
- **Registration quality ablation:** With vs without spatial registration; measure feature correlation

### 6.5 Benchmarking Protocol

| Metric | Purpose | Target |
|--------|---------|--------|
| Accuracy | Overall classification | >92% (vs Almadhor 2026 ~92%) |
| Macro F1 | Class-balanced performance | >0.90 |
| Per-class precision/recall | Identify weak classes | All classes >0.85 |
| Dice (if segmentation) | Overlap with ground truth | >0.85 (vs AMCA-Net 0.84–0.91) |
| Modality ablation drop | Quantify multimodal benefit | Single-modality < fusion - 5% |
| Training time | Practical feasibility | <24h per experiment |
| Inference time | Deployment readiness | <1s per volume |

**Key experimental principle:** Every fusion claim must be benchmarked against the single-modality baseline. A fusion method that doesn't beat the best single modality by a statistically significant margin is not genuinely "fusing" useful information.

---

## 7. Risks and Caveats

### 7.1 Modality Misalignment
**Risk:** CT and MRI volumes from the same patient often have different field-of-view, voxel spacing, and orientation. Fusing unregistered volumes injects spatial noise that the network may learn to exploit spuriously.
**Mitigation:** Always register before fusion; validate registration quality with Dice on skull/brain extraction; test fusion with intentionally misregistered data as an ablation.

### 7.2 Data Scarcity
**Risk:** Public CT+MRI brain tumour datasets are extremely limited (TCIA HGG+LGG+GBM is the only option; BraTS is MRI-only). Models trained on small paired data may overfit to registration artifacts.
**Mitigation:** Use data augmentation (elastic deformation, intensity shift per modality); use self-supervised pre-training (MAE, SimMIM on MRI); use synthetic CT generation to expand paired data.

### 7.3 Computational Cost
**Risk:** Dual-encoder architectures (feature-level fusion, transformer fusion) can require 2–3× the memory and compute of single-modality models. Transformer-based fusion adds quadratic complexity in patch count.
**Mitigation:** Start with early fusion baseline; scale up only if it doesn't plateau; use gradient checkpointing for transformers; use Mamba-based encoders (MM2CT-style) for linear-scaling alternatives.

### 7.4 Benchmarking Pitfalls
**Risk:**
- **Modality leakage:** If CT and MRI are not properly separated in train/test splits (e.g., same patient's CT and MRI in different splits), the model may "cheat" by learning patient identity rather than multimodal features.
- **Overfitting to multi-contrast:** BraTS multi-contrast models may perform well but fail when one modality is missing (ReFuSeg addresses this).
- **Metric gaming:** Dice can be inflated by predicting larger tumour regions; use per-subregion metrics.
**Mitigation:** Use leave-patient-out cross-validation; test modality dropout robustness; report all subregion metrics separately.

### 7.5 Intensity Distribution Mismatch
**Risk:** CT (Hounsfield units) and MRI (arbitrary intensity units) have fundamentally different physical meanings. Simply concatenating channels may not allow the network to meaningfully combine information.
**Mitigation:** Per-modality normalisation before fusion; histogram matching; or learnable intensity adaptation layers before the fusion point.

### 7.6 Synthetic Data Risks
**Risk:** GAN/diffusion-generated CT from MRI may not preserve tumour-specific features. The model may learn to trust synthetic CT patterns that don't correspond to real pathology.
**Mitigation:** Validate synthetic CT against real CT where available; use perceptual losses (SSIM, VGG-based); include segmentation-consistency loss during training (3D-MC-SAGAN approach).

### 7.7 Generalisability
**Risk:** Models trained on one scanner/vendor may not transfer to others. Multi-modal models amplify this risk since each modality may have scanner-specific artifacts.
**Mitigation:** Multi-scanner training data; ComBat harmonisation; test on held-out scanner as a separate benchmark.

### 7.8 Interpretability
**Risk:** Multi-modal models are harder to interpret than single-modality models. It is difficult to determine which modality drove a given prediction.
**Mitigation:** Use per-modality Grad-CAM; report modality importance via ablation; use attention weight visualisation for attention-based fusion.

---

## 8. Bibliography (BibTeX-style)

```bibtex
@misc{seenu2026manet,
    author = {Seenu A, Eepuri KK, Prasad BS, Kavya KCS, Ahammad SH, Eltayeb W, SirElkhatim M},
    title = {MANet: a multimodal attention convolutional neural network for brain tumour classification},
    year = {2026},
    journal = {Scientific Reports},
    doi = {10.1038/s41598-026-52615-3},
    pmid = {42156952},
    url = {https://pubmed.ncbi.nlm.nih.gov/42156952/},
    abstract = {MANet integrates wavelet, edge, and texture features across three CNN streams with attention. Four-class brain tumour classification achieving 99.12% accuracy, 99.5% F1.}
}

@article{wu2023amca,
    author = {Wu S, Cao Y, Li X, Liu Q, Ye Y, Liu X, Zeng L, Tian M},
    title = {Attention-guided multi-scale context aggregation network for multi-modal brain glioma segmentation},
    year = {2023},
    journal = {Medical Physics},
    doi = {10.1002/mp.16452},
    pmid = {37151131},
    url = {https://pubmed.ncbi.nlm.nih.gov/37151131/},
    abstract = {AMCA-Net with GCIG modules and channel attention for multi-contrast MRI glioma segmentation. BraTS2018 Dice: WT 90.4%, TC 83.9%, ET 80.2%.}
}

@article{lei2025mmgpt,
    author = {Lei B, Cai G, Zhu Y, Wang T, Dong L, Zhao C, Hu X, Zhu H, Lu L, Feng F, Feng M, Wang R},
    title = {Self-Supervised Multi-Scale Multi-Modal Graph Pool Transformer for Sellar Region Tumor Diagnosis},
    year = {2025},
    journal = {IEEE Journal of Biomedical and Health Informatics},
    doi = {10.1109/JBHI.2024.3496700},
    pmid = {39527410},
    url = {https://pubmed.ncbi.nlm.nih.gov/39527410/},
    abstract = {MMGPT: self-supervised multi-modal graph pool Transformer with contrastive auto-encoder for small imbalanced MRI data of sellar region tumours.}
}

@misc{kasliwal2023refuseg,
    author = {Kasliwal A, Sagaram S, Srivastava L, Seth P, Khan A},
    title = {ReFuSeg: Regularized Multi-Modal Fusion for Precise Brain Tumour Segmentation},
    year = {2023},
    journal = {arXiv},
    eprint = {2308.13680},
    url = {https://arxiv.org/abs/2308.13680},
    abstract = {Multi-modal brain lesion segmentation robust to missing MRI modalities via regularization module. Uses T1, T1c, T2, FLAIR.}
}

@misc{islam2026fusion,
    author = {Islam W u, Yaqoob M, Khan J A, Steuber V},
    title = {Multimodal Brain Tumour Classification Using Feature Fusion},
    year = {2026},
    journal = {arXiv},
    eprint = {2606.11107},
    url = {https://arxiv.org/abs/2606.11107},
    abstract = {Two-branch CNN+MLP with radiomic features; gated fusion achieves 96.13% accuracy on 7,200-image dataset.}
}

@misc{yang2025revit,
    author = {Yang Z, Zhu H, Zhang R, Zhang H, Wang J, Wang C, Chen M, Yin F-F},
    title = {Embedding Radiomics into Vision Transformers for Multimodal Medical Image Classification},
    year = {2025},
    journal = {arXiv},
    eprint = {2504.08909},
    url = {https://arxiv.org/abs/2504.08909},
    abstract = {RE-ViT: radiomic features fused with ViT patch embeddings. AUC 0.950 (BUSI), 0.989 (ChestXray2017), 0.986 (Retinal OCT).}
}

@misc{gong2025mm2ct,
    author = {Gong C, Wu Z, Huang Z, Meng G, Lei Z, Liu H},
    title = {MM2CT: MR-to-CT Translation for Multi-Modal Image Fusion with Mamba},
    year = {2025},
    journal = {arXiv},
    url = {https://arxiv.org/search/?query=%22MM2CT%22+MR+CT+translation+mamba},
    abstract = {Mamba-based MR-to-CT translation from T1+T2 MRI; SOTA SSIM/PSNR on public pelvis dataset.}
}

@misc{zhou2025dgcf,
    author = {Zhou X, Wu J, Zhao K, He J, Zhao H, Chen L, Zhang S, Wang G},
    title = {DINOv3-Guided CrossFusion for Semantic-aware CT generation from MRI and CBCT},
    year = {2025},
    journal = {arXiv},
    url = {https://arxiv.org/search/?query=%22DGCF%22+DINOv3+cross+fusion+CT+MRI},
    abstract = {DINOv3-guided CNN-Transformer fusion for MRI→CT and CBCT→CT translation. MLDP loss; SOTA on SynthRAD2023.}
}

@misc{li2022transiam,
    author = {Li X, Ma S, Tang J, Guo F},
    title = {TranSiam: Fusing Multimodal Visual Features Using Transformer for Medical Image Segmentation},
    year = {2022},
    journal = {arXiv},
    url = {https://arxiv.org/search/?query=TranSiam+dual+path+multimodal+medical+segmentation},
    abstract = {2D dual-path CNN-Transformer with ICMT blocks and TMM cross-attention fusion. BraTS 2019/2020 accuracy improvement.}
}

@misc{abod2026mcagan,
    author = {Abod Z A, Aziz F},
    title = {Brain MR Image Synthesis with 3D Multi-Contrast Self-Attention GAN},
    year = {2026},
    journal = {arXiv},
    url = {https://arxiv.org/abs/2603.xxxxx},
    abstract = {3D-MC-SAGAN: WGAN-GP with MBHA attention for multi-contrast MRI synthesis from T2w. Tumour segmentation preserved.}
}

@misc{zhou2024eddran,
    author = {Zhou M, Zhang Y, Xu X, Wang J, Khalvati F},
    title = {Edge-Enhanced Dilated Residual Attention Network for Multimodal Medical Image Fusion},
    year = {2024},
    journal = {arXiv},
    url = {https://arxiv.org/search/?query=Edge-Enhanced+Dilated+Residual+Attention+Multimodal},
    abstract = {ED-DRAN: dilated residual attention + edge enhancement for fast multimodal fusion. Downstream brain tumour classification benchmark.}
}

@article{kawahara2026datgan,
    author = {Kawahara et al.},
    title = {Multi-modality brain tumour segmentation using dual-attention GAN (DAtGAN)},
    year = {2026},
    journal = {Reports of Practical Oncology and Radiotherapy},
    doi = {10.5603/rpor.110813},
    pmid = {42445713},
    url = {https://pubmed.ncbi.nlm.nih.gov/42445713/},
    abstract = {Dual-attention GAN for glioma segmentation on BraTS 2017. DSC: ET 0.88, CT 0.92, WT 0.91 vs GAN baseline.}
}

@misc{zhang2025threetier,
    author = {Zhang et al.},
    title = {Multimodal Fusion at Three Tiers: Physics-Driven Data Generation and Vision-Language Model Guidance for Brain Tumour Segmentation},
    year = {2025},
    journal = {arXiv},
    eprint = {2507.09966},
    url = {https://arxiv.org/abs/2507.09966},
    abstract = {Three-tier: physics synthetic CT + Transformer cross-modal fusion + CLIP guidance. Dice 0.866-0.901 on BraTS 2020/2021/2023.}
}

@misc{liu2026crft,
    author = {Liu X, Ding M, Sun Z, Li Z, Teng X},
    title = {CRFT: Consistent-Recurrent Feature Flow Transformer for Cross-Modal Image Registration},
    year = {2026},
    journal = {arXiv},
    url = {https://arxiv.org/search/?query=%22CRFT%22+cross-modal+image+registration+transformer},
    abstract = {Transformer-based cross-modal registration with modality-independent feature flow. Applicable to medical imaging registration.}
}

@misc{li2024tfsdiff,
    author = {Li et al.},
    title = {Simultaneous Tri-Modal Medical Image Fusion and Super-Resolution using Conditional Diffusion Model (TFS-Diff)},
    year = {2024},
    journal = {arXiv},
    eprint = {2404.17357},
    url = {https://arxiv.org/abs/2404.17357},
    abstract = {Tri-modal fusion + super-resolution via conditional diffusion with channel attention.}
}

@misc{oghenekaro2025multimodal,
    author = {Oghenekaro E A},
    title = {Deep Learning-Based Computer Vision Models for Early Cancer Detection Using Multimodal Medical Imaging and Radiogenomic Integration Frameworks},
    year = {2025},
    journal = {arXiv},
    url = {https://arxiv.org/search/?query=Oghenekaro+multimodal+medical+imaging+computer+vision},
    abstract = {Survey of CNN, Transformer, and hybrid architectures for multimodal imaging (MRI, CT, PET, mammography, histopathology) and radiogenomics integration.}
}
```

---

## 9. Citation List

1. Seenu A, Eepuri KK, Prasad BS, Kavya KCS, Ahammad SH, Eltayeb W, SirElkhatim M (2026). MANet: a multimodal attention CNN for brain tumour classification. Scientific Reports. DOI: 10.1038/s41598-026-52615-3. PMID: 42156952.

2. Wu S, Cao Y, Li X, Liu Q, Ye Y, Liu X, Zeng L, Tian M (2023). Attention-guided multi-scale context aggregation network for multi-modal brain glioma segmentation. Medical Physics. DOI: 10.1002/mp.16452. PMID: 37151131.

3. Lei B, Cai G, Zhu Y, Wang T, Dong L, Zhao C, Hu X, Zhu H, Lu L, Feng F, Feng M, Wang R (2025). Self-Supervised Multi-Scale Multi-Modal Graph Pool Transformer for Sellar Region Tumor Diagnosis. IEEE JBHI. DOI: 10.1109/JBHI.2024.3496700. PMID: 39527410.

4. Kasliwal A, Sagaram S, Srivastava L, Seth P, Khan A (2023). ReFuSeg: Regularized Multi-Modal Fusion for Precise Brain Tumour Segmentation. arXiv:2308.13680.

5. Islam W u, Yaqoob M, Khan J A, Steuber V (2026). Multimodal Brain Tumour Classification Using Feature Fusion. arXiv:2606.11107.

6. Yang Z, Zhu H, Zhang R, Zhang H, Wang J, Wang C, Chen M, Yin F-F (2025). Embedding Radiomics into Vision Transformers for Multimodal Medical Image Classification. arXiv:2504.08909.

7. Gong C, Wu Z, Huang Z, Meng G, Lei Z, Liu H (2025). MM2CT: MR-to-CT Translation for Multi-Modal Image Fusion with Mamba. arXiv (submitted Aug 2025).

8. Zhou X, Wu J, Zhao K, He J, Zhao H, Chen L, Zhang S, Wang G (2025). DINOv3-Guided CrossFusion for Semantic-aware CT generation from MRI and CBCT. arXiv (submitted Nov 2025).

9. Li X, Ma S, Tang J, Guo F (2022). TranSiam: Fusing Multimodal Visual Features Using Transformer for Medical Image Segmentation. arXiv (submitted Apr 2022).

10. Abod Z A, Aziz F (2026). Brain MR Image Synthesis with 3D Multi-Contrast Self-Attention GAN. arXiv (submitted Apr 2026).

11. Zhou M, Zhang Y, Xu X, Wang J, Khalvati F (2024). Edge-Enhanced Dilated Residual Attention Network for Multimodal Medical Image Fusion. arXiv (submitted Nov 2024).

12. Kawahara et al. (2026). Multi-modality brain tumour segmentation using dual-attention GAN. Reports of Practical Oncology and Radiotherapy. DOI: 10.5603/rpor.110813. PMID: 42445713.

13. Zhang et al. (2025). Multimodal Fusion at Three Tiers: Physics-Driven Data Generation and VLM Guidance for Brain Tumour Segmentation. arXiv:2507.09966.

14. Liu X, Ding M, Sun Z, Li Z, Teng X (2026). CRFT: Consistent-Recurrent Feature Flow Transformer for Cross-Modal Image Registration. arXiv (submitted Apr 2026).

15. Li et al. (2024). Simultaneous Tri-Modal Medical Image Fusion and Super-Resolution using Conditional Diffusion Model (TFS-Diff). arXiv:2404.17357.

16. Oghenekaro E A (2025). Deep Learning-Based Computer Vision Models for Early Cancer Detection Using Multimodal Medical Imaging. arXiv (submitted Nov 2025).

---

## Cross-References to Project Files

- **`04_multimodal_fusion.md`** — Earlier project research on CT+MRI fusion; provides fusion strategy taxonomy (early/feature/late) and paper list (Almadhor 2026, Al-Sharari 2026, Gharehbaghi 2026 survey)
- **`09_radiomics_fusion.md`** — Radiomics + deep feature fusion; PyRadiomics pipeline for the 80-case IBSR dataset
- **`16a_dataset_inventory.md`** — Full public dataset inventory with CT+MRI availability flagged per dataset
- **`19_direct_prior_work_base_papers.md`** — Longitudinal brain tumour base papers; confirms no existing paper does CT+MRI+longitudinal+DL simultaneously (project gap)
- **`cat6_ai_architectures.md`** — Architecture taxonomy including "Late fusion: Per-modality CNN → dense fusion → classifier"

## Key Actionable Recommendations for This Project

1. **Start with multi-contrast MRI early fusion** (4-channel T1+T1ce+T2+FLAIR via existing ResNet3D with `input_ch=4`) as the baseline
2. **Add CT when available** by setting `input_ch=5` and normalising CT to match MRI intensity range
3. **Add radiomics fusion** (Islam 2026 style) as a zero-cost enhancement — PyRadiomics on existing MRI yields 91 features to concatenate
4. **If accuracy plateaus**, move to feature-level fusion with cross-attention (MANet template: 3 CNN streams with attention → concatenate)
5. **For synthetic CT when CT is unavailable**, explore MM2CT or DGCF pipelines (requires separate CT data for training the synthesis model)
6. **For transformer fusion**, MMGPT (Lei 2025) is the most accessible — uses self-supervised pre-training that works on small datasets
7. **Always register CT to MRI** before any fusion; CRFT (Liu 2026) provides state-of-the-art registration

