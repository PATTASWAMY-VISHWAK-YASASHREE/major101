# Brain Tumour DL Research — Complete Mind Map

**All 19 research categories with cross-references and decision flow.**

---

## Visual Structure

```mermaid
mindmap
  root((Brain Tumour DL))
    Core Decisions
      Late Fusion Strategy
      3D ResNet3D-18
      4-Class WHO Grading
      Patch-Based Training
    Data Layer
      IBSR 80 Cases Primary
      C-BRATS 600 Cases Secondary
      BraTS Dropped Too Large
    Imaging Foundation
      MRI Modalities T1 T1ce T2 FLAIR
      CT Imaging Basics
      PET Integration Review
    Preprocessing
      Skull-Stripping
      CTN Normalisation
      Rigid Registration
      ComBat Harmonisation
    Architecture
      CNN Backbones
      Transformer Hybrids
      DenseNet ResNet EfficientNet
    Segmentation
      U-Net V-Net nnU-Net
    Fusion Strategies
      Early Late Hybrid
      CT+MRI Multimodal
      PET+MRI Comparison
    Longitudinal
      Temporal CNN-LSTM
      Change Detection
      Progression Monitoring
    Self-Supervised
      MAE Pretraining
      SimMIM
    Generative
      GAN Diffusion Rejected
      Standard Augmentation Selected
    Explainability
      Grad-CAM
      SHAP
      Grad-CAM++
    Uncertainty
      MC Dropout
      Temperature Scaling
    Survival
      DeepSurv
      DeepHit
    Federated
      Multi-Institutional
      Out of Scope
    Radiomics
      PyRadiomics
    Registration
      ANTs MICE
    Radiologist
      AI-Assisted Workflow
    Datasets
      Inventory Gap Analysis
    Metrics
      Dice IoU Macro-F1
```

## Cross-Reference Matrix

Each category links to its related research files:

```
cat1_brain_tumour_imaging_basics.md ──┬── PET+MRI fusion
                                       └── Registration
cat2_tumour_progression_monitoring.md ──┬── Longitudinal analysis
                                       └── Change detection
cat4_pet_mri_fusion.md ──┬── Multimodal fusion
                          └── CT+MRI fusion
cat5_brain_tumor_segmentation.md ──┬── AI architectures
                                   └── Registration
cat6_ai_architectures.md ──┬── Fusion strategies
                            └── Self-supervised
cat7_registration_segmentation.md ──┬── Preprocessing
                                     └── Multimodal registration
cat8_ct_mri_multimodal_fusion.md ──┬── Core decision file
                                   └── Explainability
16a_dataset_inventory.md ──┬── Datasets
                           └── Gap analysis
16b_key_dataset_profiles.md ──┬── Primary dataset
                               └── Secondary dataset
16d_longitudinal_datasets.md ──┬── Longitudinal
                                └── Progression
12_explainability.md ── Grad-CAM / SHAP
13_uncertainty_calibration.md ── Temperature scaling / MC dropout
14_survival_analysis.md ── DeepSurv / DeepHit
03_transformer_hybrid.md ── ViT / TransUNet (not used)
04_multimodal_fusion.md ── Late fusion selected
05_longitudinal_analysis.md ── Temporal architecture
06_final_report.md ── Comprehensive summary
07_efficient_data_loading.md ── 4GB VRAM strategy
09_radiomics_fusion.md ── PyRadiomics
10_self_supervised.md ── MAE pretraining
11_generative_augmentation.md ── GAN rejected
15_federated_learning.md ── FL (out of scope)
```

## Decision Flow

```mermaid
flowchart TD
    A[Research Complete] --> B{Hardware?}
    B -->|RTX 2050 4GB| C[3D ResNet3D-18 + FP16]
    B -->|Higher VRAM| D[ViT / Swin-UNet]
    C --> E{Fusion?}
    E -->|Late| F[Per-modality encoders → Dense head]
    E -->|Early| G[Channel-stacked input]
    F --> H[IBSR Training]
    G --> I[Not recommended]
    H --> J{Explainability?}
    J -->|Yes| K[Grad-CAM + SHAP]
    J -->|Later| L[Add post-baseline]
    F --> M{Calibration?}
    M -->|Yes| N[Temperature Scaling]
    M -->|Later| O[Add post-baseline]
    F --> P{Survival?}
    P -->|Yes| Q[DeepSurv extension]
    P -->|Later| R[Add post-baseline]
    F --> S[4-Class WHO Grading]
```

## Category Relationships

```mermaid
graph LR
    A[cat1 Imaging Basics] --> B[cat2 Progression]
    A --> C[cat4 PET+MRI Fusion]
    A --> D[cat7 Registration]
    B --> E[cat5 Segmentation]
    B --> F[cat2 Progression Monitoring]
    C --> G[cat8 CT+MRI Fusion]
    C --> H[cat4 Multimodal Fusion]
    D --> I[cat6 AI Architectures]
    E --> J[cat5 Segmentation]
    F --> K[cat2 Progression]
    G --> L[cat8 CT+MRI Experimental]
    H --> M[cat4 Fusion Strategies]
    I --> N[cat6 CNN Backbones]
    I --> O[cat3 Transformer Hybrids]
    J --> P[cat5 U-Net/V-Net]
    K --> Q[cat2 Temporal Analysis]
    L --> R[cat8 Late Fusion Selected]
    M --> S[cat4 Early/Late/Hybrid]
    N --> T[cat1 Foundations]
    O --> U[cat3 ViT/TransUNet]
    P --> V[cat5 nnU-Net]
    Q --> W[cat2 CNN-LSTM]
    R --> X[cat8 Core Decision]
    S --> Y[cat4 Fusion Review]
    T --> Z[cat1 Deep Learning Basics]
    U --> AA[cat3 ViT/UNETR]
    V --> AB[cat5 Auto-Segmentation]
    W --> AC[cat2 Longitudinal]
    X --> AD[cat8 CT+MRI Strategy]
    Y --> AE[cat4 Multi-Modal Review]
    Z --> AF[cat1 DL Fundamentals]
    AA --> AG[cat3 Transformer Review]
    AB --> AH[cat5 nnU-Net Deep Dive]
    AC --> AI[cat2 Temporal CNN]
    AD --> AJ[cat8 Final Architecture]
    AE --> AK[cat4 Fusion Methods]
    AF --> AL[cat1 CNN Foundation]
```

---

**Legend:**
- `→` = feeds into / influences
- `[brackets]` = research category/file
- **Bold** = our project decisions