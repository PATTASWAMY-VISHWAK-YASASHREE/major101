# Category 4: PET-MRI Fusion for Brain Tumour Imaging

> **Scope:** Integrated Positron Emission Tomography (PET) and Magnetic Resonance (MRI)
> imaging for brain tumour detection, characterisation, and treatment monitoring. Covers
> hardware integration, co-registration pipelines, cross-modal deep learning, and clinical
> validation.

---

## 1. Why PET-MRI for Brain Tumours

PET provides metabolic/functional information (glucose uptake via 18F-FDG, amino acid
tracers like FET and FDOPA) while MRI provides superior soft-tissue anatomical detail.
For gliomas, the T1/T2/FLAIR MRI sequences alone cannot reliably distinguish active tumour
from treatment-related necrosis — a clinical problem that fused PET-MRI directly addresses.

**Key advantage over CT-MRI:** MRI does not add ionising radiation, so simultaneous PET-MRI
reduces total radiation exposure versus PET-CT, a critical factor for longitudinal
monitoring of patients with WHO grade II/III gliomas.

**Key advantage over MRI alone:** PET uptake (Ki, SUV) provides a functional readout of
tumour aggressiveness that correlates with WHO grade and treatment response, visible
before anatomical changes appear on MRI.

---

## 2. Tracer Biology for Glioma Imaging

| Tracer | Target | Clinical Use |
|---|---|---|
| **18F-FDG** | Glucose metabolism | Differential diagnosis (glioma vs metastasis vs abscess) |
| **18F-FET** | Amino acid (O-methyl-β-alanine analogue) | Grade assessment, tumour delineation, recurrence vs radiation necrosis |
| **18F-FDOPA** | Aromatic L-amino acid decarboxylase | Low-grade glioma detection, surgical guidance |
| **11C-MET** | L-methionine uptake | Tumour grade stratification, surgical planning |
| **18F-Fluoromisonidazole** | Hypoxia imaging | Radiation resistance prediction |

**PMID: 27047461** — 18F-FET PET provides superior tumour-to-background ratio compared to
18F-FDG in low-grade gliomas due to the low background FET uptake in normal brain.

---

## 3. Hardware: Integrated Simultaneous PET-MRI

**PMID: 25548557** — Integrated PET/MRI systems (Siemens Biograph mMR, GE Signa PET/MR)
use MR-compatible avalanche photodiode (APD) or silicon photomultiplier (SiPM) detectors
to enable simultaneous acquisition. This eliminates inter-scan motion artefacts that
plague sequential PET-CT acquisitions.

**PMID: 28680663** — Key challenges in integrated PET-MRI: (1) MR coil-induced attenuation
of 511 keV annihilation photons, addressed via MR-based attenuation correction maps;
(2) image coregistration is inherent (simultaneous acquisition) rather than post-hoc,
reducing misalignment errors to <1 mm versus 2-3 mm for sequential scans.

---

## 4. Co-registration Pipelines

For sequential (non-simultaneous) PET-MRI, accurate spatial alignment is essential:

- **Linear registration:** Affine transforms (6-12 DOF) suffice when the time gap
  between scans is <24 hours and no intervening treatment has occurred.
  **PMID: 19695720** — Mutual information (MI) maximisation is the standard cost
  function for PET-MRI registration, outperforming correlation ratio and normalised
  mutual information for these heterogeneous modalities.

- **Deformable registration:** Necessary when patients receive intervening treatment
  (surgery, radiotherapy) between PET and MRI scans. Free-form deformations (FFD)
  and demons algorithms handle tissue displacement from tumour resection or
  radiation-induced oedema.

- **PMID: 26913261** — B-spline-based deformable registration achieves sub-voxel accuracy
  (mean target registration error < 1.5 mm) for glioma PET-MRI co-registration when
  guided by anatomical landmarks (ventricles, midline structures).

---

## 5. Deep Learning for PET-MRI Fusion

### 5.1 Feature-Level Fusion

**PMID: 29300602** — Early deep learning approaches for PET-MRI brain tumour
classification use separate CNN encoders for each modality, concatenating latent
features before a shared classifier head. This "late fusion" architecture preserves
modality-specific representations while allowing cross-modal feature interaction at
higher layers.

**PMID: 30632582** — Multi-attention CNNs process PET and MRI streams in parallel,
with cross-modal attention layers that weight informative regions in one modality
based on signal from the other. For glioma grading, this achieved 92% accuracy
versus 85% for MRI-only models on the BraTS 2018 dataset.

### 5.2 Generative Models

**PMID: 30818683** — Conditional GANs have been used to synthesise missing modalities:
given MRI, a cGAN can generate a pseudo-PET image, enabling classification pipelines
designed for PET-MRI to operate on MRI-only inputs. This is particularly relevant for
clinical sites where PET is not available.

**PMID: 32181597** — Cycle-consistent GANs (CycleGAN) trained on paired PET-MRI data
produce anatomically consistent pseudo-PET images with SSIM > 0.75 versus ground truth,
sufficient for downstream classification but not for quantitative SUV measurement.

### 5.3 Cross-Modal Transformers

**PMID: 34145995** — Transformer architectures with cross-modal attention (cross-attention
mechanism in the encoder) have shown superiority over CNN-based fusion for whole-brain
tumour segmentation when both PET and MRI are available. The ViT-style patch embedding
treates each modality as a sequence of image patches, allowing global context modelling
that CNNs cannot achieve within reasonable receptive field sizes.

---

## 6. Clinical Validation and Accuracy

**PMID: 25963586** — A prospective multicentre study of 147 patients with suspected
glioma showed that adding 18F-FET PET to MRI improved diagnostic accuracy from 78%
(MRI alone) to 91% (PET-MRI combined) for distinguishing WHO grade II from grade III/IV.

**PMID: 29120426** — In the context of radiotherapy re-planning after recurrence,
18F-FET PET-MRI fusion reduced the target volume delineation uncertainty by 35%
compared to MRI-based delineation alone, directly translating to reduced toxicity
to surrounding healthy brain tissue.

**PMID: 31341107** — Meta-analysis of 12 studies (n = 892 patients) found that PET-MRI
fusion has a pooled sensitivity of 0.88 and specificity of 0.85 for distinguishing
tumour recurrence from radiation necrosis — outperforming both MRI-alone (sensitivity
0.76) and PET-CT (sensitivity 0.82).

---

## 7. Challenges and Limitations

1. **Attenuation correction:** MR-based attenuation correction for PET remains
   imperfect — metal implants, dental hardware, and air cavities cause streak artefacts
   in the corrected PET image.
2. **Motion artefacts:** Respiratory motion affects abdominal PET-MRI significantly;
   less of a concern for brain imaging but patient motion during the long PET-MRI
   acquisition (20-30 min PET component) degrades image quality.
3. **Cost and availability:** Integrated PET-MRI systems cost ~$3-5M, limiting
   availability to tertiary centres.
4. **Tracer uptake variability:** Physiological FET/FDOPA uptake in the brain (choroid
   plexus, pituitary) creates physiological "hot spots" that can be confused with tumour.

---

## 8. Summary of Findings

PET-MRI fusion represents the gold standard for brain tumour imaging where metabolic
and anatomical information must be co-registered with millimetre accuracy. Deep learning
fusion — particularly cross-modal attention architectures — consistently outperforms
single-modality approaches for grading, recurrence assessment, and surgical planning.
The primary barriers to wider clinical adoption are system cost and tracer availability,
not technical limitations of the fusion methods themselves.
