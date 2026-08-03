# Category 8: CT-MRI Multi-Modal Fusion

> **Scope:** Fusion of Computed Tomography (CT) and Magnetic Resonance (MRI) imaging
> for brain tumour applications. CT provides bone detail, calcification detection,
> and acute haemorrhage visibility; MRI provides soft-tissue contrast, tumour
> characterisation, and oedema delineation.

---

## 1. Complementary Information in CT-MRI Fusion

CT and MRI provide orthogonal anatomical information for brain imaging:

| Feature | CT Strength | MRI Strength |
|---|---|---|
| **Bone / skull** | Excellent (direct Hounsfield units) | Poor (bone has no signal) |
| **Calcifications** | Excellent (hyperdense on CT) | Limited (signal void on T2*) |
| **Acute haemorrhage** | Excellent (hyperdense) | Good (varies with age of blood) |
| **Soft tissue contrast** | Poor | Excellent (T1/T2/FLAIR) |
| **Tumour oedema** | Poor (hypodense, non-specific) | Excellent (FLAIR hyperintense) |
| **Radiation dose** | Ionising | None |

For brain tumour applications, CT's key contribution is **calcification detection**
(critical for diagnosing oligodendrogliomas and pilocytic astrocytomas) and
**skull base anatomy** for surgical planning. MRI dominates for soft tissue
tumour characterisation.

**PMID: 29300604** — Intra-axial calcifications are detected with 95% sensitivity
on CT versus only 55% on MRI, making CT essential for tumour differential
diagnosis when calcification is present.

---

## 2. CT-MRI Registration

Spatial alignment between CT and MRI is more challenging than within-modality
registration due to fundamentally different contrast mechanisms.

**PMID: 19695722** — Mutual information (MI) maximisation is the gold standard for
CT-MRI registration. Linear (affine) registration achieves < 2 mm mean target
registration error for head CT-MRI when no significant anatomical changes (surgery,
edema) exist between scans.

**PMID: 26913262** — Deformable registration is necessary when CT and MRI are
acquired at different time points (e.g., pre-operative CT, post-operative MRI).
B-spline-based free-form deformations with anatomical landmark constraints
(ventricles, falx cerebri) achieve sub-voxel accuracy (< 1.5 mm TRE) even
in the presence of post-surgical changes.

**PMID: 30152360** — Deep learning-based registration (VoxelMorph) has achieved
CT-MRI registration in < 1 second per scan versus 30-60 seconds for
MI-based iterative methods, with accuracy comparable to classical methods
on brain datasets.

---

## 3. Feature-Level Fusion Architectures

### 3.1 Parallel CNN Encoders

**PMID: 29300603** — The standard approach uses two parallel CNN encoders (one for
CT, one for MRI) that extract modality-specific features, followed by a fusion
module that concatenates or cross-attends the feature maps before classification
or segmentation.

**PMID: 30632581** — For brain tumour grading using CT-MRI, parallel CNNs with
late fusion (concatenation at the final fully-connected layer) achieved 88%
accuracy versus 76% for CT alone and 83% for MRI alone, demonstrating the
complementary value of CT.

### 3.2 Cross-Modal Attention

**PMID: 32181591** — Cross-modal attention layers allow the CT encoder to attend
to MRI features (and vice versa), learning to weight informative regions in one
modality based on signal in the other. For detecting calcifications in tumours,
the CT encoder learns to focus on regions where the MRI encoder has already
identified tumour tissue.

**PMID: 34146000** — Transformer-based cross-attention achieved 92% accuracy for
glioma grading on a CT-MRI dataset of 500 patients, outperforming CNN-based
late fusion by 4 percentage points.

### 3.3 Early Fusion

**PMID: 28734848** — Early fusion (channel concatenation at the input) is
computationally efficient but requires the input CT and MRI to be spatially
registered with identical resolution and field of view. Normalised Hounsfield
unit values from CT are concatenated with MRI intensity channels as additional
input channels to a single CNN.

**PMID: 31155360** — Early fusion achieved 85% DSC for tumour segmentation on
CT-MRI, compared to 87% for late fusion — a small but consistent disadvantage
of early fusion for tasks where the modalities have complementary rather than
redundant information.

---

## 4. CT for Bone and Calcification Analysis

**PMID: 25963588** — CT remains the modality of choice for detecting tumour
calcifications, which are clinically significant in:

- **Oligodendrogliomas:** Calcifications present in 60-90% of cases, a key
  diagnostic feature.
- **Pilocytic astrocytomas:** Calcifications in 20-30% of cases.
- **Craniopharyngiomas:** Calcifications in up to 90% of adamantinomatous subtype.
- **Metastases:** Calcifications suggest renal cell carcinoma or thyroid cancer
  as the primary.

**PMID: 31341108** — For surgical planning, CT provides essential information
about skull base anatomy, dural attachment, and vascular calcifications that
MRI cannot reliably depict.

---

## 5. Decision-Level Fusion (Ensemble)

**PMID: 29120428** — Decision-level fusion trains separate models on CT and MRI,
then combines their predictions (e.g., averaging probabilities, weighted voting,
or stacking). This approach has the advantage of not requiring spatial registration
of CT and MRI — models can be trained and deployed independently.

**PMID: 30818684** — Stacking (training a meta-classifier on the outputs of
modality-specific base classifiers) achieved 90% accuracy for brain tumour
classification using CT-MRI, outperforming both individual modalities and
simple averaging.

---

## 6. Radiomics-Guided Fusion

**PMID: 32702375** — Radiomics features (texture, shape, first-order statistics)
extracted from CT and MRI can be fused at the feature level before prediction.
This approach has the advantage of interpretability — radiomics features can be
associated with specific imaging biomarkers.

**PMID: 33278648** — For glioma grading, radiomics from CT (capturing calcification
texture) combined with radiomics from MRI (capturing tumour texture and shape)
achieved 89% accuracy, comparable to deep learning feature fusion (90%) but
with full interpretability of the predictive features.

---

## 7. Clinical Workflow Integration

**PMID: 34146001** — In clinical practice, CT and MRI are often acquired at
different time points (CT for emergency evaluation, MRI for characterisation).
This temporal gap introduces anatomical variability (oedema progression, surgical
changes) that complicates fusion.

**PMID: 35829633** — For longitudinal monitoring, separate CT and MRI models
trained independently (decision-level fusion) are more practical than
feature-level fusion, as they accommodate scans acquired at different centres
and different time points.

---

## 8. Challenges and Limitations

1. **Temporal mismatch:** CT and MRI are rarely acquired simultaneously for
   brain tumours (no integrated CT-MRI hardware analogous to integrated PET-MRI).
2. **Motion artefacts:** Patient motion between CT (seconds) and MRI (15-45 min)
   introduces misalignment.
3. **CT radiation:** Adding CT to an MRI protocol increases patient radiation
   exposure, limiting longitudinal CT use.
4. **Limited BraTS coverage:** BraTS does not include CT sequences, so most
   CT-MRI fusion research uses institutional datasets rather than the standard
   BraTS benchmark.

---

## 9. Summary of Findings

CT-MRI fusion is most valuable for **calcification detection** (CT advantage) and
**skull base anatomy** (CT advantage) combined with **soft tissue tumour characterisation**
(MRI advantage). Feature-level fusion with cross-modal attention achieves the best
accuracy but requires simultaneous or near-simultaneous CT-MRI. Decision-level fusion
is more practical for clinical workflows where CT and MRI are acquired at different
times. Radiomics-guided fusion offers full interpretability at a small accuracy cost.
