# Research Direction Decision: Grade Proxy or Clinically Meaningful Model?

**Project:** `major101`  
**Author:** Manus AI  
**Question:** Is the current grade-proxy model a good research endpoint, or should the project pivot to a clinically meaningful model?

## Decision in one sentence

For a serious research contribution, a **clinically defined endpoint is better** than the current grade proxy; however, the project should not claim a “clinically usable model” until it has the right labels, external testing, calibration, and clinical-workflow evaluation. The most defensible near-term position is to keep the grade proxy as a **baseline and engineering scaffold**, while pivoting the main research question to a clinically meaningful endpoint that the data can genuinely support.

## 1. Why the current grade proxy is weak as a final scientific target

The active code predicts a binary `grade_proxy` from preprocessed MRI volumes.[1] The label is dataset-derived and is not the same as a clinical tumour grade established from histology, immunohistochemistry, molecular testing, or an integrated diagnostic process. The current model can therefore answer a narrow methodological question: **Can this MRI pipeline learn the dataset’s proxy signal under a subject-disjoint split?** It cannot answer the stronger clinical question: **Can MRI reliably determine a patient’s tumour grade for diagnosis or treatment planning?**

That distinction matters because modern CNS tumour classification is not purely an imaging-intensity problem. The 2025 review of deep learning for brain-tumour MRI describes clinical and research applications across segmentation, quantitative measurement, tumour classification, molecular/genetic prediction, and survival outcomes, while noting that CNS tumour grading is based on histological, immunohistochemical, and molecular features.[2] A proxy may be correlated with the intended target, but correlation is not equivalence.

The grade proxy is still useful in three situations. It is useful as a **smoke-test target** for checking data loading, leakage controls, balanced sampling, memory limits, cross-validation, and checkpoint reproducibility. It is useful as a **baseline** for a later clinically labelled task. It is also useful if the project explicitly studies proxy-label learning, label noise, or weak supervision. It is not strong enough as the headline claim for a clinical AI paper.

## 2. What “clinically usable” would require

A clinically usable model is not defined by a high AUROC on one retrospective split. It needs a clinically meaningful outcome, a clear intended use, robust discrimination, calibrated probabilities, testing on data separated by patient and preferably by site or institution, failure analysis, and evidence that the output can support a real workflow. CLAIM 2024 is a reporting guideline for transparent medical-imaging AI studies, while TRIPOD+AI provides reporting guidance for diagnostic and prognostic prediction models.[3] [4] These guidelines improve transparency; they do not substitute for external or prospective evaluation.

For neuro-oncology response assessment, the AI-RANO recommendations further reinforce the need for standardisation, validation, and good clinical practice before AI outputs are treated as clinical evidence.[5] Therefore, the project should use the term **clinically meaningful research model** until it has clinical validation. The phrase **clinically usable** should be reserved for a later stage.

## 3. Candidate research directions compared

| Endpoint | Clinical meaning | Data required | Fit with current data/code | Scientific value | Recommendation |
|---|---|---|---|---|---|
| ET-derived grade proxy | Weak-to-moderate; reflects a dataset construction rather than definitive diagnosis | Existing binary proxy labels and MRI volumes | High; current pipeline already runs | Low-to-moderate as a final contribution; high as a baseline | Keep only as baseline or weak-supervision study |
| Tumour segmentation and volumetry | Strong; delineates tumour burden and subregions for quantification, planning, and monitoring | Voxel-level WT/TC/ET masks, preferably with quality review | Low-to-moderate; requires a new segmentation model and mask pipeline | High and directly aligned with available BraTS-style annotations | **Best near-term clinically meaningful pivot if masks are available** |
| Molecular biomarker prediction | Strong; can support integrated diagnosis or treatment stratification when labels are reliable | Patient-level IDH, 1p/19q, MGMT, or other molecular labels linked to MRI | Unknown-to-low; current label contract has only `grade_proxy` | High, but label acquisition and confounding are substantial | Strong option if a properly labelled cohort is obtained |
| True histological/integrated grade | Stronger than a proxy and closer to diagnostic practice | Pathology and molecular reference labels linked to pre-treatment MRI | Low with the current dataset contract | High, but difficult and prone to label/timing confounds | Use only after obtaining verified clinical labels |
| Response/progression assessment | Strong; directly tied to monitoring and treatment decisions | Longitudinal scans, timepoints, treatment metadata, segmentation or RANO-aligned outcomes | Low; current code is single-timepoint classification | Very high, but high implementation and validation burden | Longer-term research direction |
| Survival/prognosis prediction | Strong; estimates future patient outcome | Survival time/censoring, treatment covariates, MRI, and leakage-safe time ordering | Low; none of these are in the current label contract | High, but statistically and clinically demanding | Do not choose without a real outcomes cohort |

The clinically meaningful endpoint with the best balance of relevance and feasibility is **tumour segmentation and quantitative volumetry**, provided the project has valid segmentation masks. The most clinically ambitious alternatives—molecular prediction, progression, and survival—are better only if the corresponding labels and metadata can be acquired and audited.

## 4. Recommended project strategy

### Stage A: Keep the current model, but rename its role

Retain the current MRI-only classifier as `M0: MRI grade-proxy baseline`. Do not present it as a WHO-grade model or clinical grading tool. Use it to demonstrate the repaired data contract, subject-disjoint splitting, memory-bounded training, balanced batches, validation threshold selection, and locked-test discipline.[1] The research report should explicitly label the target as **ET-derived binary proxy**.

### Stage B: Select one clinically defined primary endpoint

If BraTS-style masks are available and trustworthy, pivot the main task to **pre-treatment tumour segmentation and volumetry**, with WT, TC, and ET masks and Dice/HD95 plus volume-error reporting. The current classification code cannot be reused unchanged, but its subject grouping, data-quality checks, memory safeguards, and evaluation discipline can be reused.

If masks are not available or segmentation is not the intended contribution, obtain a dataset with patient-level molecular labels and define one primary biomarker endpoint. IDH mutation status is a reasonable research candidate because it is clinically meaningful and has a clearer binary target than the current proxy, but it still requires label provenance, imaging timepoint control, class-balance analysis, and external-style testing.

### Stage C: Add multimodal fusion only after the endpoint is valid

The modality-fusion research should follow the label decision, not lead it. First establish a single-modality baseline for the clinically defined endpoint. Then add a second modality or model using the staged approach in [`21_multimodal_fusion_and_dataset_integration.md`](21_multimodal_fusion_and_dataset_integration.md): verified pairing and modality-specific preprocessing, CT-only or second-modality baseline, calibrated late fusion, then intermediate fusion if justified.

### Stage D: Use clinical-validation language only at the appropriate stage

The project may claim a **clinically meaningful endpoint** after the label is clinically defined and the experiment is designed around it. It may claim **external performance** only after testing on a truly separate cohort or domain. It may claim **clinical utility** only after workflow-level or decision-impact evidence. Until then, use terms such as “research prototype,” “retrospective development study,” and “clinically relevant endpoint.”

## 5. Final recommendation

The project should **not** make the grade-proxy classifier the final research contribution. It should keep it as the first baseline because it is already implemented and useful for proving pipeline integrity. The main research should pivot to a clinically meaningful endpoint, with the preferred order:

1. **Tumour segmentation and volumetry** if valid BraTS masks are available.
2. **Molecular biomarker prediction** if a reliable patient-level labelled cohort can be obtained.
3. **Response/progression modelling** only after longitudinal scans and treatment-aware outcomes are available.
4. **Survival prediction** only with a sufficiently large outcomes cohort and proper time-to-event design.

The best immediate research title would be something like **“Leakage-Aware Deep Learning for Clinically Grounded Brain-Tumour MRI Analysis: From an ET-Derived Proxy Baseline to Segmentation or Molecular Prediction.”** This is more honest and scientifically stronger than claiming that the current proxy model performs clinical tumour grading.

> **Bottom line:** The grade proxy is the better *engineering starting point*. A clinically defined endpoint is the better *research destination*. With the current data, claiming a clinically usable model would be premature; the right move is to preserve the proxy baseline and change the headline task once the labels support it.

## References

[1]: [Current implementation-aligned MRI grade-proxy pipeline](20_implemented_mri_grade_pipeline.md)

[2]: [Dorfner et al., “A review of deep learning for brain tumor analysis in MRI,” *npj Precision Oncology* (2025)](https://www.nature.com/articles/s41698-024-00789-2)

[3]: [Tejani et al., “Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update,” EQUATOR Network record](https://www.equator-network.org/reporting-guidelines/checklist-for-artificial-intelligence-in-medical-imaging-claim-a-guide-for-authors-and-reviewers/)

[4]: [Collins et al., “TRIPOD+AI statement,” *The BMJ* (2024)](https://www.bmj.com/content/385/bmj-2023-078378)

[5]: [Bakas et al. for the RANO group, “AI-RANO part 2: recommendations for standardisation, validation, and good clinical practice,” *The Lancet Oncology* (2024)](https://www.thelancet.com/journals/lanonc/article/PIIS1470-2045(24)00315-2/abstract)
