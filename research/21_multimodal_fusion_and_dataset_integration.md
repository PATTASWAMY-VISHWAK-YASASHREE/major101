# Web Research: Combining Models, Modalities, and Heterogeneous Datasets

**Project:** `major101`  
**Author:** Manus AI  
**Research question:** What are defensible ways to combine the current MRI model with a second model, a second imaging modality, or a second dataset without creating alignment, leakage, or calibration problems?

## Executive conclusion

The safest extension path is **not** to concatenate every available image into one larger tensor. The current code already has a compact MRI-only classifier, subject-disjoint splits, memory-bounded patch loading, and a locked-test protocol. The next research step should preserve that baseline and add one controlled factor at a time: first a CT-only baseline on a verified paired subset, then calibrated late fusion, and only then an intermediate feature-fusion model with explicit missing-modality handling.

The literature distinguishes three broad fusion points. **Early fusion** combines modalities at the input or very shallow feature level. **Intermediate fusion** keeps modality-specific encoders for part of the network and combines learned features later. **Late fusion** combines independently produced logits or calibrated probabilities. A systematic review describes early, joint/intermediate, and late/decision fusion as distinct design choices rather than interchangeable names.[1] Recent work on heterogeneous biomedical data similarly argues for preserving modality-specific structure while learning cross-modal interactions and handling missing modalities explicitly.[2]

For combining datasets, the key distinction is whether the datasets share the same label definition, whether modalities are paired at the patient level, and whether the data come from different scanner/site domains. Pooled training, transfer learning, domain adaptation, and domain generalization are different experiments. A random split over a pooled dataset cannot establish cross-site generalization when source or scanner identity is correlated with the label.[3] [4]

## 1. Fusion taxonomy

| Strategy | What is combined | Strength | Main risk | Fit for current project |
|---|---|---|---|---|
| Early/input fusion | Stack MRI and CT channels before the first convolution | Simple and parameter-efficient when inputs are aligned and complete | Highly sensitive to registration, resampling, intensity scale, and missing channels | Not first choice |
| Intermediate/feature fusion | Separate MRI and CT encoders, then concatenate, gate, or cross-attend pooled features | Preserves modality-specific representations while learning interactions | More parameters; requires paired examples and careful ablations | Second multimodal experiment |
| Late/logit fusion | Combine outputs from independently trained MRI and CT models | Lowest integration risk; supports modality-specific preprocessing and separate training | May miss fine-grained cross-modal interactions; requires calibration | Best first extension |
| Model ensemble | Average or stack predictions from seeds, folds, architectures, or modalities | Reduces variance and can improve reliability | More compute; a meta-model can overfit if trained on in-sample predictions | Useful after stable folds |
| Knowledge distillation | Train a student to imitate a teacher with more modalities or models | Produces a deployable single-modality model and can use privileged information during training | Teacher errors and overconfidence can be transferred | Optional missing-modality path |
| Shared-specific representation | Learn common and modality-specific features, then fuse them | Explicitly models shared versus modality-specific information and missingness | More complex objectives and more tuning | Research option if paired data are incomplete |

The choice should be made against the data contract, not solely against reported benchmark accuracy. A recent brain-tumour fusion study identifies scanner/sequence discrepancy, large labelled-data requirements, computational cost, and feature redundancy as practical limitations of multimodal and CNN/Transformer hybrids.[8]

## 2. How to combine two models

### 2.1 Calibrated late fusion

Train the MRI and CT models independently using the same subject-level development split and the same binary label definition. Freeze both checkpoints. On the validation set only, calibrate each model and learn a fusion rule such as a weighted probability average:

```text
p_fused = w * p_MRI + (1 - w) * p_CT
```

The weight `w`, decision threshold, and any calibration parameters must be selected without reading the locked test set. A stronger but still transparent alternative is a logistic-regression stacker trained on out-of-fold predictions from the development data. The stacker must never be trained on predictions from cases used to fit the base models.

Late fusion is attractive here because the current MRI classifier already has a stable interface that returns one binary logit. A CT model can expose the same interface without forcing CT preprocessing into the MRI loader. It also makes ablations clear: MRI-only, CT-only, and MRI+CT are all directly measurable.

### 2.2 Feature-level model fusion

For a second experiment, replace the single MRI encoder with two modality-specific encoders:

```text
z_MRI = Encoder_MRI(x_MRI)
z_CT  = Encoder_CT(x_CT)
z     = Fusion([z_MRI, z_CT, mask_MRI, mask_CT])
y_hat = Classifier(z)
```

The fusion block can start as concatenation followed by a small MLP. A gated sum or cross-attention block should be an ablation, not the initial implementation. Include modality-presence masks so that the network knows whether a feature is observed or imputed. HEALNet provides a recent example of preserving modality-specific structure while learning cross-modal interactions and explicitly handling missing modalities.[2]

This design is only valid when the MRI and CT inputs correspond to the same case or when a method has been selected specifically for unpaired learning. Concatenating a CT from one patient with an MRI from another creates a synthetic training example with no valid joint label evidence.

### 2.3 Deep ensembles and stacking

A model ensemble can combine different random seeds, folds, architectures, or modalities. A simple ensemble averages calibrated probabilities. A stacked ensemble feeds out-of-fold predictions into a low-capacity meta-classifier. The key evaluation requirement is calibration: an ensemble can improve ranking while still producing unreliable probabilities. A medical-imaging ensemble-calibration study reports that standard dropout and identical-model ensembles do not automatically approximate classification probabilities well, and recommends calibration procedures and k-fold strategies to reduce dependence on a held-out calibration set.[7]

For this project, the minimum credible ensemble is three to five independently trained models from completed subject-disjoint folds, with a validation-only calibration step and a locked-test evaluation performed once. Do not call repeated checkpoints from the same training trajectory an independent ensemble unless the independence claim is justified.

### 2.4 Knowledge distillation

If CT is available for training but unavailable at deployment, a multimodal teacher can supervise an MRI-only student. The student receives the MRI input and is trained with a mixture of the hard label loss and a distillation loss on teacher logits or softened probabilities. This is useful when the deployment contract is MRI-only but CT provides privileged information during development.

Distillation should be treated as a separate experiment. It does not prove that the student has access to CT information at inference; it only tests whether CT supervision improves the MRI representation. Compare the distilled student against the original MRI baseline and report calibration, not only accuracy.

## 3. How to combine imaging modalities

### 3.1 MRI sequences already form one multimodal input

The current implementation already combines four MRI sequences: T1, T1ce, T2, and FLAIR. These channels are available within one preprocessed case and share the current volume shape. Adding CT is therefore not the first multimodal problem in the repository; it is a new modality with different intensity semantics, resolution, acquisition timing, and likely availability.

The MRI and CT branches should have modality-specific preprocessing and encoders. MRI normalization should not be reused as CT normalization. CT requires a documented intensity policy, such as a justified Hounsfield-unit window and normalization, and both modalities require a verified spatial relationship before voxel- or feature-level fusion.

### 3.2 Early fusion

Early fusion would resample CT to the MRI grid, register it to MRI, normalize both, and pass a five-channel tensor to the first convolution. This is appropriate only if the paired cases have reliable alignment and the registration error is small relative to the tumour features of interest. It is easy to implement but makes the first layer responsible for resolving modality-specific scale and alignment differences.

Early fusion should be included as a baseline, not assumed to be optimal. A fair comparison requires the same cases, split, label, augmentations, and evaluation threshold across early, intermediate, and late fusion.

### 3.3 Intermediate fusion

Intermediate fusion should keep the MRI and CT encoders separate through several downsampling blocks. Fuse either pooled embeddings or matched-resolution feature maps. A concatenation-plus-MLP design is the appropriate minimum baseline; a gated or attention-based design can follow after establishing whether cross-modal interaction improves over late fusion.

A useful ablation matrix is:

| Model | MRI | CT | Fusion | Question answered |
|---|---:|---:|---|---|
| M1 | Yes | No | None | Current MRI baseline |
| M2 | No | Yes | None | CT-only value |
| M3 | Yes | Yes | Early | Does aligned input stacking help? |
| M4 | Yes | Yes | Intermediate concat | Does joint representation help? |
| M5 | Yes | Yes | Late calibrated | Does decision-level combination help? |
| M6 | Yes | Yes | Late + missing-mask logic | Is performance robust when one modality is absent? |

## 4. How to combine two datasets

### 4.1 Same task, same label, compatible preprocessing

If two datasets share the same task and label semantics, they may be pooled after a case-level audit. Retain `dataset_id`, scanner/site identifiers, acquisition metadata, and label provenance. Use source-aware sampling so a larger dataset does not dominate the gradient updates. Preserve subject-level grouping and split by patient before any patch extraction.

The evaluation should include both a pooled subject-disjoint split and an unseen-domain test, such as leave-one-dataset-out evaluation. Report overall and per-dataset metrics. The domain-generalization literature warns that medical-image shifts arise from modality, protocol, scanner, site, patient population, software, and acquisition variability.[4]

### 4.2 Same task, different label definitions

Do not merge labels merely because their names look similar. If one dataset contains clinical WHO grade and another contains an ET-derived proxy, the target is not identical. Options include a harmonized label subset, a multi-head model with dataset-specific heads, partial-label or masked loss, pretraining on one label system followed by fine-tuning on the target label, or an explicit label-noise model.

The current project should keep `grade_proxy` separate from clinical grade until the mapping has been independently reviewed. A larger sample with incompatible labels may reduce validity rather than improve it.

### 4.3 Paired versus unpaired modalities

A paired dataset contains MRI and CT for the same case, with a defensible correspondence between the images and the label. An unpaired collection contains MRI and CT from different cases or different cohorts. Unpaired data can support modality-specific pretraining or methods designed for shared representation learning, but it cannot support naive voxel concatenation.

The ShaSpec study models shared and modality-specific features and uses auxiliary distribution-alignment and domain-classification objectives to address missing modalities.[5] A separate brain-tumour study investigates learning from unpaired images, showing that unpaired learning is an explicit methodological setting rather than a license to combine arbitrary folders.[6]

### 4.4 Different scanner or site domains

The data should be treated as multi-domain if acquisition hardware, field strength, protocol, reconstruction, annotation process, or patient population differs. The practical options are:

| Strategy | Uses target-domain data during training? | Appropriate question |
|---|---:|---|
| Pooled training | Usually yes, without explicit domain objective | Can one model fit the combined sample? |
| Domain adaptation | Yes, often with labelled or unlabelled target data | Can a source model adapt to a known target site? |
| Domain generalization | No target data required during training | Can the model handle an unseen site? |
| Harmonization | Preprocesses or transforms distributions | Can acquisition differences be reduced before learning? |
| Transfer learning | Sequentially reuses weights | Can knowledge from a related dataset improve a smaller target set? |

The brain-MRI domain-adaptation benchmark across vendors found that no single adaptation method consistently dominates and that hyperparameter and compute costs remain important barriers.[3] Therefore, the first multi-dataset experiment should be a domain-stratified audit and leave-one-domain-out baseline, not a complex adversarial adaptation model.

## 5. Recommended implementation sequence for `major101`

### Step 0: Dataset and pairing audit

Create one case-level table with `case_id`, `subject_id`, `dataset_id`, `site_id`, `scanner_id` when available, MRI path, CT path, label, label provenance, and modality-presence flags. Measure how many cases are MRI-only, CT-only, and paired. Verify shape, orientation, spacing, finite values, and label conflicts before creating patches. Keep the locked test split frozen.

### Step 1: Preserve the current MRI baseline

Do not change the current MRI model while adding a modality. Re-run or retain its locked-test candidate, then evaluate all multimodal models on the same subject-disjoint cases. The current code’s `MemoryMappedPatchDataset`, `BalancedBatchSampler`, and binary metric helpers provide the correct starting contract.[9] [10]

### Step 2: Add a CT-only baseline

Implement a CT-specific dataset and encoder with the same output interface as the MRI model. Train it on the paired subset and report performance on the same paired validation/test cases. This establishes whether CT contains incremental signal before any fusion model is built.

### Step 3: Add calibrated late fusion

Freeze the MRI and CT checkpoints. Fit calibration and a fusion weight on validation data only. Compare equal-weight probability averaging, validation-fitted weighted averaging, and a low-capacity logistic stacker trained on out-of-fold development predictions. Evaluate once on the locked test.

### Step 4: Add intermediate fusion

Use two modality-specific encoders, concatenate pooled features, add presence masks, and train a small fusion head. Start with modality dropout so the model sees MRI-only, CT-only, and paired conditions. Compare against late fusion under identical splits and compute limits.

### Step 5: Test missing-modality robustness

Report four conditions: MRI only, CT only, both modalities, and the modality pattern expected at deployment. If the production contract is MRI-only, do not report paired-case performance as deployment performance. If CT is optional, define the fallback and escalation behavior before evaluation.

### Step 6: Extend to multiple datasets

Use dataset-aware sampling, preserve domain labels, and run leave-one-dataset-out validation. If labels are incompatible, use separate heads or target fine-tuning rather than a blind merge. Only after the pooled and external-style baselines are stable should domain adaptation or adversarial alignment be considered.

## 6. What not to do

Do not stack CT and MRI into one tensor before checking patient-level pairing, registration, orientation, spacing, intensity policies, and missingness. Do not join images by filename similarity alone. Do not pool datasets with different labels without a documented harmonization rule. Do not fit fusion weights, calibration, thresholds, or stacking coefficients on the locked test. Do not compare a paired-modality model against an MRI-only model on different cases and call the difference a modality gain. Do not claim that a model trained on two datasets generalizes unless at least one dataset or site is held out as an external-style domain.

## 7. Evidence quality and remaining uncertainty

The strongest evidence used here is the systematic fusion review, the peer-reviewed domain-adaptation benchmark, the NeurIPS primary architecture paper, and the peer-reviewed ensemble-calibration study.[1] [2] [3] [7] The domain-generalization survey and ShaSpec paper were accessed as arXiv full texts and are valuable methodological sources, but their claims should be triangulated with task-specific experiments before becoming project requirements.[4] [5] The unpaired brain-tumour study was verified through its PubMed record; its full methods and exact metrics should be retrieved from the publisher before using quantitative claims.[6]

No source establishes that one fusion point is universally superior. The project should therefore treat the following as hypotheses to test: calibrated late fusion will be the strongest low-risk baseline; intermediate fusion may add cross-modal value if the paired cohort is large and well registered; and dataset-aware training plus an unseen-domain evaluation will be more informative than a larger random pooled split.

## References

[1]: [Huang et al., “Fusion of medical imaging and electronic health records using deep learning: a systematic review and implementation guidelines,” *npj Digital Medicine* (2020)](https://www.nature.com/articles/s41746-020-00341-z)

[2]: [Hemker, Simidjievski, and Jamnik, “HEALNet: Multimodal Fusion for Heterogeneous Biomedical Data,” *NeurIPS 2024*](https://proceedings.neurips.cc/paper_files/paper/2024/hash/765871e77d2ca65126d3d64d31aa6908-Abstract-Conference.html)

[3]: [Saat et al., “A domain adaptation benchmark for T1-weighted brain magnetic resonance image segmentation,” *Frontiers in Neuroinformatics* (2022)](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.919779/full)

[4]: [Yoon et al., “Domain Generalization for Medical Image Analysis: A Survey,” arXiv:2310.08598v2 (2024)](https://arxiv.org/html/2310.08598v2)

[5]: [Wang et al., “Multi-modal Learning with Missing Modality via Shared-Specific Feature Modelling,” arXiv:2307.14126v2 (2024)](https://arxiv.org/html/2307.14126v2)

[6]: [Liu et al., “Learning multi-modal brain tumor segmentation from unpaired images,” PubMed PMID 37105113](https://pubmed.ncbi.nlm.nih.gov/37105113/)

[7]: [Buddenkotte et al., “Calibrating ensembles for scalable uncertainty quantification in deep learning-based medical image segmentation,” *Computers in Biology and Medicine* (2023)](https://www.cruk.cam.ac.uk/publications/calibrating-ensembles-for-scalable-uncertainty-quantification-in-deep-learningbased-medical-image-segmentation/)

[8]: [Pajany et al., “Multimodal deep feature fusion with transformer for brain tumor classification from magnetic resonance imaging,” *Scientific Reports* (2026)](https://www.nature.com/articles/s41598-026-44957-9)

[9]: [`src/grade_data.py` in the current implementation](../src/grade_data.py)

[10]: [`src/grade_model.py` and `scripts/train_ultra_light.py` in the current implementation](../src/grade_model.py)


## Code alignment

The research recommendations have been checked against the active implementation in [`22_code_alignment_fusion_gap.md`](22_code_alignment_fusion_gap.md). That comparison identifies which ideas are already supported by existing data integrity, split, cross-validation, and checkpoint infrastructure, and which require new modality, manifest, fusion, calibration, or domain-aware code.
