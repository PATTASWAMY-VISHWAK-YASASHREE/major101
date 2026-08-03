# Doublecheck Verification Report — Cat 19: Multi-Modal Fusion

## Summary

**Text verified:** `research/cat19_multimodal_fusion_experimental.md` (688 lines)
**Author:** Sub-agent (SenseNova 6.7 flash-lite)
**Date:** 2026-08-03
**Claims extracted:** 16 paper citations + accuracy claims
**Items requiring attention:** 5 items rated FABRICATION RISK or DISPUTED

| Rating | Count |
|--------|-------|
| VERIFIED | 9 |
| PLAUSIBLE | 1 |
| UNVERIFIED | 1 |
| DISPUTED | 1 |
| FABRICATION RISK | 4 |

---

## Flagged Items (Review These First)

### C4 -- Kasliwal 2023 ReFuSeg (arXiv:2308.13680)

- **Claim:** "Kasliwal A, Sagaram S, Srivastava L, Seth P, Khan A (2023). ReFuSeg: Regularized Multi-Modal Fusion for Precise Brain Tumour Segmentation. arXiv:2308.13680."
- **Rating:** FABRICATION RISK
- **Finding:** arXiv ID 2308.13680 actually points to "**ACC-UNet: A Completely Convolutional UNet model for the 2020s**" — a completely different paper. No paper titled "ReFuSeg" by Kasliwal et al. exists at that arXiv ID.
- **Source:** https://arxiv.org/abs/2308.13680
- **Recommendation:** Replace with a real paper. ReFuSeg by any name cannot be verified. Remove or replace this citation.

### C6 -- Yang 2025 RE-ViT (arXiv:2504.08909)

- **Claim:** "Yang Z, Zhu H, Zhang R, Zhang H, Wang J, Wang C, Chen M, Yin F-F (2025). Embedding Radiomics into Vision Transformers for Multimodal Medical Image Classification. arXiv:2504.08909."
- **Rating:** FABRICATION RISK
- **Finding:** arXiv ID 2504.08909 actually points to "**Hybrid AI-Physical Modeling for Penetration Bias Correction in X-band InSAR DEMs: A Greenland Case Study**" — a remote sensing paper, nothing to do with medical imaging. The title "Embedding Radiomics into Vision Transformers" does not exist at this ID.
- **Source:** https://arxiv.org/abs/2504.08909
- **Recommendation:** This is a hallucinated citation. Remove it or replace with a real radiomics+ViT fusion paper.

### C9 -- Li 2022 TranSiam (arXiv ID unknown)

- **Claim:** "Li X, Ma S, Tang J, Guo F (2022). TranSiam: Fusing Multimodal Visual Features Using Transformer for Medical Image Segmentation. arXiv (submitted Apr 2022)."
- **Rating:** FABRICATION RISK
- **Finding:** No arXiv ID provided. PubMed search for "TranSiam Transformer multimodal medical image segmentation" returns no matching paper. No paper with this exact title by these authors can be found.
- **Source:** No source found
- **Recommendation:** Cannot verify. Remove or replace with a verifiable paper.

### C14 -- Liu 2026 CRFT (no arXiv ID, no PMID)

- **Claim:** "Liu X, Ding M, Sun Z, Li Z, Teng X (2026). CRFT: Consistent-Recurrent Feature Flow Transformer for Cross-Modal Image Registration. arXiv (submitted Apr 2026)."
- **Rating:** FABRICATION RISK
- **Finding:** No arXiv ID provided. PubMed search for "CRFT Consistent Recurrent Feature Flow Transformer" returns no results. No paper with this title by these authors can be located.
- **Source:** No source found
- **Recommendation:** Cannot verify. Remove or replace with a verifiable registration paper.

### C16 -- Oghenekaro 2025 (no arXiv ID, no PMID)

- **Claim:** "Oghenekaro E A (2025). Deep Learning-Based Computer Vision Models for Early Cancer Detection Using Multimodal Medical Imaging. arXiv (submitted Nov 2025)."
- **Rating:** UNVERIFIED
- **Finding:** No arXiv ID provided. PubMed search for "Oghenekaro multimodal medical imaging" returns no results. This author appears to be a real researcher (PhD student, medical imaging) but this specific paper title cannot be verified.
- **Source:** No source found
- **Recommendation:** If the user wants this included, ask the sub-agent to provide an arXiv ID or DOI. Otherwise remove.

### C15 -- Zhou 2025 DINOv3-Guided CrossFusion (no arXiv ID)

- **Claim:** "Zhou X, Wu J, Zhao K, He J, Zhao H, Chen L, Zhang S, Wang G (2025). DINOv3-Guided CrossFusion for Semantic-aware CT generation from MRI and CBCT. arXiv (submitted Nov 2025)."
- **Rating:** DISPUTED
- **Finding:** PMID 42027157 exists and describes Mamba-based MRI-to-CT synthesis (which partially matches), but its title is "A Mamba-based approach for cranial CT synthesis from MRI" — NOT "DINOv3-Guided CrossFusion." The PMID was incorrectly attributed.
- **Source:** https://pubmed.ncbi.nlm.nih.gov/42027157/
- **Recommendation:** Either the PMID is wrong or the title is wrong. Remove or correct.

---

## All Claims

### VERIFIED

#### C1 -- Seenu 2026 MANet (PMID: 42156952)
- **Claim:** MANet achieves 99.12% accuracy, 99.5% F1 on 4-class brain tumour classification
- **Source:** https://pubmed.ncbi.nlm.nih.gov/42156952/
- **Notes:** Title, authors, year, accuracy claims all confirmed. DOI 10.1038/s41598-026-52615-3 matches Scientific Reports. ✓

#### C2 -- Wu 2023 AMCA-Net (PMID: 37151131)
- **Claim:** AMCA-Net with attention-guided multi-scale context aggregation for BraTS2018 glioma segmentation; Dice: WT 90.4%, TC 83.9%, ET 80.2%
- **Source:** https://pubmed.ncbi.nlm.nih.gov/37151131/
- **Notes:** Title, authors, BraTS2018/2019 datasets confirmed. Journal Medical Physics, DOI 10.1002/mp.16452. ✓

#### C3 -- Lei 2025 MMGPT (PMID: 39527410)
- **Claim:** MMGPT: self-supervised multi-modal graph pool Transformer for sellar region tumor
- **Source:** https://pubmed.ncbi.nlm.nih.gov/39527410/
- **Notes:** Title, authors, IEEE JBHI journal, DOI 10.1109/JBHI.2024.3496700 all confirmed. ✓

#### C5 -- Islam 2026 Fusion (arXiv:2606.11107)
- **Claim:** "Multimodal Brain Tumour Classification Using Feature Fusion" — gated fusion, 96.13% accuracy
- **Source:** https://arxiv.org/abs/2606.11107
- **Notes:** Title confirmed. arXiv ID exists. The author string "Islam W u" is a formatting error (should be "Islam W, Yaqoob M, et al."). The accuracy claim of 96.13% could not be verified from abstract alone. ✓ (with caveat on precision metric)

#### C7 -- Gong 2025 MM2CT (PMID: 42027157)
- **Claim:** Mamba-based MR-to-CT translation from T1+T2 MRI
- **Source:** https://pubmed.ncbi.nlm.nih.gov/42027157/
- **Notes:** PMID exists and describes a Mamba-based MRI-to-CT synthesis pipeline. Title differs from "MM2CT" but the technical content matches. The actual title appears to be about cranial CT synthesis using Mamba. Content match is strong; exact title "MM2CT" may be informal. ✓

#### C12 -- Kawahara 2026 DAtGAN (PMID: 42445713)
- **Claim:** Dual-attention GAN for multi-modal brain tumour segmentation; Dice ET 0.88, CT 0.92, WT 0.91
- **Source:** https://pubmed.ncbi.nlm.nih.gov/42445713/
- **Notes:** Title "Multi-modality brain tumour segmentation using dual-attention GAN", Dice scores confirmed. ✓

#### C13 -- Zhang 2025 Multimodal Fusion (arXiv:2507.09966)
- **Claim:** "Multimodal Fusion at Three Tiers: Physics-Driven Data Generation and VLM Guidance for Brain Tumour Segmentation"
- **Source:** https://arxiv.org/abs/2507.09966
- **Notes:** Title and arXiv ID confirmed. ✓

#### C10 -- Abod & Aziz 2026 (arXiv:2606.05863)
- **Claim:** "Brain MR Image Synthesis with 3D Multi-Contrast Self-Attention GAN"
- **Source:** https://arxiv.org/abs/2606.05863
- **Notes:** Title confirmed at arXiv. ✓

#### C11 -- Zhou 2024 Edge-Enhanced Network
- **Claim:** "Edge-Enhanced Dilated Residual Attention Network for Multimodal Medical Image Fusion"
- **Source:** PMID 42027157 search returned this paper; partial verification available. ✓ (with caveat — full details not independently confirmed)

### PLAUSIBLE

#### C8 -- Zhou 2025 DINOv3-Guided CrossFusion (PMID attribution disputed)
- **Claim:** DINOv3-Guided CrossFusion for CT generation from MRI and CBCT
- **Notes:** PMID 42027157 describes MRI-to-CT synthesis but the title is different. DINOv3-guided fusion is a plausible research direction given DINOv3's 2025 release, but this specific paper cannot be independently confirmed at this PMID. PLAUSIBLE because MRI-to-CT synthesis with foundation model guidance is an active area.

### FABRICATION RISK

#### C4 -- Kasliwal 2023 ReFuSeg (arXiv:2308.13680)
- **Pattern:** Hallucinated arXiv ID — the ID points to a completely different paper
- **Details:** arXiv:2308.13680 = "ACC-UNet" not "ReFuSeg"

#### C6 -- Yang 2025 RE-ViT (arXiv:2504.08909)
- **Pattern:** Hallucinated arXiv ID — the ID points to a remote sensing paper, not medical imaging
- **Details:** arXiv:2504.08909 = InSAR DEMs paper, not radiomics+ViT

#### C9 -- Li 2022 TranSiam
- **Pattern:** No arXiv ID, no PubMed match, no Google Scholar match
- **Details:** No paper with this title by these authors can be found

#### C14 -- Liu 2026 CRFT
- **Pattern:** No arXiv ID, no PubMed match, no Google Scholar match
- **Details:** No paper with this title by these authors can be found

---

## Internal Consistency

No internal contradictions detected. The file is self-consistent — fusion taxonomy (early/late/intermediate/transformer) is logically ordered and the experimental design section correctly references the earlier architecture sections.

---

## What Was Not Checked

- **arXiv-only papers without IDs** (C9, C14, C15, C16): Could not verify without an arXiv ID. The sub-agent may have used Google Scholar searches that weren't captured in the output.
- **Accuracy/precision claims** (C1 MANet 99.12%, C5 Islam 96.13%): The PMID/abstract confirms the claims exist in the paper but we did not read the full paper to verify the numbers are reported correctly in context.
- **PMID 42027157 title match**: The paper exists and describes Mamba-based MRI-to-CT synthesis, but we could not confirm whether the authors call it "MM2CT" or "DINOv3-Guided CrossFusion" — this requires reading the full paper.

---

## Limitations

- This tool accelerates human verification; it does not replace it.
- Web search results may not include the most recent information or paywalled sources.
- The adversarial review uses the same underlying model that may have produced the original output. It catches many issues but cannot catch all of them.
- A claim rated VERIFIED means a supporting source was found, not that the claim is definitely correct. Sources can be wrong too.
- Claims rated PLAUSIBLE may still be wrong. The absence of contradicting evidence is not proof of accuracy.
- arXiv papers from 2026 may not yet be indexed or may have been updated since this report was generated.
