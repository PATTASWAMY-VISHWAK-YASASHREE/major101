# Category 1: Brain Tumour Imaging Basics

**Status: ✅ RESEARCH COMPLETE**

---

## 1.1 The Imaging Problem

Brain tumours are classified by the WHO into **Grades I–IV**, based on histopathology and molecular markers (IDH, 1p/19q, MGMT). Imaging must predict this grade from **non-invasive scans**.

| Grade | Name | MRI appearance | CT appearance |
|---|---|---|---|
| I | Pilocytic astrocytoma, meningioma | Well-defined, cystic/solid | Often isodense |
| II | Diffuse astrocytoma (IDH-mutant) | Infiltrative, T2-hyperintense | Hypodense, no enhancement |
| III | Anaplastic astrocytoma | Moderate enhancement, edema | Heterogeneous |
| IV | Glioblastoma | Ring-enhancing, necrotic core | Mixed hypo/hyperdense |

---

## 1.2 MRI vs CT: What Each Sees

| Feature | MRI | CT |
|---|---|---|
| Soft tissue contrast | Excellent (T1, T2, FLAIR, T1ce) | Poor |
| Edema detection | FLAIR is gold standard | Subtle |
| Hemorrhage | Susceptibility (SWI/T2*) | Hyperdense (acute) |
| Bone involvement | Poor | Excellent |
| Calcification | Missed | Easily seen |
| Necrosis | Ring enhancement (T1ce) | Hypodense core |
| Acute scan time | 30-45 min | 2-5 min |
| Radiation | None | ~2 mSv |

**Key insight:** MRI and CT are **complementary**, not redundant. MRI sees soft tissue; CT sees bone, blood, and calcification. Fusion captures both.

---

## 1.3 Standard MRI Sequences for Tumour Imaging

| Sequence | What it shows |
|---|---|
| **T1-weighted** | Anatomy, post-contrast enhancement |
| **T1-weighted post-contrast (T1ce)** | Blood-brain barrier breakdown — the hallmark of high-grade tumours |
| **T2-weighted** | Water content — oedema and tumour extent |
| **FLAIR** | Oedema without CSF signal — crucial for infiltrative tumour boundary |
| **DWI/ADC** | Cellularity — restricted diffusion in high-grade tumours |

---

## 1.4 Why Multimodal (CT+MRI) Is Rare

- **99% of published tumour work is MRI-only.** CT is almost never included.
- CT is ordered in trauma/acute settings, MRI in oncology — **different clinical workflows**.
- This gap is exactly what our project addresses.
