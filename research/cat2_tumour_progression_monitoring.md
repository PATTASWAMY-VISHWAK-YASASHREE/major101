# Category 2: Tumour Progression Monitoring

**Status: ✅ RESEARCH COMPLETE**

---

## 2.1 What Progression Monitoring Means

Longitudinal scans track tumour size and shape over time. Key questions:
- **Growing?** (progression)
- **Shrinking?** (response to therapy)
- **Stable?** (pseudoprogression?)

---

## 2.2 RANO Criteria (Response Assessment in Neuro-Oncology)

| Criteria | How it works | Limitation |
|---|---|---|
| **RANO (2010)** | Measures longest diameter on T1ce; 25% change = progression/response | Only works for enhancing tumours |
| **RANO-LGG (2014)** | Includes FLAIR for non-enhancing tumours | Manual measurement |
| **RANO-BBM (2019)** | For brain metastases | Multiple lesion counting |

**AI contribution:** Automate RANO measurements + flag subtle changes missed by radiologists.

---

## 2.3 Pseudoprogression Problem

Post-radiation oedema mimics tumour growth on MRI. Radiologists struggle to distinguish:
- **True progression** — tumour is growing
- **Pseudoprogression** — treatment effect, not tumour

**AI angle:** CT adds information MRI alone cannot — bone integrity, acute haemorrhage patterns — that can disambiguate.

---

## 2.4 Our Project's Approach

- **Not the primary goal** — classification comes first
- **Built-in from day one:** per-timepoint feature extractor in the architecture
- Add Temporal Transformer + RANO measurement automation when longitudinal data arrives
- See `research/05_longitudinal_analysis.md` for architectural details
