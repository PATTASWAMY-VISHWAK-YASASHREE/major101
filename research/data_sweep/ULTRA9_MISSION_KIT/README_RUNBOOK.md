# Ultra 9 Mission Kit — PROTEAS Download + Audit

**Target machine:** Core Ultra 9, 32GB RAM, 1TB storage (the workhorse).
**This kit needs:** Python 3.9+ ONLY. No pip installs, no GPU, no internet beyond zenodo.org.
**Get this kit there:** copy the `ULTRA9_MISSION_KIT` folder via OneDrive (it syncs) or USB.

---

## STEP-BY-STEP (run on the Ultra 9)

### 1. Stage 1 — download the small verification set (~700 MB)

```bat
python download_proteas.py --out D:\proteas_raw
```

Downloads (with md5 verification + resume):
- `PROTEAS-Clinical_and_demographic_data.xlsx` (+ radiomics xlsx if present)
- `P01.zip`, `P02.zip`, `P03.zip`

### 2. Audit gate — verify BEFORE committing the full 15 GB

```bat
python audit_proteas.py --src D:\proteas_raw
```

- Audits the zips DIRECTLY (no extraction needed — saves disk)
- Verifies: patient folder structure, per-timepoint 4 sequences (t1/t1c/t2/fla),
  NIfTI headers parsed with a built-in pure-Python reader (shape, dtype, spacing),
  segmentation/CT/RT files present, md5 integrity
- Writes `proteas_audit_report.json` and prints a per-patient timeline table

**STOP CHECK:** if the report says any patient FAIL, bring the JSON back here before downloading more.

### 3. Open the clinical xlsx in Excel

Note these things and tell me: patient count, timepoint columns, treatment fields,
survival/date fields, anything that looks like per-lesion IDs. (One screenshot is fine.)

### 4. Stage 2 — full download (~14.4 GB more, leave overnight if slow)

```bat
python download_proteas.py --out D:\proteas_raw --all
```

Re-runs are safe: already-verified files are skipped.

### 5. Re-audit everything

```bat
python audit_proteas.py --src D:\proteas_raw
```

### 6. (Optional companion) Brain-Tumor-Progression (3.16 GB, glioma)

- Browser: https://www.kaggle.com/datasets/andrewmvd/brain-tumor-progression → Download
  (needs free Kaggle account) — or -
- TCIA original (no signup): https://www.cancerimagingarchive.net/collection/brain-tumor-progression
- Save as `D:\btp\brain-tumor-progression.zip` (DICOM audit comes after PROTEAS passes)

---

## WHAT TO BRING BACK HERE (to the i5 command center)

1. `proteas_audit_report.json` (both stages)
2. The xlsx observations (step 3)
3. A `dir D:\proteas_raw` listing

Then we do the go/no-go decision HERE, and the training scripts get built for the Ultra 9 next.

---

## DISK MATH (Ultra 9)

| Item | Size |
|---|---|
| PROTEAS zips (all 45) | 15.1 GB |
| Extracted (later, training time) | ~15 GB |
| Brain-Tumor-Progression | 3.2 GB |
| BraTS-Reg (later, optional) | 4.3 GB |
| **Total for core plan** | **~38 GB** (1TB drive: no problem) |

Keep the zips after extraction — the audit reads zips, and zips are the md5-verified source of truth.

## IF A DOWNLOAD FAILS

- Re-run the same command — verified files are skipped, broken ones redownload
- md5 mismatch 3× on one file = tell me, I'll find an alternate route
- Zenodo is sometimes slow (EU servers) — nights are faster
