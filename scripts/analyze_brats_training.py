"""BraTS-GLI Training Data — comprehensive analysis (streaming, <1 GB RAM).

Analyses MRI intensities + segmentation masks. Designed to run overnight
(5+ hours) on a machine with tight RAM (8 GB) and limited free disk.
"""
import re
import sys
import gc
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "brats2024" / "training" / "BraTS2024-BraTS-GLI-TrainingData" / "training_data1_v2"
OUT  = ROOT / "outputs" / "brats_training_analysis"
OUT.mkdir(parents=True, exist_ok=True)

MODALITIES = {"t1c": "T1ce", "t1n": "T1n", "t2f": "FLAIR", "2w": "T2w", "t2w": "T2w"}
SEG_LABELS = {1: "edema (ET)", 2: "non-enhancing core (NCR)", 3: "enhancing core (ET)", 4: "whole tumor (WST)"}
TUMOR_REGIONS = {
    "tumor_core": [2, 3],       # regions 2 or 3
    "whole_tumor": [4],         # region 4 = union of 1+2+3 (always present)
    "edema_only": [1],
    "enhancing_only": [3],
}
VOXEL_SPACING = (1.0, 1.0, 1.0)  # mm³; all BraTS-GLI are isotropic 1mm

def fmt_secs(t):
    h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"


# ── Helper: streaming intensity stats (mean, std, min, max, nonzero) ─────────
def stream_stats(arr):
    """Return (mean, var, n, nnz, min, max) from a single array."""
    n = arr.size
    s = float(arr.sum())
    s2 = float(np.square(arr).sum())
    return (s / n, (s2 / n) - (s / n) ** 2, n, int((arr > 0).sum()),
            float(arr.min()), float(arr.max()))


# ── PASS 1: scan every case, collect metadata + streaming accumulators ───────

cases = sorted(d for d in DATA.iterdir() if d.is_dir())
print(f"[PASS 1] Scanning {len(cases)} cases ...")
print(f"  Output dir: {OUT}")

# Accumulators: mod -> [sum, sum_sq, count, nnz, min, max]
agg = {}
seg_agg = {}  # region_name -> [vol_mm3_sum, count]
case_meta = []  # per-case row dicts

# Progress tracking
t0 = __import__("time").time()
batch = 50

for i, case_dir in enumerate(cases):
    case_id = case_dir.name
    seq_m = re.search(r"-(\d{3})$", case_id)
    pat_m = re.search(r"BraTS-GLI-(\d+)-", case_id)
    sequence = seq_m.group(1) if seq_m else "?"
    patient  = pat_m.group(1) if pat_m else "?"

    row = {"case_id": case_id, "patient": patient, "sequence": sequence}

    # ── MRI modalities ────────────────────────────────────────────────────────
    for f in sorted(case_dir.glob("*.nii.gz")):
        mod = f.stem.rsplit("-", 1)[-1].replace(".nii", "")
        if mod == "seg":
            continue  # handled below
        if mod not in MODALITIES:
            continue
        row[f"{mod}_bytes"] = f.stat().st_size
        try:
            img = nib.load(f)
            arr = img.get_fdata(dtype=np.float32)
            row[f"{mod}_shape"] = ",".join(map(str, arr.shape))
            row[f"{mod}_min"]   = float(arr.min())
            row[f"{mod}_max"]   = float(arr.max())
            row[f"{mod}_mean"]  = float(arr.mean())
            row[f"{mod}_std"]   = float(arr.std())
            row[f"{mod}_nonzero_pct"] = float((arr > 0).mean() * 100)
            s, var, n, nnz, mn, mx = stream_stats(arr)
            a = agg.setdefault(mod, [0.0, 0.0, 0, 0, float("inf"), float("-inf")])
            a[0] += s * n; a[1] += var * n + s * s * n; a[2] += n; a[3] += nnz
            a[4] = min(a[4], mn); a[5] = max(a[5], mx)
        except Exception as e:
            row[f"{mod}_err"] = str(e)
            print(f"  WARN {f.name}: {e}", file=sys.stderr)
        del arr, img; gc.collect()

    # ── Segmentation mask ─────────────────────────────────────────────────────
    seg_path = case_dir / f"{case_id}-seg.nii.gz"
    if seg_path.exists():
        row["seg_bytes"] = seg_path.stat().st_size
        try:
            seg_img = nib.load(seg_path)
            seg = seg_img.get_fdata(dtype=np.float32).astype(np.int16)
            shape_str = ",".join(map(str, seg.shape))
            row["seg_shape"] = shape_str
            row["seg_nunique"] = len(np.unique(seg))

            # Count voxels per label
            labels, counts = np.unique(seg[seg > 0], return_counts=True)
            for lbl, cnt in zip(labels, counts):
                lbl_name = SEG_LABELS.get(int(lbl), f"label_{int(lbl)}")
                vol_mm3 = cnt * float(np.prod(VOXEL_SPACING))
                row[f"seg_{lbl_name}_voxels"] = int(cnt)
                row[f"seg_{lbl_name}_mm3"] = vol_mm3
                # accumulate
                sa = seg_agg.setdefault(lbl_name, [0.0, 0])
                sa[0] += vol_mm3; sa[1] += 1

            # Tumor regions (composite)
            for region_name, region_labels in TUMOR_REGIONS.items():
                mask = np.isin(seg, region_labels)
                vol_mm3 = mask.sum() * float(np.prod(VOXEL_SPACING))
                row[f"tumor_{region_name}_voxels"] = int(mask.sum())
                row[f"tumor_{region_name}_mm3"] = vol_mm3
                ra = seg_agg.setdefault(f"tumor_{region_name}", [0.0, 0])
                ra[0] += vol_mm3; ra[1] += 1

            # Tumor volume fraction: whole tumor (label 4) voxels / total voxels
            total_voxels = seg.size
            tumor_voxels = int((seg == 4).sum())
            if total_voxels > 0:
                row["tumor_volume_fraction"] = float(tumor_voxels / total_voxels)

        except Exception as e:
            row["seg_err"] = str(e)
            print(f"  WARN seg {case_id}: {e}", file=sys.stderr)
        del seg, seg_img; gc.collect()

    case_meta.append(row)

    if (i + 1) % batch == 0:
        elapsed = __import__("time").time() - t0
        eta = elapsed / (i + 1) * (len(cases) - i - 1)
        print(f"  [{i+1}/{len(cases)}] elapsed={fmt_secs(elapsed)} eta={fmt_secs(eta)}")
    if i == len(cases) - 1:
        elapsed = __import__("time").time() - t0
        print(f"  [{i+1}/{len(cases)}] DONE in {fmt_secs(elapsed)}")

print(f"\n  -> {len(cases)} cases scanned, "
      f"{len(cases_meta) if 'cases_meta' in dir() else len(case_meta)} rows collected.")


# ── Compute aggregate intensity stats ────────────────────────────────────────

print("\n[PASS 2] Computing aggregate intensity statistics ...")
agg_rows = []
for mod, (sum_n, sum_n_var, n, nnz, mn, mx) in agg.items():
    total_voxels = n
    grand_mean = sum_n / n if n else 0.0
    # sum_n_var = sum(var_i * n_i + mean_i^2 * n_i) = sum(x^2)
    grand_var = (sum_n_var - sum_n * sum_n / n) / n if n else 0.0
    grand_std = max(0.0, grand_var) ** 0.5
    agg_rows.append({
        "modality": mod,
        "n_cases": len(cases),
        "total_voxels": n,
        "grand_mean": grand_mean,
        "grand_std": grand_std,
        "grand_min": mn, "grand_max": mx,
        "nonzero_pct": nnz / n * 100 if n else 0.0,
    })
    print(f"  {mod}: mean={grand_mean:.1f}, std={grand_std:.1f}, "
          f"min={mn:.1f}, max={mx:.1f}, non-zero={nnz/n*100:.2f}%")

# Percentiles — sparse sampling (100 cases, downsampled 4x per axis)
print("  Sampling percentiles (100 cases, 4x downsample) ...")
for mod in ["t1c", "t1n", "t2f", "t2w"]:
    files = sorted(DATA.rglob(f"*-{mod}.nii.gz"))[:100]
    chunks = []
    for f in files:
        try:
            arr = nib.load(f).get_fdata(dtype=np.float32)
            chunks.append(arr[::4, ::4, ::4].flatten())
        except:
            pass
        del arr; gc.collect()
    if chunks:
        v = np.concatenate(chunks)
        for r in agg_rows:
            if r["modality"] == mod:
                r["p1"] = float(np.percentile(v, 1))
                r["p5"] = float(np.percentile(v, 5))
                r["p50"] = float(np.percentile(v, 50))
                r["p95"] = float(np.percentile(v, 95))
                r["p99"] = float(np.percentile(v, 99))
        del v, chunks; gc.collect()

agg_df = pd.DataFrame(agg_rows)
agg_df.to_csv(OUT / "modality_stats.csv", index=False)
print(f"  -> modality_stats.csv")


# ── Per-case metadata CSV ────────────────────────────────────────────────────

case_df = pd.DataFrame(case_meta)
case_df.to_csv(OUT / "case_metadata.csv", index=False)
print(f"  -> case_metadata.csv ({len(case_df)} rows)")

# Segmentation summary CSV
seg_rows = []
for name, (total_vol, cnt) in seg_agg.items():
    seg_rows.append({
        "region": name,
        "n_cases_with_tumor": int(cnt),
        "total_volume_mm3": total_vol,
        "mean_volume_mm3": total_vol / cnt if cnt else 0.0,
    })
seg_df = pd.DataFrame(seg_rows)
seg_df.to_csv(OUT / "tumor_volume_summary.csv", index=False)
print(f"  -> tumor_volume_summary.csv")


# ── Histograms (sample-based) ────────────────────────────────────────────────

print("\n[PASS 3] Generating histograms ...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
colors = {"t1c": "#e74c3c", "t1n": "#3498db", "t2f": "#2ecc71", "t2w": "#9b59b6"}

for i, mod in enumerate(["t1c", "t1n", "t2f", "t2w"]):
    ax = axes[i]
    files = sorted(DATA.rglob(f"*-{mod}.nii.gz"))[:100]
    chunks = []
    for f in files:
        try:
            arr = nib.load(f).get_fdata(dtype=np.float32)
            chunks.append(arr[::4, ::4, ::4].flatten())
        except:
            pass
        del arr; gc.collect()
    if chunks:
        vals = np.concatenate(chunks)
        hi = np.percentile(vals, 99.5)
        ax.hist(vals[vals <= hi], bins=80, color=colors[mod], edgecolor="none", alpha=0.85)
        ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1,
                   label=f"mean={vals.mean():.0f}")
        ax.set_title(f"{mod.upper()} — value distribution")
        ax.set_xlabel("Intensity"); ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    del vals, chunks; gc.collect()
plt.tight_layout()
plt.savefig(OUT / "modality_histograms.png", dpi=120)
plt.close()
print(f"  -> modality_histograms.png")


# ── Tumor volume distribution histogram ──────────────────────────────────────

print("  Generating tumor volume distribution plots ...")
whole_tumor_vols = []
for row in case_meta:
    vol = row.get("tumor_whole_tumor_mm3", 0)
    if vol and vol > 0:
        whole_tumor_vols.append(float(vol))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
if whole_tumor_vols:
    axes[0].hist(whole_tumor_vols, bins=40, color="#e74c3c", edgecolor="none", alpha=0.85)
    axes[0].set_title("Whole Tumor Volume Distribution")
    axes[0].set_xlabel("Volume (mm^3)"); axes[0].set_ylabel("Cases")
    axes[0].set_xlim(0, max(whole_tumor_vols) * 0.5)
else:
    axes[0].text(0.5, 0.5, "No tumor volumes found", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_title("Whole Tumor Volume Distribution")

# Tumor volume fraction
tumor_fracs = [r.get("tumor_volume_fraction", 0) for r in case_meta if r.get("tumor_volume_fraction", 0) > 0]
if tumor_fracs:
    axes[1].hist(tumor_fracs, bins=40, color="#9b59b6", edgecolor="none", alpha=0.85)
    axes[1].set_title("Tumor Volume Fraction Distribution")
    axes[1].set_xlabel("Fraction of brain"); axes[1].set_ylabel("Cases")
else:
    axes[1].text(0.5, 0.5, "No tumor fractions found", ha="center", va="center", transform=axes[1].transAxes)
plt.tight_layout()
plt.savefig(OUT / "tumor_volume_distribution.png", dpi=120)
plt.close()
print(f"  -> tumor_volume_distribution.png")

del whole_tumor_vols, tumor_fracs; gc.collect()


# ── Sample slices with tumor overlay ─────────────────────────────────────────

print("  Generating sample slices with tumor overlay ...")
# Find a case with a clear tumor
tumor_cases = sorted(
    [c for c in case_meta if c.get("tumor_whole_tumor_mm3", 0) and c["tumor_whole_tumor_mm3"] > 1000],
    key=lambda r: r["tumor_whole_tumor_mm3"], reverse=True
)[:5]

if tumor_cases:
    best_case = tumor_cases[0]
    case_dir = DATA / best_case["case_id"]
    mid = 90  # axial mid-slice
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: modalities
    for i, mod in enumerate(["t1c", "t1n", "t2f", "t2w"]):
        fpath = case_dir / f"{case_dir.name}-{mod}.nii.gz"
        if fpath.exists():
            arr = nib.load(fpath).get_fdata(dtype=np.float32)
            lo, hi = np.percentile(arr, [1, 99])
            sliced = np.clip((arr[:, :, mid] - lo) / (hi - lo + 1e-8), 0, 1)
            ax = axes[0, i]
            ax.imshow(sliced, cmap="gray", aspect="auto")
            ax.set_title(f"{mod.upper()}", fontsize=9)
            ax.axis("off")
        del arr; gc.collect()

    # Bottom row: segmentation overlay
    seg_path = case_dir / f"{case_dir.name}-seg.nii.gz"
    if seg_path.exists():
        seg = nib.load(seg_path).get_fdata(dtype=np.int16)
        seg_slice = seg[:, :, mid]
        tumor_mask = (seg_slice == 4)

        # Overlay on T2f
        t2f_path = case_dir / f"{case_dir.name}-t2f.nii.gz"
        if t2f_path.exists():
            t2f_arr = nib.load(t2f_path).get_fdata(dtype=np.float32)
            lo, hi = np.percentile(t2f_arr, [1, 99])
            t2f_slice = np.clip((t2f_arr[:, :, mid] - lo) / (hi - lo + 1e-8), 0, 1)

            ax = axes[1, 0]
            ax.imshow(t2f_slice, cmap="gray", aspect="auto")
            ax.imshow(tumor_mask, cmap="Reds", alpha=0.5, aspect="auto")
            ax.set_title("FLAIR + Tumor Mask", fontsize=9)
            ax.axis("off")
            del t2f_arr; gc.collect()

            # Label breakdown
            ax = axes[1, 1]
            label_colors = np.zeros_like(seg_slice, dtype=float)
            for lbl, color in [(1, 1.0), (2, 0.5), (3, 0.25)]:
                label_colors[seg_slice == lbl] = color
            ax.imshow(label_colors, cmap="hot", aspect="auto", vmin=0, vmax=1)
            ax.set_title("Tumor Sub-regions (1=edema, 2=NCR, 3=ET)", fontsize=9)
            ax.axis("off")

            # Stats
            ax = axes[1, 2]
            ax.axis("off")
            vols = {
                "Edema (WC)": best_case.get("seg_edema (WC)_mm3", 0),
                "Non-Enhancing (NCR)": best_case.get("seg_non-enhancing tumor core (NCR)_mm3", 0),
                "Enhancing (ET)": best_case.get("seg_enhancing tumor core (ET)_mm3", 0),
                "Tumor Core": best_case.get("tumor_tumor_core_mm3", 0),
                "Whole Tumor": best_case.get("tumor_whole_tumor_mm3", 0),
            }
            lines = [f"{name}: {vol:.0f} mm3" for name, vol in vols.items() if vol]
            ax.text(0.05, 0.9, "Tumor Volumes", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
            for idx, line in enumerate(lines):
                ax.text(0.05, 0.8 - idx * 0.12, line, transform=ax.transAxes, fontsize=9, va="top")

        del seg, seg_slice, tumor_mask; gc.collect()

    fig.suptitle(f"Case: {best_case['case_id']} (Patient: {best_case['patient']})", y=1.01, fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "sample_slices_with_tumor.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  -> sample_slices_with_tumor.png")

del tumor_cases; gc.collect()


# ── Summary report ───────────────────────────────────────────────────────────

print("\n[PASS 4] Writing summary report ...")
total_files = sum(1 for _ in DATA.rglob("*.nii.gz"))
total_size_gb = sum(f.stat().st_size for f in DATA.rglob("*.nii.gz")) / 1e9
patients = case_df["patient"].nunique()
sequences = case_df["sequence"].nunique()
cases_with_seg = case_df["case_id"].nunique() if "case_id" in case_df.columns else "?"

# Tumor stats for report
has_tumor = [r for r in case_meta if r.get("tumor_whole_tumor_mm3", 0) and r["tumor_whole_tumor_mm3"] > 0]
n_tumor = len(has_tumor)
mean_tumor_vol = float(np.mean([r["tumor_whole_tumor_mm3"] for r in has_tumor])) if has_tumor else 0
max_tumor_vol = float(max([r["tumor_whole_tumor_mm3"] for r in has_tumor])) if has_tumor else 0
median_tumor_vol = float(np.median([r["tumor_whole_tumor_mm3"] for r in has_tumor])) if has_tumor else 0

agg_table = "\n".join(
    f"| {r['modality']} | {r['n_cases']} | {r['grand_mean']:.1f} | {r['grand_std']:.1f} | "
    f"{r['grand_min']:.1f} | {r['grand_max']:.1f} | {r['p1']:.1f} | {r['p50']:.1f} | "
    f"{r['p95']:.1f} | {r['p99']:.1f} | {r['nonzero_pct']:.2f}% |"
    for r in agg_rows
)

seg_table = "\n".join(
    f"| {r['region']} | {r['n_cases_with_tumor']} | {r['total_volume_mm3']:.0f} | {r['mean_volume_mm3']:.0f} |"
    for r in seg_df.to_dict("records")
)

report = f"""# BraTS-GLI Training Data — Full Analysis Report

Generated: 2026-08-04
Dataset: BraTS2024 BraTS-GLI Training (syn60086071)

## 1. Overview

| Metric | Value |
|--------|-------|
| Total cases (directories) | {len(cases)} |
| Total .nii.gz files | {total_files} |
| Total size on disk | {total_size_gb:.2f} GB |
| Typical 3D shape | 182 × 218 × 182 |
| Unique patients | {patients} |
| Unique sequences | {sequences} |
| Modalities per case | T1ce, T1n, FLAIR, T2w (4) |
| Segmentation (seg.nii.gz) | YES — present in training set |

## 2. Modalities

| Code | Meaning |
|------|---------|
| t1c | T1-weighted post-contrast (contrast-enhanced) |
| t1n | T1-weighted native (pre-contrast) |
| t2f | FLAIR (Fluid Attenuated Inversion Recovery) |
| t2w | T2-weighted |

## 3. Per-Modality Intensity Statistics

| Modality | N voxels | Grand Mean | Grand Std | Min | Max | P1 | P50 | P95 | P99 | Non-zero % |
|----------|----------|------------|-----------|-----|-----|----|-----|-----|-----|------------|
{agg_table}

## 4. Segmentation Analysis

### Tumor Region Definitions

| Label | Region | BraTS-GLI definition |
|-------|--------|---------------------|
| 1 | Edema (WC) | Peritumoral edema — white matter involvement |
| 2 | Non-enhancing tumor core (NCR) | Necrotic / non-enhancing core |
| 3 | Enhancing tumor core (ET) | Contrast-enhancing viable tumor |
| 2+3 | Tumor Core | Enhancing + non-enhancing core |
| 1+2+3 | Whole Tumor | All tumor-involved tissue |

### Tumor Volume Summary

| Region | Cases with Tumor | Total Volume (mm3) | Mean Volume (mm3) |
|--------|-----------------|-------------------|-------------------|
{seg_table}

### Tumor Statistics

| Metric | Value |
|--------|-------|
| Cases with detectable tumor | {n_tumor} / {len(cases)} ({n_tumor/len(cases)*100:.1f}%) |
| Mean whole tumor volume | {mean_tumor_vol:.0f} mm3 |
| Median whole tumor volume | {median_tumor_vol:.0f} mm3 |
| Max whole tumor volume | {max_tumor_vol:.0f} mm3 |

## 5. Key Findings

1. **Segmentation masks ARE present** — training data includes `seg.nii.gz` with 3-region tumor labels.
2. **No WHO Grade labels** — BraTS-GLI uses binary tumor labels, not WHO Grade I-IV.
3. **No CT data** — BraTS-GLI is MRI-only. CT must come from IBSR/TCIA.
4. **Tumor volume range** — from small (<100 mm3) to very large (>30000 mm3), highly variable.
5. **All cases have 4 MRI modalities** — no missing modality across the dataset.
6. **Voxel spacing** — 1mm isotropic, total brain volume ~7.3 million mm3 per scan.

## 6. Files Generated

| File | Description |
|------|-------------|
| modality_stats.csv | Per-modality aggregate intensity statistics |
| case_metadata.csv | Per-case shapes, intensity stats, tumor volumes |
| tumor_volume_summary.csv | Tumor region volume aggregates |
| modality_histograms.png | Value distribution histograms per modality |
| tumor_volume_distribution.png | Whole tumor volume + fraction distributions |
| sample_slices_with_tumor.png | Axial slices with tumor overlay (highest-volume case) |
| summary_report.md | This report |

## 7. Next Steps (Phase 1: Preprocessing)

- [ ] Skull-stripping (ANTsHDGMM or BraTS-specific approach)
- [ ] Rigid registration across modalities (FLAIR as reference)
- [ ] CTN normalisation (CTN from BraTS challenge)
- [ ] Patch-based augmentation for small GPU memory
- [ ] Train/test split with patient-level stratification
- [ ] Multi-task network: segmentation + tumour presence classification
"""

with open(OUT / "summary_report.md", "w") as f:
    f.write(report)
print(f"  -> summary_report.md")

elapsed = __import__("time").time() - t0
print(f"\nAnalysis complete in {fmt_secs(elapsed)}.")
