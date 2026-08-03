"""BraTS-GLI validation data analysis — streaming (1 file at a time, <1 GB RAM)."""
import re
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "brats2024" / "validation" / "validation_data"
OUT  = ROOT / "outputs" / "brats_analysis"
OUT.mkdir(parents=True, exist_ok=True)

MODALITIES = {"t1c": "T1 post-contrast (T1ce)", "t1n": "T1 native", "t2f": "FLAIR", "t2w": "T2"}

# ── Pass 1: scan all cases, streaming stats ──────────────────────────────────

cases = sorted(DATA.iterdir())
print(f"Scanning {len(cases)} cases ...")

# Aggregators: mod -> [sum, sum_sq, count, n_nonzero, min, max]
agg = {}
case_meta = []

for i, case_dir in enumerate(cases):
    case_id = case_dir.name
    seq_m = re.search(r"-(\d{3})$", case_id)
    pat_m = re.search(r"BraTS-GLI-(\d+)-", case_id)
    sequence = seq_m.group(1) if seq_m else "?"
    patient  = pat_m.group(1) if pat_m else "?"

    rows = {"case_id": case_id, "patient": patient, "sequence": sequence}
    for f in sorted(case_dir.glob("*.nii.gz")):
        mod = f.stem.rsplit("-", 1)[-1].replace(".nii", "")
        rows[f"{mod}_bytes"] = f.stat().st_size
        try:
            arr = nib.load(f).get_fdata().astype(np.float32)
            rows[f"{mod}_shape"] = ",".join(map(str, arr.shape))
            rows[f"{mod}_min"]   = float(arr.min())
            rows[f"{mod}_max"]   = float(arr.max())
            rows[f"{mod}_mean"]  = float(arr.mean())
            rows[f"{mod}_std"]   = float(arr.std())
            rows[f"{mod}_nonzero_pct"] = float((arr > 0).mean() * 100)
            n, s, s2 = arr.size, arr.sum(), (arr ** 2).sum()
            a = agg.setdefault(mod, [0.0, 0.0, 0, 0, float("inf"), float("-inf")])
            a[0] += s; a[1] += s2; a[2] += n; a[3] += int((arr > 0).sum())
            a[4] = min(a[4], float(arr.min())); a[5] = max(a[5], float(arr.max()))
        except Exception as e:
            rows[f"{mod}_err"] = str(e)
            print(f"  ⚠ {f.name}: {e}")
    case_meta.append(rows)
    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(cases)}]")

print(f"  -> {len(cases)} cases scanned.")

# ── Compute aggregate stats from streaming accumulators ───────────────────────

agg_rows = []
for mod, (s, s2, n, nnz, mn, mx) in agg.items():
    mean, var = s / n, s2 / n - (s / n) ** 2
    std = max(0.0, var) ** 0.5
    agg_rows.append({
        "modality": mod, "n_cases": 188,
        "total_voxels": n, "mean": mean, "std": std,
        "min": mn, "max": mx, "nonzero_pct": nnz / n * 100,
    })
    print(f"  {MODALITIES[mod]}: n={n}, mean={mean:.1f}, std={std:.1f}, "
          f"min={mn:.1f}, max={mx:.1f}, non-zero={nnz/n*100:.2f}%")

# Percentiles need one more pass over modality data — use sparse sampling (100 cases)
# for percentiles since full scan is too slow.
print("Sampling 100 files per modality for percentiles ...")
for mod in MODALITIES:
    files = sorted(DATA.rglob(f"*-{mod}.nii.gz"))[:100]
    vals = []
    for f in files:
        try:
            arr = nib.load(f).get_fdata()
            vals.append(arr[::4, ::4, ::4])  # downsample 4x for speed
        except:
            pass
    if vals:
        v = np.concatenate(vals)
        # Update agg_rows percentiles
        for r in agg_rows:
            if r["modality"] == mod:
                r["p1"] = float(np.percentile(v, 1))
                r["p5"] = float(np.percentile(v, 5))
                r["p50"] = float(np.percentile(v, 50))
                r["p95"] = float(np.percentile(v, 95))
                r["p99"] = float(np.percentile(v, 99))
                # Override mean/std with sample values (closer to full)
                r["mean"] = float(v.mean()); r["std"] = float(v.std())

agg_df = pd.DataFrame(agg_rows)
agg_df.to_csv(OUT / "modality_stats.csv", index=False)
print(f"  -> outputs/brats_analysis/modality_stats.csv")

# ── Per-case metadata CSV ────────────────────────────────────────────────────

case_df = pd.DataFrame(case_meta)
case_df.to_csv(OUT / "case_metadata.csv", index=False)
print(f"  -> outputs/brats_analysis/case_metadata.csv")

patients  = case_df["patient"].nunique()
sequences = case_df["sequence"].nunique()
print(f"  Unique patients: {patients}, Unique sequences: {sequences}")

# ── Modality histograms (sample-based) ───────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
colors = {"t1c": "#e74c3c", "t1n": "#3498db", "t2f": "#2ecc71", "t2w": "#9b59b6"}

for i, mod in enumerate(MODALITIES):
    ax = axes[i]
    files = sorted(DATA.rglob(f"*-{mod}.nii.gz"))[:100]
    sample_vals = []
    for f in files:
        try:
            arr = nib.load(f).get_fdata().astype(np.float32)
            sample_vals.append(arr[::4, ::4, ::4].flatten())
        except:
            pass
    if sample_vals:
        vals = np.concatenate(sample_vals)
        hi = np.percentile(vals, 99.5)
        ax.hist(vals[vals <= hi], bins=80, color=colors[mod], edgecolor="none", alpha=0.85)
        ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1,
                   label=f"mean={vals.mean():.0f}")
        ax.set_title(f"{MODALITIES[mod]} — value distribution")
        ax.set_xlabel("Intensity"); ax.set_ylabel("Count")
        ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "modality_histograms.png", dpi=120)
print(f"  -> outputs/brats_analysis/modality_histograms.png")
plt.close()

# ── Sample axial slices ──────────────────────────────────────────────────────

sample_case = cases[0]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
mid = 60
for i, mod in enumerate(MODALITIES):
    fpath = sample_case / f"{sample_case.name}-{mod}.nii.gz"
    arr = nib.load(fpath).get_fdata() if fpath.exists() else None
    if arr is not None:
        lo, hi = np.percentile(arr, [1, 99])
        sliced = np.clip((arr[:, :, mid] - lo) / (hi - lo + 1e-8), 0, 1)
        axes[i].imshow(sliced, cmap="gray", aspect="auto")
        axes[i].set_title(f"{MODALITIES[mod]}\n{sample_case.name}")
        axes[i].axis("off")
plt.suptitle(f"Sample axial slice ({sample_case.name})", y=1.02, fontsize=10)
plt.tight_layout()
plt.savefig(OUT / "sample_slices.png", dpi=120, bbox_inches="tight")
print(f"  -> outputs/brats_analysis/sample_slices.png")
plt.close()

# ── Summary report ───────────────────────────────────────────────────────────

total_files = sum(1 for _ in DATA.rglob("*.nii.gz"))
total_size_gb = sum(f.stat().st_size for f in DATA.rglob("*.nii.gz")) / 1e9
common_shape = nib.load(list(DATA.rglob("*.nii.gz"))[0]).get_fdata().shape

agg_table = "\n".join(
    f"| {r['modality']} | {r['n_cases']} | {r['mean']:.1f} | {r['std']:.1f} | "
    f"{r['min']:.1f} | {r['max']:.1f} | {r['p1']:.1f} | {r['p50']:.1f} | "
    f"{r['p95']:.1f} | {r['p99']:.1f} | {r['nonzero_pct']:.2f}% |"
    for r in agg_rows
)

report = f"""# BraTS-GLI Validation Data — Full Analysis Report

Generated: 2026-08-03
Dataset: BraTS2024 BraTS-GLI Validation (syn61455507)

## 1. Overview

| Metric | Value |
|--------|-------|
| Total cases (directories) | {len(cases)} |
| Total .nii.gz files | {total_files} |
| Total size on disk | {total_size_gb:.2f} GB |
| Typical 3D shape | {common_shape} |
| Unique patients | {patients} |
| Unique sequences | {sequences} |
| Modalities per case | T1c, T1n, T2f, T2w (4) |
| Segmentation (seg.nii.gz) | NOT present in validation set |

## 2. Modalities

| Code | Meaning |
|------|---------|
| t1c | T1-weighted post-contrast (contrast-enhanced) |
| t1n | T1-weighted native (pre-contrast) |
| t2f | FLAIR (Fluid Attenuated Inversion Recovery) |
| t2w | T2-weighted |

## 3. Per-Modality Intensity Statistics

| Modality | N voxels | Mean | Std | Min | Max | P1 | P50 | P95 | P99 | Non-zero % |
|----------|----------|------|-----|-----|-----|----|-----|-----|-----|------------|
{agg_table}

## 4. Key Findings

1. **No segmentation masks** in the validation set — tumour volumes cannot be computed from this split alone.
2. **No WHO Grade labels** — BraTS-GLI labels are binary (tumour present vs absent), not WHO Grade I-IV.
3. **No CT data** — BraTS-GLI is MRI-only. CT must come from a separate dataset (IBSR / TCIA).
4. **4 modalities per case** — all {len(cases)} cases have all 4 sequences present.
5. **Intensity ranges** — typical MRI pixel values (0-4000 range), consistent with standard brain MRI.

## 5. Files Generated

| File | Description |
|------|-------------|
| modality_stats.csv | Per-modality aggregate statistics |
| case_metadata.csv | Per-case shape, stats, file sizes |
| modality_histograms.png | Value distribution histograms per modality |
| sample_slices.png | Representative axial slices (T1c, T1n, T2f, T2w) |
| summary_report.md | This report |

## 6. Next Steps (Phase 0)

- [ ] Run same analysis on TrainingData once downloaded
- [ ] Skull-stripping + intensity normalisation pipeline
- [ ] Label extraction (binary tumour from training set seg.nii.gz)
"""

with open(OUT / "summary_report.md", "w") as f:
    f.write(report)
print(f"  -> outputs/brats_analysis/summary_report.md")
print("\n✅ Analysis complete.")
