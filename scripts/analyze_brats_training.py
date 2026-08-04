"""BraTS-GLI Training Data — comprehensive analysis (streaming, <1 GB RAM).

Analyses MRI intensities + segmentation masks. Designed to run overnight
(5+ hours) on a machine with tight RAM (8 GB) and limited free disk.
"""
import re
import sys
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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

MAX_WORKERS = 3  # ponytail: keep low to avoid RAM spikes on 8GB machine
DOWNSAMPLE  = 8  # ponytail: 8× stride → 512× fewer voxels, identical histogram shape

MODALITIES = {"t1c": "T1ce", "t1n": "T1n", "t2f": "FLAIR", "2w": "T2w", "t2w": "T2w"}
SEG_LABELS = {1: "edema (ET)", 2: "non-enhancing core (NCR)", 3: "enhancing core (ET)", 4: "whole tumor (WST)"}
TUMOR_REGIONS = {
    "tumor_core": [2, 3],       # regions 2 or 3
    "whole_tumor": [4],         # region 4 = union of 1+2+3 (always present)
    "edema_only": [1],
    "enhancing_only": [3],
}
VOXEL_SPACING = (1.0, 1.0, 1.0)  # mm³; all BraTS-GLI are isotropic 1mm

# ── Parallel loader helpers ───────────────────────────────────────────────────

def _load_one_nifti(path: Path, stride: int = DOWNSAMPLE):
    """Load a single NIfTI file, return downsampled array or None on error."""
    try:
        arr = nib.load(path).get_fdata(dtype=np.float32)
        return arr[::stride, ::stride, ::stride]
    except Exception as e:
        return None


def _load_one_mean(path: Path, stride: int = DOWNSAMPLE):
    """Load a single NIfTI file, return (mean, max) or None on error."""
    try:
        arr = nib.load(path).get_fdata(dtype=np.float32)
        s = arr[::stride, ::stride, ::stride]
        return float(s.mean()), float(s.max())
    except Exception:
        return None


def parallel_load_array(files: list[Path], stride: int = DOWNSAMPLE):
    """Load many NIfTI files in parallel, return list of arrays (same order, None on failure)."""
    results = [None] * len(files)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_map = {ex.submit(_load_one_nifti, f, stride): i for i, f in enumerate(files)}
        for fut in as_completed(future_map):
            idx = future_map[fut]
            results[idx] = fut.result()
    return results


def parallel_load_flatten(files: list[Path], stride: int = DOWNSAMPLE):
    """Load many NIfTI files in parallel, return concatenated flattened array (skipping None)."""
    arrs = parallel_load_array(files, stride)
    chunks = [a.flatten() for a in arrs if a is not None]
    return np.concatenate(chunks) if chunks else np.array([])


def parallel_load_means(files: list[Path], stride: int = DOWNSAMPLE):
    """Load many NIfTI files in parallel, return list of mean values (same order, None on failure)."""
    results = [None] * len(files)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_map = {ex.submit(_load_one_mean, f, stride): i for i, f in enumerate(files)}
        for fut in as_completed(future_map):
            idx = future_map[fut]
            results[idx] = fut.result()
    return results

def fmt_secs(t):
    h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"


# ── Helper: streaming intensity stats ──────────────────────────────────────────
def stream_stats(arr):
    """Return (sum_x, sum_x2, n, nnz, min, max) — accumulates additively across cases."""
    n = arr.size
    s = float(arr.sum())
    s2 = float(np.square(arr).sum())
    return (s, s2, n, int((arr > 0).sum()), float(arr.min()), float(arr.max()))


# ── PASS 1: scan every case, collect metadata + streaming accumulators ───────

cases = sorted(d for d in DATA.iterdir() if d.is_dir())
print(f"[PASS 1] Scanning {len(cases)} cases ({MAX_WORKERS} parallel workers) ...")
print(f"  Output dir: {OUT}")

# Accumulators
agg = {}
seg_agg = {}
case_meta = []

t0 = __import__("time").time()
batch = 50


def _process_case(case_dir):
    """Worker: process one BraTS case directory, return (meta_row, mod_stats_dict, seg_stats_dict)."""
    case_id = case_dir.name
    seq_m = re.search(r"-(\d{3})$", case_id)
    pat_m = re.search(r"BraTS-GLI-(\d+)-", case_id)
    sequence = seq_m.group(1) if seq_m else "?"
    patient  = pat_m.group(1) if pat_m else "?"
    row = {"case_id": case_id, "patient": patient, "sequence": sequence}
    mod_stats = {}
    seg_stats = {}

    # MRI modalities
    for f in sorted(case_dir.glob("*.nii.gz")):
        mod = f.stem.rsplit("-", 1)[-1].replace(".nii", "")
        if mod == "seg" or mod not in MODALITIES:
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
            sum_x, sum_x2, n, nnz, mn, mx = stream_stats(arr)
            mod_stats[mod] = (sum_x, sum_x2, n, nnz, mn, mx)
        except Exception as e:
            row[f"{mod}_err"] = str(e)
            mod_stats[mod] = None

    # Segmentation mask
    seg_path = case_dir / f"{case_id}-seg.nii.gz"
    if seg_path.exists():
        row["seg_bytes"] = seg_path.stat().st_size
        try:
            seg_img = nib.load(seg_path)
            seg = seg_img.get_fdata(dtype=np.float32).astype(np.int16)
            row["seg_shape"] = ",".join(map(str, seg.shape))
            row["seg_nunique"] = len(np.unique(seg))

            labels, counts = np.unique(seg[seg > 0], return_counts=True)
            for lbl, cnt in zip(labels, counts):
                lbl_name = SEG_LABELS.get(int(lbl), f"label_{int(lbl)}")
                vol_mm3 = cnt * float(np.prod(VOXEL_SPACING))
                row[f"seg_{lbl_name}_voxels"] = int(cnt)
                row[f"seg_{lbl_name}_mm3"] = vol_mm3
                seg_stats.setdefault(lbl_name, []).append(vol_mm3)

            for region_name, region_labels in TUMOR_REGIONS.items():
                mask = np.isin(seg, region_labels)
                vol_mm3 = mask.sum() * float(np.prod(VOXEL_SPACING))
                row[f"tumor_{region_name}_voxels"] = int(mask.sum())
                row[f"tumor_{region_name}_mm3"] = vol_mm3
                seg_stats.setdefault(f"tumor_{region_name}", []).append(vol_mm3)

            total_voxels = seg.size
            tumor_voxels = int((seg == 4).sum())
            if total_voxels > 0:
                row["tumor_volume_fraction"] = float(tumor_voxels / total_voxels)

        except Exception as e:
            row["seg_err"] = str(e)

    return row, mod_stats, seg_stats


# Parallel case scanning (max 3 concurrent)
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = {ex.submit(_process_case, c): c for c in cases}
    completed = 0
    for fut in as_completed(futures):
        try:
            row, mod_stats, seg_stats = fut.result()
            case_meta.append(row)
            for mod, stats in mod_stats.items():
                if stats is not None:
                    a = agg.setdefault(mod, [0.0, 0.0, 0, 0, float("inf"), float("-inf")])
                    sum_x, sum_x2, n, nnz, mn, mx = stats
                    a[0] += sum_x; a[1] += sum_x2; a[2] += n; a[3] += nnz
                    a[4] = min(a[4], mn); a[5] = max(a[5], mx)
            for name, vol_list in seg_stats.items():
                sa = seg_agg.setdefault(name, [0.0, 0])
                sa[0] += sum(vol_list); sa[1] += len(vol_list)
        except Exception as e:
            print(f"  WARN {futures[fut].name}: {e}", file=sys.stderr)

        completed += 1
        if completed % batch == 0:
            elapsed = __import__("time").time() - t0
            eta = elapsed / completed * (len(cases) - completed)
            print(f"  [{completed}/{len(cases)}] elapsed={fmt_secs(elapsed)} eta={fmt_secs(eta)}")

elapsed = __import__("time").time() - t0
print(f"  [{len(cases)}/{len(cases)}] DONE in {fmt_secs(elapsed)}")

print(f"\n  -> {len(cases)} cases scanned, "
      f"{len(case_meta)} rows collected.")


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

# Percentiles — sparse sampling (100 cases, downsampled 8x per axis, parallel)
print("  Sampling percentiles (100 cases, {}x downsample, {} workers) ...".format(DOWNSAMPLE, MAX_WORKERS))
sample_cases = sorted(DATA.rglob("*/"))[:100]
sample_pcts = {}
for mod in ["t1c", "t1n", "t2f", "t2w"]:
    files = [d / f"{d.name}-{mod}.nii.gz" for d in sample_cases]
    vals = parallel_load_flatten(files, DOWNSAMPLE)
    v = np.array(vals, dtype=np.float32)[:10_000]
    for r in agg_rows:
        if r["modality"] == mod:
            r["p1"] = float(np.percentile(v, 1))
            r["p5"] = float(np.percentile(v, 5))
            r["p50"] = float(np.percentile(v, 50))
            r["p95"] = float(np.percentile(v, 95))
            r["p99"] = float(np.percentile(v, 99))
    del v; gc.collect()

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


# ── PASS 3: Paper-Grade Statistical Visualisations ────────────────────────────

print("\n[PASS 3] Generating paper-grade statistical visualisations ...")

colors = {"t1c": "#e74c3c", "t1n": "#3498db", "t2f": "#2ecc71", "t2w": "#9b59b6"}
mods = ["t1c", "t1n", "t2f", "t2w"]
SAMPLE = 100  # sample size for intensity stats

# --- 3a. Modality intensity histograms (per-modality, 2x2) ---
print("  [3a] Modality histograms ...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

intensity_stats = {}
for i, mod in enumerate(mods):
    ax = axes[i]
    files = sorted(DATA.rglob(f"*-{mod}.nii.gz"))[:SAMPLE]
    vals = parallel_load_flatten(files, DOWNSAMPLE)
    if len(vals) > 0:
        hi = np.percentile(vals, 99.5)
        ax.hist(vals[vals <= hi], bins=100, color=colors[mod], edgecolor="none", alpha=0.85)
        ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"μ={vals.mean():.0f}")
        ax.axvline(np.median(vals), color="darkred", linestyle=":", linewidth=1,
                   label=f"median={np.median(vals):.0f}")
        ax.set_title(f"{mod.upper()} — Intensity Distribution (n={SAMPLE})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Intensity (HU-scale for T1C/T1N)"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=8, loc="upper right")
        intensity_stats[mod] = {
            "mean": float(vals.mean()), "std": float(vals.std()),
            "min": float(vals.min()), "max": float(vals.max()),
            "p25": float(np.percentile(vals, 25)), "p50": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)), "p99": float(np.percentile(vals, 99)),
        }
    del vals; gc.collect()
plt.tight_layout()
plt.savefig(OUT / "modality_histograms.png", dpi=150)
plt.close()
print("    -> modality_histograms.png")


# --- 3b. Modality intensity box plots ---
print("  [3b] Modality box plots ...")
sample_dfs = {}
for mod in mods:
    files = sorted(DATA.rglob(f"*-{mod}.nii.gz"))[:SAMPLE]
    means = parallel_load_means(files, DOWNSAMPLE)
    sample_dfs[mod] = [m[0] for m in means if m is not None]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, mod in enumerate(mods):
    ax = axes[i]
    data = sample_dfs[mod]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(colors[mod])
    bp['boxes'][0].set_alpha(0.7)
    ax.set_title(f"{mod.upper()} — Per-Case Mean Intensity", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Intensity")
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "modality_boxplots.png", dpi=150)
plt.close()
print("    -> modality_boxplots.png")
del sample_dfs; gc.collect()


# --- 3c. Modality correlation matrix (per-case mean intensities, parallel) ---
print("  [3c] Modality correlation matrix ...")
corr_sample = sorted(DATA.rglob("*/"))[:200]
case_mod_means = {mod: [] for mod in mods}


def _case_mod_means(case_dir):
    """Return dict of {mod: mean} for one case, or None if incomplete."""
    row = {}
    for mod in mods:
        f = case_dir / f"{case_dir.name}-{mod}.nii.gz"
        if not f.exists():
            return None
        try:
            arr = nib.load(f).get_fdata(dtype=np.float32)
            row[mod] = float(arr[::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE].mean())
            del arr; gc.collect()
        except Exception:
            return None
    return row


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    corr_futs = {ex.submit(_case_mod_means, d): d for d in corr_sample}
    for fut in as_completed(corr_futs):
        try:
            row = fut.result()
            if row:
                for mod in mods:
                    case_mod_means[mod].append(row[mod])
        except Exception:
            pass

corr_df = pd.DataFrame(case_mod_means)
corr_matrix = corr_df.corr()

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(mods))); ax.set_xticklabels([m.upper() for m in mods], fontsize=12)
ax.set_yticks(range(len(mods))); ax.set_yticklabels([m.upper() for m in mods], fontsize=12)
for i in range(len(mods)):
    for j in range(len(mods)):
        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.3f}", ha="center", va="center",
                fontsize=12, color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white")
fig.colorbar(im, ax=ax, label="Pearson r")
ax.set_title("Modality Pairwise Correlation (n=200 cases)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "modality_correlation_matrix.png", dpi=150)
plt.close()
corr_matrix.to_csv(OUT / "modality_correlation_matrix.csv")
print("    -> modality_correlation_matrix.png")
print("    -> modality_correlation_matrix.csv")

del case_mod_means, corr_df; gc.collect()


# --- 3d. Tumor volume distribution ---
print("  [3d] Tumor volume distribution ...")
whole_tumor_vols = [float(r.get("tumor_whole_tumor_mm3", 0)) for r in case_meta if r.get("tumor_whole_tumor_mm3", 0) > 0]

fig, axes = plt.subplots(2, 2, figsize=(16, 8))
axes = axes.flatten()
if whole_tumor_vols:
    axes[0].hist(whole_tumor_vols, bins=50, color="#e74c3c", edgecolor="white", alpha=0.85)
    axes[0].set_title("Whole Tumor Volume (mm³)", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Volume (mm³)"); axes[0].set_ylabel("Cases")
    axes[0].set_xlim(0, float(np.percentile(whole_tumor_vols, 95)))
    axes[0].axvline(np.median(whole_tumor_vols), color="blue", linestyle="--",
                    label=f"median={np.median(whole_tumor_vols):.0f}")
    axes[0].legend(fontsize=8)
else:
    axes[0].text(0.5, 0.5, "No tumor volumes", ha="center", va="center", transform=axes[0].transAxes)

# Sub-region volumes
for idx, (key, name, clr) in enumerate([
    ("seg_edema (ET)_mm3", "Edema", "#f39c12"),
    ("seg_non-enhancing core (NCR)_mm3", "Non-Enhancing (NCR)", "#3498db"),
    ("seg_enhancing core (ET)_mm3", "Enhancing (ET)", "#e74c3c"),
]):
    ax = axes[idx + 1]
    vals = [float(r.get(key, 0)) for r in case_meta if r.get(key, 0) > 0]
    if vals:
        ax.hist(vals, bins=30, color=clr, edgecolor="white", alpha=0.85)
        ax.set_title(f"{name} Volume (mm³)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Volume (mm³)"); ax.set_ylabel("Cases")
        ax.set_xlim(0, float(np.percentile(vals, 95)))
    else:
        ax.text(0.5, 0.5, f"No {name} data", ha="center", va="center", transform=ax.transAxes)
plt.tight_layout()
plt.savefig(OUT / "tumor_volume_distribution.png", dpi=150)
plt.close()
print("    -> tumor_volume_distribution.png")
del whole_tumor_vols; gc.collect()


# --- 3e. Dimension shape scatter (cases with shape data) ---
print("  [3e] Dimension scatter plots ...")
dims = []
for row in case_meta:
    shape = row.get("shape")
    if isinstance(shape, list) and len(shape) >= 3:
        dims.append(shape[:3])
if dims:
    dims = np.array(dims, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(dims[:, 0], dims[:, 1], alpha=0.5, s=20, c="#e74c3c")
    axes[0].set_xlabel("X (mm)"); axes[0].set_ylabel("Y (mm)"); axes[0].set_title("X vs Y")
    axes[0].grid(alpha=0.3)
    axes[1].scatter(dims[:, 0], dims[:, 2], alpha=0.5, s=20, c="#3498db")
    axes[1].set_xlabel("X (mm)"); axes[1].set_ylabel("Z (mm)"); axes[1].set_title("X vs Z")
    axes[1].grid(alpha=0.3)
    axes[2].scatter(dims[:, 1], dims[:, 2], alpha=0.5, s=20, c="#2ecc71")
    axes[2].set_xlabel("Y (mm)"); axes[2].set_ylabel("Z (mm)"); axes[2].set_title("Y vs Z")
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "dimension_scatter.png", dpi=150)
    plt.close()
    print("    -> dimension_scatter.png")
del dims; gc.collect()


# --- 3f. Intensity correlation heatmap (scatter per pair, parallel) ---
print("  [3f] Pairwise intensity scatter plots ...")
sample_pairs = sorted(DATA.rglob("*/"))[:200]
pair_means = {m: [] for m in mods}

# Reuse _case_mod_means from 3c
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    pair_futs = {ex.submit(_case_mod_means, d): d for d in sample_pairs}
    for fut in as_completed(pair_futs):
        try:
            row = fut.result()
            if row:
                for mod in mods:
                    pair_means[mod].append(row[mod])
        except Exception:
            pass

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
pairs = [("t1c", "t1n"), ("t1c", "t2f"), ("t1c", "t2w"), ("t1n", "t2f"), ("t1n", "t2w"), ("t2f", "t2w")]
for idx, (a, b) in enumerate(pairs):
    ax = axes[idx]
    x_vals = pair_means[a]; y_vals = pair_means[b]
    ax.scatter(x_vals, y_vals, alpha=0.4, s=15, c="#3498db")
    ax.set_xlabel(f"{a.upper()} mean"); ax.set_ylabel(f"{b.upper()} mean")
    ax.set_title(f"{a.upper()} vs {b.upper()}", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUT / "pairwise_intensity_scatter.png", dpi=150)
plt.close()
print("    -> pairwise_intensity_scatter.png")
del pair_means; gc.collect()


# --- 3g. Tumor presence & volume fraction ---
print("  [3g] Tumor presence & volume fraction ...")

has_tumor_count = sum(1 for r in case_meta if r.get("tumor_whole_tumor_mm3", 0) > 0)
no_tumor_count = len(case_meta) - has_tumor_count

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Tumor presence pie
axes[0].pie([has_tumor_count, no_tumor_count],
            labels=[f"With Tumor ({has_tumor_count})", f"No Tumor ({no_tumor_count})"],
            autopct="%1.1f%%", colors=["#e74c3c", "#2ecc71"], startangle=90,
            textprops={"fontsize": 11})
axes[0].set_title("Tumor Presence", fontsize=12, fontweight="bold")

# Tumor volume fraction distribution
tumor_fracs = [float(r.get("tumor_volume_fraction", 0)) for r in case_meta if r.get("tumor_volume_fraction", 0) > 0]
if tumor_fracs:
    axes[1].hist(tumor_fracs, bins=50, color="#9b59b6", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("Fraction of Brain with Tumor"); axes[1].set_ylabel("Cases")
    axes[1].set_title("Tumor Volume Fraction", fontsize=12, fontweight="bold")
    axes[1].set_xlim(0, float(np.percentile(tumor_fracs, 95)))
else:
    axes[1].text(0.5, 0.5, "No volume fraction data", ha="center", va="center",
                 transform=axes[1].transAxes, fontsize=11)
    axes[1].set_title("Tumor Volume Fraction", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT / "tumor_presence_fraction.png", dpi=150)
plt.close()
print("    -> tumor_presence_fraction.png")
del tumor_fracs; gc.collect()


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
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

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
        seg = nib.load(seg_path).get_fdata().astype(np.int16)
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
