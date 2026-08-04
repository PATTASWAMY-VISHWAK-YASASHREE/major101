"""Phase 1 — BraTS Preprocessing Pipeline.

Processes 1350 BraTS cases:
  - Load 4 modalities (t1, t1c, t2, t2w) + segmentation
  - CTN normalisation with brain mask from T1-native
  - Percentile clipping (99.5%)
  - Stack into (4, D, H, W) .pt files
  - Extract tumor subregion presence labels

Output: data/brats_preprocessed/
  - train/{case}.pt  — 4-channel normalised volume (numpy float32)
  - labels.csv       — case, et/tc/wt presence, volumes, grade proxy

Usage:
  python scripts/preprocess_brats.py [--workers 3] [--max-cases 100]

Expected: 1350 cases → ~1350 .pt files + labels.csv
"""

import gc, csv, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import nibabel as nib


# ── Constants ───────────────────────────────────────────────────────────────
NIFTI_DIR = Path("data/brats/training")
OUTPUT_DIR = Path("data/brats_preprocessed")
MODALITIES = ["t1", "t1c", "t2", "t2w"]  # BraTS filename convention
BRAIN_THRESHOLD = 50


# ── Normalisation ───────────────────────────────────────────────────────────
def ctn_normalize(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Contrast-Transformed Normalisation: percentile clip + z-score + max-bounds."""
    out = volume.astype(np.float32)
    p99 = np.percentile(out[mask > 0], 99.5)
    out = np.clip(out, None, p99)
    mean = out[mask > 0].mean()
    std = out[mask > 0].std() + 1e-8
    out = (out - mean) / std
    max_abs = np.max(np.abs(out)) + 1e-8
    out = out / max_abs
    return out


def brain_mask(t1n: np.ndarray) -> np.ndarray:
    """Extract brain mask from T1-native via thresholding + largest component."""
    mask = (t1n > BRAIN_THRESHOLD).astype(np.float32)
    try:
        from scipy import ndimage
        labelled, n_comp = ndimage.label(mask)
        sizes = ndimage.sum(mask, labelled, range(1, n_comp + 1))
        brain_label = sizes.argmax() + 1
        mask = (labelled == brain_label).astype(np.float32)
    except ImportError:
        pass
    return mask


# ── Single case processing ─────────────────────────────────────────────────
def process_case(case_name: str, output_dir: Path) -> dict:
    """Process one case: normalise 4 modalities, save .pt, extract labels."""
    case_dir = NIFTI_DIR / case_name
    vols = {}
    for mod in MODALITIES:
        f = case_dir / f"{case_name}_{mod}.nii.gz"
        if not f.exists():
            return {"status": "missing", "case": case_name, "mod": mod}
        try:
            vols[mod] = nib.load(f).get_fdata(dtype=np.float32)
        except Exception as e:
            return {"status": "error", "case": case_name, "error": str(e)}
    gc.collect()

    # Brain mask from T1-native
    mask = brain_mask(vols["t1n"])
    del vols["t1n"]
    gc.collect()

    # Normalise each modality
    normalised = []
    stats = {}
    for mod in MODALITIES:
        vol = ctn_normalize(vols[mod], mask)
        normalised.append(vol)
        stats[f"{mod}_min"] = float(vol.min())
        stats[f"{mod}_max"] = float(vol.max())
        del vols[mod]
    del mask
    gc.collect()

    # Stack into (4, D, H, W) — BraTS standard
    vol_4d = np.stack(normalised, axis=0).astype(np.float32)
    del normalised
    gc.collect()

    # Parse segmentation labels
    # BraTS 2024 labels (non-overlapping): 1=Edema, 2=NCR, 3=ET, 4=Whole Tumor (mask)
    seg_path = case_dir / f"{case_name}_seg.nii.gz"
    et_present = tc_present = wt_present = False
    volumes = {"et": 0.0, "tc": 0.0, "wt": 0.0}
    if seg_path.exists():
        seg = nib.load(seg_path).get_fdata(dtype=np.float32)
        # ET = enhancing tumor core (label 3) — hallmark of high-grade glioma
        et_present = bool(np.any(seg == 3))
        # TC = tumor core = NCR (2) + ET (3) — clinically defined core
        tc_present = bool(np.any(seg == 2) or np.any(seg == 3))
        # WT = whole tumor = Edema (1) + NCR (2) + ET (3)
        wt_present = bool(np.any(seg == 1) or np.any(seg == 2) or np.any(seg == 3))
        volumes["et"] = float(np.sum(seg == 3))
        volumes["tc"] = float(np.sum(seg == 2) + np.sum(seg == 3))
        volumes["wt"] = float(np.sum(seg == 1) + np.sum(seg == 2) + np.sum(seg == 3))
        del seg
        gc.collect()

    # Grade proxy: high-grade = contrast-enhancing (label 3)
    # Clinical basis: GBM (WHO IV) always enhances; low-grade (I-III) typically do not
    grade_proxy = int(et_present)

    # Save .npy file (numpy format, loadable by torch/np.load)
    npy_path = output_dir / "train" / f"{case_name}.npy"
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_path), vol_4d)
    del vol_4d
    gc.collect()

    return {
        "status": "ok",
        "case": case_name,
        "et": int(et_present),
        "tc": int(tc_present),
        "wt": int(wt_present),
        "volumes": volumes,
        "grade": grade_proxy,
        "stats": stats,
    }


# ── Main ────────────────────────────────────────────────────────────────────
def run_preprocessing(n_workers: int = 3, max_cases: int = None):
    print("=" * 70)
    print("BRAINS TUMOUR PREPROCESSING PIPELINE")
    print("=" * 70)

    all_cases = sorted([d.name for d in NIFTI_DIR.iterdir() if d.is_dir()])
    if max_cases:
        all_cases = all_cases[:max_cases]
    print(f"  Cases to process: {len(all_cases)}")
    print(f"  Workers:          {n_workers}")

    (OUTPUT_DIR / "train").mkdir(parents=True, exist_ok=True)

    results, errors = [], []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_case, c, OUTPUT_DIR): c
            for c in all_cases
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                r = future.result()
                if r["status"] == "ok":
                    results.append(r)
                else:
                    errors.append(r)
            except Exception as e:
                errors.append({"status": "exception", "case": futures[future], "error": str(e)})
            if done % 100 == 0:
                print(f"  [{done}/{len(all_cases)}]  OK={len(results)}  ERR={len(errors)}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"PREPROCESSING COMPLETE")
    print(f"  Processed: {len(results)}/{len(all_cases)}")
    print(f"  Errors:    {len(errors)}")
    if errors:
        for e in errors[:10]:
            print(f"    {e['case']}: {e.get('error', e.get('mod', 'unknown'))}")

    # Write labels.csv
    labels_path = OUTPUT_DIR / "labels.csv"
    with open(labels_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "et", "tc", "wt", "wt_volume", "tc_volume",
                     "et_volume", "grade_proxy"])
        for r in results:
            w.writerow([r["case"], r["et"], r["tc"], r["wt"],
                         r["volumes"]["wt"], r["volumes"]["tc"],
                         r["volumes"]["et"], r["grade"]])
    print(f"  Labels: {labels_path}")

    # Class balance
    high = sum(1 for r in results if r["grade"] == 1)
    low = len(results) - high
    print(f"\n  Grade proxy distribution:")
    print(f"    High-grade (ET/TC present): {high}")
    print(f"    Low-grade  (neither):       {low}")
    print(f"\n  Subregion presence:")
    for sub in ["et", "tc", "wt"]:
        cnt = sum(1 for r in results if r[sub] == 1)
        pct = cnt / len(results) * 100 if results else 0
        print(f"    {sub.upper()}: {cnt}/{len(results)} ({pct:.1f}%)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--max-cases", type=int, default=None)
    args = p.parse_args()
    run_preprocessing(args.workers, args.max_cases)