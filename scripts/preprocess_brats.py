"""Phase 1 — BraTS Preprocessing Pipeline.

Processes BraTS 2024 GLI cases:
  - Load 4 modalities (t1c, t1n, t2f, t2w) + segmentation
  - CTN normalisation with brain mask from T1-native
  - Percentile clipping (99.5th percentile)
  - Stack into (4, D, H, W) .npy files
  - Extract tumor subregion presence labels

Output: data/brats_preprocessed/
  - train/{case}.npy      — 4-channel normalised volume (numpy float32)
  - labels.csv            — case, et/tc/wt presence, volumes, grade proxy
  - preprocessing_log.txt — progress, errors, summary log
  - preprocessing_errors.csv — per-case errors (if any)

Usage:
  python scripts/preprocess_brats.py [--workers 3] [--max-cases 100]

Expected: ~1350 cases → ~1350 .npy files + labels.csv
"""

import gc, csv, sys, logging, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import nibabel as nib


# ── Constants ───────────────────────────────────────────────────────────────
# BraTS 2024 GLI: case dirs like "BraTS-GLI-00005-100"
# Each contains: <name>-t1c.nii.gz, <name>-t1n.nii.gz, <name>-t2f.nii.gz, <name>-t2w.nii.gz, <name>-seg.nii.gz
ROOT = Path(__file__).resolve().parents[1]
NIFTI_DIR = ROOT / "data" / "raw" / "brats2024" / "training" / "BraTS2024-BraTS-GLI-TrainingData" / "training_data1_v2"
OUTPUT_DIR = ROOT / "data" / "brats_preprocessed"
MODALITIES = ["t1c", "t1n", "t2f", "t2w"]  # BraTS 2024 GLI naming: T1ce, T1 native, FLAIR, T2
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
    """Process one case: normalise 4 modalities, save .npy, extract labels."""
    case_dir = NIFTI_DIR / case_name
    vols = {}
    for mod in MODALITIES:
        f = case_dir / f"{case_name}-{mod}.nii.gz"
        if not f.exists():
            candidates = list(NIFTI_DIR.rglob(f"{case_name}-{mod}.nii.gz"))
            if candidates:
                f = candidates[0]
        if not f.exists():
            return {"status": "missing", "case": case_name, "mod": mod}
        try:
            vols[mod] = nib.load(f).get_fdata(dtype=np.float32)
        except Exception as e:
            return {"status": "error", "case": case_name, "mod": mod, "error": str(e)}
    gc.collect()

    # Brain mask from T1-native
    mask = brain_mask(vols["t1n"])

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
    seg_path = case_dir / f"{case_name}-seg.nii.gz"
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
def run_preprocessing(n_workers: int = 3, max_cases: int = None, resume: bool = False):
    # File logger — writes progress, errors, and summary to log file + console
    log_path = OUTPUT_DIR / "preprocessing_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    # Append mode if resuming so prior log is preserved
    log_mode = "a" if resume else "w"
    fh = logging.FileHandler(log_path, mode=log_mode, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    def log(msg):
        logger.info(msg)

    log("=" * 70)
    log("BRAINS TUMOUR PREPROCESSING PIPELINE")
    log(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if resume:
        log("  MODE: RESUME — skipping cases with existing .npy output")
    log("=" * 70)

    all_cases = sorted([d.name for d in NIFTI_DIR.iterdir() if d.is_dir()])
    if max_cases:
        all_cases = all_cases[:max_cases]

    # ── Resume: skip cases with valid .npy ──────────────────────────────
    if resume:
        valid = 0
        corrupted = []
        for c in all_cases:
            npy = (OUTPUT_DIR / "train" / f"{c}.npy")
            if npy.exists():
                try:
                    arr = np.load(str(npy), mmap_mode="r")
                    if arr.shape == (4, 182, 218, 182) and arr.dtype == np.float32:
                        valid += 1
                    else:
                        corrupted.append((c, "unexpected shape/dtype"))
                except Exception as e:
                    corrupted.append((c, str(e)))
        all_cases = [c for c in all_cases if (OUTPUT_DIR / "train" / f"{c}.npy").exists() is False]
        if valid:
            log(f"  Resuming: {valid} already processed, {len(all_cases)} remain")
        if corrupted:
            log(f"  Corrupted files: {len(corrupted)}")
            for c, reason in corrupted:
                log(f"    SKIP {c}: {reason}")
    log(f"  Cases to process: {len(all_cases)}")
    log(f"  Workers:          {n_workers}")

    (OUTPUT_DIR / "train").mkdir(parents=True, exist_ok=True)

    # ── Incremental labels writer ───────────────────────────────────────
    labels_path = OUTPUT_DIR / "labels.csv"
    if not labels_path.exists():
        with open(labels_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["case", "et", "tc", "wt", "wt_volume", "tc_volume", "et_volume", "grade_proxy"])
    # ponytail: file-lock skipped (single-process run), multi-node lock if scaling up later

    t0 = time.time()
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
                    # Write label row immediately — survives crash
                    with open(labels_path, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(
                            [r["case"], r["et"], r["tc"], r["wt"],
                             r["volumes"]["wt"], r["volumes"]["tc"],
                             r["volumes"]["et"], r["grade"]])
                else:
                    errors.append(r)
            except Exception as e:
                errors.append({"status": "exception", "case": futures[future], "error": str(e)})
            if done % 100 == 0:
                elapsed = time.time() - t0
                eta = (elapsed / done * (len(all_cases) - done)) if done else 0
                h, m = divmod(int(elapsed), 3600)
                eh, em = divmod(int(eta), 3600)
                log(f"  [{done}/{len(all_cases)}] elapsed={h}h {m}m  eta={eh}h {em}m  OK={len(results)}  ERR={len(errors)}")

    elapsed = time.time() - t0
    h, m = divmod(int(elapsed), 3600)
    log(f"  [{len(all_cases)}/{len(all_cases)}] DONE in {h}h {m}m")

    # Write errors to CSV
    if errors:
        err_path = OUTPUT_DIR / "preprocessing_errors.csv"
        with open(err_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["case", "status", "mod", "error"])
            for e in errors:
                w.writerow([e["case"], e["status"], e.get("mod", ""), e.get("error", "")])
        log(f"  Errors log: {err_path} ({len(errors)} rows)")

    # Summary
    log(f"\n{'=' * 70}")
    log("PREPROCESSING COMPLETE")
    log(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Total time: {h}h {m}m")
    log(f"  Processed: {len(results)}/{len(all_cases)}")
    log(f"  Errors:    {len(errors)}")

    log(f"  Labels: {labels_path}")

    # Class balance
    high = sum(1 for r in results if r["grade"] == 1)
    low = len(results) - high
    log(f"\n  Grade proxy distribution:")
    log(f"    High-grade (ET/TC present): {high}")
    log(f"    Low-grade  (neither):       {low}")
    log(f"\n  Subregion presence:")
    for sub in ["et", "tc", "wt"]:
        cnt = sum(1 for r in results if r[sub] == 1)
        pct = cnt / len(results) * 100 if results else 0
        log(f"    {sub.upper()} {cnt}/{len(results)} ({pct:.1f}%)")
    log(f"  Log file: {log_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Resume crashed run: skip cases with existing .npy, append labels")
    args = p.parse_args()
    run_preprocessing(args.workers, args.max_cases, args.resume)