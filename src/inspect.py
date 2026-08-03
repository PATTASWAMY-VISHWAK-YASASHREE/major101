"""Inspect MRI/CT dataset: class balance, scan quality, intensity stats, bad files."""

from pathlib import Path
from collections import Counter
import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("nibabel not installed — skipping volume inspection")
    nib = None


def inspect_dataset(root: Path):
    stats = {"n_scans": 0, "n_classes": 0, "bad_files": [], "class_counts": Counter()}
    intensity_means, intensity_stds = [], []

    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = int(class_dir.name)
            for f in sorted(class_dir.glob("*.nii.gz")):
                stats["n_scans"] += 1
                stats["class_counts"][label] += 1
                try:
                    if nib:
                        img = nib.load(f)
                        vol = img.get_fdata(dtype=np.float32)
                        non_zero = vol[vol != 0]
                        if non_zero.size == 0:
                            stats["bad_files"].append((str(f), "all-zero volume"))
                            continue
                        stats["class_counts"][label] -= 1  # recount below
                        # re-add as valid
                        mean, std = non_zero.mean(), non_zero.std()
                        intensity_means.append(mean)
                        intensity_stds.append(std)
                        stats["class_counts"][label] += 1
                except Exception as e:
                    stats["bad_files"].append((str(f), str(e)))

    stats["n_classes"] = len(stats["class_counts"])
    stats["intensity_mean"] = float(np.mean(intensity_means)) if intensity_means else None
    stats["intensity_std"] = float(np.mean(intensity_stds)) if intensity_stds else None

    # Summary report
    print("=" * 60)
    print("BRAIN TUMOUR DATASET INSPECTION")
    print("=" * 60)
    print(f"Total valid scans:  {stats['n_scans']}")
    print(f"Classes:            {stats['n_classes']}")
    print(f"Class distribution:")
    for cls, cnt in sorted(stats["class_counts"].items()):
        pct = cnt / stats["n_scans"] * 100 if stats["n_scans"] else 0
        print(f"  Class {cls}: {cnt:5d}  ({pct:.1f}%)")
    print(f"Intensity mean:     {stats['intensity_mean']:.2f}" if stats["intensity_mean"] else "Intensity mean:     N/A (no nibabel)")
    print(f"Intensity std:      {stats['intensity_std']:.2f}" if stats["intensity_std"] else "Intensity std:      N/A")
    if stats["bad_files"]:
        print(f"\n⚠ {len(stats['bad_files'])} bad file(s):")
        for path, reason in stats["bad_files"]:
            print(f"  {path}: {reason}")
    else:
        print("\n✓ All files readable")
    print("=" * 60)
    return stats


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    args = p.parse_args()
    inspect_dataset(Path(args.data))
