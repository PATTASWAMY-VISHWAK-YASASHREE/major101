#!/usr/bin/env python3
"""Stream-check every preprocessed BraTS .npy file without exceeding RAM limits."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.grade_data import EXPECTED_SHAPE, REQUIRED_LABEL_COLUMNS


def inspect_file(path: Path, chunk_depth: int) -> dict:
    """Read a file in <= roughly 11 MiB chunks and return structural/value stats."""
    result = {"file": path.name, "valid": False, "errors": []}
    try:
        array = np.load(path, mmap_mode="r")
    except (OSError, ValueError) as exc:
        result["errors"].append(f"load_error: {exc}")
        return result
    result["shape"] = list(array.shape)
    result["dtype"] = str(array.dtype)
    if tuple(array.shape) != EXPECTED_SHAPE:
        result["errors"].append(f"shape={tuple(array.shape)} expected={EXPECTED_SHAPE}")
    if array.dtype != np.float32:
        result["errors"].append(f"dtype={array.dtype} expected=float32")
    if result["errors"]:
        return result

    channel_min = np.full(4, np.inf, dtype=np.float64)
    channel_max = np.full(4, -np.inf, dtype=np.float64)
    channel_sum = np.zeros(4, dtype=np.float64)
    channel_count = np.zeros(4, dtype=np.int64)
    for start in range(0, array.shape[1], chunk_depth):
        chunk = np.asarray(array[:, start:start + chunk_depth, :, :])
        if not np.isfinite(chunk).all():
            result["errors"].append(f"non_finite_values_depth={start}:{min(start + chunk_depth, array.shape[1])}")
            return result
        flattened = chunk.reshape(4, -1)
        channel_min = np.minimum(channel_min, flattened.min(axis=1))
        channel_max = np.maximum(channel_max, flattened.max(axis=1))
        channel_sum += flattened.sum(axis=1, dtype=np.float64)
        channel_count += flattened.shape[1]
    result["channel_min"] = channel_min.tolist()
    result["channel_max"] = channel_max.tolist()
    result["channel_mean"] = (channel_sum / channel_count).tolist()
    result["valid"] = True
    return result


def validate_labels(labels_csv: Path, npy_dir: Path) -> dict:
    frame = pd.read_csv(labels_csv)
    result: dict = {"row_count": int(len(frame)), "errors": []}
    missing = sorted(REQUIRED_LABEL_COLUMNS.difference(frame.columns))
    if missing:
        result["errors"].append(f"missing_columns={missing}")
        return result
    frame["case"] = frame["case"].astype(str).str.strip()
    frame["grade_proxy"] = pd.to_numeric(frame["grade_proxy"], errors="coerce")
    result["empty_case_count"] = int(frame["case"].eq("").sum())
    result["invalid_label_count"] = int((~frame["grade_proxy"].isin([0, 1])).sum())
    duplicate_labels = frame[frame["case"].duplicated(keep=False)]
    result["duplicate_row_count"] = int(len(duplicate_labels))
    conflict_cases = (
        frame.groupby("case")["grade_proxy"].nunique().loc[lambda values: values > 1].index.tolist()
    )
    result["conflicting_label_cases"] = conflict_cases
    deduped = frame.drop_duplicates(subset="case", keep="first")
    result["unique_case_count"] = int(len(deduped))
    result["class_counts"] = {str(int(label)): int(count) for label, count in deduped.groupby("grade_proxy").size().items()}
    labelled = set(deduped["case"])
    files = {path.stem for path in npy_dir.glob("*.npy")}
    result["missing_labelled_files"] = sorted(labelled.difference(files))
    result["orphan_files"] = sorted(files.difference(labelled))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy-dir", type=Path, default=Path("data/brats_preprocessed/train"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/brats_preprocessed/labels.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/data_quality/preprocessed_data_report.json"))
    parser.add_argument("--chunk-depth", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Only inspect this many files; 0 means every file")
    args = parser.parse_args()
    if args.chunk_depth <= 0:
        parser.error("--chunk-depth must be positive")

    paths = sorted(args.npy_dir.glob("*.npy"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        parser.error(f"No .npy files found in {args.npy_dir}")

    started = time.perf_counter()
    checks = []
    for index, path in enumerate(paths, start=1):
        check = inspect_file(path, args.chunk_depth)
        checks.append(check)
        if index % 25 == 0 or index == len(paths):
            print(f"checked {index}/{len(paths)} files")

    invalid = [check for check in checks if not check["valid"]]
    label_report = validate_labels(args.labels_csv, args.npy_dir)
    report = {
        "npy_dir": str(args.npy_dir),
        "labels_csv": str(args.labels_csv),
        "expected_shape": list(EXPECTED_SHAPE),
        "inspected_file_count": len(checks),
        "invalid_file_count": len(invalid),
        "invalid_files": invalid,
        "label_report": label_report,
        "elapsed_seconds": time.perf_counter() - started,
        "complete_scan": args.limit == 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("inspected_file_count", "invalid_file_count", "elapsed_seconds", "complete_scan")}, indent=2))
    print(f"report: {args.output}")
    has_label_error = bool(label_report["errors"] or label_report["conflicting_label_cases"] or label_report["missing_labelled_files"])
    return 1 if invalid or has_label_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
