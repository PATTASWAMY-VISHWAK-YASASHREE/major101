#!/usr/bin/env python3
"""Compatibility entry point for the retired ET-feature experiment.

The old implementation was incomplete and its scalar inputs reproduce the
``grade_proxy`` construction rule. Keep the command name working, but route it
to the leakage-safe, memory-bounded image-only trainer.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER = REPO_ROOT / "scripts" / "train_ultra_light.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy-dir", type=Path, default=Path("data/brats_preprocessed/train"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/brats_preprocessed/labels.csv"))
    parser.add_argument("--data-report", type=Path, default=Path("outputs/data_quality/preprocessed_data_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training/M1_image_only"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    command = [
        sys.executable,
        str(TRAINER),
        "--npy-dir",
        str(args.npy_dir),
        "--labels-csv",
        str(args.labels_csv),
        "--data-report",
        str(args.data_report),
        "--output-dir",
        str(args.output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
    ]
    if args.full:
        command.append("--full")
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
