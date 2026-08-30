#!/usr/bin/env python3
"""Compatibility entry point for the repaired memory-bounded classifier.

The former implementation loaded full 182x218x182 volumes, ignored the CLI
batch size, hard-coded class counts, broadcast focal weights across samples,
and evaluated the test set on every run. This command now forwards to the
single canonical trainer.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER = REPO_ROOT / "scripts" / "train_ultra_light.py"


def build_command(args: argparse.Namespace) -> list[str]:
    output_dir = Path(args.output_dir) / args.model
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
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--steps-per-epoch",
        str(args.steps_per_epoch),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--patch-size",
        str(args.patch_size),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--patience",
        str(args.patience),
        "--focal-gamma",
        "0",
        "--label-smoothing",
        "0",
    ]
    if args.full:
        command.append("--full")
    if args.evaluate_test:
        command.append("--evaluate-test")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("M1", "M3"), default="M1")
    parser.add_argument("--augment", action="store_true", help="Accepted for compatibility; training augmentation is always enabled")
    parser.add_argument("--npy-dir", type=Path, default=Path("data/brats_preprocessed/train"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/brats_preprocessed/labels.csv"))
    parser.add_argument("--data-report", type=Path, default=Path("outputs/data_quality/preprocessed_data_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.augment:
        print("--augment is retained for compatibility; the canonical image-only trainer already augments training crops.")
    return subprocess.call(build_command(args), cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
