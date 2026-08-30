#!/usr/bin/env python3
"""Fit the final repaired MRI checkpoint on the development partition only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate import apply_temperature
from scripts.train_ultra_light import load_verified_report, run_epoch, runtime_memory, set_seed
from src.grade_data import (
    BalancedBatchSampler,
    MemoryMappedPatchDataset,
    assert_case_disjoint,
    assert_subject_disjoint,
    build_stratified_splits,
    load_case_table,
    records_from_table,
    save_split_manifest,
)
from src.grade_model import BinaryFocalLoss, TinyGradeClassifier3D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy-dir", type=Path, default=Path("data/brats_preprocessed/train"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/brats_preprocessed/labels.csv"))
    parser.add_argument("--data-report", type=Path, default=Path("outputs/data_quality/preprocessed_data_report.json"))
    parser.add_argument("--calibration", type=Path, default=Path("outputs/calibration/repaired/calibration.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training/repaired_final"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--base-channels", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-vram-gib", type=float, default=2.0)
    parser.add_argument("--target-ram-gib", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for final development training")
    if args.epochs <= 0 or args.steps_per_epoch < 0 or args.grad_accum <= 0:
        raise ValueError("epochs and gradient accumulation must be positive; steps-per-epoch cannot be negative")
    if args.batch_size < 2 or args.batch_size % 2 or args.patch_size > 96:
        raise ValueError("batch size must be even and patch size must be <= 96")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    temperature = float(calibration["temperature"])
    threshold = float(calibration["threshold"])
    apply_temperature([0.5], temperature)

    load_verified_report(args.data_report, npy_dir=args.npy_dir, labels_csv=args.labels_csv)
    set_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    cases = load_case_table(args.labels_csv, args.npy_dir)
    locked_splits = build_stratified_splits(cases, seed=args.seed)
    development = pd.concat([locked_splits["train"], locked_splits["val"]], ignore_index=True)
    assert_case_disjoint({"development": development, "locked_test": locked_splits["test"]})
    assert_subject_disjoint({"development": development, "locked_test": locked_splits["test"]})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    development_manifest = args.output_dir / "development_manifest.json"
    save_split_manifest({"development": development, "locked_test": locked_splits["test"]}, development_manifest, seed=args.seed)
    (args.output_dir / "run_config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2),
        encoding="utf-8",
    )

    dataset = MemoryMappedPatchDataset(records_from_table(development), patch_size=args.patch_size, training=True)
    sampler = BalancedBatchSampler(
        development["grade_proxy"].tolist(), args.batch_size, seed=args.seed, steps_per_epoch=args.steps_per_epoch or None
    )
    sample_batch = next(iter(sampler))
    sample_labels = development.iloc[sample_batch]["grade_proxy"].value_counts().to_dict()
    if sample_labels.get(0) != args.batch_size // 2 or sample_labels.get(1) != args.batch_size // 2:
        raise AssertionError(f"Sampler produced an unbalanced batch: {sample_labels}")
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)
    model = TinyGradeClassifier3D(base_channels=args.base_channels, dropout=args.dropout).to(device)
    model = model.to(memory_format=torch.channels_last_3d)
    criterion = BinaryFocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing).to(device)
    initial_lr = args.lr / args.warmup_epochs if args.warmup_epochs else args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history: list[dict[str, Any]] = []

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for group in optimizer.param_groups:
                group["lr"] = warmup_lr
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        train_loss, train_metrics = run_epoch(
            model, loader, sampler, optimizer, scaler, criterion, device, epoch=epoch, grad_accum=args.grad_accum
        )
        memory = runtime_memory(device)
        if memory["reserved_gib"] > args.target_vram_gib or memory["ram_gib"] > args.target_ram_gib:
            raise MemoryError(f"Memory budget exceeded: {memory}")
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "train_auroc": train_metrics["auroc"],
                "lr": optimizer.param_groups[0]["lr"],
                "peak_vram_gib": memory["reserved_gib"],
                "process_ram_gib": memory["ram_gib"],
                "seconds": time.perf_counter() - started,
            }
        )
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        print(f"development epoch {epoch + 1}/{args.epochs}: loss={train_loss:.4f} auroc={train_metrics['auroc']} memory={memory}")

    checkpoint_path = args.output_dir / "best_checkpoint.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": {"base_channels": args.base_channels, "dropout": args.dropout, "patch_size": args.patch_size, "whole_volume": False},
            "seed": args.seed,
            "split_strategy": "development-only subject-disjoint StratifiedGroupKFold",
            "development_manifest": str(development_manifest),
            "calibration": str(args.calibration),
            "temperature": temperature,
            "threshold": threshold,
            "training": {"batch_size": args.batch_size, "grad_accum": args.grad_accum, "learning_rate": args.lr, "epochs": args.epochs},
            "selection": "fixed epoch count selected by development cross-validation; threshold and temperature fit on out-of-fold development predictions",
        },
        str(checkpoint_path),
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "development_case_count": len(development),
        "locked_test_case_count": len(locked_splits["test"]),
        "epochs_completed": args.epochs,
        "temperature": temperature,
        "threshold": threshold,
        "development_manifest": str(development_manifest),
        "calibration": str(args.calibration),
        "test_evaluated": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
