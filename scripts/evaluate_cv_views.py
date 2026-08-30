#!/usr/bin/env python3
"""Re-evaluate completed development CV checkpoints with deterministic multi-view crops."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cross_validate_repaired import bootstrap_metric_intervals
from scripts.train_ultra_light import evaluate, load_verified_report, runtime_memory
from src.grade_data import (
    MemoryMappedPatchDataset,
    assert_case_disjoint,
    assert_subject_disjoint,
    build_cross_validation_folds,
    build_stratified_splits,
    load_case_table,
    records_from_table,
)
from src.grade_model import BinaryFocalLoss, TinyGradeClassifier3D, binary_metrics, choose_validation_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-dir", type=Path, default=Path("outputs/cv/full_epoch_baseline_5fold_5ep"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cv/full_epoch_baseline_5fold_5ep_views5"))
    parser.add_argument("--npy-dir", type=Path, default=Path("data/brats_preprocessed/train"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/brats_preprocessed/labels.csv"))
    parser.add_argument("--data-report", type=Path, default=Path("outputs/data_quality/preprocessed_data_report.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--views", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-vram-gib", type=float, default=2.0)
    parser.add_argument("--target-ram-gib", type=float, default=3.0)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture", {})
    model = TinyGradeClassifier3D(
        base_channels=int(architecture.get("base_channels", 12)),
        dropout=float(architecture.get("dropout", 0.25)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(memory_format=torch.channels_last_3d), checkpoint


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for CV view evaluation")
    if not 1 <= args.views <= 8 or not 0 < args.patch_size <= 96:
        raise ValueError("views must be 1..8 and patch-size must be 1..96")
    if args.batch_size < 2 or args.batch_size % 2:
        raise ValueError("batch-size must be even and at least 2")

    load_verified_report(args.data_report, npy_dir=args.npy_dir, labels_csv=args.labels_csv)
    device = torch.device("cuda")
    cases = load_case_table(args.labels_csv, args.npy_dir)
    locked_splits = build_stratified_splits(cases, seed=args.seed)
    assert_case_disjoint(locked_splits)
    assert_subject_disjoint(locked_splits)
    development = pd.concat([locked_splits["train"], locked_splits["val"]], ignore_index=True)
    folds = build_cross_validation_folds(development, seed=args.seed, n_splits=args.folds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2),
        encoding="utf-8",
    )

    fold_summaries: list[dict[str, Any]] = []
    out_of_fold: list[pd.DataFrame] = []
    for fold_number, split in sorted(folds.items()):
        checkpoint_path = args.cv_dir / f"fold_{fold_number:02d}" / "best_checkpoint.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing completed fold checkpoint: {checkpoint_path}")
        dataset = MemoryMappedPatchDataset(
            records_from_table(split["val"]),
            patch_size=args.patch_size,
            training=False,
            noise_std=0.0,
            crop_candidates=args.views,
            evaluation_views=args.views > 1,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        model, checkpoint = load_model(checkpoint_path, device)
        criterion = BinaryFocalLoss(gamma=0.0, label_smoothing=0.0).to(device)
        torch.cuda.reset_peak_memory_stats(device)
        _, _, predictions = evaluate(
            model,
            loader,
            criterion,
            device,
            progress_label=f"fold {fold_number} {args.views}-view evaluation",
            aggregate_cases=args.views > 1,
        )
        memory = runtime_memory(device)
        if memory["reserved_gib"] > args.target_vram_gib or memory["ram_gib"] > args.target_ram_gib:
            raise MemoryError(f"Memory budget exceeded: {memory}")
        threshold, metrics = choose_validation_threshold(
            predictions["probability_high"].to_numpy(), predictions["true_label"].to_numpy()
        )
        predictions.insert(0, "fold", fold_number)
        predictions.to_csv(args.output_dir / f"fold_{fold_number:02d}_validation_predictions.csv", index=False)
        fold_summaries.append(
            {
                "fold": fold_number,
                "validation_case_count": len(split["val"]),
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
                "threshold": threshold,
                "metrics": metrics,
                "memory": memory,
            }
        )
        out_of_fold.append(predictions)
        del model, loader, dataset, criterion
        torch.cuda.empty_cache()

    oof = pd.concat(out_of_fold, ignore_index=True)
    if oof["case"].duplicated().any() or len(oof) != len(development) or set(oof["case"]) != set(development["case"]):
        raise AssertionError("Multi-view OOF predictions do not cover each development case exactly once")
    oof.to_csv(args.output_dir / "out_of_fold_predictions.csv", index=False)
    fixed_metrics = binary_metrics(oof["probability_high"].to_numpy(), oof["true_label"].to_numpy(), threshold=0.5)
    threshold, selected_metrics = choose_validation_threshold(
        oof["probability_high"].to_numpy(), oof["true_label"].to_numpy()
    )
    summary = {
        "source_cv_dir": str(args.cv_dir),
        "split_strategy": "development-only subject-disjoint StratifiedGroupKFold",
        "evaluation_views": args.views,
        "patch_size": args.patch_size,
        "development_case_count": len(development),
        "out_of_fold_case_count": len(oof),
        "complete_oof": True,
        "folds": fold_summaries,
        "mean_best_balanced_accuracy": float(np.mean([item["metrics"]["balanced_accuracy"] for item in fold_summaries])),
        "mean_best_auroc": float(np.mean([item["metrics"]["auroc"] for item in fold_summaries])),
        "out_of_fold_metrics_at_threshold_0_5": fixed_metrics,
        "out_of_fold_threshold": threshold,
        "out_of_fold_metrics_at_selected_threshold": selected_metrics,
        "out_of_fold_bootstrap_95ci": bootstrap_metric_intervals(
            oof["probability_high"].to_numpy(), oof["true_label"].to_numpy(), threshold=threshold, seed=args.seed
        ),
        "locked_test_case_count": len(locked_splits["test"]),
        "locked_test_evaluated": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
