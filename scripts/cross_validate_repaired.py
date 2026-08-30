#!/usr/bin/env python3
"""Run MRI-only subject-disjoint cross-validation without touching the locked test set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_ultra_light import (
    evaluate,
    load_verified_report,
    plot_diagnostics,
    run_epoch,
    runtime_memory,
    set_seed,
)
from src.grade_data import (
    BalancedBatchSampler,
    MemoryMappedPatchDataset,
    assert_case_disjoint,
    assert_subject_disjoint,
    build_cross_validation_folds,
    build_stratified_splits,
    load_case_table,
    records_from_table,
    subject_id,
)
from src.grade_model import BinaryFocalLoss, TinyGradeClassifier3D, binary_metrics, choose_validation_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy-dir", type=Path, default=Path("data/brats_preprocessed/train"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/brats_preprocessed/labels.csv"))
    parser.add_argument("--data-report", type=Path, default=Path("outputs/data_quality/preprocessed_data_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cv/repaired"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-limit", type=int, default=0, help="Run only the first N folds; 0 runs all folds")
    parser.add_argument("--fold-list", default="", help="Comma-separated fold numbers to run, for example 3,4,5")
    parser.add_argument("--allow-incomplete", action="store_true", help="Aggregate only selected folds for a smoke run; never treat OOF as complete")
    parser.add_argument("--resume", action="store_true", help="Reuse completed fold artifacts in the output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 uses a full balanced epoch")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--evaluation-views", type=int, default=1, help="Deterministic crops averaged per validation case")
    parser.add_argument("--base-channels", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3, help="Stop after this many non-improving validation epochs; 0 disables early stopping")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum selection-score improvement for early stopping")
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None, help="Seed for fixed fold construction; defaults to --seed")
    parser.add_argument("--target-vram-gib", type=float, default=2.0)
    parser.add_argument("--target-ram-gib", type=float, default=3.0)
    return parser.parse_args()


def parse_fold_list(value: str, available: list[int]) -> list[int]:
    if not value.strip():
        return available
    try:
        selected = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--fold-list must contain comma-separated integers") from exc
    if not selected or len(set(selected)) != len(selected) or any(item not in available for item in selected):
        raise ValueError(f"--fold-list must contain unique fold numbers from {available}")
    return selected


def fold_artifacts_complete(fold_dir: Path, epochs: int, *, allow_early_stop: bool = False) -> bool:
    required = (fold_dir / "best_checkpoint.pth", fold_dir / "best_validation_predictions.csv", fold_dir / "history.csv")
    if not all(path.is_file() for path in required):
        return False
    try:
        history_length = len(pd.read_csv(fold_dir / "history.csv"))
        if history_length >= epochs:
            return True
        if allow_early_stop:
            run_summary = fold_run_summary(fold_dir)
            return bool(run_summary.get("early_stopped")) and int(run_summary.get("epochs_completed", 0)) == history_length
        return False
    except (OSError, pd.errors.ParserError):
        return False


def fold_run_summary(fold_dir: Path) -> dict[str, Any]:
    path = fold_dir / "run_summary.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def fold_summary_from_artifacts(
    fold_number: int,
    split: dict[str, pd.DataFrame],
    fold_dir: Path,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    checkpoint = torch.load(fold_dir / "best_checkpoint.pth", map_location=device, weights_only=False)
    metrics = checkpoint["validation_metrics"]
    selection_score = metrics.get("auroc")
    if selection_score is None:
        selection_score = metrics["balanced_accuracy"]
    predictions = pd.read_csv(fold_dir / "best_validation_predictions.csv")
    predictions.insert(0, "fold", fold_number)
    run_summary = fold_run_summary(fold_dir)
    return (
        {
            "fold": fold_number,
            "train_case_count": len(split["train"]),
            "validation_case_count": len(split["val"]),
            "best_epoch": int(checkpoint["epoch"]),
            "best_selection_score": float(selection_score),
            "best_selection_metric": "auroc",
            "best_balanced_accuracy": metrics["balanced_accuracy"],
            "best_validation_metrics": metrics,
            "checkpoint": str(fold_dir / "best_checkpoint.pth"),
            "epochs_completed": int(run_summary.get("epochs_completed", len(pd.read_csv(fold_dir / "history.csv")))),
            "early_stopped": bool(run_summary.get("early_stopped", False)),
        },
        predictions,
    )


def bootstrap_metric_intervals(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    threshold: float,
    seed: int,
    samples: int = 1000,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {}
    for _ in range(samples):
        indexes = rng.integers(0, len(targets), size=len(targets))
        metrics = binary_metrics(probabilities[indexes], targets[indexes], threshold=threshold)
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                values.setdefault(name, []).append(float(value))
    return {
        name: {"lower_95": float(np.quantile(scores, 0.025)), "upper_95": float(np.quantile(scores, 0.975))}
        for name, scores in values.items()
    }


def manifest_payload(folds: dict[int, dict[str, pd.DataFrame]], *, seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "split_strategy": "development-only subject-disjoint StratifiedGroupKFold",
        "folds": {
            str(number): {
                name: [
                    {"case": str(row.case), "subject": subject_id(row.case), "label": int(row.grade_proxy)}
                    for row in frame.itertuples(index=False)
                ]
                for name, frame in split.items()
            }
            for number, split in folds.items()
        },
    }


def main() -> int:
    args = parse_args()
    if args.split_seed is None:
        args.split_seed = args.seed
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for cross-validation")
    if args.folds < 2 or args.fold_limit < 0 or args.epochs <= 0 or args.steps_per_epoch < 0:
        raise ValueError("folds and epochs must be positive; steps-per-epoch cannot be negative")
    if args.patience < 0 or args.min_delta < 0:
        raise ValueError("patience and min-delta cannot be negative")
    if args.batch_size < 2 or args.batch_size % 2:
        raise ValueError("--batch-size must be even and at least 2")
    if args.patch_size > 96 or args.evaluation_views < 1 or args.evaluation_views > 8:
        raise ValueError("Use patches <=96 while enforcing the 2 GiB VRAM budget")

    load_verified_report(args.data_report, npy_dir=args.npy_dir, labels_csv=args.labels_csv)
    set_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cases = load_case_table(args.labels_csv, args.npy_dir)
    locked_splits = build_stratified_splits(cases, seed=args.split_seed)
    assert_case_disjoint(locked_splits)
    assert_subject_disjoint(locked_splits)
    development = pd.concat([locked_splits["train"], locked_splits["val"]], ignore_index=True)
    all_folds = build_cross_validation_folds(development, seed=args.split_seed, n_splits=args.folds)
    selected_fold_numbers = parse_fold_list(args.fold_list, sorted(all_folds))
    if args.fold_limit:
        selected_fold_numbers = selected_fold_numbers[: args.fold_limit]
    folds = {number: all_folds[number] for number in sorted(all_folds)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "fold_manifest.json").write_text(json.dumps(manifest_payload(folds, seed=args.split_seed), indent=2), encoding="utf-8")
    print(f"Development cases: {len(development)} | locked test cases excluded: {len(locked_splits['test'])}")
    aggregate_fold_numbers = selected_fold_numbers if args.allow_incomplete else sorted(folds)
    print(f"Requested folds: {selected_fold_numbers} | aggregate folds: {aggregate_fold_numbers}")

    fold_summaries: list[dict[str, Any]] = []
    out_of_fold: list[pd.DataFrame] = []
    trained_folds: list[int] = []
    reused_folds: list[int] = []
    for fold_number in selected_fold_numbers:
        split = folds[fold_number]
        fold_dir = args.output_dir / f"fold_{fold_number:02d}"
        if args.resume and fold_artifacts_complete(fold_dir, args.epochs, allow_early_stop=args.patience > 0):
            print(f"Reusing completed fold {fold_number}")
            reused_folds.append(fold_number)
            continue
        fold_seed = args.seed + fold_number
        set_seed(fold_seed)
        fold_dir.mkdir(parents=True, exist_ok=True)
        trained_folds.append(fold_number)
        train_dataset = MemoryMappedPatchDataset(
            records_from_table(split["train"]), patch_size=args.patch_size, training=True
        )
        val_dataset = MemoryMappedPatchDataset(
            records_from_table(split["val"]),
            patch_size=args.patch_size,
            training=False,
            noise_std=0.0,
            crop_candidates=args.evaluation_views,
            evaluation_views=args.evaluation_views > 1,
        )
        sampler = BalancedBatchSampler(
            split["train"]["grade_proxy"].tolist(),
            args.batch_size,
            seed=fold_seed,
            steps_per_epoch=args.steps_per_epoch or None,
        )
        loader_kwargs = {"num_workers": 0, "pin_memory": True}
        train_loader = DataLoader(train_dataset, batch_sampler=sampler, **loader_kwargs)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        model = TinyGradeClassifier3D(base_channels=args.base_channels, dropout=args.dropout).to(device)
        model = model.to(memory_format=torch.channels_last_3d)
        criterion = BinaryFocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing).to(device)
        initial_lr = args.lr / args.warmup_epochs if args.warmup_epochs else args.lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        history: list[dict[str, Any]] = []
        best_score = -float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        early_stopped = False
        checkpoint_path = fold_dir / "best_checkpoint.pth"
        prediction_path = fold_dir / "best_validation_predictions.csv"
        for epoch in range(args.epochs):
            if epoch < args.warmup_epochs:
                warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = warmup_lr
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            train_loss, train_metrics = run_epoch(
                model, train_loader, sampler, optimizer, scaler, criterion, device,
                epoch=epoch, grad_accum=args.grad_accum,
            )
            val_loss, fixed_metrics, predictions = evaluate(
                model,
                val_loader,
                criterion,
                device,
                progress_label=f"fold {fold_number} validation {epoch + 1}",
                aggregate_cases=args.evaluation_views > 1,
            )
            threshold, selected_metrics = choose_validation_threshold(
                predictions["probability_high"].to_numpy(), predictions["true_label"].to_numpy()
            )
            memory = runtime_memory(device)
            if memory["reserved_gib"] > args.target_vram_gib:
                raise MemoryError(f"VRAM budget exceeded: {memory['reserved_gib']:.2f} GiB > {args.target_vram_gib:.2f} GiB")
            if memory["ram_gib"] > args.target_ram_gib:
                raise MemoryError(f"RAM budget exceeded: {memory['ram_gib']:.2f} GiB > {args.target_ram_gib:.2f} GiB")
            score = selected_metrics["auroc"]
            if score is None:
                score = selected_metrics["balanced_accuracy"]
            if epoch >= args.warmup_epochs:
                scheduler.step(score)
            record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_loss,
                "val_accuracy_fixed_0_5": fixed_metrics["accuracy"],
                "val_balanced_accuracy": selected_metrics["balanced_accuracy"],
                "val_f1": selected_metrics["f1"],
                "val_sensitivity": selected_metrics["sensitivity"],
                "val_specificity": selected_metrics["specificity"],
                "val_auroc": selected_metrics["auroc"],
                "val_average_precision": selected_metrics["average_precision"],
                "threshold": threshold,
                "lr": optimizer.param_groups[0]["lr"],
                "peak_vram_gib": memory["reserved_gib"],
                "process_ram_gib": memory["ram_gib"],
                "seconds": time.perf_counter() - started,
            }
            history.append(record)
            pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
            print(
                f"fold {fold_number} epoch {epoch + 1}/{args.epochs}: "
                f"bal_acc={selected_metrics['balanced_accuracy']:.4f} auroc={score:.4f} "
                f"vram={memory['reserved_gib']:.2f}GiB ram={memory['ram_gib']:.2f}GiB"
            )
            improved = score > best_score + args.min_delta
            if improved:
                best_score = score
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "architecture": {"base_channels": args.base_channels, "dropout": args.dropout, "patch_size": args.patch_size},
                        "seed": fold_seed,
                        "fold": fold_number,
                        "split_strategy": "development-only subject-disjoint StratifiedGroupKFold",
                        "threshold": threshold,
                        "epoch": best_epoch,
                        "validation_metrics": selected_metrics,
                    },
                    checkpoint_path,
                )
                predictions.to_csv(prediction_path, index=False)
            else:
                epochs_without_improvement += 1
                if args.patience and epochs_without_improvement >= args.patience:
                    early_stopped = True
                    print(
                        f"fold {fold_number}: early stopping after {epoch + 1} epochs "
                        f"({args.patience} epochs without >{args.min_delta:g} improvement)"
                    )
                    break

        (fold_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "epochs_requested": args.epochs,
                    "epochs_completed": len(history),
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "early_stopped": early_stopped,
                    "patience": args.patience,
                    "min_delta": args.min_delta,
                    "evaluation_views": args.evaluation_views,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        fold_summary, best_predictions = fold_summary_from_artifacts(fold_number, split, fold_dir, device=device)
        out_of_fold.append(best_predictions)
        plot_diagnostics(history, best_predictions, fold_dir, f"Fold {fold_number} validation evidence (best epoch {best_epoch})")
        fold_summaries.append(fold_summary)
        del model, optimizer, scheduler, scaler, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()

    for fold_number in aggregate_fold_numbers:
        fold_dir = args.output_dir / f"fold_{fold_number:02d}"
        if not fold_artifacts_complete(fold_dir, args.epochs, allow_early_stop=args.patience > 0):
            raise RuntimeError(
                f"Fold {fold_number} is incomplete. Run it with --fold-list {fold_number} "
                f"or select the missing folds before aggregating."
            )
        if fold_number not in {item["fold"] for item in fold_summaries}:
            summary, predictions = fold_summary_from_artifacts(fold_number, folds[fold_number], fold_dir, device=device)
            fold_summaries.append(summary)
            out_of_fold.append(predictions)

    fold_summaries.sort(key=lambda item: item["fold"])
    oof = pd.concat(out_of_fold, ignore_index=True)
    expected_cases = pd.concat([folds[number]["val"] for number in aggregate_fold_numbers], ignore_index=True)
    expected_oof_cases = len(expected_cases)
    if oof["case"].duplicated().any() or len(oof) != expected_oof_cases or set(oof["case"]) != set(expected_cases["case"]):
        raise AssertionError("Out-of-fold predictions do not cover each development case exactly once")
    oof.to_csv(args.output_dir / "out_of_fold_predictions.csv", index=False)
    oof_fixed_metrics = binary_metrics(
        oof["probability_high"].to_numpy(), oof["true_label"].to_numpy(), threshold=0.5
    )
    oof_threshold, oof_selected_metrics = choose_validation_threshold(
        oof["probability_high"].to_numpy(), oof["true_label"].to_numpy()
    )
    metric_names = (
        "accuracy", "balanced_accuracy", "f1", "sensitivity", "specificity", "auroc",
        "average_precision", "predicted_positive_rate",
    )
    fold_metric_summary = {}
    for name in metric_names:
        values = [item["best_validation_metrics"].get(name) for item in fold_summaries]
        values = [float(value) for value in values if value is not None]
        fold_metric_summary[name] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
    majority_class = int(round(float(oof["true_label"].mean())))
    majority_metrics = binary_metrics(
        np.full(len(oof), majority_class, dtype=np.float64), oof["true_label"].to_numpy(), threshold=0.5
    )
    summary = {
        "split_strategy": "development-only subject-disjoint StratifiedGroupKFold",
        "seed": args.seed,
        "split_seed": args.split_seed,
        "folds_requested": args.folds,
        "folds_run": aggregate_fold_numbers,
        "folds_trained_this_run": sorted(trained_folds),
        "folds_reused_this_run": sorted(reused_folds),
        "early_stopping": {"patience": args.patience, "min_delta": args.min_delta},
        "evaluation_views": args.evaluation_views,
        "development_case_count": len(development),
        "out_of_fold_case_count": len(oof),
        "complete_oof": not args.allow_incomplete and len(oof) == len(development),
        "locked_test_case_count": len(locked_splits["test"]),
        "folds": fold_summaries,
        "mean_best_balanced_accuracy": float(np.mean([item["best_balanced_accuracy"] for item in fold_summaries])),
        "checkpoint_selection_metric": "auroc (balanced accuracy is reported at a validation-only threshold)",
        "mean_best_auroc": float(np.mean([item["best_validation_metrics"]["auroc"] for item in fold_summaries])),
        "fold_metric_mean_std": fold_metric_summary,
        "out_of_fold_metrics_at_threshold_0_5": oof_fixed_metrics,
        "out_of_fold_threshold": oof_threshold,
        "out_of_fold_metrics_at_selected_threshold": oof_selected_metrics,
        "out_of_fold_bootstrap_95ci": bootstrap_metric_intervals(
            oof["probability_high"].to_numpy(),
            oof["true_label"].to_numpy(),
            threshold=oof_threshold,
            seed=args.seed,
        ),
        "majority_class_baseline": {"class": majority_class, "metrics": majority_metrics},
        "locked_test_evaluated": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
