#!/usr/bin/env python3
"""Memory-bounded, leakage-safe BraTS image-only grade training.

Default mode is a three-epoch validation smoke test. It never evaluates the
locked test partition unless ``--evaluate-test`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.grade_data import (
    BalancedBatchSampler,
    MemoryMappedPatchDataset,
    build_stratified_splits,
    load_case_table,
    records_from_table,
    save_split_manifest,
)
from src.grade_model import BinaryFocalLoss, TinyGradeClassifier3D, binary_metrics, choose_validation_threshold


def load_labels(csv_path: Path) -> list[tuple[str, int]]:
    """Read one unambiguous binary label per case; retained as a fast regression seam."""
    seen: dict[str, int] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"case", "grade_proxy"}.issubset(reader.fieldnames):
            raise ValueError("labels CSV must contain case and grade_proxy")
        for row in reader:
            case = row["case"].strip()
            label = int(row["grade_proxy"])
            if label not in (0, 1):
                raise ValueError(f"{case}: grade_proxy must be 0 or 1")
            if case in seen and seen[case] != label:
                raise ValueError(f"Conflicting duplicate labels for {case}")
            seen[case] = label
    return list(seen.items())


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_verified_report(path: Path, *, npy_dir: Path | None = None, labels_csv: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            f"Missing data-quality report: {path}. Run scripts/verify_preprocessed_data.py before training."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    labels = report.get("label_report", {})
    failures = []
    if npy_dir is not None and Path(report.get("npy_dir", "")).resolve() != Path(npy_dir).resolve():
        failures.append("report npy_dir does not match the requested data directory")
    if labels_csv is not None and Path(report.get("labels_csv", "")).resolve() != Path(labels_csv).resolve():
        failures.append("report labels_csv does not match the requested labels file")
    if not report.get("complete_scan"):
        failures.append("scan was not complete")
    if report.get("invalid_file_count", 0):
        failures.append(f"invalid files={report['invalid_file_count']}")
    if labels.get("missing_labelled_files"):
        failures.append(f"missing labelled files={len(labels['missing_labelled_files'])}")
    if labels.get("conflicting_label_cases"):
        failures.append(f"conflicting labels={len(labels['conflicting_label_cases'])}")
    if labels.get("invalid_label_count", 0) or labels.get("empty_case_count", 0):
        failures.append("invalid label rows")
    if failures:
        raise RuntimeError(f"Data-quality report is not usable: {', '.join(failures)}")
    return report


def runtime_memory(device: torch.device) -> dict[str, float]:
    try:
        import psutil
        ram_gib = psutil.Process().memory_info().rss / 1024 ** 3
    except ImportError:
        ram_gib = float("nan")
    result = {"ram_gib": ram_gib}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        result.update(
            allocated_gib=torch.cuda.max_memory_allocated(device) / 1024 ** 3,
            reserved_gib=torch.cuda.max_memory_reserved(device) / 1024 ** 3,
        )
    return result


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    sampler: BalancedBatchSampler,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    epoch: int,
    grad_accum: int,
) -> tuple[float, dict[str, Any]]:
    model.train()
    sampler.set_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_examples = 0
    all_probabilities: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    pending_steps = 0
    amp_enabled = device.type == "cuda"

    for step, (inputs, targets, _) in enumerate(tqdm(loader, desc=f"train {epoch + 1}", leave=False), start=1):
        inputs = inputs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last_3d)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(inputs)
            unscaled_loss = criterion(logits, targets)
            loss = unscaled_loss / grad_accum
        scaler.scale(loss).backward()
        pending_steps += 1
        if pending_steps == grad_accum:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pending_steps = 0

        batch_size = targets.numel()
        total_loss += float(unscaled_loss.detach()) * batch_size
        total_examples += batch_size
        all_probabilities.append(torch.sigmoid(logits.detach()).float().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    if pending_steps:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    probabilities = np.concatenate(all_probabilities)
    targets = np.concatenate(all_targets)
    return total_loss / total_examples, binary_metrics(probabilities, targets)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    threshold: float = 0.5,
    progress_label: str = "validation",
    aggregate_cases: bool = False,
) -> tuple[float, dict[str, Any], pd.DataFrame]:
    model.eval()
    amp_enabled = device.type == "cuda"
    total_loss = 0.0
    total_examples = 0
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    case_ids: list[str] = []
    for inputs, batch_targets, batch_cases in tqdm(loader, desc=progress_label, leave=False):
        inputs = inputs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last_3d)
        batch_targets = batch_targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(inputs)
            loss = criterion(logits, batch_targets)
        total_loss += float(loss) * batch_targets.numel()
        total_examples += batch_targets.numel()
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
        targets.append(batch_targets.cpu().numpy())
        case_ids.extend(batch_cases)
    probability_array = np.concatenate(probabilities)
    target_array = np.concatenate(targets)
    metrics = binary_metrics(probability_array, target_array, threshold)
    prediction_table = pd.DataFrame(
        {
            "case": case_ids,
            "true_label": target_array.astype(int),
            "probability_high": probability_array,
            "predicted_label": (probability_array >= threshold).astype(int),
        }
    )
    prediction_table["correct"] = (prediction_table["true_label"] == prediction_table["predicted_label"]).astype(int)
    if aggregate_cases:
        prediction_table = aggregate_case_predictions(prediction_table, threshold=threshold)
        metrics = binary_metrics(
            prediction_table["probability_high"].to_numpy(),
            prediction_table["true_label"].to_numpy(),
            threshold,
        )
    return total_loss / total_examples, metrics, prediction_table


def aggregate_case_predictions(predictions: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    """Average repeated deterministic crop predictions into one row per case."""
    if predictions["case"].duplicated().sum() == 0:
        return predictions
    aggregated = (
        predictions.groupby("case", sort=False, as_index=False)
        .agg(true_label=("true_label", "first"), probability_high=("probability_high", "mean"))
    )
    aggregated["predicted_label"] = (aggregated["probability_high"] >= threshold).astype(int)
    aggregated["correct"] = (aggregated["true_label"] == aggregated["predicted_label"]).astype(int)
    return aggregated


def plot_diagnostics(history: list[dict[str, Any]], predictions: pd.DataFrame, output_dir: Path, title: str) -> None:
    """Create compact visual evidence for every completed smoke/full run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["val_balanced_accuracy"], label="balanced accuracy")
    axes[1].plot(frame["epoch"], frame["val_f1"], label="F1")
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylim=(0, 1))
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)

    truth = predictions["true_label"].to_numpy()
    predicted = predictions["predicted_label"].to_numpy()
    matrix = np.array(
        [
            [np.sum((truth == 0) & (predicted == 0)), np.sum((truth == 0) & (predicted == 1))],
            [np.sum((truth == 1) & (predicted == 0)), np.sum((truth == 1) & (predicted == 1))],
        ]
    )
    fig, axis = plt.subplots(figsize=(5, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set(xticks=(0, 1), yticks=(0, 1), xticklabels=("Low", "High"), yticklabels=("Low", "High"), xlabel="Predicted", ylabel="True", title=f"{title}: confusion matrix")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_confusion_matrix.png", dpi=180)
    plt.close(fig)

    if len(np.unique(truth)) == 2:
        probabilities = predictions["probability_high"].to_numpy()
        false_positive_rate, true_positive_rate, _ = roc_curve(truth, probabilities)
        precision, recall, _ = precision_recall_curve(truth, probabilities)
        observed, expected = calibration_curve(truth, probabilities, n_bins=min(8, len(predictions)))
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        axes[0].plot(false_positive_rate, true_positive_rate)
        axes[0].plot([0, 1], [0, 1], "--", color="gray")
        axes[0].set(title="ROC", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
        axes[1].plot(recall, precision)
        axes[1].set(title="Precision-recall", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
        axes[2].plot(expected, observed, marker="o")
        axes[2].plot([0, 1], [0, 1], "--", color="gray")
        axes[2].set(title="Calibration", xlabel="Predicted probability", ylabel="Observed positive rate", xlim=(0, 1), ylim=(0, 1))
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output_dir / "validation_probability_diagnostics.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy-dir", type=Path, default=Path("data/brats_preprocessed/train"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/brats_preprocessed/labels.csv"))
    parser.add_argument("--data-report", type=Path, default=Path("outputs/data_quality/preprocessed_data_report.json"))
    parser.add_argument("--skip-data-report", action="store_true", help="Only for debugging a new dataset; never use for final runs")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training/ultra_light_repaired"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--full", action="store_true", help="Run at least 100 epochs instead of the smoke-test default")
    parser.add_argument("--steps-per-epoch", type=int, default=64, help="0 uses a full balanced epoch")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--whole-volume", action="store_true", help="Downsample the full mmap volume to the model input size")
    parser.add_argument("--base-channels", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--focal-gamma", type=float, default=0.0, help="Keep at 0 for balanced BCE; focal modulation is optional")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--target-vram-gib", type=float, default=2.0)
    parser.add_argument("--target-ram-gib", type=float, default=3.0)
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate the locked test split once after training")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.full:
        args.epochs = max(args.epochs, 100)
        if args.steps_per_epoch == 64:
            args.steps_per_epoch = 0
    if args.epochs <= 0 or args.grad_accum <= 0 or args.warmup_epochs < 0 or args.steps_per_epoch < 0:
        raise ValueError("epochs and grad accumulation must be positive; warmup and steps per epoch cannot be negative")
    if args.batch_size < 2 or args.batch_size % 2:
        raise ValueError("--batch-size must be even and at least 2 for balanced batches")
    if args.patch_size > 96:
        raise ValueError("Use patches <=96 while enforcing the 2 GiB VRAM budget")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required for this training command")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if not args.skip_data_report:
        report = load_verified_report(args.data_report, npy_dir=args.npy_dir, labels_csv=args.labels_csv)
        print(f"Using verified data report: {args.data_report} ({report['inspected_file_count']} files)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2),
        encoding="utf-8",
    )
    cases = load_case_table(args.labels_csv, args.npy_dir)
    splits = build_stratified_splits(cases, seed=args.seed)
    save_split_manifest(splits, args.output_dir / "split_manifest.json", seed=args.seed)
    print("Split counts:", {name: len(frame) for name, frame in splits.items()})
    print("Train classes:", splits["train"]["grade_proxy"].value_counts().sort_index().to_dict())
    print("Subject counts:", {name: splits[name]["case"].map(lambda case: str(case).rsplit("-", 1)[0]).nunique() for name in splits})

    train_dataset = MemoryMappedPatchDataset(records_from_table(splits["train"]), patch_size=args.patch_size, training=True, whole_volume=args.whole_volume)
    val_dataset = MemoryMappedPatchDataset(records_from_table(splits["val"]), patch_size=args.patch_size, training=False, noise_std=0.0, whole_volume=args.whole_volume)
    test_dataset = MemoryMappedPatchDataset(records_from_table(splits["test"]), patch_size=args.patch_size, training=False, noise_std=0.0, whole_volume=args.whole_volume)
    sampler = BalancedBatchSampler(
        splits["train"]["grade_proxy"].tolist(),
        args.batch_size,
        seed=args.seed,
        steps_per_epoch=args.steps_per_epoch or None,
    )
    sample_batch = next(iter(sampler))
    sample_labels = splits["train"].iloc[sample_batch]["grade_proxy"].value_counts().to_dict()
    if sample_labels.get(0) != args.batch_size // 2 or sample_labels.get(1) != args.batch_size // 2:
        raise AssertionError(f"Sampler produced an unbalanced batch: {sample_labels}")
    loader_kwargs = {"num_workers": 0, "pin_memory": True}
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = TinyGradeClassifier3D(base_channels=args.base_channels, dropout=args.dropout).to(device)
    model = model.to(memory_format=torch.channels_last_3d)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {torch.cuda.get_device_name(device)} | model parameters: {parameter_count:,}")
    # Batches are already 1:1 balanced; applying effective-number weights here would
    # double-count the minority and was a likely source of the earlier oscillation.
    criterion = BinaryFocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing).to(device)
    initial_lr = args.lr / args.warmup_epochs if args.warmup_epochs else args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = args.output_dir / "best_checkpoint.pth"
    best_prediction_path = args.output_dir / "best_validation_predictions.csv"
    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = warmup_lr
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        train_loss, train_metrics = run_epoch(model, train_loader, sampler, optimizer, scaler, criterion, device, epoch=epoch, grad_accum=args.grad_accum)
        val_loss, val_fixed_metrics, val_predictions = evaluate(model, val_loader, criterion, device, progress_label=f"validation {epoch + 1}")
        threshold, val_selected_metrics = choose_validation_threshold(
            val_predictions["probability_high"].to_numpy(), val_predictions["true_label"].to_numpy()
        )
        val_selected_metrics["loss"] = val_loss
        memory = runtime_memory(device)
        if memory["reserved_gib"] > args.target_vram_gib:
            raise MemoryError(f"VRAM budget exceeded: {memory['reserved_gib']:.2f} GiB > {args.target_vram_gib:.2f} GiB")
        if memory["ram_gib"] > args.target_ram_gib:
            raise MemoryError(f"RAM budget exceeded: {memory['ram_gib']:.2f} GiB > {args.target_ram_gib:.2f} GiB")
        score = val_selected_metrics["balanced_accuracy"]
        if epoch >= args.warmup_epochs:
            scheduler.step(score)
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_loss,
            "val_accuracy_fixed_0_5": val_fixed_metrics["accuracy"],
            "val_balanced_accuracy": val_selected_metrics["balanced_accuracy"],
            "val_f1": val_selected_metrics["f1"],
            "val_sensitivity": val_selected_metrics["sensitivity"],
            "val_specificity": val_selected_metrics["specificity"],
            "val_auroc": val_selected_metrics["auroc"],
            "threshold": threshold,
            "predicted_positive_rate": val_selected_metrics["predicted_positive_rate"],
            "lr": optimizer.param_groups[0]["lr"],
            "peak_vram_gib": memory["reserved_gib"],
            "process_ram_gib": memory["ram_gib"],
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        print(
            f"epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f} val_bal_acc={score:.4f} "
            f"val_f1={record['val_f1']:.4f} sens={record['val_sensitivity']:.4f} spec={record['val_specificity']:.4f} "
            f"threshold={threshold:.2f} vram={memory['reserved_gib']:.2f}GiB ram={memory['ram_gib']:.2f}GiB {record['seconds']:.1f}s"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "architecture": {"base_channels": args.base_channels, "dropout": args.dropout, "patch_size": args.patch_size, "whole_volume": args.whole_volume},
                    "seed": args.seed,
                    "split_strategy": "subject-disjoint StratifiedGroupKFold",
                    "training": {
                        "batch_size": args.batch_size,
                        "grad_accum": args.grad_accum,
                        "learning_rate": args.lr,
                        "focal_gamma": args.focal_gamma,
                        "label_smoothing": args.label_smoothing,
                    },
                    "threshold": threshold,
                    "epoch": best_epoch,
                    "validation_metrics": val_selected_metrics,
                    "normalization": "preprocessed CTN volumes; no per-volume renormalization; patch-first memory-map loader",
                },
                str(checkpoint_path),
            )
            val_predictions.to_csv(best_prediction_path, index=False)
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    best_predictions = pd.read_csv(best_prediction_path)
    plot_diagnostics(history, best_predictions, args.output_dir, f"Validation evidence (best epoch {best_epoch})")
    summary: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": best_score,
        "checkpoint": str(checkpoint_path),
        "validation_predictions": str(best_prediction_path),
        "test_evaluated": False,
    }
    if args.evaluate_test:
        test_loss, test_metrics, test_predictions = evaluate(
            model, test_loader, criterion, device, threshold=float(checkpoint["threshold"]), progress_label="locked test"
        )
        test_metrics["loss"] = test_loss
        test_predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
        plot_diagnostics(history, test_predictions, args.output_dir / "test_evidence", "Locked test evidence")
        summary.update(test_evaluated=True, test_metrics=test_metrics, test_predictions=str(args.output_dir / "test_predictions.csv"))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
