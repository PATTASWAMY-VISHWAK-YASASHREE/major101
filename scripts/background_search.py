#!/usr/bin/env python3
"""Bounded, resumable search for a subject-disjoint 75% balanced-accuracy candidate."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cross_validate_repaired import bootstrap_metric_intervals
from scripts.train_ultra_light import evaluate, load_verified_report, run_epoch, runtime_memory, set_seed
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/search/ba75"))
    parser.add_argument("--npy-dir", type=Path, default=Path("data/brats_preprocessed/train"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/brats_preprocessed/labels.csv"))
    parser.add_argument("--data-report", type=Path, default=Path("outputs/data_quality/preprocessed_data_report.json"))
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--max-hours", type=float, default=30.0)
    parser.add_argument("--screen-folds", type=int, default=3)
    parser.add_argument("--screen-epochs", type=int, default=2)
    parser.add_argument("--screen-steps-per-epoch", type=int, default=16)
    parser.add_argument("--confirmation-epochs", type=int, default=10)
    parser.add_argument("--confirmation-steps-per-epoch", type=int, default=16)
    parser.add_argument("--target-balanced-accuracy", type=float, default=0.75)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--locked-split-seed", type=int, default=42)
    parser.add_argument("--model-seed-start", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--target-vram-gib", type=float, default=2.0)
    parser.add_argument("--target-ram-gib", type=float, default=3.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    temporary.replace(path)


def table_for_cases(cases: pd.DataFrame, case_ids: list[str]) -> pd.DataFrame:
    by_case = cases.set_index("case")
    missing = sorted(set(case_ids) - set(by_case.index))
    if missing:
        raise AssertionError(f"Manifest contains unknown cases: {missing[:5]}")
    return by_case.loc[case_ids].reset_index()


def manifest_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"case": str(row.case), "subject": subject_id(row.case), "label": int(row.grade_proxy)}
        for row in frame.itertuples(index=False)
    ]


def build_search_manifest(cases: pd.DataFrame, *, split_seed: int, locked_split_seed: int, folds: int) -> tuple[dict[str, Any], dict[str, Any]]:
    locked = build_stratified_splits(cases, seed=locked_split_seed)
    development = pd.concat([locked["train"], locked["val"]], ignore_index=True)
    outer = build_stratified_splits(development, seed=split_seed, val_fraction=0.10, test_fraction=0.10)
    search_pool = pd.concat([outer["train"], outer["val"]], ignore_index=True)
    confirmation = outer["test"]
    search_folds = build_cross_validation_folds(search_pool, seed=split_seed, n_splits=folds)
    assert_case_disjoint({"search_pool": search_pool, "confirmation": confirmation, "locked_test": locked["test"]})
    assert_subject_disjoint({"search_pool": search_pool, "confirmation": confirmation, "locked_test": locked["test"]})
    manifest = {
        "split_seed": split_seed,
        "locked_split_seed": locked_split_seed,
        "locked_test": manifest_rows(locked["test"]),
        "search_pool": manifest_rows(search_pool),
        "confirmation": manifest_rows(confirmation),
        "folds": {
            str(number): {"train": manifest_rows(split["train"]), "val": manifest_rows(split["val"])}
            for number, split in sorted(search_folds.items())
        },
    }
    partitions = {
        "locked": locked,
        "locked_test": locked["test"],
        "search_pool": search_pool,
        "confirmation": confirmation,
        "folds": search_folds,
    }
    return manifest, partitions


def load_search_manifest(cases: pd.DataFrame, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    search_pool = table_for_cases(cases, [row["case"] for row in manifest["search_pool"]])
    confirmation = table_for_cases(cases, [row["case"] for row in manifest["confirmation"]])
    locked_test = table_for_cases(cases, [row["case"] for row in manifest["locked_test"]])
    folds = {
        int(number): {
            "train": table_for_cases(cases, [row["case"] for row in split["train"]]),
            "val": table_for_cases(cases, [row["case"] for row in split["val"]]),
        }
        for number, split in manifest["folds"].items()
    }
    assert_case_disjoint({"search_pool": search_pool, "confirmation": confirmation, "locked_test": locked_test})
    assert_subject_disjoint({"search_pool": search_pool, "confirmation": confirmation, "locked_test": locked_test})
    expected = set(search_pool["case"])
    observed = [case for split in folds.values() for case in split["val"]["case"]]
    if len(observed) != len(expected) or set(observed) != expected or len(observed) != len(set(observed)):
        raise AssertionError("Search folds do not cover search_pool exactly once")
    return manifest, {"locked_test": locked_test, "search_pool": search_pool, "confirmation": confirmation, "folds": folds}


def candidate_configs(model_seed_start: int) -> list[dict[str, Any]]:
    values = itertools.product(
        (5e-5, 1e-4, 2e-4),
        (1e-5, 1e-4, 1e-3),
        (0.15, 0.25, 0.35),
        (0.0, 1.0),
        (0.0, 0.05),
    )
    return [
        {
            "attempt": index + 1,
            "seed": model_seed_start + index,
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "focal_gamma": focal_gamma,
            "label_smoothing": label_smoothing,
            "base_channels": 12,
            "patch_size": 96,
            "evaluation_views": 1,
        }
        for index, (lr, weight_decay, dropout, focal_gamma, label_smoothing) in enumerate(values)
    ]


def selection_score(metrics: dict[str, Any]) -> float:
    value = metrics.get("auroc")
    return float(value) if value is not None and np.isfinite(value) else float(metrics["balanced_accuracy"])


def make_model(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = TinyGradeClassifier3D(base_channels=int(config["base_channels"]), dropout=float(config["dropout"])).to(device)
    return model.to(memory_format=torch.channels_last_3d)


def save_checkpoint(path: Path, model: torch.nn.Module, config: dict[str, Any], *, epoch: int, metrics: dict[str, Any] | None) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": {"base_channels": config["base_channels"], "dropout": config["dropout"], "patch_size": config["patch_size"]},
            "seed": config["seed"],
            "epoch": epoch,
            "validation_metrics": metrics,
            "locked_test_evaluated": False,
        },
        str(path),
    )


def train_model(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame | None,
    config: dict[str, Any],
    output_dir: Path,
    *,
    epochs: int,
    steps_per_epoch: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Path, list[dict[str, Any]]]:
    set_seed(int(config["seed"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = MemoryMappedPatchDataset(records_from_table(train_frame), patch_size=args.patch_size, training=True)
    sampler = BalancedBatchSampler(
        train_frame["grade_proxy"].tolist(), args.batch_size, seed=int(config["seed"]), steps_per_epoch=steps_per_epoch
    )
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = None
    if val_frame is not None:
        val_dataset = MemoryMappedPatchDataset(records_from_table(val_frame), patch_size=args.patch_size, training=False, noise_std=0.0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    else:
        val_dataset = None
    model = make_model(config, device)
    criterion = BinaryFocalLoss(gamma=float(config["focal_gamma"]), label_smoothing=float(config["label_smoothing"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    checkpoint_path = output_dir / "best_checkpoint.pth"
    prediction_path = output_dir / "best_validation_predictions.csv"

    for epoch in range(epochs):
        warmup_epochs = min(3, epochs)
        if epoch < warmup_epochs:
            for group in optimizer.param_groups:
                group["lr"] = float(config["lr"]) * (epoch + 1) / warmup_epochs
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        train_loss, train_metrics = run_epoch(
            model, train_loader, sampler, optimizer, scaler, criterion, device, epoch=epoch, grad_accum=args.grad_accum
        )
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_auroc": train_metrics["auroc"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        if val_loader is not None:
            _, fixed_metrics, predictions = evaluate(
                model, val_loader, criterion, device, progress_label=f"search epoch {epoch + 1}", aggregate_cases=False
            )
            threshold, metrics = choose_validation_threshold(predictions["probability_high"].to_numpy(), predictions["true_label"].to_numpy())
            score = selection_score(metrics)
            record.update({"val_balanced_accuracy": metrics["balanced_accuracy"], "val_auroc": metrics["auroc"], "val_threshold": threshold, "val_accuracy_fixed_0_5": fixed_metrics["accuracy"]})
            if epoch >= warmup_epochs:
                scheduler.step(score)
            if score > best_score:
                best_score = score
                best_epoch = epoch + 1
                save_checkpoint(checkpoint_path, model, config, epoch=best_epoch, metrics=metrics)
                predictions.to_csv(prediction_path, index=False)
        else:
            save_checkpoint(checkpoint_path, model, config, epoch=epoch + 1, metrics=None)
        memory = runtime_memory(device)
        record.update({"peak_vram_gib": memory["reserved_gib"], "process_ram_gib": memory["ram_gib"]})
        if memory["reserved_gib"] > args.target_vram_gib or memory["ram_gib"] > args.target_ram_gib:
            raise MemoryError(f"Memory budget exceeded: {memory}")
        history.append(record)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    write_json(output_dir / "run_summary.json", {"epochs_requested": epochs, "epochs_completed": len(history), "best_epoch": best_epoch, "best_score": best_score, "locked_test_evaluated": False})
    del model, optimizer, scheduler, scaler, train_loader, train_dataset, val_loader, val_dataset
    torch.cuda.empty_cache()
    return checkpoint_path, history


def run_screen_attempt(config: dict[str, Any], partitions: dict[str, Any], args: argparse.Namespace, attempt_dir: Path, device: torch.device) -> dict[str, Any]:
    predictions: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold_number, split in sorted(partitions["folds"].items()):
        fold_dir = attempt_dir / f"fold_{fold_number:02d}"
        checkpoint, history = train_model(
            split["train"], split["val"], config, fold_dir, epochs=args.screen_epochs, steps_per_epoch=args.screen_steps_per_epoch, args=args, device=device
        )
        fold_predictions = pd.read_csv(fold_dir / "best_validation_predictions.csv")
        predictions.append(fold_predictions)
        fold_summaries.append({"fold": fold_number, "validation_case_count": len(split["val"]), "best_epoch": int(json.loads((fold_dir / "run_summary.json").read_text())["best_epoch"]), "checkpoint": str(checkpoint), "epochs_completed": len(history)})
    oof = pd.concat(predictions, ignore_index=True)
    expected_cases = set(partitions["search_pool"]["case"])
    if len(oof) != len(expected_cases) or oof["case"].duplicated().any() or set(oof["case"]) != expected_cases:
        raise AssertionError("Search OOF predictions do not cover search_pool exactly once")
    oof.to_csv(attempt_dir / "search_oof_predictions.csv", index=False)
    threshold, selected_metrics = choose_validation_threshold(oof["probability_high"].to_numpy(), oof["true_label"].to_numpy())
    summary = {
        "attempt": config["attempt"],
        "config": config,
        "search_pool_case_count": len(partitions["search_pool"]),
        "oof_case_count": len(oof),
        "complete_oof": True,
        "threshold": threshold,
        "metrics_at_threshold_0_5": binary_metrics(oof["probability_high"].to_numpy(), oof["true_label"].to_numpy(), threshold=0.5),
        "metrics_at_selected_threshold": selected_metrics,
        "bootstrap_95ci": bootstrap_metric_intervals(oof["probability_high"].to_numpy(), oof["true_label"].to_numpy(), threshold=threshold, seed=int(config["seed"])),
        "folds": fold_summaries,
        "locked_test_evaluated": False,
    }
    write_json(attempt_dir / "summary.json", summary)
    return summary


def run_confirmation(config: dict[str, Any], partitions: dict[str, Any], args: argparse.Namespace, attempt_dir: Path, threshold: float, device: torch.device) -> dict[str, Any]:
    confirmation_dir = attempt_dir / "confirmation"
    checkpoint, history = train_model(
        partitions["search_pool"], None, config, confirmation_dir, epochs=args.confirmation_epochs, steps_per_epoch=args.confirmation_steps_per_epoch, args=args, device=device
    )
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    model = make_model(config, device)
    model.load_state_dict(checkpoint_data["state_dict"])
    dataset = MemoryMappedPatchDataset(records_from_table(partitions["confirmation"]), patch_size=args.patch_size, training=False, noise_std=0.0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    criterion = BinaryFocalLoss(gamma=float(config["focal_gamma"]), label_smoothing=float(config["label_smoothing"])).to(device)
    _, metrics, predictions = evaluate(model, loader, criterion, device, progress_label="confirmation", threshold=threshold)
    predictions.to_csv(confirmation_dir / "predictions.csv", index=False)
    result = {"threshold_from_search_oof": threshold, "metrics": metrics, "epochs_completed": len(history), "checkpoint": str(checkpoint), "locked_test_evaluated": False}
    write_json(confirmation_dir / "summary.json", result)
    del model, loader, dataset, criterion
    torch.cuda.empty_cache()
    return result


def pid_alive(pid: int) -> bool:
    return pid > 0 and psutil.pid_exists(pid)


def acquire_lock(output_dir: Path) -> Path:
    lock_path = output_dir / "search.lock"
    if lock_path.is_file():
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        try:
            existing_pid = int(existing.get("pid", -1))
        except (TypeError, ValueError):
            existing_pid = -1
        if pid_alive(existing_pid):
            raise RuntimeError(f"Another search process is active: PID {existing_pid}")
        lock_path.unlink()
    write_json(lock_path, {"pid": os.getpid(), "started": time.time(), "command": sys.argv})
    return lock_path


def main() -> int:
    args = parse_args()
    if args.max_attempts < 0 or args.max_hours <= 0 or args.screen_folds < 2 or args.screen_epochs <= 0 or args.screen_steps_per_epoch <= 0:
        raise ValueError("attempts may be zero for dry-run; hours, folds, epochs, and steps must be positive")
    if args.batch_size < 2 or args.batch_size % 2 or args.patch_size > 96:
        raise ValueError("batch-size must be even and patch-size must be <=96")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_verified_report(args.data_report, npy_dir=args.npy_dir, labels_csv=args.labels_csv)
    cases = load_case_table(args.labels_csv, args.npy_dir)
    manifest_path = args.output_dir / "search_manifest.json"
    if manifest_path.is_file():
        manifest, partitions = load_search_manifest(cases, manifest_path)
    else:
        manifest, partitions = build_search_manifest(cases, split_seed=args.split_seed, locked_split_seed=args.locked_split_seed, folds=args.screen_folds)
        write_json(manifest_path, manifest)
    configs = candidate_configs(args.model_seed_start)
    write_json(args.output_dir / "run_config.json", vars(args))
    write_json(args.output_dir / "candidate_configs.json", {"count": len(configs), "configs": configs})
    if args.dry_run:
        print(json.dumps({"manifest": str(manifest_path), "search_pool_cases": len(partitions["search_pool"]), "confirmation_cases": len(partitions["confirmation"]), "candidate_count": len(configs), "locked_test_cases": len(partitions["locked_test"]), "locked_test_evaluated": False}, indent=2))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for background search")
    lock_path = acquire_lock(args.output_dir)
    device = torch.device("cuda")
    started = time.monotonic()
    deadline = started + args.max_hours * 3600
    stop_reason = "attempt_limit"
    best: dict[str, Any] | None = None
    status = {"state": "running", "started": time.time(), "pid": os.getpid(), "attempts_completed": 0, "max_attempts": args.max_attempts, "max_hours": args.max_hours, "target_balanced_accuracy": args.target_balanced_accuracy, "locked_test_evaluated": False}
    write_json(args.output_dir / "status.json", status)
    try:
        for config in configs[: args.max_attempts]:
            if time.monotonic() >= deadline:
                stop_reason = "time_limit"
                break
            attempt_dir = args.output_dir / f"attempt_{config['attempt']:04d}"
            summary_path = attempt_dir / "summary.json"
            if args.resume and summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                try:
                    summary = run_screen_attempt(config, partitions, args, attempt_dir, device)
                except Exception as exc:
                    summary = {"attempt": config["attempt"], "config": config, "state": "failed", "error": f"{type(exc).__name__}: {exc}", "locked_test_evaluated": False}
                    write_json(attempt_dir / "summary.json", summary)
            if summary.get("state") == "failed":
                status["attempts_completed"] = config["attempt"]
                status["last_error"] = summary["error"]
                write_json(args.output_dir / "status.json", status)
                continue
            status["attempts_completed"] = config["attempt"]
            status["last_attempt"] = summary
            if best is None or summary["metrics_at_selected_threshold"]["balanced_accuracy"] > best["metrics_at_selected_threshold"]["balanced_accuracy"]:
                best = summary
                status["best_attempt"] = config["attempt"]
            write_json(args.output_dir / "status.json", status)
            if summary["metrics_at_selected_threshold"]["balanced_accuracy"] >= args.target_balanced_accuracy:
                if time.monotonic() >= deadline:
                    stop_reason = "time_limit"
                    break
                confirmation = run_confirmation(config, partitions, args, attempt_dir, float(summary["threshold"]), device)
                status["confirmation"] = confirmation
                write_json(args.output_dir / "status.json", status)
                if confirmation["metrics"]["balanced_accuracy"] >= args.target_balanced_accuracy:
                    winner = {"state": "succeeded", "attempt": config["attempt"], "search": summary, "confirmation": confirmation, "locked_test_evaluated": False}
                    write_json(args.output_dir / "winner.json", winner)
                    status.update({"state": "succeeded", "finished": time.time(), "winner_attempt": config["attempt"]})
                    write_json(args.output_dir / "status.json", status)
                    return 0
        status.update({"state": "exhausted", "finished": time.time(), "best_attempt": best.get("attempt") if best else None, "stop_reason": stop_reason})
        write_json(args.output_dir / "status.json", status)
        return 0
    finally:
        if lock_path.is_file():
            lock_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
