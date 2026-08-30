#!/usr/bin/env python3
"""Evaluate one repaired checkpoint on the locked unseen test partition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate import apply_temperature
from scripts.train_ultra_light import evaluate, load_verified_report
from src.grade_data import MemoryMappedPatchDataset, build_stratified_splits, load_case_table, records_from_table
from src.grade_model import BinaryFocalLoss, TinyGradeClassifier3D, binary_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--npy-dir", type=Path, default=Path("data/brats_preprocessed/train"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/brats_preprocessed/labels.csv"))
    parser.add_argument("--data-report", type=Path, default=Path("outputs/data_quality/preprocessed_data_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation/repaired_test"))
    parser.add_argument("--calibration", type=Path, help="Optional development-only calibration.json; checkpoint metadata is used otherwise")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the final evaluation command")
    load_verified_report(args.data_report, npy_dir=args.npy_dir, labels_csv=args.labels_csv)
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    architecture = checkpoint["architecture"]
    seed = int(checkpoint.get("seed", 42))
    model = TinyGradeClassifier3D(
        base_channels=int(architecture["base_channels"]),
        dropout=float(architecture["dropout"]),
    ).to("cuda").to(memory_format=torch.channels_last_3d)
    model.load_state_dict(checkpoint["state_dict"])
    cases = load_case_table(args.labels_csv, args.npy_dir)
    splits = build_stratified_splits(cases, seed=seed)
    test_dataset = MemoryMappedPatchDataset(
        records_from_table(splits["test"]),
        patch_size=int(architecture["patch_size"]),
        training=False,
        noise_std=0.0,
        whole_volume=bool(architecture.get("whole_volume", False)),
    )
    loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    criterion = BinaryFocalLoss()
    loss, _, predictions = evaluate(
        model,
        loader,
        criterion,
        torch.device("cuda"),
        threshold=0.5,
        progress_label="locked unseen test",
    )
    calibration_path = args.calibration or Path(checkpoint.get("calibration", ""))
    temperature = float(checkpoint.get("temperature", 1.0))
    if args.calibration:
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
        temperature = float(calibration["temperature"])
    threshold = float(checkpoint.get("threshold", 0.5))
    predictions["probability_high"] = apply_temperature(predictions["probability_high"].to_numpy(), temperature)
    predictions["predicted_label"] = (predictions["probability_high"] >= threshold).astype(int)
    predictions["correct"] = (predictions["true_label"] == predictions["predicted_label"]).astype(int)
    metrics = binary_metrics(predictions["probability_high"].to_numpy(), predictions["true_label"].to_numpy(), threshold=threshold)
    metrics["loss"] = loss
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    summary = {
        "checkpoint": str(args.checkpoint),
        "split_strategy": checkpoint.get("split_strategy", "subject-disjoint StratifiedGroupKFold"),
        "seed": seed,
        "case_count": len(predictions),
        "loss": loss,
        "metrics": metrics,
        "temperature": temperature,
        "calibration": str(calibration_path) if calibration_path.is_file() else None,
        "predictions": str(args.output_dir / "test_predictions.csv"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
