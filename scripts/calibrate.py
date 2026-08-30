#!/usr/bin/env python3
"""Fit development-only temperature scaling for repaired MRI predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.grade_model import binary_metrics, choose_validation_threshold


def probability_logits(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(probabilities).all() or ((probabilities <= 0) | (probabilities >= 1)).any():
        probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(probabilities / (1.0 - probabilities))


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    logits = probability_logits(probabilities) / float(temperature)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((np.asarray(probabilities) - np.asarray(targets)) ** 2))


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1] if index == bins - 1 else probabilities < edges[index + 1])
        if mask.any():
            error += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(targets[mask].mean()))
    return error


def fit_temperature(probabilities: np.ndarray, targets: np.ndarray) -> float:
    logits = torch.tensor(probability_logits(probabilities), dtype=torch.float64)
    labels = torch.tensor(np.asarray(targets, dtype=np.float64), dtype=torch.float64)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(1e-3, 100.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(1e-3, 100.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("outputs/cv/full_epoch_baseline_5fold_5ep/out_of_fold_predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/calibration/repaired"))
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    required = {"case", "true_label", "probability_high"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing columns: {sorted(missing)}")
    if predictions["case"].duplicated().any():
        raise ValueError("Calibration predictions must contain each development case once")
    targets = predictions["true_label"].to_numpy(dtype=np.int64)
    raw = predictions["probability_high"].to_numpy(dtype=np.float64)
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError("Calibration requires both binary classes")

    temperature = fit_temperature(raw, targets)
    calibrated = apply_temperature(raw, temperature)
    threshold, selected_metrics = choose_validation_threshold(calibrated, targets)
    raw_metrics = binary_metrics(raw, targets, threshold=0.5)
    calibrated_metrics = binary_metrics(calibrated, targets, threshold=0.5)
    raw_ece = expected_calibration_error(raw, targets)
    calibrated_ece = expected_calibration_error(calibrated, targets)
    raw_brier = brier_score(raw, targets)
    calibrated_brier = brier_score(calibrated, targets)

    observed_raw, expected_raw = calibration_curve(targets, raw, n_bins=min(10, len(predictions)), strategy="uniform")
    observed_calibrated, expected_calibrated = calibration_curve(targets, calibrated, n_bins=min(10, len(predictions)), strategy="uniform")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.5, 5))
    axis.plot(expected_raw, observed_raw, "o-", label="raw")
    axis.plot(expected_calibrated, observed_calibrated, "o-", label=f"temperature={temperature:.3f}")
    axis.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed positive rate", title="Development calibration")
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "reliability.png", dpi=180)
    plt.close(fig)

    payload = {
        "predictions": str(args.predictions),
        "case_count": len(predictions),
        "temperature": temperature,
        "threshold": threshold,
        "raw_metrics_at_threshold_0_5": raw_metrics,
        "calibrated_metrics_at_threshold_0_5": calibrated_metrics,
        "calibrated_metrics_at_selected_threshold": selected_metrics,
        "raw_brier": raw_brier,
        "calibrated_brier": calibrated_brier,
        "raw_ece": raw_ece,
        "calibrated_ece": calibrated_ece,
        "method": "single scalar temperature optimized on development out-of-fold logits",
    }
    (args.output_dir / "calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
