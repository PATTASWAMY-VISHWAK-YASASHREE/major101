"""Small 3D image-only classifier and binary evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score, roc_auc_score


def _groups(channels: int) -> int:
    for groups in (8, 6, 4, 3, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class TinyGradeClassifier3D(nn.Module):
    """GroupNorm 3D CNN sized for 64^3 patches on a 2 GB VRAM budget."""

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 12,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, widths[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(widths[0]), widths[0]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.blocks = nn.Sequential(
            self._block(widths[0], widths[1]),
            self._block(widths[1], widths[2]),
            self._block(widths[2], widths[3]),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(widths[3], widths[3]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(widths[3], 1),
        )
        self.reset_parameters()

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu")
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.blocks(self.stem(inputs)))).squeeze(-1)


class BinaryFocalLoss(nn.Module):
    """Correct per-sample focal BCE; optional class weights are never broadcast across samples."""

    def __init__(
        self,
        *,
        gamma: float = 0.0,
        label_smoothing: float = 0.0,
        class_weights: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if not 0 <= label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().reshape_as(logits)
        smoothed = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = functional.binary_cross_entropy_with_logits(logits, smoothed, reduction="none")
        probabilities = torch.sigmoid(logits)
        pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        focal_weight = (1.0 - pt).pow(self.gamma)
        sample_weight = torch.where(targets > 0.5, self.class_weights[1], self.class_weights[0])
        return (bce * focal_weight * sample_weight).mean()


def binary_metrics(probabilities: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Return explicit binary metrics, including collapse-detecting class counts."""
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predictions = (probabilities >= threshold).astype(np.int64)
    tn = int(np.sum((predictions == 0) & (targets == 0)))
    fp = int(np.sum((predictions == 1) & (targets == 0)))
    fn = int(np.sum((predictions == 0) & (targets == 1)))
    tp = int(np.sum((predictions == 1) & (targets == 1)))
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    result: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(targets) if len(targets) else 0.0,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "predicted_positive_rate": float(predictions.mean()) if len(predictions) else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    if len(np.unique(targets)) == 2:
        result["auroc"] = float(roc_auc_score(targets, probabilities))
        result["average_precision"] = float(average_precision_score(targets, probabilities))
    else:
        result["auroc"] = None
        result["average_precision"] = None
    return result


def choose_validation_threshold(probabilities: np.ndarray, targets: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Select a threshold on validation data only, prioritising balanced accuracy then F1."""
    candidates = np.linspace(0.05, 0.95, 91)
    scored = [(binary_metrics(probabilities, targets, float(threshold)), float(threshold)) for threshold in candidates]
    best_metrics, best_threshold = max(
        scored,
        key=lambda item: (item[0]["balanced_accuracy"], item[0]["f1"], -abs(item[1] - 0.5)),
    )
    return best_threshold, best_metrics
