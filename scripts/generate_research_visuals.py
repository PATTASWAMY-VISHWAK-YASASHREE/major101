#!/usr/bin/env python3
"""Generate slice, Grad-CAM, saliency, and prediction evidence for a repaired checkpoint."""

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
import torch.nn.functional as functional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate import apply_temperature
from src.grade_data import EXPECTED_SHAPE, MemoryMappedPatchDataset
from src.grade_model import TinyGradeClassifier3D


MODALITY_NAMES = ("T1ce", "T1n", "T2f", "T2w")


def center_start(patch_size: int) -> tuple[int, int, int]:
    return tuple((size - patch_size) // 2 for size in EXPECTED_SHAPE[1:])


def load_checkpoint(path: Path, device: torch.device) -> tuple[TinyGradeClassifier3D, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "state_dict" not in checkpoint or "architecture" not in checkpoint:
        raise ValueError("Expected a repaired checkpoint dictionary with state_dict and architecture")
    architecture = checkpoint["architecture"]
    model = TinyGradeClassifier3D(
        base_channels=int(architecture["base_channels"]), dropout=float(architecture["dropout"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def compute_explanations(model: TinyGradeClassifier3D, patch: torch.Tensor, device: torch.device, threshold: float, temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Return patch-local Grad-CAM and saliency, targeting the predicted class."""
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_inputs, grad_outputs):
        gradients.append(grad_outputs[0])

    target_layer = model.blocks[-1]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    inputs = patch.unsqueeze(0).to(device).contiguous(memory_format=torch.channels_last_3d)
    inputs.requires_grad_(True)
    logits = model(inputs)
    probability = float(apply_temperature([float(torch.sigmoid(logits).detach().cpu())], temperature)[0])
    prediction = int(probability >= threshold)
    target = logits if prediction else -logits
    model.zero_grad(set_to_none=True)
    target.backward()

    activation = activations[0]
    gradient = gradients[0]
    weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
    cam = functional.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = functional.interpolate(cam, size=patch.shape[1:], mode="trilinear", align_corners=False)
    cam = cam.squeeze().detach().float().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    saliency = inputs.grad.detach().abs().amax(dim=1).squeeze().float().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    forward_handle.remove()
    backward_handle.remove()
    return cam, saliency, probability, prediction


def select_examples(predictions: pd.DataFrame, maximum: int) -> pd.DataFrame:
    """Cover correct and incorrect predictions from both true classes before filling slots."""
    picks = []
    for label in (0, 1):
        for correct in (1, 0):
            subset = predictions[(predictions["true_label"] == label) & (predictions["correct"] == correct)]
            if not subset.empty:
                confidence = np.abs(subset["probability_high"] - 0.5)
                picks.append(subset.loc[confidence.idxmax()])
    selected = pd.DataFrame(picks).drop_duplicates(subset="case") if picks else predictions.iloc[0:0]
    if len(selected) < maximum:
        remaining = predictions[~predictions["case"].isin(selected["case"])]
        selected = pd.concat([selected, remaining.head(maximum - len(selected))], ignore_index=True)
    return selected.head(maximum).reset_index(drop=True)


def save_case_visuals(
    case_id: str,
    true_label: int,
    recorded_probability: float,
    volume_path: Path,
    model: TinyGradeClassifier3D,
    patch_size: int,
    device: torch.device,
    output_dir: Path,
    threshold: float,
    temperature: float,
    whole_volume: bool,
) -> dict:
    volume = np.load(volume_path, mmap_mode="r")
    if tuple(volume.shape) != EXPECTED_SHAPE or volume.dtype != np.float32:
        raise ValueError(f"{volume_path} does not match the verified preprocessing contract")
    if whole_volume:
        patch = torch.nn.functional.interpolate(
            torch.from_numpy(np.array(volume, dtype=np.float32, copy=True)).unsqueeze(0),
            size=(patch_size, patch_size, patch_size),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).contiguous()
        cam, saliency, recomputed_probability, predicted_label = compute_explanations(model, patch, device, threshold, temperature)
        cam = torch.nn.functional.interpolate(
            torch.from_numpy(cam).unsqueeze(0).unsqueeze(0),
            size=EXPECTED_SHAPE[1:],
            mode="trilinear",
            align_corners=False,
        ).squeeze().numpy()
        saliency = torch.nn.functional.interpolate(
            torch.from_numpy(saliency).unsqueeze(0).unsqueeze(0),
            size=EXPECTED_SHAPE[1:],
            mode="trilinear",
            align_corners=False,
        ).squeeze().numpy()
        patch_np = np.asarray(volume)
        z, y, x = 0, 0, 0
    else:
        z, y, x = MemoryMappedPatchDataset([], patch_size=patch_size, training=False)._random_foreground_start(volume)
        patch_np = np.array(volume[:, z:z + patch_size, y:y + patch_size, x:x + patch_size], copy=True, dtype=np.float32)
        patch = torch.from_numpy(patch_np)
        cam, saliency, recomputed_probability, predicted_label = compute_explanations(model, patch, device, threshold, temperature)
    cam_indices = (
        z + int(np.argmax(cam.sum(axis=(1, 2)))),
        y + int(np.argmax(cam.sum(axis=(0, 2)))),
        x + int(np.argmax(cam.sum(axis=(0, 1)))),
    )

    fig, axes = plt.subplots(4, 3, figsize=(12, 15), constrained_layout=True)
    for modality in range(4):
        plots = (
            volume[modality, cam_indices[0], :, :],
            volume[modality, :, cam_indices[1], :],
            volume[modality, :, :, cam_indices[2]],
        )
        for column, image in enumerate(plots):
            axes[modality, column].imshow(image.T, cmap="gray", origin="lower")
            axes[modality, column].set_title(f"{MODALITY_NAMES[modality]} | view {column + 1}")
            axes[modality, column].axis("off")
    fig.suptitle(f"{case_id} | true={'HIGH' if true_label else 'LOW'} | p(high)={recomputed_probability:.3f}")
    fig.savefig(output_dir / f"{case_id}_modalities.png", dpi=180)
    plt.close(fig)

    local_indices = (cam_indices[0] - z, cam_indices[1] - y, cam_indices[2] - x)
    t1ce = patch_np[0]
    views = (
        (t1ce[local_indices[0], :, :], cam[local_indices[0], :, :], saliency[local_indices[0], :, :], "axial"),
        (t1ce[:, local_indices[1], :], cam[:, local_indices[1], :], saliency[:, local_indices[1], :], "coronal"),
        (t1ce[:, :, local_indices[2]], cam[:, :, local_indices[2]], saliency[:, :, local_indices[2]], "sagittal"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for column, (background, cam_view, saliency_view, name) in enumerate(views):
        axes[0, column].imshow(background.T, cmap="gray", origin="lower")
        image = axes[0, column].imshow(cam_view.T, cmap="jet", alpha=0.55, origin="lower", vmin=0, vmax=1)
        axes[0, column].set_title(f"Grad-CAM | {name}")
        axes[0, column].axis("off")
        axes[1, column].imshow(background.T, cmap="gray", origin="lower")
        axes[1, column].imshow(saliency_view.T, cmap="hot", alpha=0.55, origin="lower", vmin=0, vmax=1)
        axes[1, column].set_title(f"Input saliency | {name}")
        axes[1, column].axis("off")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7)
    fig.suptitle(f"{case_id} | predicted={'HIGH' if predicted_label else 'LOW'} | recorded p(high)={recorded_probability:.3f}")
    fig.savefig(output_dir / f"{case_id}_attention.png", dpi=180)
    plt.close(fig)

    result = {
        "case": case_id,
        "true_label": int(true_label),
        "recorded_probability_high": float(recorded_probability),
        "recomputed_model_input_probability_high": recomputed_probability,
        "predicted_label": predicted_label,
        "temperature": temperature,
        "crop_start": None if whole_volume else [z, y, x],
        "whole_volume_input": whole_volume,
        "attention_peak": list(cam_indices),
        "modalities_figure": f"{case_id}_modalities.png",
        "attention_figure": f"{case_id}_attention.png",
    }
    (output_dir / f"{case_id}_evidence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--npy-dir", type=Path, default=Path("data/brats_preprocessed/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/explainability"))
    parser.add_argument("--max-cases", type=int, default=4)
    args = parser.parse_args()
    if args.max_cases <= 0:
        parser.error("--max-cases must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    predictions = pd.read_csv(args.predictions)
    required = {"case", "true_label", "probability_high", "correct"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing columns: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = select_examples(predictions, args.max_cases)
    evidence = []
    for row in examples.itertuples(index=False):
        volume_path = args.npy_dir / f"{row.case}.npy"
        evidence.append(
            save_case_visuals(
                row.case,
                int(row.true_label),
                float(row.probability_high),
                volume_path,
                model,
                int(checkpoint["architecture"]["patch_size"]),
                device,
                args.output_dir,
                float(checkpoint.get("threshold", 0.5)),
                float(checkpoint.get("temperature", 1.0)),
                bool(checkpoint["architecture"].get("whole_volume", False)),
            )
        )
        print(f"saved research visuals for {row.case}")
    pd.DataFrame(evidence).to_csv(args.output_dir / "visual_evidence_manifest.csv", index=False)
    (args.output_dir / "visual_evidence_manifest.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    (args.output_dir / "visual_run.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "predictions": str(args.predictions),
                "threshold": float(checkpoint.get("threshold", 0.5)),
                "temperature": float(checkpoint.get("temperature", 1.0)),
                "architecture": checkpoint["architecture"],
                "split_strategy": "subject-disjoint StratifiedGroupKFold",
                "interpretation": "Qualitative research evidence only; Grad-CAM and saliency are not clinical validation.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
