#!/usr/bin/env python3
"""Run inference with a repaired checkpoint and its recorded input mode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.grade_data import CaseRecord, EXPECTED_SHAPE, MemoryMappedPatchDataset
from src.grade_model import TinyGradeClassifier3D


def load_checkpoint(path: Path, device: torch.device) -> tuple[TinyGradeClassifier3D, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not {"state_dict", "architecture"}.issubset(checkpoint):
        raise ValueError("Checkpoint is not from the repaired trainer; use a repaired best_checkpoint.pth")
    architecture = checkpoint["architecture"]
    model = TinyGradeClassifier3D(
        base_channels=int(architecture["base_channels"]),
        dropout=float(architecture["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


def load_model(checkpoint_path: Path, device: torch.device) -> TinyGradeClassifier3D:
    """Compatibility wrapper returning only the canonical model."""
    return load_checkpoint(Path(checkpoint_path), device)[0]


def preprocess_npy(npy_path: Path, target_shape: tuple[int, int, int] = (96, 96, 96), *, whole_volume: bool = False) -> torch.Tensor:
    path = Path(npy_path)
    array = np.load(path, mmap_mode="r")
    if tuple(array.shape) != EXPECTED_SHAPE or array.dtype != np.float32:
        raise ValueError(f"{path} must be float32 with shape {EXPECTED_SHAPE}; got {array.shape}, {array.dtype}")
    dataset = MemoryMappedPatchDataset(
        [CaseRecord(path.stem, 0, path)],
        patch_size=target_shape,
        training=False,
        noise_std=0.0,
        whole_volume=whole_volume,
    )
    return dataset[0][0].unsqueeze(0)


def predict(model: TinyGradeClassifier3D, input_tensor: torch.Tensor, device: torch.device, *, threshold: float = 0.5) -> tuple[float, int]:
    input_tensor = input_tensor.to(device).contiguous(memory_format=torch.channels_last_3d)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        probability = float(torch.sigmoid(model(input_tensor)).item())
    return probability, int(probability >= threshold)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/training/repaired_candidate/best_checkpoint.pth"))
    parser.add_argument("--case-id")
    parser.add_argument("--npy-file", type=Path)
    parser.add_argument("--npy-dir", type=Path, default=Path("data/brats_preprocessed/train"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if not args.case_id and not args.npy_file:
        parser.error("provide --case-id or --npy-file")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    path = args.npy_file or args.npy_dir / f"{args.case_id}.npy"
    architecture = checkpoint["architecture"]
    input_tensor = preprocess_npy(
        path,
        (int(architecture["patch_size"]),) * 3,
        whole_volume=bool(architecture.get("whole_volume", False)),
    )
    probability, prediction = predict(model, input_tensor, device, threshold=float(checkpoint.get("threshold", 0.5)))
    case_id = path.stem
    grade = "HIGH (GBM proxy)" if prediction else "LOW (LGG proxy)"
    print(f"CASE: {case_id}\nPREDICTION: {grade}\nPROBABILITY_HIGH: {probability:.4f}\nDEVICE: {device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
