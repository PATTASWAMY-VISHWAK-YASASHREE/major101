#!/usr/bin/env python3
"""Run blind, one-case-at-a-time inference on raw BraTS validation MRI."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import psutil
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate import apply_temperature
from scripts.preprocess_brats import brain_mask, ctn_normalize
from src.grade_model import TinyGradeClassifier3D


MODALITIES = ("t1c", "t1n", "t2f", "t2w")
EXPECTED_SHAPE = (182, 218, 182)
ANCHORS = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)


def fixed_crop_starts(shape: tuple[int, int, int, int], patch_size: int, views: int) -> list[tuple[int, int, int]]:
    """Return the same center-plus-anchor crop policy used by validation."""
    if len(shape) != 4 or tuple(shape[1:]) != EXPECTED_SHAPE:
        raise ValueError(f"Expected four-channel volume with shape (4, 182, 218, 182), got {shape}")
    if not 1 <= views <= len(ANCHORS):
        raise ValueError("views must be between 1 and 8")
    if not 0 < patch_size <= min(EXPECTED_SHAPE):
        raise ValueError(f"patch_size must be in (0, {min(EXPECTED_SHAPE)}]")
    max_starts = [size - patch_size for size in EXPECTED_SHAPE]
    center = tuple((size - patch_size) // 2 for size in EXPECTED_SHAPE)
    starts = [center]
    starts.extend(
        tuple(maximum if side else 0 for maximum, side in zip(max_starts, anchor))
        for anchor in ANCHORS[: views - 1]
    )
    return starts


def case_directories(raw_dir: Path) -> list[Path]:
    cases = sorted(path for path in Path(raw_dir).iterdir() if path.is_dir())
    if not cases:
        raise FileNotFoundError(f"No case directories found under {raw_dir}")
    return cases


def modality_paths(case_dir: Path) -> dict[str, Path]:
    case = case_dir.name
    paths = {mod: case_dir / f"{case}-{mod}.nii.gz" for mod in MODALITIES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{case} is missing modalities: {missing}")
    return paths


def load_raw_case(case_dir: Path) -> np.ndarray:
    """Normalize one raw case without writing a persistent intermediate file."""
    paths = modality_paths(case_dir)
    t1n = nib.load(str(paths["t1n"])).get_fdata(dtype=np.float32)
    if tuple(t1n.shape) != EXPECTED_SHAPE:
        raise ValueError(f"{case_dir.name} t1n has shape {t1n.shape}, expected {EXPECTED_SHAPE}")
    mask = brain_mask(t1n)
    if not np.any(mask):
        raise ValueError(f"{case_dir.name} produced an empty T1n brain mask")

    normalized: list[np.ndarray] = []
    for modality in MODALITIES:
        volume = t1n if modality == "t1n" else nib.load(str(paths[modality])).get_fdata(dtype=np.float32)
        if tuple(volume.shape) != EXPECTED_SHAPE:
            raise ValueError(f"{case_dir.name} {modality} has shape {volume.shape}, expected {EXPECTED_SHAPE}")
        normalized.append(ctn_normalize(volume, mask).astype(np.float32, copy=False))
        if modality != "t1n":
            del volume
    result = np.ascontiguousarray(np.stack(normalized, axis=0), dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{case_dir.name} produced non-finite normalized values")
    del t1n, mask, normalized
    return result


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture", {})
    model = TinyGradeClassifier3D(
        base_channels=int(architecture.get("base_channels", 12)),
        dropout=float(architecture.get("dropout", 0.25)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(memory_format=torch.channels_last_3d)
    model.eval()
    return model, checkpoint


def infer_case(
    model: torch.nn.Module,
    volume: np.ndarray,
    *,
    device: torch.device,
    patch_size: int,
    views: int,
    temperature: float,
) -> dict[str, float | int]:
    logits: list[float] = []
    for z, y, x in fixed_crop_starts(volume.shape, patch_size, views):
        patch = np.ascontiguousarray(volume[:, z:z + patch_size, y:y + patch_size, x:x + patch_size])
        tensor = torch.from_numpy(patch).unsqueeze(0).to(device, memory_format=torch.channels_last_3d)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True):
            logit = float(model(tensor).float().item())
        logits.append(logit)
        del patch, tensor

    mean_logit = float(np.mean(logits))
    raw_probability = float(1.0 / (1.0 + np.exp(-np.clip(mean_logit, -60.0, 60.0))))
    calibrated_probability = float(apply_temperature(np.asarray([raw_probability]), temperature)[0])
    return {
        "raw_probability_high": raw_probability,
        "calibrated_probability_high": calibrated_probability,
        "view_count": views,
    }


def memory_snapshot(device: torch.device) -> dict[str, float]:
    process_ram = psutil.Process().memory_info().rss / (1024 ** 3)
    reserved_vram = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    return {"process_ram_gib": float(process_ram), "reserved_vram_gib": float(reserved_vram)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/brats2024/validation/validation_data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/training/repaired_final/best_checkpoint.pth"))
    parser.add_argument("--calibration", type=Path, default=Path("outputs/calibration/repaired/calibration.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/external_validation"))
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--views", type=int, default=5, help="Deterministic crops per case; 1 through 8")
    parser.add_argument("--max-cases", type=int, default=0, help="Process only the first N cases for a smoke run")
    parser.add_argument("--max-vram-gib", type=float, default=2.5)
    parser.add_argument("--max-ram-gib", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for raw validation inference")
    if args.max_cases < 0:
        raise ValueError("max-cases cannot be negative")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    cases = case_directories(args.raw_dir)
    if args.max_cases:
        cases = cases[: args.max_cases]
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    temperature = float(calibration["temperature"])
    threshold = float(calibration["threshold"])
    model, checkpoint = load_model(args.checkpoint, device)
    patch_size = int(checkpoint.get("architecture", {}).get("patch_size", args.patch_size))
    if patch_size != args.patch_size:
        raise ValueError(f"Checkpoint patch size is {patch_size}; pass --patch-size {patch_size}")
    fixed_crop_starts((4, *EXPECTED_SHAPE), patch_size, args.views)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.csv"
    errors: list[dict[str, str]] = []
    processed = 0
    peak = {"process_ram_gib": 0.0, "reserved_vram_gib": 0.0}
    started = time.perf_counter()
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "raw_probability_high", "calibrated_probability_high", "threshold", "predicted_label", "view_count"),
        )
        writer.writeheader()
        for case_dir in cases:
            try:
                volume = load_raw_case(case_dir)
                result = infer_case(
                    model,
                    volume,
                    device=device,
                    patch_size=patch_size,
                    views=args.views,
                    temperature=temperature,
                )
                result.update(
                    {
                        "case": case_dir.name,
                        "threshold": threshold,
                        "predicted_label": int(float(result["calibrated_probability_high"]) >= threshold),
                    }
                )
                writer.writerow(result)
                handle.flush()
                processed += 1
                current = memory_snapshot(device)
                for key in peak:
                    peak[key] = max(peak[key], current[key])
                if peak["reserved_vram_gib"] > args.max_vram_gib:
                    raise MemoryError(f"VRAM limit exceeded: {peak['reserved_vram_gib']:.2f} GiB > {args.max_vram_gib:.2f} GiB")
                if peak["process_ram_gib"] > args.max_ram_gib:
                    raise MemoryError(f"RAM limit exceeded: {peak['process_ram_gib']:.2f} GiB > {args.max_ram_gib:.2f} GiB")
                if processed % 10 == 0 or processed == len(cases):
                    print(f"[{processed}/{len(cases)}] {case_dir.name} p_high={float(result['calibrated_probability_high']):.4f}")
            except Exception as exc:
                errors.append({"case": case_dir.name, "error": f"{type(exc).__name__}: {exc}"})
                print(f"ERROR {case_dir.name}: {errors[-1]['error']}", file=sys.stderr)
            finally:
                if "volume" in locals():
                    del volume
                gc.collect()
                torch.cuda.empty_cache()

    summary = {
        "raw_dir": str(args.raw_dir),
        "checkpoint": str(args.checkpoint),
        "calibration": str(args.calibration),
        "processed_cases": processed,
        "requested_cases": len(cases),
        "failed_cases": errors,
        "views": args.views,
        "patch_size": patch_size,
        "temperature": temperature,
        "threshold": threshold,
        "peak_memory": peak,
        "elapsed_seconds": time.perf_counter() - started,
        "labels_available": False,
        "metrics_available": False,
        "persistent_processed_cache_created": False,
    }
    (args.output_dir / "validation_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "run_config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if errors:
        raise RuntimeError(f"Raw validation inference failed for {len(errors)} case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
