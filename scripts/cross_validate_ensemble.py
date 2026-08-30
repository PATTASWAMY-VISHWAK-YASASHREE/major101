#!/usr/bin/env python3
"""Train independent short members on fixed folds and aggregate their OOF probabilities."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cross_validate_repaired import bootstrap_metric_intervals
from src.grade_data import build_cross_validation_folds, build_stratified_splits, load_case_table
from src.grade_model import binary_metrics, choose_validation_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cv/ensemble_5member_2ep_16steps"))
    parser.add_argument("--npy-dir", type=Path, default=Path("data/brats_preprocessed/train"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/brats_preprocessed/labels.csv"))
    parser.add_argument("--data-report", type=Path, default=Path("outputs/data_quality/preprocessed_data_report.json"))
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=16)
    parser.add_argument("--views", type=int, default=1, help="Validation crops per member; keep 1 to isolate ensemble diversity")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--member-seed", type=int, default=100)
    parser.add_argument("--target-vram-gib", type=float, default=2.0)
    parser.add_argument("--target-ram-gib", type=float, default=3.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.members < 2 or args.folds < 2 or args.epochs <= 0 or args.steps_per_epoch <= 0:
        raise ValueError("members/folds must be >=2; epochs and steps-per-epoch must be positive")
    if not 1 <= args.views <= 8:
        raise ValueError("views must be between 1 and 8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_case_table(args.labels_csv, args.npy_dir)
    locked_splits = build_stratified_splits(cases, seed=args.split_seed)
    development = pd.concat([locked_splits["train"], locked_splits["val"]], ignore_index=True)
    folds = build_cross_validation_folds(development, seed=args.split_seed, n_splits=args.folds)
    expected_cases = set(development["case"])
    member_oof: dict[int, list[pd.DataFrame]] = {member: [] for member in range(1, args.members + 1)}
    member_logs: list[dict[str, object]] = []

    for member in range(1, args.members + 1):
        model_seed = args.member_seed + member - 1
        for fold in sorted(folds):
            member_dir = args.output_dir / f"member_{member:02d}"
            fold_prediction_path = member_dir / f"fold_{fold:02d}" / "best_validation_predictions.csv"
            log_path = member_dir / f"fold_{fold:02d}.log"
            if not (args.resume and fold_prediction_path.is_file()):
                member_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "cross_validate_repaired.py"),
                    "--output-dir", str(member_dir),
                    "--npy-dir", str(args.npy_dir),
                    "--labels-csv", str(args.labels_csv),
                    "--data-report", str(args.data_report),
                    "--folds", str(args.folds),
                    "--fold-list", str(fold),
                    "--allow-incomplete",
                    "--epochs", str(args.epochs),
                    "--steps-per-epoch", str(args.steps_per_epoch),
                    "--evaluation-views", str(args.views),
                    "--patience", "0",
                    "--seed", str(model_seed),
                    "--split-seed", str(args.split_seed),
                    "--target-vram-gib", str(args.target_vram_gib),
                    "--target-ram-gib", str(args.target_ram_gib),
                ]
                with log_path.open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    raise RuntimeError(f"Member {member} fold {fold} failed; see {log_path}")
            prediction = pd.read_csv(fold_prediction_path)
            member_oof[member].append(prediction)
            member_logs.append({"member": member, "fold": fold, "seed": model_seed, "log": str(log_path)})
            print(f"member {member}/{args.members}, fold {fold}/{args.folds} ready")

    all_member_frames: dict[int, pd.DataFrame] = {}
    for member, frames in member_oof.items():
        combined = pd.concat(frames, ignore_index=True)
        if set(combined["case"]) != expected_cases or combined["case"].duplicated().any():
            raise AssertionError(f"Member {member} does not have complete development OOF coverage")
        all_member_frames[member] = combined.sort_values("case").reset_index(drop=True)

    member_columns = [f"member_{member:02d}" for member in all_member_frames]
    pooled = all_member_frames[1][["case", "true_label"]].copy()
    for member, frame in all_member_frames.items():
        pooled[f"member_{member:02d}"] = frame["probability_high"].to_numpy()
    pooled["probability_high"] = pooled[member_columns].mean(axis=1)
    threshold, selected = choose_validation_threshold(pooled["probability_high"].to_numpy(), pooled["true_label"].to_numpy())
    pooled["predicted_label"] = (pooled["probability_high"] >= threshold).astype(int)
    pooled["correct"] = (pooled["predicted_label"] == pooled["true_label"]).astype(int)
    pooled.to_csv(args.output_dir / "out_of_fold_ensemble_predictions.csv", index=False)

    member_metrics = {}
    for member, frame in all_member_frames.items():
        member_threshold, metrics = choose_validation_threshold(frame["probability_high"].to_numpy(), frame["true_label"].to_numpy())
        member_metrics[f"member_{member:02d}"] = {"threshold": member_threshold, "metrics": metrics}
    probabilities = pooled[member_columns].to_numpy()
    pairwise_correlations = [float(np.corrcoef(probabilities[:, i], probabilities[:, j])[0, 1]) for i in range(args.members) for j in range(i + 1, args.members)]
    hard_predictions = probabilities >= 0.5
    errors = hard_predictions != pooled["true_label"].to_numpy()[:, None]
    error_overlap = [float(np.sum(errors[:, i] & errors[:, j]) / max(1, np.sum(errors[:, i] | errors[:, j]))) for i in range(args.members) for j in range(i + 1, args.members)]
    summary = {
        "split_strategy": "development-only subject-disjoint StratifiedGroupKFold",
        "members": args.members,
        "folds": args.folds,
        "epochs_per_member": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "evaluation_views": args.views,
        "split_seed": args.split_seed,
        "member_seed_start": args.member_seed,
        "development_case_count": len(development),
        "out_of_fold_case_count": len(pooled),
        "complete_oof": True,
        "member_metrics": member_metrics,
        "ensemble_metrics_at_threshold_0_5": binary_metrics(pooled["probability_high"].to_numpy(), pooled["true_label"].to_numpy(), threshold=0.5),
        "ensemble_threshold": threshold,
        "ensemble_metrics_at_selected_threshold": selected,
        "ensemble_bootstrap_95ci": bootstrap_metric_intervals(pooled["probability_high"].to_numpy(), pooled["true_label"].to_numpy(), threshold=threshold, seed=args.split_seed),
        "pairwise_probability_correlation": {"mean": float(np.mean(pairwise_correlations)), "min": float(np.min(pairwise_correlations)), "max": float(np.max(pairwise_correlations))},
        "pairwise_error_overlap_jaccard": {"mean": float(np.mean(error_overlap)), "min": float(np.min(error_overlap)), "max": float(np.max(error_overlap))},
        "member_logs": member_logs,
        "locked_test_case_count": len(locked_splits["test"]),
        "locked_test_evaluated": False,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
