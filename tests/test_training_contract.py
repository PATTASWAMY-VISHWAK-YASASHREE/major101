"""Fast regression checks for the memory-constrained BraTS training path."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ULTRA_PATH = REPO_ROOT / "scripts" / "train_ultra_light.py"


def load_ultra_module():
    spec = importlib.util.spec_from_file_location("train_ultra_light_under_test", ULTRA_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {ULTRA_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UltraLightDataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ultra = load_ultra_module()

    def test_load_labels_deduplicates_repeated_case_ids(self) -> None:
        """A case must never appear twice or leak between train and validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            labels_csv = Path(temp_dir) / "labels.csv"
            with labels_csv.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["case", "grade_proxy"])
                writer.writeheader()
                writer.writerow({"case": "BraTS-GLI-001", "grade_proxy": 0})
                writer.writerow({"case": "BraTS-GLI-001", "grade_proxy": 0})
                writer.writerow({"case": "BraTS-GLI-002", "grade_proxy": 1})

            labels = self.ultra.load_labels(labels_csv)

        self.assertEqual(labels, [("BraTS-GLI-001", 0), ("BraTS-GLI-002", 1)])

    def test_load_labels_rejects_conflicting_duplicate_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            labels_csv = Path(temp_dir) / "labels.csv"
            with labels_csv.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["case", "grade_proxy"])
                writer.writeheader()
                writer.writerow({"case": "BraTS-GLI-001", "grade_proxy": 0})
                writer.writerow({"case": "BraTS-GLI-001", "grade_proxy": 1})

            with self.assertRaises(ValueError):
                self.ultra.load_labels(labels_csv)

    def test_balanced_sampler_requires_even_binary_batch_size(self) -> None:
        with self.assertRaises(ValueError):
            self.ultra.BalancedBatchSampler([0, 1, 0, 1], batch_size=3)

    def test_split_keeps_repeated_subject_acquisitions_together(self) -> None:
        from src.grade_data import assert_subject_disjoint, build_stratified_splits

        rows = []
        for subject in range(20):
            label = subject % 2
            for visit in (100, 101):
                rows.append({"case": f"BraTS-GLI-{subject:05d}-{visit}", "grade_proxy": label, "path": "unused"})
        splits = build_stratified_splits(pd.DataFrame(rows), seed=42)
        assert_subject_disjoint(splits)
        owners = {}
        for name, frame in splits.items():
            for case in frame["case"]:
                subject = case.rsplit("-", 1)[0]
                previous = owners.setdefault(subject, name)
                self.assertEqual(previous, name)

    def test_cross_validation_folds_are_subject_disjoint_and_cover_cases(self) -> None:
        from src.grade_data import build_cross_validation_folds, subject_id

        rows = []
        for subject in range(20):
            label = subject % 2
            for visit in (100, 101):
                rows.append({"case": f"BraTS-GLI-{subject:05d}-{visit}", "grade_proxy": label, "path": "unused"})
        cases = pd.DataFrame(rows)
        folds = build_cross_validation_folds(cases, seed=42, n_splits=5)

        validation_cases = []
        for fold in folds.values():
            train_subjects = {subject_id(case) for case in fold["train"]["case"]}
            val_subjects = {subject_id(case) for case in fold["val"]["case"]}
            self.assertTrue(train_subjects.isdisjoint(val_subjects))
            validation_cases.extend(fold["val"]["case"].tolist())
        self.assertEqual(sorted(validation_cases), sorted(cases["case"].tolist()))

    def test_binary_loss_applies_weights_per_sample(self) -> None:
        from src.grade_model import BinaryFocalLoss

        loss = BinaryFocalLoss(gamma=0, label_smoothing=0, class_weights=(2.0, 1.0))
        actual = loss(torch.zeros(2), torch.tensor([0.0, 1.0]))
        expected = torch.log(torch.tensor(2.0)) * 1.5
        self.assertTrue(torch.allclose(actual, expected))

    def test_multi_crop_predictions_are_averaged_per_case(self) -> None:
        predictions = pd.DataFrame(
            {
                "case": ["a", "a", "b"],
                "true_label": [1, 1, 0],
                "probability_high": [0.2, 0.8, 0.1],
                "predicted_label": [0, 1, 0],
                "correct": [0, 1, 1],
            }
        )
        actual = self.ultra.aggregate_case_predictions(predictions, threshold=0.5)
        self.assertEqual(actual["case"].tolist(), ["a", "b"])
        self.assertAlmostEqual(float(actual.loc[0, "probability_high"]), 0.5)
        self.assertEqual(actual["predicted_label"].tolist(), [1, 0])

    def test_cv_fold_list_is_explicit_and_unique(self) -> None:
        from scripts.cross_validate_repaired import parse_fold_list

        self.assertEqual(parse_fold_list("3, 4,5", [1, 2, 3, 4, 5]), [3, 4, 5])
        with self.assertRaises(ValueError):
            parse_fold_list("3,3", [1, 2, 3, 4, 5])

    def test_background_search_configs_are_deterministic_and_bounded(self) -> None:
        from scripts.background_search import candidate_configs

        configs = candidate_configs(10000)
        self.assertEqual(len(configs), 108)
        self.assertEqual(configs[0]["attempt"], 1)
        self.assertEqual(configs[-1]["attempt"], 108)
        self.assertEqual([config["seed"] for config in configs[:3]], [10000, 10001, 10002])
        self.assertEqual(configs, candidate_configs(10000))

    def test_background_search_manifest_is_disjoint_and_oof_complete(self) -> None:
        from scripts.background_search import build_search_manifest
        from src.grade_data import subject_id

        rows = []
        for subject in range(40):
            label = subject % 2
            for visit in (100, 101):
                rows.append({"case": f"BraTS-GLI-{subject:05d}-{visit}", "grade_proxy": label, "path": "unused"})
        cases = pd.DataFrame(rows)
        manifest, partitions = build_search_manifest(cases, split_seed=2026, locked_split_seed=42, folds=3)

        self.assertEqual(len(manifest["locked_test"]), len(partitions["locked_test"]))
        search_cases = set(partitions["search_pool"]["case"])
        confirmation_cases = set(partitions["confirmation"]["case"])
        locked_cases = set(partitions["locked_test"]["case"])
        self.assertTrue(search_cases.isdisjoint(confirmation_cases))
        self.assertTrue(search_cases.isdisjoint(locked_cases))
        self.assertTrue(confirmation_cases.isdisjoint(locked_cases))

        validation_cases = [case for fold in partitions["folds"].values() for case in fold["val"]["case"]]
        self.assertEqual(len(validation_cases), len(search_cases))
        self.assertEqual(set(validation_cases), search_cases)
        for fold in partitions["folds"].values():
            train_subjects = {subject_id(case) for case in fold["train"]["case"]}
            val_subjects = {subject_id(case) for case in fold["val"]["case"]}
            self.assertTrue(train_subjects.isdisjoint(val_subjects))

    def test_temperature_scaling_stays_in_probability_range(self) -> None:
        from scripts.calibrate import apply_temperature

        scaled = apply_temperature([0.1, 0.5, 0.9], 2.0)
        self.assertTrue(((scaled > 0) & (scaled < 1)).all())
        self.assertLess(float(scaled[0]), float(scaled[1]))
        self.assertLess(float(scaled[1]), float(scaled[2]))

    def test_raw_validation_crop_policy_is_bounded_and_deterministic(self) -> None:
        from scripts.infer_raw_validation_stream import fixed_crop_starts

        shape = (4, 182, 218, 182)
        starts = fixed_crop_starts(shape, patch_size=96, views=5)
        self.assertEqual(len(starts), 5)
        self.assertEqual(starts, fixed_crop_starts(shape, patch_size=96, views=5))
        self.assertEqual(starts[0], (43, 61, 43))
        self.assertEqual(len(set(starts)), 5)
        with self.assertRaises(ValueError):
            fixed_crop_starts(shape, patch_size=96, views=9)


if __name__ == "__main__":
    unittest.main()
