"""Leakage-safe, memory-bounded data utilities for BraTS grade experiments."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, Sampler


EXPECTED_SHAPE = (4, 182, 218, 182)
REQUIRED_LABEL_COLUMNS = {"case", "grade_proxy"}


@dataclass(frozen=True)
class CaseRecord:
    """One valid, unique labelled input volume."""

    case_id: str
    label: int
    path: Path


def subject_id(case: str) -> str:
    """Return the subject key shared by repeated acquisition case IDs."""
    prefix, suffix = str(case).rsplit("-", 1)
    return prefix if suffix.isdigit() else str(case)


def load_case_table(labels_csv: Path, npy_dir: Path) -> pd.DataFrame:
    """Return one valid label row per case and reject ambiguous training labels."""
    labels_csv = Path(labels_csv)
    npy_dir = Path(npy_dir)
    df = pd.read_csv(labels_csv)
    missing_columns = REQUIRED_LABEL_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{labels_csv} is missing columns: {sorted(missing_columns)}")

    df = df.copy()
    df["case"] = df["case"].astype(str).str.strip()
    df["grade_proxy"] = pd.to_numeric(df["grade_proxy"], errors="raise").astype(int)
    if df["case"].eq("").any():
        raise ValueError(f"{labels_csv} contains an empty case identifier")
    invalid_labels = sorted(set(df.loc[~df["grade_proxy"].isin([0, 1]), "grade_proxy"]))
    if invalid_labels:
        raise ValueError(f"grade_proxy must be binary; found {invalid_labels}")

    label_counts = df.groupby("case", sort=False)["grade_proxy"].nunique()
    conflicts = label_counts[label_counts > 1].index.tolist()
    if conflicts:
        preview = ", ".join(conflicts[:10])
        raise ValueError(f"Conflicting duplicate labels for {len(conflicts)} cases: {preview}")

    df = df.drop_duplicates(subset="case", keep="first").reset_index(drop=True)
    df["path"] = df["case"].map(lambda case: str(npy_dir / f"{case}.npy"))
    missing_files = df.loc[~df["path"].map(lambda value: Path(value).is_file()), "case"].tolist()
    if missing_files:
        preview = ", ".join(missing_files[:10])
        raise FileNotFoundError(f"Missing .npy files for {len(missing_files)} labelled cases: {preview}")
    return df[["case", "grade_proxy", "path"]].copy()


def build_stratified_splits(
    cases: pd.DataFrame,
    *,
    seed: int,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
) -> dict[str, pd.DataFrame]:
    """Create reproducible, subject-disjoint train/validation/test partitions.

    BraTS case IDs may contain multiple acquisitions (for example ``-100`` and
    ``-101``). Keeping those acquisitions together prevents a subject from
    appearing in both a fit and an evaluation partition.
    """
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be positive and sum to less than one")
    if cases["case"].duplicated().any():
        raise ValueError("Split input contains duplicate case identifiers")
    if not 0 < test_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("val_fraction and test_fraction must be positive")

    groups = cases["case"].map(subject_id).to_numpy()
    indices = np.arange(len(cases))

    def choose_group_fold(frame_indices: np.ndarray, frame_groups: np.ndarray, fraction: float, random_seed: int) -> tuple[np.ndarray, np.ndarray]:
        n_splits = max(2, min(len(np.unique(frame_groups)), round(1.0 / fraction)))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        target_size = len(frame_indices) * fraction
        target_positive = cases.iloc[frame_indices]["grade_proxy"].mean() * target_size
        candidates = list(splitter.split(frame_indices, cases.iloc[frame_indices]["grade_proxy"], frame_groups))
        _, local_test = min(
            candidates,
            key=lambda split: (
                abs(len(split[1]) - target_size),
                abs(cases.iloc[frame_indices[split[1]]]["grade_proxy"].sum() - target_positive),
            ),
        )
        test = frame_indices[local_test]
        train = np.setdiff1d(frame_indices, test, assume_unique=True)
        return train, test

    train_val_idx, test_idx = choose_group_fold(indices, groups, test_fraction, seed)
    train_val_groups = groups[train_val_idx]
    val_ratio_within_train_val = val_fraction / (1.0 - test_fraction)
    train_idx, val_idx = choose_group_fold(train_val_idx, train_val_groups, val_ratio_within_train_val, seed + 1)
    splits = {
        "train": cases.iloc[train_idx].reset_index(drop=True),
        "val": cases.iloc[val_idx].reset_index(drop=True),
        "test": cases.iloc[test_idx].reset_index(drop=True),
    }
    assert_case_disjoint(splits)
    assert_subject_disjoint(splits)
    return splits


def build_cross_validation_folds(
    cases: pd.DataFrame,
    *,
    seed: int,
    n_splits: int = 5,
) -> dict[int, dict[str, pd.DataFrame]]:
    """Build subject-disjoint stratified folds for development-only evaluation."""
    if cases["case"].duplicated().any():
        raise ValueError("Cross-validation input contains duplicate case identifiers")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    groups = cases["case"].map(subject_id).to_numpy()
    if n_splits > len(np.unique(groups)):
        raise ValueError("n_splits cannot exceed the number of unique subjects")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: dict[int, dict[str, pd.DataFrame]] = {}
    indices = np.arange(len(cases))
    for fold_number, (train_idx, val_idx) in enumerate(
        splitter.split(indices, cases["grade_proxy"], groups), start=1
    ):
        fold = {
            "train": cases.iloc[train_idx].reset_index(drop=True),
            "val": cases.iloc[val_idx].reset_index(drop=True),
        }
        assert_case_disjoint(fold)
        assert_subject_disjoint(fold)
        folds[fold_number] = fold
    return folds


def assert_case_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    """Raise if a case is present in more than one split."""
    combined = pd.concat(
        [frame.assign(_split=name) for name, frame in splits.items()], ignore_index=True
    )
    duplicated = combined[combined["case"].duplicated(keep=False)]
    if not duplicated.empty:
        examples = ", ".join(sorted(duplicated["case"].unique())[:10])
        raise AssertionError(f"Case leakage across splits: {examples}")


def assert_subject_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    """Raise if repeated acquisitions of one subject cross split boundaries."""
    owners: dict[str, str] = {}
    for split_name, frame in splits.items():
        for case in frame["case"]:
            subject = subject_id(case)
            previous = owners.setdefault(subject, split_name)
            if previous != split_name:
                raise AssertionError(f"Subject leakage across splits: {subject}")


def save_split_manifest(splits: dict[str, pd.DataFrame], path: Path, *, seed: int) -> None:
    """Persist the exact case assignment used by a run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "splits": {
            name: [
                {
                    "case": str(row.case),
                    "subject": str(row.case).rsplit("-", 1)[0],
                    "label": int(row.grade_proxy),
                }
                for row in frame.itertuples(index=False)
            ]
            for name, frame in splits.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def records_from_table(table: pd.DataFrame) -> list[CaseRecord]:
    return [
        CaseRecord(str(row.case), int(row.grade_proxy), Path(row.path))
        for row in table.itertuples(index=False)
    ]


def _crop_start_for_center(shape: Sequence[int], patch_size: Sequence[int]) -> tuple[int, int, int]:
    return tuple((size - patch) // 2 for size, patch in zip(shape, patch_size))


class MemoryMappedPatchDataset(Dataset):
    """Load one BraTS patch at a time without materialising the full 4-channel volume."""

    def __init__(
        self,
        records: Sequence[CaseRecord],
        *,
        patch_size: int | tuple[int, int, int] = 64,
        training: bool,
        crop_candidates: int = 8,
        noise_std: float = 0.01,
        whole_volume: bool = False,
        evaluation_views: bool = False,
    ) -> None:
        self.records = list(records)
        self.patch_size = (patch_size,) * 3 if isinstance(patch_size, int) else tuple(patch_size)
        self.training = training
        self.crop_candidates = crop_candidates
        self.noise_std = noise_std
        self.whole_volume = whole_volume
        self.evaluation_views = evaluation_views
        if len(self.patch_size) != 3 or any(value <= 0 for value in self.patch_size):
            raise ValueError(f"Invalid patch size: {self.patch_size}")
        if any(patch > size for patch, size in zip(self.patch_size, EXPECTED_SHAPE[1:])):
            raise ValueError(f"Patch {self.patch_size} exceeds volume shape {EXPECTED_SHAPE[1:]}")

    def __len__(self) -> int:
        if self.evaluation_views and not self.training:
            return len(self.records) * self.crop_candidates
        return len(self.records)

    def _random_foreground_start(self, volume: np.ndarray) -> tuple[int, int, int]:
        """Pick an informative candidate patch without allocating the volume."""
        spatial_shape = volume.shape[1:]
        max_starts = [size - patch for size, patch in zip(spatial_shape, self.patch_size)]
        candidates = [_crop_start_for_center(spatial_shape, self.patch_size)]
        if self.training:
            for _ in range(max(self.crop_candidates - 1, 0)):
                candidates.append(tuple(np.random.randint(0, maximum + 1) for maximum in max_starts))
        else:
            # Fixed anchors make validation/test deterministic while still
            # giving small enhancing regions a chance to enter the patch.
            anchors = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
            candidates.extend(
                tuple(maximum if side else 0 for maximum, side in zip(max_starts, anchor))
                for anchor in anchors[: max(self.crop_candidates - 1, 0)]
            )

        best_start = candidates[0]
        best_score = (-1, -1)
        for start in candidates:
            z, y, x = start
            dz, dy, dx = self.patch_size
            # CTN preprocessing leaves outside-brain T1n voxels at -1 rather than
            # zero. T1ce bright-voxel coverage is image-only and helps avoid
            # training on an arbitrary background patch when the proxy label is
            # defined by enhancing tumour presence.
            t1ce = volume[0, z:z + dz, y:y + dy, x:x + dx]
            t1n = volume[1, z:z + dz, y:y + dy, x:x + dx]
            score = (int(np.count_nonzero(t1ce > 0.25)), int(np.count_nonzero(t1n > -0.95)))
            if score > best_score:
                best_score = score
                best_start = start
        return best_start

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        view = None
        if self.evaluation_views and not self.training:
            record_index, view = divmod(index, self.crop_candidates)
            record = self.records[record_index]
        else:
            record = self.records[index]
        try:
            volume = np.load(record.path, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not memory-map {record.path}: {exc}") from exc
        if volume.shape != EXPECTED_SHAPE:
            raise ValueError(f"{record.path} has shape {volume.shape}, expected {EXPECTED_SHAPE}")
        if volume.dtype != np.float32:
            raise ValueError(f"{record.path} has dtype {volume.dtype}, expected float32")

        if self.whole_volume:
            # A read-only memmap cannot safely back a PyTorch tensor. Copy one
            # case, then immediately downsample it to the compact model input.
            # At 4x182x218x182 this bounded copy is about 116 MB.
            patch_tensor = functional.interpolate(
                torch.from_numpy(np.array(volume, dtype=np.float32, copy=True)).unsqueeze(0),
                size=self.patch_size,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0).contiguous()
        else:
            if view is None:
                start = self._random_foreground_start(volume)
            else:
                spatial_shape = volume.shape[1:]
                max_starts = [size - patch for size, patch in zip(spatial_shape, self.patch_size)]
                anchors = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
                starts = [_crop_start_for_center(spatial_shape, self.patch_size)]
                starts.extend(
                    tuple(maximum if side else 0 for maximum, side in zip(max_starts, anchor))
                    for anchor in anchors[: max(self.crop_candidates - 1, 0)]
                )
                start = starts[view]
            z, y, x = start
            dz, dy, dx = self.patch_size
            patch = np.array(volume[:, z:z + dz, y:y + dy, x:x + dx], dtype=np.float32, copy=True, order="C")
            if not np.isfinite(patch).all():
                raise ValueError(f"{record.path} has non-finite values in sampled patch {start}")
            patch_tensor = torch.from_numpy(patch)

        if not torch.isfinite(patch_tensor).all():
            raise ValueError(f"{record.path} has non-finite values in the model input")
        if self.training:
            for axis in (1, 2, 3):
                if np.random.random() < 0.5:
                    patch_tensor = torch.flip(patch_tensor, dims=(axis,))
            if self.noise_std > 0:
                patch_tensor = patch_tensor + torch.randn_like(patch_tensor) * self.noise_std

        return patch_tensor.contiguous(), torch.tensor(record.label, dtype=torch.float32), record.case_id


class BalancedBatchSampler(Sampler[list[int]]):
    """Yield equal low/high-grade batches and oversample only inside the training split."""

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        *,
        seed: int = 42,
        steps_per_epoch: int | None = None,
    ) -> None:
        self.labels = np.asarray(labels, dtype=np.int64)
        if batch_size < 2 or batch_size % 2:
            raise ValueError("BalancedBatchSampler requires an even batch size of at least 2")
        self.class_indices = {label: np.flatnonzero(self.labels == label) for label in (0, 1)}
        if not len(self.class_indices[0]) or not len(self.class_indices[1]):
            raise ValueError("BalancedBatchSampler requires both binary classes in the training split")
        self.batch_size = batch_size
        self.samples_per_class = batch_size // 2
        natural_steps = math.ceil(max(len(indexes) for indexes in self.class_indices.values()) / self.samples_per_class)
        self.steps_per_epoch = steps_per_epoch if steps_per_epoch is not None else natural_steps
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.steps_per_epoch):
            batch = np.concatenate(
                [
                    rng.choice(self.class_indices[0], size=self.samples_per_class, replace=True),
                    rng.choice(self.class_indices[1], size=self.samples_per_class, replace=True),
                ]
            )
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self) -> int:
        return self.steps_per_epoch
