"""BraTS3D dataset — loads preprocessed .npy + labels.csv with stratified splits."""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class BraTS3DDataset(Dataset):
    """Dataset for preprocessed BraTS .npy volumes.

    Args:
        npy_dir: Directory containing <case_id>.npy files (shape 4×182×218×182).
        labels_csv: Path to labels.csv (case, et, tc, wt, et_vol, tc_vol, wt_vol, grade_proxy).
        indices: List of case indices (0..N-1) for this split.
        labels_df: Full labels DataFrame (aligned by row index).
        augment: If True, apply online augmentation (random flips, rotation, intensity jitter).
    """

    def __init__(
        self,
        npy_dir: Path,
        labels_csv: Path,
        indices: list[int],
        labels_df: pd.DataFrame,
        augment: bool = False,
    ):
        self.npy_dir = npy_dir
        self.indices = indices
        self.labels_df = labels_df
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        case_idx = self.indices[idx]
        case_id = self.labels_df.iloc[case_idx]["case"]
        npy_path = self.npy_dir / f"{case_id}.npy"

        # Load .npy — shape (4, 182, 218, 182), CTN-normalised, float32
        img = np.load(npy_path, mmap_mode="r")  # lazy load on disk
        img = img.astype(np.float32)  # copy to memory — mmap on small files is slower
        img = torch.from_numpy(img)  # (4, 182, 218, 182)

        # Label: grade_proxy (0=low, 1=high)
        label = int(self.labels_df.iloc[case_idx]["grade_proxy"])

        if self.augment:
            img = _augment(img)

        return img, torch.tensor(label, dtype=torch.long)


def _augment(img: torch.Tensor) -> torch.Tensor:
    """Online augmentation: random flips, rotation, intensity jitter."""
    # Random flip along any axis (x, y, or z)
    for dim in range(2, 5):  # skip channel dim
        if random.random() < 0.5:
            img = torch.flip(img, [dim])

    # Random 90° rotation in axial plane (last 2 dims)
    k = random.randint(0, 3)
    img = torch.rot90(img, k, dims=(3, 2))

    # Intensity jitter: additive Gaussian noise σ=0.02 (CTN range [-1,1])
    noise = torch.randn_like(img) * 0.02
    img = img + noise

    return img


def build_split_indices(
    labels_csv: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Stratified 80/10/10 split by grade_proxy.

    Returns dict: {"train": [indices], "val": [indices], "test": [indices]}
    """
    df = pd.read_csv(labels_csv)
    rng = np.random.default_rng(seed)

    train_idx, val_idx, test_idx = [], [], []
    for grade in sorted(df["grade_proxy"].unique()):
        group = df[df["grade_proxy"] == grade].index.tolist()
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(group[:n_train])
        val_idx.extend(group[n_train:n_train + n_val])
        test_idx.extend(group[n_train + n_val:])

    return {
        "train": sorted(train_idx),
        "val": sorted(val_idx),
        "test": sorted(test_idx),
    }


def make_dataloaders(
    npy_dir: Path,
    labels_csv: Path,
    batch_size: int = 2,
    augment: bool = False,
    seed: int = 42,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders with stratified splits.

    Args:
        npy_dir: Directory with <case_id>.npy files.
        labels_csv: Path to labels.csv.
        batch_size: Batch size (default 2 — 4GB RAM safe on CPU).
        augment: Enable online augmentation (M3).
        seed: Random seed for split reproducibility.
        num_workers: DataLoader workers (0 = main process, avoids Windows spawn overhead).

    Returns:
        dict with "train", "val", "test" DataLoader keys + "splits" dict.
    """
    splits = build_split_indices(labels_csv, seed=seed)
    labels_df = pd.read_csv(labels_csv)

    loaders = {}
    for split_name, indices in splits.items():
        ds = BraTS3DDataset(
            npy_dir=npy_dir,
            labels_csv=labels_csv,
            indices=indices,
            labels_df=labels_df,
            augment=augment if split_name == "train" else False,
        )
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=False,
        )

    return loaders, splits


if __name__ == "__main__":
    # Self-check: verify the dataset loads correctly
    import sys
    npy_dir = Path("C:/Users/pvish/copilot-worktrees/major101/pattaswamy-vishwak-yasashree-cuddly-lamp/data/brats_preprocessed/train")
    labels_csv = Path("C:/Users/pvish/copilot-worktrees/major101/pattaswamy-vishwak-yasashree-cuddly-lamp/data/brats_preprocessed/labels.csv")

    loaders, splits = make_dataloaders(npy_dir, labels_csv, batch_size=2, augment=False, seed=42)

    print(f"Train: {len(loaders['train'].dataset)} samples")
    print(f"Val:   {len(loaders['val'].dataset)} samples")
    print(f"Test:  {len(loaders['test'].dataset)} samples")

    # Verify first batch
    x, y = next(iter(loaders["train"]))
    print(f"Image shape: {x.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Grade distribution: {y.bincount().tolist()}")

    assert x.shape == (2, 4, 182, 218, 182), f"Unexpected image shape: {x.shape}"
    assert x.dtype == torch.float32
    assert y.dtype == torch.long
    print("Dataset check passed ✓")
