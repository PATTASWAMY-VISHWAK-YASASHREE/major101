"""Dataset and DataLoader for 3D MRI/CT brain tumour scans."""

from pathlib import Path
from typing import Optional
import random

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class BrainTumourDataset(Dataset):
    """
    Loads 3D volumes from .nii.gz files.
    Expects: root/{split}/{class_label}/<file>.nii.gz
    Class labels auto-inferred from directory names (integer).
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        img_size: tuple[int, int, int] = (96, 96, 96),
        normalize: bool = True,
        transform: Optional[transforms.Compose] = None,
    ):
        self.root = Path(root) / split
        self.img_size = img_size
        self.normalize = normalize
        self.transform = transform
        self.files: list[Path] = []
        self.labels: list[int] = []
        self._build_index()

    def _build_index(self):
        """Scan directory for .nii.gz files and assign class labels."""
        for label_dir in sorted(self.root.iterdir()):
            if not label_dir.is_dir():
                continue
            try:
                label = int(label_dir.name)
            except ValueError:
                continue
            for f in sorted(label_dir.glob("*.nii.gz")):
                self.files.append(f)
                self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def _load_volume(self, path: Path) -> np.ndarray:
        img = nib.load(path)
        return img.get_fdata(dtype=np.float32)

    def _preprocess(self, vol: np.ndarray) -> np.ndarray:
        if self.normalize:
            non_zero = vol[vol != 0]
            if non_zero.size == 0:
                vol = np.zeros_like(vol)
            else:
                mean, std = non_zero.mean(), non_zero.std() + 1e-8
                vol = (vol - mean) / std
        if vol.shape != self.img_size:
            vol = torch.nn.functional.interpolate(
                torch.from_numpy(vol).unsqueeze(0).unsqueeze(0),
                size=self.img_size,
                mode="trilinear",
                align_corners=False,
            ).squeeze().numpy()
        return vol

    def __getitem__(self, idx: int):
        vol = self._load_volume(self.files[idx])
        vol = self._preprocess(vol)
        x = torch.from_numpy(vol).unsqueeze(0)  # (1, D, H, W)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform:
            x = self.transform(x)
        return x, y


def build_file_index(
    root: Path,
    img_size: tuple[int, int, int] = (96, 96, 96),
    normalize: bool = True,
) -> tuple[list[Path], list[int]]:
    """
    Build (files, labels) index from a flat root directory.
    Expects: root/{class_label}/<file>.nii.gz
    """
    files, labels = [], []
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        try:
            label = int(label_dir.name)
        except ValueError:
            continue
        for f in sorted(label_dir.glob("*.nii.gz")):
            files.append(f)
            labels.append(label)
    return files, labels


def split_index(
    files: list[Path],
    labels: list[int],
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42,
) -> tuple[tuple[list[Path], list[int]], tuple[list[Path], list[int]], tuple[list[Path], list[int]]]:
    """
    Stratified split of (files, labels) into train/val/test.
    No data mutation — callers pass split indices to BrainTumourDataset.
    """
    rng = np.random.default_rng(seed)
    n = len(files)
    indices = list(range(n))
    rng.shuffle(indices)

    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def slice_idx(idx_list):
        return [files[i] for i in idx_list], [labels[i] for i in idx_list]

    return slice_idx(train_idx), slice_idx(val_idx), slice_idx(test_idx)


def make_datasets_from_split(
    root: Path,
    train_split: float = 0.7,
    val_split: float = 0.15,
    img_size: tuple[int, int, int] = (96, 96, 96),
    normalize: bool = True,
    seed: int = 42,
) -> tuple[BrainTumourDataset, BrainTumourDataset, BrainTumourDataset, int]:
    """
    Build train/val/test datasets from a flat root dir.
    Returns (train_ds, val_ds, test_ds, n_classes).
    """
    files, labels = build_file_index(root, img_size, normalize)
    train_files, train_labels = split_index(files, labels, train_split, val_split, seed)[0]
    val_files, val_labels = split_index(files, labels, train_split, val_split, seed)[1]
    test_files, test_labels = split_index(files, labels, train_split, val_split, seed)[2]

    # Write split subdirs (non-destructive, just a symlink/manifest approach)
    def write_split(subdir: str, f_list: list[Path], l_list: list[int]):
        target = root / subdir
        target.mkdir(parents=True, exist_ok=True)
        unique_labels = sorted(set(l_list))
        for lbl in unique_labels:
            (target / str(lbl)).mkdir(parents=True, exist_ok=True)
        for f, lbl in zip(f_list, l_list):
            src_name = f.name
            dest = target / str(lbl) / src_name
            if not dest.exists():
                # hard-link if on same filesystem, else copy
                try:
                    dest.hardlink_to(f)
                except OSError:
                    import shutil
                    shutil.copy2(f, dest)

    write_split("train", train_files, train_labels)
    write_split("val", val_files, val_labels)
    write_split("test", test_files, test_labels)

    train_ds = BrainTumourDataset(root / "train", img_size=img_size, normalize=normalize)
    val_ds = BrainTumourDataset(root / "val", img_size=img_size, normalize=normalize)
    test_ds = BrainTumourDataset(root / "test", img_size=img_size, normalize=normalize)

    n_classes = max(train_labels + val_labels + test_labels) + 1
    return train_ds, val_ds, test_ds, n_classes


def make_dataloaders(
    root: Path,
    batch_size: int = 4,
    num_workers: int = 4,
    img_size: tuple[int, int, int] = (96, 96, 96),
    normalize: bool = True,
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42,
):
    train_ds, val_ds, test_ds, n_classes = make_datasets_from_split(
        root, train_split, val_split, img_size, normalize, seed,
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        n_classes,
    )
