#!/usr/bin/env python
"""
BraTS 3D CNN with Explicit Tumor Features - Training Script
Adds ET/WT ratio, volumes as scalar features to fix "large volume but low-grade" problem
"""

import logging
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data import BraTS3DDataset, make_dataloaders, build_split_indices


logger = logging.getLogger("classifier")


class BalancedBatchSampler(torch.utils.data.BatchSampler):
    """Yield batches with balanced high/low representation."""

    def __init__(self, dataset: BraTS3DDataset, batch_size: int, seed: int):
        pos = [i for i, idx in enumerate(dataset.indices) if dataset.labels[idx][1] == 1]
        neg = [i for i, idx in enumerate(dataset.indices) if dataset.labels[idx][1] == 0]
        rng = torch.Generator().manual_seed(seed)
        super().__init__(dataset.indices, batch_size, drop_last=True)
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos = torch.tensor(pos)
        self.neg = torch.tensor(neg)
        self.rng = rng

    def __iter__(self):
        b = self.batch_size
        half = b // 2
        n_batches = len(self.dataset.indices) // b
        for _ in range(n_batches):
            pos_idx = torch.randint(0, len(self.pos), (half,), generator=self.rng)
            neg_idx = torch.randint(0, len(self.neg), (half,), generator=self.rng)
            batch = torch.cat([self.pos[pos_idx], self.neg[neg_idx]])
            batch = batch[torch.randperm(b, generator=self.rng)]
            yield batch.tolist()

    def __len__(self):
        return len(self.dataset.indices) // self.batch_size


def _setup_logging(log_path: Path) -> None:
    if not logger.handlers:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.FileHandler(log_path, mode="a"),
                logging.StreamHandler(),
            ],
        )


class GradeClassifier3DWithFeatures(nn.Module):
    """3D CNN + Explicit Tumor Features for BraTS grade classification.

    Adds scalar tumor characteristics (ET/WT ratio, volumes) as additional features
    to solve the "large non-enhancing tumor = low grade" confusion.
    """

    def __init__(self, n_scalar_features=3):
        super().__init__()
        channels = [32, 64, 128, 256]

        blocks = []
        in_ch = 4
        for out_ch in channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(num_groups=8, num_channels=out_ch, affine=True),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch

        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Image features: 256, Scalar features: n_scalar_features
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 + n_scalar_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, scalar_features=None):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # (B, 256)
        
        if scalar_features is not None:
            x = torch.cat([x, scalar_features], dim=1)  # (B, 256 + n_scalar)
        
        return self.head(x).squeeze(-1)


def compute_class_balanced_weights(labels_csv: Path, beta=0.999) -> torch.Tensor:
    """Class-Balanced Loss weights (Cui et al. 2019)"""
    df = pd.read_csv(labels_csv).drop_duplicates(subset='case')
    n_pos = int(df["grade_proxy"].sum())
    n_neg = len(df) - n_pos
    
    eff_pos = (1 - beta ** n_pos) / (1 - beta)
    eff_neg = (1 - beta ** n_neg) / (1 - beta)
    w_pos = (1 - beta) / eff_pos
    w_neg = (1 - beta) / eff_neg
    
    # Normalize so mean weight = 1
    scale = 2.0 / (w_pos + w_neg)
    weights = torch.tensor([w_neg * scale, w_pos * scale])
    logger.info(f"Class-Balanced weights (β={beta}): neg={weights[0]:.3f}, pos={weights[1]:.3f}")
    return weights


def build_scalar_features(labels_csv: Path, indices) -> torch.Tensor:
    """Build scalar tumor features for each case: [ET_volume, WT_volume, ET/WT_ratio]"""
    df = pd.read_csv(labels_csv).drop_duplicates(subset='case').reset_index(drop=True)
    df.columns = [c.strip().lower() for c in df.columns]
    df['label'] = df['grade_proxy'].astype(int)
    df = df.rename(columns={'case': 'case_id'})
    
    features = []
    for idx in indices:
        row = df.iloc[idx]
        et_vol = row['et_volume']
        wt_vol = row['wt_volume']
        tc_vol = row['tc_volume']
        
        # Normalize volumes (log scale)
        et_log = np.log1p(et_vol) / 12.0  # ~max log(100k) ≈ 11.5
        wt_log = np.log1p(wt_vol) / 12.0
        tc_log = np.log1p(tc_vol) / 12.0
        
        # ET/WT ratio (key discriminative feature)
        et_wt_ratio = et_vol / (wt_vol + 1e-8)
        
        features.append([et_log, wt_log, et_wt_ratio])
    
    return torch.tensor(features, dtype=torch.float32)


def make_dataloaders_with_features(
    npy_dir: Path,
    labels_csv: Path,
    train_indices,
    val_indices,
    batch_size: int = 2,
    num_workers: int = 0,
    seed: int = 42,
):
    """Create dataloaders with scalar tumor features"""
    df = pd.read_csv(labels_csv).drop_duplicates(subset='case').reset_index(drop=True)
    df.columns = [c.strip().lower() for c in df.columns]
    df['label'] = df['grade_proxy'].astype(int)
    df = df.rename(columns={'case': 'case_id'})
    
    labels = [(row['case_id'], int(row['label'])) for _, row in df.iterrows()]
    
    # Build scalar features
    train_scalar = build_scalar_features(labels_csv, train_indices)
    val_scalar = build_scalar_features(labels_csv, val_indices)
    
    train_ds = BraTS3DDataset(
        npy_dir=npy_dir,
        labels_csv=labels_csv,
        indices=train_indices,
        labels=labels,
        augment=True,
    )
    val_ds = BraTS3DDataset(
        npy_dir=npy_dir,
        labels_csv=labels_csv,
        indices=val_indices,
        labels=labels,
        augment=False,
    )
    
    train_sampler = BalancedBatchSampler(train_ds, batch_size, seed)
    
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    
    return train_loader, val_loader, train_scalar, val_scalar


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss combining CB weights + Focal modulation"""
    def __init__(self, class_weights: torch.Tensor, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('class_weights', class_weights)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.gamma
        
        # Per-sample class weights
        sample_weights = torch.where(
            targets == 1, 
            self.class_weights[1], 
            self.class_weights[0]
        ).to(targets.device)
        
        loss = focal_weight * bce * sample_weights
        return loss.mean()


def train_model(
    npy_dir: Path,
    labels_csv: Path,
    model_name: str,
    epochs: int = 100,
    batch_size: int = 2,
    lr: float = 5e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    seed: int = 42,
    output_dir: Path = Path("outputs/training"),
):
    """Train model with explicit tumor features"""
    _setup_logging(output_dir / f"{model_name}_training.log")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training {model_name} on {device}")
    
    # Build splits
    train_idx, val_idx = build_split_indices(labels_csv, seed=seed)
    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # Data loaders with scalar features
    train_loader, val_loader, train_scalar, val_scalar = make_dataloaders_with_features(
        npy_dir, labels_csv, train_idx, val_idx, batch_size, num_workers, seed
    )
    
    # Move scalar features to device
    train_scalar = train_scalar.to(device)
    val_scalar = val_scalar.to(device)
    
    # Model
    model = GradeClassifier3DWithFeatures(n_scalar_features=3).to(device)
    
    # Class-Balanced Focal Loss
    class_weights = compute_class_balanced_weights(labels_csv, beta=0.999).to(device)
    criterion = ClassBalancedFocalLoss(class_weights, gamma=1.0).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    history = []
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device).float()
            # Get scalar features for this batch
            batch_indices = train_sampler.__iter__().__next__()  # This won't work directly
            # Instead, we need to track indices differently
            # For now, use the order from the sampler
            pass
        
        # Simpler approach: use the fact that sampler yields indices
        # We'll need to modify the approach
        
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy-dir', type=Path, default=Path('data/brats_preprocessed/train'))
    parser.add_argument('--labels-csv', type=Path, default=Path('data/brats_preprocessed/labels.csv'))
    parser.add_argument('--model-name', type=str, default='M1_features')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/training'))
    args = parser.parse_args()
    
    train_model(
        npy_dir=args.npy_dir,
        labels_csv=args.labels_csv,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )