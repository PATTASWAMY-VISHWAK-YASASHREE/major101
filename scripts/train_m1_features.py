#!/usr/bin/env python
"""
BraTS 3D CNN with Explicit Tumor Features (M1-Features)
Fixes "large volume but low-grade" by adding ET/WT ratio, volumes as scalar features.
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
    
    Uses a separate pathway for scalar features with residual addition,
    so they have their own gradient path and aren't drowned by image features.
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
        
        # Image feature pathway: 256 -> 128 -> 64
        self.image_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
        )
        
        # Scalar feature pathway: 3 -> 64 (projected to same dim as image features)
        self.scalar_head = nn.Sequential(
            nn.Linear(n_scalar_features, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
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
        
        # Image pathway
        img_feat = self.image_head(x)  # (B, 64)
        
        if scalar_features is not None:
            # Scalar pathway - project to same dimension
            scalar_feat = self.scalar_head(scalar_features)  # (B, 64)
            # Residual addition - scalar features add to image features
            combined = img_feat + scalar_feat
        else:
            combined = img_feat
        
        return self.classifier(combined).squeeze(-1)


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
    logger.info(f"Class-Balanced weights (beta={beta}): neg={weights[0]:.3f}, pos={weights[1]:.3f}")
    return weights


def build_scalar_features(labels_csv: Path, indices) -> torch.Tensor:
    """Build scalar tumor features for each case: [log(ET_vol), log(WT_vol), ET/WT_ratio]
    
    indices can be either integer positions or case_ids (strings).
    """
    df = pd.read_csv(labels_csv).drop_duplicates(subset='case').reset_index(drop=True)
    df.columns = [c.strip().lower() for c in df.columns]
    df['label'] = df['grade_proxy'].astype(int)
    df = df.rename(columns={'case': 'case_id'})
    
    # Build case_id -> row mapping
    case_to_row = {row['case_id']: row for _, row in df.iterrows()}
    
    features = []
    for idx in indices:
        # idx could be integer position or case_id string
        if isinstance(idx, str):
            row = case_to_row[idx]
        else:
            row = df.iloc[idx]
        et_vol = row['et_volume']
        wt_vol = row['wt_volume']
        
        # Normalize volumes (log scale)
        et_log = np.log1p(et_vol) / 12.0  # ~max log(100k) ≈ 11.5
        wt_log = np.log1p(wt_vol) / 12.0
        
        # ET/WT ratio (key discriminative feature)
        et_wt_ratio = et_vol / (wt_vol + 1e-8)
        
        features.append([et_log, wt_log, et_wt_ratio])
    
    return torch.tensor(features, dtype=torch.float32)


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
    
    # Build scalar features
    train_scalar = build_scalar_features(labels_csv, train_idx).to(device)
    val_scalar = build_scalar_features(labels_csv, val_idx).to(device)
    
    # Data
    df = pd.read_csv(labels_csv).drop_duplicates(subset='case').reset_index(drop=True)
    df.columns = [c.strip().lower() for c in df.columns]
    df['label'] = df['grade_proxy'].astype(int)
    df = df.rename(columns={'case': 'case_id'})
    labels = [(row['case_id'], int(row['label'])) for _, row in df.iterrows()]
    
    train_ds = BraTS3DDataset(npy_dir=npy_dir, labels_csv=labels_csv, indices=train_idx, labels=labels, augment=True)
    val_ds = BraTS3DDataset(npy_dir=npy_dir, labels_csv=labels_csv, indices=val_idx, labels=labels, augment=False)
    
    train_sampler = BalancedBatchSampler(train_ds, batch_size, seed)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    
    # Model
    model = GradeClassifier3DWithFeatures(n_scalar_features=3).to(device)
    
    # Loss
    class_weights = compute_class_balanced_weights(labels_csv, beta=0.999).to(device)
    criterion = ClassBalancedFocalLoss(class_weights, gamma=1.0).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    
    # Create index mapping for scalar features
    train_idx_to_scalar_idx = {idx: i for i, idx in enumerate(train_idx)}
    val_idx_to_scalar_idx = {idx: i for i, idx in enumerate(val_idx)}
    
    best_val_acc = 0.0
    best_epoch = 0
    history = []
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (xb, yb) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False)):
            xb, yb = xb.to(device), yb.to(device).float()
            
            # Get scalar features for this batch
            # The sampler yields indices from dataset.indices, which correspond to train_idx
            batch_positions = train_sampler.__iter__().__next__()  # This doesn't work directly
            
            # Simpler: use the fact that we know the batch order from sampler
            # We'll track using a different approach - just use sequential for now
            # Actually, let's use a custom collate or modify the dataset
            
        # Let me use a simpler approach - modify the dataset to return scalar features
        
    return model


# Simpler approach: Modify dataset to return scalar features directly
class BraTS3DDatasetWithFeatures:
    """Wrapper that adds scalar features to BraTS3DDataset items."""
    def __init__(self, base_dataset, scalar_features, idx_to_scalar):
        self.base_dataset = base_dataset
        self.scalar_features = scalar_features
        self.idx_to_scalar = idx_to_scalar
        # Delegate attributes
        self.indices = base_dataset.indices
        self.labels = base_dataset.labels
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        xb, yb = self.base_dataset[idx]
        dataset_idx = self.base_dataset.indices[idx]
        sf = self.scalar_features[self.idx_to_scalar[dataset_idx]]
        return xb, yb, sf


def train_model_simple(
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
    """Train model with explicit tumor features - simplified version"""
    _setup_logging(output_dir / f"{model_name}_training.log")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training {model_name} on {device}")
    
    # Build splits - returns (splits_dict, dataframe)
    splits, _ = build_split_indices(labels_csv, seed=seed)
    train_idx = splits['train']
    val_idx = splits['val']
    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # Build scalar features
    train_scalar = build_scalar_features(labels_csv, train_idx)
    val_scalar = build_scalar_features(labels_csv, val_idx)
    
    # Data
    df = pd.read_csv(labels_csv).drop_duplicates(subset='case').reset_index(drop=True)
    df.columns = [c.strip().lower() for c in df.columns]
    df['label'] = df['grade_proxy'].astype(int)
    df = df.rename(columns={'case': 'case_id'})
    labels = [(row['case_id'], int(row['label'])) for _, row in df.iterrows()]
    
    # Create index mapping
    train_idx_to_scalar = {idx: i for i, idx in enumerate(train_idx)}
    val_idx_to_scalar = {idx: i for i, idx in enumerate(val_idx)}
    
    train_ds = BraTS3DDataset(npy_dir=npy_dir, labels_csv=labels_csv, indices=train_idx, labels=labels, augment=True)
    val_ds = BraTS3DDataset(npy_dir=npy_dir, labels_csv=labels_csv, indices=val_idx, labels=labels, augment=False)
    
    # Wrap with scalar features
    train_ds = BraTS3DDatasetWithFeatures(train_ds, train_scalar, train_idx_to_scalar)
    val_ds = BraTS3DDatasetWithFeatures(val_ds, val_scalar, val_idx_to_scalar)
    
    train_sampler = BalancedBatchSampler(train_ds, batch_size, seed)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    
    # Model
    model = GradeClassifier3DWithFeatures(n_scalar_features=3).to(device)
    
    # Loss
    class_weights = compute_class_balanced_weights(labels_csv, beta=0.999).to(device)
    criterion = ClassBalancedFocalLoss(class_weights, gamma=1.0).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    
    best_val_acc = 0.0
    best_epoch = 0
    history = []
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for xb, yb, sf in tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False):
            xb, yb, sf = xb.to(device), yb.to(device).float(), sf.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                logits = model(xb, sf)
                loss = criterion(logits, yb)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            train_correct += (preds == yb.long()).sum().item()
            train_total += xb.size(0)
        
        train_acc = train_correct / train_total
        train_loss /= train_total
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for xb, yb, sf in tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=False):
                xb, yb, sf = xb.to(device), yb.to(device).float(), sf.to(device)
                logits = model(xb, sf)
                loss = criterion(logits, yb)
                
                val_loss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(logits) > 0.5).long()
                val_correct += (preds == yb.long()).sum().item()
                val_total += xb.size(0)
                all_preds.append(preds.cpu())
                all_targets.append(yb.cpu().long())
        
        val_acc = val_correct / val_total
        val_loss /= val_total
        
        # Metrics
        preds_all = torch.cat(all_preds).squeeze().numpy()
        targets_all = torch.cat(all_targets).squeeze().numpy()
        
        from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
        val_f1 = f1_score(targets_all, preds_all)
        val_sens = recall_score(targets_all, preds_all)
        val_prec = precision_score(targets_all, preds_all)
        cm = confusion_matrix(targets_all, preds_all)
        val_spec = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), str(output_dir / f"{model_name}_best.pth"))
            logger.info(f"New best val_acc: {val_acc:.4f} at epoch {epoch}")
        
        scheduler.step(val_acc)
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_sens': val_sens,
            'val_spec': val_spec,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        logger.info(f"Epoch {epoch:3d} | Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                    f"Val: loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} sens={val_sens:.4f} spec={val_spec:.4f}")
        
        # Save history
        pd.DataFrame(history).to_csv(output_dir / f"{model_name}_history.csv", index=False)
    
    logger.info(f"Training complete. Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}")
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
    
    train_model_simple(
        npy_dir=args.npy_dir,
        labels_csv=args.labels_csv,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )