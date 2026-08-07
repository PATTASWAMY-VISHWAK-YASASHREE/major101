"""3D CNN Grade Classifier (M1) + Augmentation variant (M3)."""

import logging
import math
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src` package is importable
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

# ponytail: class-balanced batch sampler — yields batches with equal high/low samples
# Each batch = half from minority (low-grade) + half from majority (high-grade), sampled with replacement
# This ensures the gradient always sees both classes, breaking the "predict all high" attractor


class BalancedBatchSampler(torch.utils.data.BatchSampler):
    """Yield batches with balanced high/low representation."""

    def __init__(self, dataset: BraTS3DDataset, batch_size: int, seed: int):
        # Build mapping: position in dataset.indices -> label
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
        # How many balanced batches fit in one epoch
        n_batches = len(self.dataset.indices) // b
        for _ in range(n_batches):
            pos_idx = torch.randint(0, len(self.pos), (half,), generator=self.rng)
            neg_idx = torch.randint(0, len(self.neg), (half,), generator=self.rng)
            batch = torch.cat([self.pos[pos_idx], self.neg[neg_idx]])
            batch = batch[torch.randperm(b, generator=self.rng)]
            yield batch.tolist()

    def __len__(self):
        return len(self.dataset.indices) // self.batch_size


logger = logging.getLogger("classifier")


def _setup_logging(log_path: Path) -> None:
    """One-time logging setup — safe to call repeatedly."""
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


class GradeClassifier3D(nn.Module):
    """3D CNN for BraTS grade classification.

    Architecture: 4 conv3d blocks with strided convs for spatial downsampling,
    adaptive pooling, then linear head for binary classification.

    Input: (4, 182, 218, 182) — 4 CTN-normalised modalities
    Output: single logit (BCEWithLogitsLoss)
    """

    def __init__(self):
        super().__init__()
        channels = [32, 64, 128, 256]

        blocks = []
        in_ch = 4
        for out_ch in channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(num_groups=8, num_channels=out_ch, affine=True),  # stable for batch=2-4
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch

        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x).squeeze(-1)


def compute_class_weights(labels_csv: Path) -> torch.Tensor:
    """Compute class weights for per-sample BCE loss from DEDUPED cases."""
    df = pd.read_csv(labels_csv).drop_duplicates(subset='case')
    n_pos = int(df["grade_proxy"].sum())
    n_neg = len(df) - n_pos
    w_pos = len(df) / (2.0 * n_pos) if n_pos > 0 else 1.0
    w_neg = len(df) / (2.0 * n_neg) if n_neg > 0 else 1.0
    weights = torch.tensor([w_neg, w_pos])
    logger.info(f"Class weights (deduped {len(df)} cases) — neg={w_neg:.3f}, pos={w_pos:.3f}")
    return weights


def train_model(
    npy_dir: Path,
    labels_csv: Path,
    model_name: str,
    augment: bool = False,
    epochs: int = 100,
    batch_size: int = 2,
    lr: float = 1e-3,
    patience: int = 10,
    seed: int = 42,
    output_dir: Path = None,
) -> dict:
    """Train M1 or M3 classifier.

    Args:
        model_name: "M1" or "M3".
        augment: True for M3 (augmentation ablation).
        patience: Early stopping patience (epochs without val improvement).

    Returns:
        dict with metrics and checkpoint paths.
    """
    output_dir = output_dir or Path("outputs/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{model_name}_best.pth"
    history_path = output_dir / f"{model_name}_history.csv"
    log_path = Path("training_log.txt")
    per_case_path = output_dir / f"{model_name}_predictions.csv"

    _setup_logging(log_path)
    logger.info(f"{'='*60}")
    logger.info(f"Starting {model_name} — augment={augment}")
    logger.info(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.mkldnn = False  # Disable MKLDNN — causes OOM on large 3D batches (CPU)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # GPU memory budgeting (pre-flight check)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1e9
        logger.info(f"GPU: {props.name}, total={total_gb:.2f} GB, SM={props.major}.{props.minor}")
        budget_gb = total_gb * 0.80
        logger.info(f"VRAM budget: {budget_gb:.2f} GB (80% of {total_gb:.2f} GB)")

        # Warn if other processes are holding VRAM
        pre_alloc = torch.cuda.memory_allocated() / 1e9
        if pre_alloc > 0.5:
            logger.warning(f"!! {pre_alloc:.2f} GB already allocated on GPU — training may OOM")
            torch.cuda.empty_cache()

        # Reset peak stats for clean measurement
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info(f"Post-clear alloc={torch.cuda.memory_allocated()/1e9:.2f} GB")

    # AMP for RTX 2050 4GB — no grad accumulation (batch=4 gives clean gradients)
    batch_size = 4  # ponytail: bigger batch gives better gradient signal to escape collapse
    grad_accum = 1
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    # Higher LR needed to escape majority-class attractor on imbalanced data
    lr = lr * 5 if lr == 1e-3 else lr
    logger.info(f"LR adjusted to {lr:.2e} for class imbalance")

    loaders, splits = make_dataloaders(
        npy_dir, labels_csv, batch_size=batch_size, augment=augment, seed=seed, num_workers=0
    )
    n_train = len(loaders["train"].dataset)
    n_val = len(loaders["val"].dataset)
    n_test = len(loaders["test"].dataset)
    logger.info(f"Split sizes — train={n_train}, val={n_val}, test={n_test}")

    model = GradeClassifier3D().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    # Weight init: Kaiming normal for Conv3d, ones/zeros for GroupNorm affine
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.GroupNorm, nn.InstanceNorm3d)):
            if m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    # Class-Balanced Loss (Cui et al. 2019) with beta=0.999
    # n_pos=704 (high-grade), n_neg=172 (low-grade) -> minority (low) gets higher weight
    beta = 0.999
    eff_pos = (1 - beta ** 704) / (1 - beta)  # ~505
    eff_neg = (1 - beta ** 172) / (1 - beta)  # ~158
    # Weight = (1-beta)/(1-beta^n) -- ratio neg:pos = 3.2:1
    w_pos = (1 - beta) / eff_pos
    w_neg = (1 - beta) / eff_neg
    # Normalize so weights sum to 2.0 (keep loss magnitude similar to BCE)
    scale = 2.0 / (w_pos + w_neg)
    class_weights = torch.tensor([w_neg * scale, w_pos * scale]).to(device)
    logger.info(f"Class-balanced weights: neg={class_weights[0]:.4f}, pos={class_weights[1]:.4f}")

    # Focal loss gamma=1.0 for binary classification (not 2.0)
    focal_gamma = 1.0
    sampler = BalancedBatchSampler(
        loaders["train"].dataset, batch_size=batch_size, seed=seed
    )
    loaders["train"] = DataLoader(
        loaders["train"].dataset, batch_sampler=sampler,
        num_workers=0, pin_memory=False
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        for step, (x, y) in enumerate(tqdm(loaders["train"], desc=f"[{model_name}] Train", leave=False)):
            try:
                x, y = x.to(device), y.to(device).float()
                # Reset gradients
                optimizer.zero_grad(set_to_none=True)
                # AMP forward pass
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    # Focal loss: reduce weight on easy samples, amplify on misclassified ones
                    p = torch.sigmoid(logits).squeeze()
                    prob = p * y.squeeze() + (1 - p) * (1 - y.squeeze())
                    focal_weight = (1 - prob) ** focal_gamma
                    bce = criterion(logits, y).squeeze()
                    loss = bce * focal_weight.unsqueeze(1)
                    # Class-balanced weights: apply per-sample based on true label
                    sample_weights = torch.where(y == 1, class_weights[1], class_weights[0]).unsqueeze(1)
                    loss = loss * sample_weights
                    loss = loss.mean() / grad_accum  # scale for accumulation
                # Scaled backward pass
                scaler.scale(loss).backward()
                # Step every grad_accum batches
                if (step + 1) % grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                train_loss += loss.item() * x.size(0) * grad_accum
                train_preds.extend((logits > 0).cpu().numpy())
                train_labels.extend(y.cpu().numpy())
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Skipping corrupt batch in train: {e}")

        # Log GPU memory snapshot per epoch (detect leaks, OOM risks)
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            util_pct = peak / (torch.cuda.get_device_properties(0).total_memory / 1e9) * 100
            logger.info(f"  GPU mem: alloc={alloc:.2f}G reserved={reserved:.2f}G peak={peak:.2f}G util={util_pct:.0f}%")
            # Leak detection: warn if VRAM usage grows monotonically across epochs
            if epoch > 0 and "prev_peak" in dir():
                if peak > prev_peak * 1.10:  # >10% growth = possible leak
                    logger.warning(f"  !! Possible memory leak: peak grew {prev_peak:.2f}G -> {peak:.2f}G")
            prev_peak = peak

            torch.cuda.empty_cache()

        # Budget check — warn if approaching 80% VRAM
        if torch.cuda.is_available():
            util_pct = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
            if util_pct > 85:
                logger.warning(f"  !! VRAM utilization {util_pct:.0f}% — approaching budget. Consider reducing batch size.")
            elif util_pct > 70:
                logger.info(f"  VRAM utilization: {util_pct:.0f}%")

        train_loss /= n_train
        train_acc = np.mean(np.array(train_preds) == np.array(train_labels))

        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in tqdm(loaders["val"], desc=f"[{model_name}] Val", leave=False):
                try:
                    x, y = x.to(device), y.to(device).float()
                    with torch.amp.autocast("cuda"):
                        logits = model(x)
                        loss = criterion(logits, y).mean()
                    val_loss += loss.item() * x.size(0)
                    val_preds.extend((logits > 0).cpu().numpy())
                    val_labels.extend(y.cpu().numpy())
                except (RuntimeError, ValueError) as e:
                    logger.warning(f"Skipping corrupt batch in val: {e}")

        val_loss /= n_val
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Convert Path to string for Windows compatibility
            torch.save(model.state_dict(), str(checkpoint_path))
        else:
            patience_counter += 1

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        pd.DataFrame(history).to_csv(history_path, index=False)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(str(checkpoint_path), weights_only=True))
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    test_preds, test_labels = [], []
    test_case_ids = loaders["test"].dataset.indices
    labels_df = pd.read_csv(labels_csv)
    with torch.no_grad():
        for x, y in loaders["test"]:
            try:
                x, y = x.to(device), y.to(device).float()
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                test_preds.extend((logits > 0).cpu().numpy())
                test_labels.extend(y.cpu().numpy())
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Skipping corrupt batch in test: {e}")

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_acc = np.mean(test_preds == test_labels)

    # Per-class metrics
    tn = int(np.sum((test_preds == 0) & (test_labels == 0)))
    tp = int(np.sum((test_preds == 1) & (test_labels == 1)))
    fn = int(np.sum((test_preds == 0) & (test_labels == 1)))
    fp = int(np.sum((test_preds == 1) & (test_labels == 0)))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    # Save per-case predictions CSV
    pred_df = pd.DataFrame({
        "case": labels_df.iloc[test_case_ids]["case"].values,
        "true_label": test_labels.astype(int),
        "predicted_label": test_preds.astype(int),
        "correct": (test_preds == test_labels).astype(int),
    })
    pred_df.to_csv(per_case_path, index=False)

    results = {
        "model": model_name,
        "augment": augment,
        "best_val_loss": best_val_loss,
        "best_val_acc": max(h["val_acc"] for h in history),
        "test_acc": float(test_acc),
        "test_f1": float(f1),
        "test_sensitivity": float(sensitivity),
        "test_specificity": float(specificity),
        "epochs_trained": len(history),
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "predictions": str(per_case_path),
    }

    logger.info(f"{model_name} done — acc={test_acc:.4f} F1={f1:.4f} "
                f"sens={sensitivity:.4f} spec={specificity:.4f} ({len(history)} epochs)")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train M1/M3 grade classifier")
    parser.add_argument("--model", choices=["M1", "M3"], required=True, help="Model variant")
    parser.add_argument("--augment", action="store_true", help="Enable augmentation (M3)")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stop patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="outputs/training", help="Output dir")
    args = parser.parse_args()

    npy_dir = Path(__file__).resolve().parent.parent / "data" / "brats_preprocessed" / "train"
    labels_csv = Path(__file__).resolve().parent.parent / "data" / "brats_preprocessed" / "labels.csv"
    output_dir = Path(args.output_dir)

    results = train_model(
        npy_dir=npy_dir,
        labels_csv=labels_csv,
        model_name=args.model,
        augment=args.augment,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        output_dir=output_dir,
    )

    print(f"\nResults: {results}")
