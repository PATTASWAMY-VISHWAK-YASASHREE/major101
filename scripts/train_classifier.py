"""3D CNN Grade Classifier (M1) + Augmentation variant (M3)."""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from src.data import BraTS3DDataset, make_dataloaders, build_split_indices

logger = logging.getLogger("classifier")


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
                    nn.BatchNorm3d(out_ch),
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


def compute_class_weights(labels_csv: Path, seed: int = 42) -> torch.Tensor:
    """Compute class weights for BCEWithLogitsLoss based on grade imbalance."""
    df = pd.read_csv(labels_csv)
    n_pos = int(df["grade_proxy"].sum())
    n_neg = len(df) - n_pos
    w_pos = len(df) / (2.0 * n_pos) if n_pos > 0 else 1.0
    w_neg = len(df) / (2.0 * n_neg) if n_neg > 0 else 1.0
    weights = torch.tensor([w_neg, w_pos])
    logger.info(f"Class weights — neg={w_neg:.3f}, pos={w_pos:.3f}")
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ],
    )
    logger.info(f"{'='*60}")
    logger.info(f"Starting {model_name} — augment={augment}")
    logger.info(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    loaders, splits = make_dataloaders(
        npy_dir, labels_csv, batch_size=batch_size, augment=augment, seed=seed,
    )
    n_train = len(loaders["train"].dataset)
    n_val = len(loaders["val"].dataset)
    n_test = len(loaders["test"].dataset)
    logger.info(f"Split sizes — train={n_train}, val={n_val}, test={n_test}")

    model = GradeClassifier3D().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    class_weights = compute_class_weights(labels_csv, seed).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1:])

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
        for x, y in tqdm(loaders["train"], desc=f"[{model_name}] Train", leave=False):
            x, y = x.to(device), y.to(device).float()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_preds.extend((logits > 0).cpu().numpy())
            train_labels.extend(y.cpu().numpy())

        train_loss /= n_train
        train_acc = np.mean(np.array(train_preds) == np.array(train_labels))

        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in tqdm(loaders["val"], desc=f"[{model_name}] Val", leave=False):
                x, y = x.to(device), y.to(device).float()
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                val_preds.extend((logits > 0).cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_loss /= n_val
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
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

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for x, y in loaders["test"]:
            x, y = x.to(device), y.to(device).float()
            logits = model(x)
            test_preds.extend((logits > 0).cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_acc = np.mean(test_preds == test_labels)

    results = {
        "model": model_name,
        "augment": augment,
        "best_val_loss": best_val_loss,
        "best_val_acc": max(h["val_acc"] for h in history),
        "test_acc": test_acc,
        "epochs_trained": len(history),
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }

    logger.info(f"{model_name} complete — test_acc={test_acc:.4f}, {len(history)} epochs")
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
