"""BraTS Brain Tumour Classification — Training Script.

Trains a 3D ResNet on preprocessed BraTS 4-modality volumes.
Uses 64³ patch sampling to fit in 4GB VRAM.

Usage:
  python train.py [--epochs 100] [--batch 2] [--workers 2]
"""

import sys, json, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

import csv

sys.path.insert(0, str(Path(__file__).parent))
from src.model import ResNet3D
from src.utils import seed_everything, get_device


class BraTSPatchDataset(Dataset):
    def __init__(self, data_dir, labels_path, patch_size=64, cache_all=False):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.cases, self.labels = [], []
        with open(labels_path) as f:
            for row in csv.DictReader(f):
                self.cases.append(row["case"])
                self.labels.append(int(row["grade_proxy"]))
        self.cache = None
        if cache_all:
            self.cache = [np.load(self.data_dir / f"{c}.npy") for c in self.cases]

    def __len__(self):
        return len(self.cases) * 50

    def __getitem__(self, idx):
        case_idx = idx // 50
        case = self.cases[case_idx]
        label = self.labels[case_idx]
        vol = np.load(self.data_dir / f"{case}.npy") if self.cache is None else self.cache[case_idx]
        vol = vol.astype(np.float32)
        if len(vol.shape) == 3:
            vol = vol[None, ...]
        if vol.shape[0] != 4:
            vol = vol[:4, ...]
        d, h, w = vol.shape[1:]
        ps = self.patch_size
        dz = int(np.random.randint(0, d - ps)) if d > ps else 0
        hy = int(np.random.randint(0, h - ps)) if h > ps else 0
        wx = int(np.random.randint(0, w - ps)) if w > ps else 0
        patch = vol[:, dz:dz+ps, hy:hy+ps, wx:wx+ps]
        for _ in range(3):
            if np.random.rand() > 0.5:
                patch = patch[:, :, :, ::-1]
        if np.random.rand() > 0.5:
            patch = patch[:, :, ::-1, :]
        if np.random.rand() > 0.5:
            patch = patch[:, ::-1, :, :]
        x = torch.tensor(patch, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class BraTSCaseDataset(Dataset):
    def __init__(self, data_dir, labels_path):
        self.data_dir = Path(data_dir)
        self.cases, self.labels = [], []
        with open(labels_path) as f:
            for row in csv.DictReader(f):
                self.cases.append(row["case"])
                self.labels.append(int(row["grade_proxy"]))

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        vol = np.load(self.data_dir / f"{self.cases[idx]}.npy")
        vol = vol.astype(np.float32)
        if len(vol.shape) == 3:
            vol = vol[None, ...]
        if vol.shape[0] != 4:
            vol = vol[:4, ...]
        return torch.tensor(vol, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total += y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        if i % 50 == 0:
            print(f"  [{i}/{len(loader)}] loss={loss.item():.4f} acc={correct/total*100:.1f}%")
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            total += y.size(0)
            correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--data-dir", type=str, default="data/brats_preprocessed/train")
    p.add_argument("--labels", type=str, default="data/brats_preprocessed/labels.csv")
    p.add_argument("--output", type=str, default="outputs/model")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = 2

    print("=" * 70)
    print("TRAINING: BraTS Brain Tumour Classification")
    print(f"  Device:  {device}")
    print(f"  Model:   resnet3d, input_channels=4, classes={num_classes}")
    print(f"  Patches: 64³ from 4-modality BraTS volumes")
    print(f"  Batch:   {args.batch}, Epochs: {args.epochs}, LR: {args.lr}")
    print("=" * 70)

    model = ResNet3D(input_ch=4, num_classes=num_classes, base_width=64).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    dataset = BraTSPatchDataset(args.data_dir, args.labels, patch_size=args.patch_size)
    dataloader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda",
    )
    print(f"  Patches/epoch: {len(dataset)}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if device.type == "cuda" else None

    eval_dataset = BraTSCaseDataset(args.data_dir, args.labels)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                              num_workers=args.workers, pin_memory=device.type == "cuda")

    history, best_acc = [], 0.0
    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        train_loss, train_acc = train_epoch(model, dataloader, criterion, optimizer, scaler, device)
        eval_loss, eval_acc = evaluate(model, eval_loader, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
                         "eval_loss": eval_loss, "eval_acc": eval_acc})
        print(f"  Train: loss={train_loss:.4f} acc={train_acc*100:.1f}%")
        print(f"  Eval:  loss={eval_loss:.4f} acc={eval_acc*100:.1f}%")
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "epoch": epoch, "best_acc": best_acc, "args": vars(args)},
                        out_dir / "best_model.pt")
            print(f"  >> New best: {eval_acc*100:.1f}%")
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE — Best eval acc: {best_acc*100:.2f}%")
    print(f"  Model: {out_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()