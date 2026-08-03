#!/usr/bin/env python3
"""Main training entry point. Usage:
    python train.py                    # use configs/default.yaml
    python train.py --cfg configs/finetune.yaml
    python train.py --data /path/to/scans --epochs 200 --batch 8
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from src.data import make_dataloaders
from src.model import build_model
from src.train import train_epoch, eval_epoch, EarlyStopper, count_params
from src.utils import load_cfg, seed_everything, get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/default.yaml", help="config file path")
    p.add_argument("--data", default=None, help="override data.root")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args.cfg)
    seed = args.seed or cfg["training"]["seed"]
    seed_everything(seed)

    device = get_device(cfg["training"]["device"])
    print(f"Device: {device}  |  Seed: {seed}")

    # --- Data ---
    root = Path(args.data or cfg["data"]["root"])
    dl_train, dl_val, dl_test, n_classes = make_dataloaders(
        root=root,
        batch_size=args.batch or cfg["training"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 4),
        img_size=cfg["data"]["img_size"],
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
    )
    # Override model num_classes if inferred
    cfg["model"]["num_classes"] = n_classes

    # --- Model ---
    model = build_model(cfg["model"]).to(device)
    print(f"Model params: {count_params(model)}")

    # --- Optimizer / Scheduler ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr or cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    if cfg["training"].get("warmup_epochs", 0) > 0:
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                     total_iters=cfg["training"]["warmup_epochs"]),
                lr_scheduler.CosineAnnealingLR(optimizer,
                                               T_max=cfg["training"].get("scheduler_t0", 30)),
            ],
            milestones=[cfg["training"]["warmup_epochs"]],
        )
    else:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=cfg["training"].get("scheduler_t0", 30))

    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopper(cfg["training"].get("patience", 20))

    # --- Training loop ---
    best_acc = 0.0
    save_dir = Path(cfg["output"]["dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs or cfg["training"]["epochs"]):
        train_loss, train_acc = train_epoch(model, dl_train, optimizer, criterion, device, scheduler)
        val_loss, val_acc, val_f1 = eval_epoch(model, dl_val, criterion, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best.pt")

        if (epoch + 1) % cfg["output"].get("save_every", 5) == 0:
            torch.save(model.state_dict(), save_dir / f"checkpoint_epoch_{epoch+1}.pt")

        print(f"[{epoch+1}] Train loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"Val loss={val_loss:.4f} acc={val_acc:.3f} f1={val_f1:.3f}")

        if stopper.step(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

    # --- Test ---
    model.load_state_dict(torch.load(save_dir / "best.pt", map_location=device))
    test_loss, test_acc, test_f1 = eval_epoch(model, dl_test, criterion, device)
    print(f"\nTest acc={test_acc:.4f}  f1={test_f1:.4f}")


if __name__ == "__main__":
    main()
