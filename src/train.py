"""Training loop, evaluation, and utility helpers."""

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm


class EarlyStopper:
    def __init__(self, patience: int = 20):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_epoch(model: nn.Module, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total += y.size(0)
        correct += (out.argmax(1) == y).sum().item()
    if scheduler:
        scheduler.step()
    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_epoch(model: nn.Module, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, labels = [], []
    for x, y in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item()
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    acc = (torch.tensor(preds) == torch.tensor(labels)).float().mean().item()
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return total_loss / len(loader), acc, f1


def count_params(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{total/1e6:.2f}M total / {trainable/1e6:.2f}M trainable"
