import torch, numpy as np
import pandas as pd
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
npy_dir = repo / "data" / "brats_preprocessed" / "npy"
labels_csv = repo / "data" / "brats_preprocessed" / "labels.csv"

print("npy_dir type:", type(npy_dir), "=", npy_dir)
print("exists:", npy_dir.exists())

from src.data import make_dataloaders

loaders, splits = make_dataloaders(npy_dir, labels_csv, batch_size=4, augment=False, seed=42, num_workers=0)
val_ds = loaders["val"].dataset
print("Val dataset len:", len(val_ds))
print("First 5 indices:", val_ds.indices[:5])
print("First 5 labels:", [val_ds.labels[i] for i in val_ds.indices[:5]])

x0, y0 = val_ds[0]
print("First item x.shape:", x0.shape, "y:", y0)

print("")
print("=== Dataloader iteration ===")
val_labels = []
for batch_i, (x, y) in enumerate(loaders["val"]):
    print("Batch", batch_i, "x=", x.shape, "y=", y.shape, "y_vals=", y.unique().tolist())
    val_labels.extend(y.cpu().numpy())
    if batch_i > 20:
        break

print("")
print("Loaded", len(val_labels), "labels")
ones = sum(1 for x in val_labels if x == 1)
zeros = sum(1 for x in val_labels if x == 0)
print("Label dist: 1=", ones, "0=", zeros)
print("Majority baseline:", round(ones / len(val_labels), 6))
print("PROBE PASSED - val data loads correctly")
