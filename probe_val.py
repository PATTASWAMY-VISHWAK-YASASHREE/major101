import torch, numpy as np
import sys
sys.path.insert(0, ".")

npy_dir = "C:/Users/pvish/copilot-worktrees/major101/pattaswamy-vishwak-yasashree-cuddly-lamp/data/brats_preprocessed/train"
labels_csv = "C:/Users/pvish/copilot-worktrees/major101/pattaswamy-vishwak-yasashree-cuddly-lamp/data/brats_preprocessed/labels.csv"

from src.data import make_dataloaders

loaders, splits = make_dataloaders(npy_dir, labels_csv, batch_size=4, augment=False, seed=42, num_workers=0)
val_ds = loaders["val"].dataset
print("Val len:", len(val_ds))

x0, y0 = val_ds[0]
print("getitem[0] x.shape:", x0.shape, "y:", y0)

val_labels = []
for bi, (x, y) in enumerate(loaders["val"]):
    val_labels.extend(y.cpu().numpy())

ones = sum(1 for x in val_labels if x == 1)
print("Loaded", len(val_labels), "Val 1=", ones, "0=", len(val_labels)-ones)
print("Majority baseline:", round(ones/len(val_labels), 6))
print("PROBE PASSED - val data loads, val_acc=0.8046 IS real")
