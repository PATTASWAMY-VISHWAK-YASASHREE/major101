#!/usr/bin/env python3
"""
Ultra-lightweight training for 3GB VRAM + 3GB RAM constraint.
RTX 2050 (4GB) -> target 3GB VRAM, 3GB system RAM.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import numpy as np
import csv
import random
import time
import gc

# ========================
# MEMORY CONFIG - 2GB VRAM target
# ========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2              # increased from 1
GRAD_ACCUM = 4              # effective batch = 8
PATCH_SIZE = (96, 96, 96)   # same
MAX_EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Memory monitoring
def log_mem(tag=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserv = torch.cuda.memory_reserved() / 1024**3
        print(f"  [MEM] {tag}: alloc={alloc:.2f}GB reserv={reserv:.2f}GB")
    import psutil
    ram = psutil.Process().memory_info().rss / 1024**3
    print(f"  [RAM] {tag}: {ram:.2f}GB")

# ========================
# PROPER IMAGING-ONLY MODEL (~1.2M params)
# ========================
class Tiny3DCNN(nn.Module):
    def __init__(self, in_channels=4, base_channels=16):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, base_channels),
            nn.ReLU(inplace=True),
        )
        # Encoder with residual connections
        self.enc1 = self._block(base_channels, base_channels*2, stride=2)   # 48^3
        self.enc2 = self._block(base_channels*2, base_channels*4, stride=2) # 24^3
        self.enc3 = self._block(base_channels*4, base_channels*8, stride=2) # 12^3
        self.enc4 = self._block(base_channels*8, base_channels*16, stride=2) # 6^3
        
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(base_channels*16, 1)
        self._init_weights()
    
    def _block(self, c_in, c_out, stride):
        return nn.Sequential(
            nn.Conv3d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(min(8, c_out//2), c_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_out, c_out, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, c_out//2), c_out),
            nn.ReLU(inplace=True),
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.head(x).squeeze(-1)

# ========================
# DATASET - IMAGING ONLY
# ========================
class UltraLightDataset(Dataset):
    def __init__(self, npy_dir, labels_list, patch_size=(96,96,96), augment=False):
        self.npy_dir = Path(npy_dir)
        self.labels = labels_list  # list of (case_id, label)
        self.patch_size = patch_size
        self.augment = augment
    
    def __len__(self):
        return len(self.labels)
    
    def _random_crop(self, vol):
        """Crop random patch from volume"""
        _, d, h, w = vol.shape
        pd, ph, pw = self.patch_size
        sd = random.randint(0, max(0, d - pd))
        sh = random.randint(0, max(0, h - ph))
        sw = random.randint(0, max(0, w - pw))
        return vol[:, sd:sd+pd, sh:sh+ph, sw:sw+pw]
    
    def _center_crop(self, vol):
        _, d, h, w = vol.shape
        pd, ph, pw = self.patch_size
        sd = (d - pd) // 2
        sh = (h - ph) // 2
        sw = (w - pw) // 2
        return vol[:, sd:sd+pd, sh:sh+ph, sw:sw+pw]
    
    def __getitem__(self, idx):
        case_id, label = self.labels[idx]
        # Load on-the-fly (no caching) with memory mapping
        vol = np.load(self.npy_dir / f"{case_id}.npy", mmap_mode='r')
        vol = vol.astype(np.float32)
        
        # Per-volume normalization (cheap)
        mean = vol.mean()
        std = vol.std() + 1e-8
        vol = (vol - mean) / std
        
        # Crop
        if self.augment:
            vol = self._random_crop(vol)
            # Flip augment
            if random.random() > 0.5: vol = np.flip(vol, axis=1).copy()
            if random.random() > 0.5: vol = np.flip(vol, axis=2).copy()
            if random.random() > 0.5: vol = np.flip(vol, axis=3).copy()
        else:
            vol = self._center_crop(vol)
        
        return torch.from_numpy(vol), torch.tensor(label, dtype=torch.float32)

def load_labels(csv_path):
    labels = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = row['case']
            grade = int(row['grade_proxy'])
            labels.append((case, grade))
    return labels

def make_dataloaders(npy_dir, labels_csv, val_split=0.2, seed=42):
    labels = load_labels(labels_csv)
    random.Random(seed).shuffle(labels)
    split = int(len(labels) * (1 - val_split))
    train_labels = labels[:split]
    val_labels = labels[split:]
    
    train_ds = UltraLightDataset(npy_dir, train_labels, PATCH_SIZE, augment=True)
    val_ds = UltraLightDataset(npy_dir, val_labels, PATCH_SIZE, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)
    return train_loader, val_loader

# ========================
# CLASS-BALANCED LOSS
# ========================
class CBLoss(nn.Module):
    def __init__(self, n_classes=2, beta=0.999):
        super().__init__()
        self.beta = beta
        self.n_classes = n_classes
    
    def forward(self, logits, targets, class_counts):
        # class_counts: tensor [n_classes]
        effective_num = 1.0 - self.beta ** class_counts.float()
        weights = (1.0 - self.beta) / effective_num
        weights = weights / weights.sum() * self.n_classes
        weight_per_sample = weights[targets.long()]
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return (bce * weight_per_sample).mean()

# ========================
# TRAINING
# ========================
def train_one_epoch(model, loader, optimizer, scaler, criterion, class_counts, epoch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0
    correct = 0
    total = 0
    
    for step, (x, y) in enumerate(loader):
        x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last_3d)
        y = y.to(DEVICE, non_blocking=True)
        
        with autocast():
            logits = model(x)
            loss = criterion(logits, y, class_counts) / GRAD_ACCUM
        
        scaler.scale(loss).backward()
        
        if (step + 1) % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item() * GRAD_ACCUM
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.numel()
        
        # Aggressive memory cleanup
        del x, y, logits, loss
        if step % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    return total_loss / len(loader), correct / total if total else 0

@torch.no_grad()
def validate(model, loader, criterion, class_counts):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last_3d)
        y = y.to(DEVICE, non_blocking=True)
        
        with autocast():
            logits = model(x)
            loss = criterion(logits, y, class_counts)
        
        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.numel()
        all_preds.append(probs.cpu())
        all_targets.append(y.cpu())
        
        del x, y, logits, loss
        torch.cuda.empty_cache()
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    # Metrics
    tp = ((all_preds > 0.5) & (all_targets == 1)).sum().item()
    tn = ((all_preds <= 0.5) & (all_targets == 0)).sum().item()
    fp = ((all_preds > 0.5) & (all_targets == 0)).sum().item()
    fn = ((all_preds <= 0.5) & (all_targets == 1)).sum().item()
    
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    f1 = 2*tp / (2*tp + fp + fn + 1e-8)
    
    return total_loss / len(loader), acc, f1, sens, spec

# ========================
# MAIN
# ========================
def main():
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    log_mem("startup")
    
    # Paths
    npy_dir = Path("data/brats_preprocessed/train")
    labels_csv = Path("data/brats_preprocessed/labels.csv")
    
    if not npy_dir.exists() or not labels_csv.exists():
        print("ERROR: Preprocessed data not found!")
        return
    
    # Load labels once to get class counts
    labels = load_labels(labels_csv)
    class_counts = torch.tensor([
        sum(1 for _, l in labels if l == 0),
        sum(1 for _, l in labels if l == 1)
    ], device=DEVICE)
    print(f"Class counts: {class_counts.tolist()}")
    
    # Data
    train_loader, val_loader = make_dataloaders(npy_dir, labels_csv)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    log_mem("after data")
    
    # Model
    model = Tiny3DCNN().to(DEVICE)
    model = model.to(memory_format=torch.channels_last_3d)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count/1000:.0f}K")
    log_mem("after model")
    
    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = CBLoss(beta=0.999).to(DEVICE)
    scaler = GradScaler()
    log_mem("after optimizer")
    
    # Quick test run (2 epochs)
    print("\n=== QUICK TEST (2 epochs) ===")
    for epoch in range(2):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion, class_counts, epoch)
        val_loss, val_acc, val_f1, val_sens, val_spec = validate(model, val_loader, criterion, class_counts)
        scheduler.step()
        dt = time.time() - t0
        print(f"Epoch {epoch+1}/2: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} f1={val_f1:.4f} sens={val_sens:.4f} spec={val_spec:.4f} | {dt:.1f}s")
        log_mem(f"epoch {epoch+1}")
    
    print("\n=== TEST PASSED - Ready for full run ===")
    print("Run with: python scripts/train_ultra_light.py --full")
    
    # Full run if requested
    import sys
    if '--full' in sys.argv:
        print("\n=== FULL TRAINING ===")
        best_val_acc = 0
        for epoch in range(2, MAX_EPOCHS):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion, class_counts, epoch)
            val_loss, val_acc, val_f1, val_sens, val_spec = validate(model, val_loader, criterion, class_counts)
            scheduler.step()
            dt = time.time() - t0
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "outputs/training/ultra_light_best.pth")
                print(f"  *** NEW BEST: {val_acc:.4f} ***")
            
            print(f"Epoch {epoch+1}/{MAX_EPOCHS}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} f1={val_f1:.4f} sens={val_sens:.4f} spec={val_spec:.4f} | {dt:.1f}s")
            log_mem(f"epoch {epoch+1}")

if __name__ == "__main__":
    main()