#!/usr/bin/env python
"""
BraTS 3D CNN Inference Script
Loads the best trained model and runs inference on new .npy files.
Usage: python scripts/inference.py --case-id BraTS-GLI-XXXXX-XXX --npy-dir data/brats_preprocessed/train
       python scripts/inference.py --npy-file path/to/case.npy
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np


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


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    model = GradeClassifier3D().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def preprocess_npy(npy_path, target_shape=(182, 218, 182)):
    """Load and preprocess a .npy file (matches BraTS3DDataset exactly)"""
    data = np.load(npy_path).astype(np.float32)  # (4, H, W, D) or (H, W, D, 4)
    
    # Handle different shapes
    if data.shape[-1] == 4 and data.ndim == 4:
        # (H, W, D, 4) -> (4, H, W, D)
        data = np.transpose(data, (3, 0, 1, 2))
    elif data.shape[0] != 4:
        raise ValueError(f"Expected 4 channels, got shape {data.shape}")
    
    # Resize if needed (simple center crop/pad)
    c, h, w, d = data.shape
    th, tw, td = target_shape
    
    # Center crop or pad each dimension
    def crop_or_pad(arr, target, dim):
        curr = arr.shape[dim]
        if curr > target:
            start = (curr - target) // 2
            slices = [slice(None)] * arr.ndim
            slices[dim] = slice(start, start + target)
            return arr[tuple(slices)]
        elif curr < target:
            pad_before = (target - curr) // 2
            pad_after = target - curr - pad_before
            pads = [(0, 0)] * arr.ndim
            pads[dim] = (pad_before, pad_after)
            return np.pad(arr, pads, mode='constant')
        return arr
    
    data = crop_or_pad(data, th, 1)
    data = crop_or_pad(data, tw, 2)
    data = crop_or_pad(data, td, 3)
    
    # NO additional normalization - .npy files are already in [-1, 1] range
    # (BraTS3DDataset loads them directly)
    
    return torch.from_numpy(data).unsqueeze(0)  # (1, 4, H, W, D)


def predict(model, input_tensor, device):
    """Run inference and return probability and prediction"""
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob > 0.5 else 0
    return prob, pred


def main():
    parser = argparse.ArgumentParser(description='BraTS 3D CNN Glioma Grade Inference')
    parser.add_argument('--checkpoint', type=str, 
                        default='outputs/training/M1_best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--case-id', type=str, 
                        help='Case ID (looks for npy_dir/case_id.npy)')
    parser.add_argument('--npy-file', type=str, 
                        help='Direct path to .npy file')
    parser.add_argument('--npy-dir', type=str, 
                        default='data/brats_preprocessed/train',
                        help='Directory containing .npy files (used with --case-id)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, or auto')
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully.")
    
    # Determine input file
    if args.npy_file:
        npy_path = Path(args.npy_file)
        case_id = npy_path.stem
    elif args.case_id:
        npy_path = Path(args.npy_dir) / f"{args.case_id}.npy"
        case_id = args.case_id
    else:
        parser.error("Must provide either --case-id or --npy-file")
    
    if not npy_path.exists():
        print(f"ERROR: File not found: {npy_path}")
        return 1
    
    print(f"Processing: {npy_path}")
    
    # Preprocess
    input_tensor = preprocess_npy(npy_path)
    print(f"Input shape: {input_tensor.shape}")
    
    # Predict
    prob, pred = predict(model, input_tensor, device)
    
    grade = "HIGH (GBM)" if pred == 1 else "LOW (LGG)"
    confidence = prob if pred == 1 else 1 - prob
    
    print(f"\n{'='*50}")
    print(f"CASE: {case_id}")
    print(f"PREDICTION: {grade}")
    print(f"PROBABILITY (High-grade): {prob:.4f}")
    print(f"CONFIDENCE: {confidence:.4f}")
    print(f"{'='*50}")
    
    return 0


if __name__ == '__main__':
    exit(main())