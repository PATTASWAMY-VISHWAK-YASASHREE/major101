#!/usr/bin/env python
"""
BraTS 3D CNN Inference + Visualization Script
Generates visual outputs: attention maps, slice overlays, saliency maps
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class GradeClassifier3D(nn.Module):
    """3D CNN for BraTS grade classification."""

    def __init__(self):
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
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x).squeeze(-1)

    def get_feature_maps(self, x):
        """Return intermediate feature maps for visualization"""
        feat_maps = []
        for block in self.features:
            x = block(x)
            feat_maps.append(x)
        return feat_maps


def load_model(checkpoint_path, device):
    """Load model from checkpoint, supporting both M1 and M1_features architectures."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Check if it's the feature-augmented model (M1_features)
    if 'scalar_head.0.weight' in ckpt:
        # Inline the EXACT model definition from train_m1_features.py
        class GradeClassifier3DWithFeatures(nn.Module):
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
        
        model = GradeClassifier3DWithFeatures().to(device)
    else:
        model = GradeClassifier3D().to(device)
    
    model.load_state_dict(ckpt)
    model.eval()
    return model


def preprocess_npy(npy_path, target_shape=(182, 218, 182)):
    data = np.load(npy_path).astype(np.float32)
    if data.shape[-1] == 4 and data.ndim == 4:
        data = np.transpose(data, (3, 0, 1, 2))
    elif data.shape[0] != 4:
        raise ValueError(f"Expected 4 channels, got shape {data.shape}")
    
    c, h, w, d = data.shape
    th, tw, td = target_shape
    
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
    
    return torch.from_numpy(data).unsqueeze(0)


def compute_gradcam_3d(model, input_tensor, target_layer_idx=-1, scalar_features=None):
    """Compute 3D Grad-CAM for the last conv block"""
    model.eval()
    
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    target_layer = model.features[target_layer_idx]
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    
    input_tensor.requires_grad_(True)
    if scalar_features is not None:
        logits = model(input_tensor, scalar_features)
    else:
        logits = model(input_tensor)
    prob = torch.sigmoid(logits)
    
    model.zero_grad()
    prob.backward()
    
    act = activations[0]
    grad = gradients[0]
    
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='trilinear', align_corners=False)
    
    handle_f.remove()
    handle_b.remove()
    
    return cam.squeeze().detach().cpu().numpy(), prob.item()


def compute_saliency_map(model, input_tensor):
    """Compute input saliency map (gradient w.r.t input)"""
    model.eval()
    input_tensor.requires_grad_(True)
    logits = model(input_tensor)
    prob = torch.sigmoid(logits)
    model.zero_grad()
    prob.backward()
    saliency = input_tensor.grad.abs().max(dim=1)[0].squeeze().detach().cpu().numpy()
    saliency = saliency / (saliency.max() + 1e-8)
    return saliency, prob.item()


def save_slice_visualization(npy_path, output_dir, cam=None, saliency=None, prob=None, pred=None, case_id=None):
    """Save multi-slice visualization with overlays"""
    data = np.load(npy_path).astype(np.float32)
    if data.shape[-1] == 4 and data.ndim == 4:
        data = np.transpose(data, (3, 0, 1, 2))
    
    modal_names = ['T1', 'T1ce', 'T2', 'FLAIR']
    H, W, D = data.shape[1], data.shape[2], data.shape[3]
    mid_h, mid_w, mid_d = H//2, W//2, D//2
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    
    for m in range(4):
        im = axes[m, 0].imshow(data[m, :, :, mid_d].T, cmap='gray', origin='lower')
        axes[m, 0].set_title(f'{modal_names[m]} - Axial (z={mid_d})')
        axes[m, 0].axis('off')
        plt.colorbar(im, ax=axes[m, 0], fraction=0.046, pad=0.04)
        
        im = axes[m, 1].imshow(data[m, :, mid_w, :].T, cmap='gray', origin='lower')
        axes[m, 1].set_title(f'{modal_names[m]} - Coronal (y={mid_w})')
        axes[m, 1].axis('off')
        plt.colorbar(im, ax=axes[m, 1], fraction=0.046, pad=0.04)
        
        im = axes[m, 2].imshow(data[m, mid_h, :, :].T, cmap='gray', origin='lower')
        axes[m, 2].set_title(f'{modal_names[m]} - Sagittal (x={mid_h})')
        axes[m, 2].axis('off')
        plt.colorbar(im, ax=axes[m, 2], fraction=0.046, pad=0.04)
    
    if case_id:
        fig.suptitle(f'{case_id} | Pred: {"HIGH (GBM)" if pred==1 else "LOW (LGG)"} | Prob: {prob:.4f}', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{case_id}_modalities.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    if cam is not None:
        save_cam_overlays(data, cam, output_dir, case_id, prob, pred)
    
    if saliency is not None:
        save_saliency_overlays(data, saliency, output_dir, case_id, prob, pred)


def save_cam_overlays(data, cam, output_dir, case_id, prob, pred):
    H, W, D = cam.shape
    mid_h, mid_w, mid_d = H//2, W//2, D//2
    bg = data[1]  # T1ce
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    slices = [
        (bg[:, :, mid_d].T, cam[:, :, mid_d].T, f'Axial (z={mid_d})'),
        (bg[:, mid_w, :].T, cam[:, mid_w, :].T, f'Coronal (y={mid_w})'),
        (bg[mid_h, :, :].T, cam[mid_h, :, :].T, f'Sagittal (x={mid_h})'),
    ]
    
    for i, (bg_slice, cam_slice, title) in enumerate(slices):
        axes[i].imshow(bg_slice, cmap='gray', origin='lower')
        im = axes[i].imshow(cam_slice, cmap='jet', alpha=0.5, origin='lower', vmin=0, vmax=1)
        axes[i].set_title(f'Grad-CAM on T1ce - {title}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    fig.suptitle(f'{case_id} | Grad-CAM Attention Map | Pred: {"HIGH" if pred==1 else "LOW"} ({prob:.4f})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'{case_id}_gradcam.png', dpi=150, bbox_inches='tight')
    plt.close()


def save_saliency_overlays(data, saliency, output_dir, case_id, prob, pred):
    H, W, D = saliency.shape
    mid_h, mid_w, mid_d = H//2, W//2, D//2
    bg = data[1]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    slices = [
        (bg[:, :, mid_d].T, saliency[:, :, mid_d].T, f'Axial (z={mid_d})'),
        (bg[:, mid_w, :].T, saliency[:, mid_w, :].T, f'Coronal (y={mid_w})'),
        (bg[mid_h, :, :].T, saliency[mid_h, :, :].T, f'Sagittal (x={mid_h})'),
    ]
    
    for i, (bg_slice, sal_slice, title) in enumerate(slices):
        axes[i].imshow(bg_slice, cmap='gray', origin='lower')
        im = axes[i].imshow(sal_slice, cmap='hot', alpha=0.6, origin='lower', vmin=0, vmax=1)
        axes[i].set_title(f'Saliency Map - {title}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    fig.suptitle(f'{case_id} | Input Saliency Map | Pred: {"HIGH" if pred==1 else "LOW"} ({prob:.4f})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'{case_id}_saliency.png', dpi=150, bbox_inches='tight')
    plt.close()


def analyze_tumor_characteristics(npy_path):
    """Extract tumor characteristics from the 4 modalities"""
    data = np.load(npy_path).astype(np.float32)
    if data.shape[-1] == 4 and data.ndim == 4:
        data = np.transpose(data, (3, 0, 1, 2))
    
    t1ce = data[1]
    t2 = data[2]
    flair = data[3]
    
    # Since data is normalized to [-1, 1], use relative thresholds
    # High values in T1ce = enhancing tumor
    et_mask = t1ce > 0.3  # More sensitive threshold
    wt_mask = (t2 > 0.1) | (flair > 0.1)  # Edema + tumor
    
    et_volume = int(et_mask.sum())
    wt_volume = int(wt_mask.sum())
    et_wt_ratio = et_volume / (wt_volume + 1e-8)
    
    return {
        'et_volume_voxels': et_volume,
        'wt_volume_voxels': wt_volume,
        'et_wt_ratio': float(et_wt_ratio),
        't1ce_mean': float(t1ce.mean()),
        't1ce_max': float(t1ce.max()),
        't1ce_std': float(t1ce.std()),
        'has_enhancing': bool(et_volume > 500),  # Convert to Python bool
    }


def main():
    parser = argparse.ArgumentParser(description='BraTS 3D CNN Inference + Visualization')
    parser.add_argument('--checkpoint', type=str, default='outputs/training/M1_best.pth')
    parser.add_argument('--case-id', type=str, help='Case ID')
    parser.add_argument('--npy-file', type=str, help='Direct path to .npy file')
    parser.add_argument('--npy-dir', type=str, default='data/brats_preprocessed/train')
    parser.add_argument('--labels-csv', type=str, default='data/brats_preprocessed/labels.csv')
    parser.add_argument('--output-dir', type=str, default='outputs/visualizations')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--no-gradcam', action='store_true', help='Skip Grad-CAM (faster)')
    parser.add_argument('--no-saliency', action='store_true', help='Skip saliency map')
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    print("Model loaded.")
    
    # Check if it's the feature-augmented model
    is_feature_model = hasattr(model, 'scalar_head')
    
    # Load labels for scalar features
    labels_df = pd.read_csv(args.labels_csv)
    labels_df.columns = [c.strip().lower() for c in labels_df.columns]
    labels_df['label'] = labels_df['grade_proxy'].astype(int)
    labels_df = labels_df.rename(columns={'case': 'case_id'})
    
    if args.npy_file:
        npy_path = Path(args.npy_file)
        case_id = npy_path.stem
    elif args.case_id:
        npy_path = Path(args.npy_dir) / f"{args.case_id}.npy"
        case_id = args.case_id
    else:
        parser.error("Must provide --case-id or --npy-file")
    
    if not npy_path.exists():
        print(f"ERROR: File not found: {npy_path}")
        return 1
    
    print(f"Processing: {npy_path}")
    
    input_tensor = preprocess_npy(npy_path).to(device)
    
    # Compute scalar features if using feature-augmented model
    scalar_features = None
    if is_feature_model:
        # Inline scalar feature computation to avoid import issues
        import numpy as np
        case_row = labels_df[labels_df['case_id'] == case_id]
        if len(case_row) > 0:
            row = case_row.iloc[0]
            et_vol = row['et_volume']
            wt_vol = row['wt_volume']
            et_log = np.log1p(et_vol) / 12.0
            wt_log = np.log1p(wt_vol) / 12.0
            et_wt_ratio = et_vol / wt_vol if wt_vol > 0 else 0.0
            scalar_features = torch.tensor([[et_log, wt_log, et_wt_ratio]], dtype=torch.float32).to(device)
            print(f"Scalar features: {scalar_features}")
    
    with torch.no_grad():
        if is_feature_model and scalar_features is not None:
            logits = model(input_tensor, scalar_features)
        else:
            logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob > 0.5 else 0
    
    chars = analyze_tumor_characteristics(npy_path)
    
    print(f"\n{'='*60}")
    print(f"CASE: {case_id}")
    print(f"PREDICTION: {'HIGH (GBM)' if pred==1 else 'LOW (LGG)'}")
    print(f"PROBABILITY: {prob:.4f}")
    print(f"CONFIDENCE: {prob if pred==1 else 1-prob:.4f}")
    print(f"TUMOR CHARS: ET_vol={chars['et_volume_voxels']}, WT_vol={chars['wt_volume_voxels']}, ET/WT={chars['et_wt_ratio']:.3f}")
    print(f"HAS ENHANCING: {chars['has_enhancing']}")
    print(f"{'='*60}\n")
    
    print("Saving modality slices...")
    save_slice_visualization(npy_path, output_dir, case_id=case_id, prob=prob, pred=pred)
    
    cam = None
    if not args.no_gradcam:
        print("Computing Grad-CAM...")
        cam, _ = compute_gradcam_3d(model, input_tensor.clone(), scalar_features=scalar_features)
        save_slice_visualization(npy_path, output_dir, cam=cam, prob=prob, pred=pred, case_id=case_id)
    
    sal = None
    if not args.no_saliency:
        print("Computing saliency map...")
        sal, _ = compute_saliency_map(model, input_tensor.clone())
        save_slice_visualization(npy_path, output_dir, saliency=sal, prob=prob, pred=pred, case_id=case_id)
    
    import json
    analysis = {
        'case_id': case_id,
        'prediction': 'HIGH' if pred==1 else 'LOW',
        'probability': prob,
        'tumor_characteristics': chars,
    }
    with open(output_dir / f'{case_id}_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nVisualizations saved to: {output_dir}/")
    print(f"  - {case_id}_modalities.png (4 modalities x 3 views)")
    if cam is not None:
        print(f"  - {case_id}_gradcam.png (Grad-CAM attention)")
    if sal is not None:
        print(f"  - {case_id}_saliency.png (Input saliency)")
    print(f"  - {case_id}_analysis.json (Quantitative analysis)")
    
    return 0


if __name__ == '__main__':
    exit(main())