# Major101 — Brain Tumour Classification from MRI/CT

Longitudinal volumetric deep learning for brain tumour detection and grading.

## Quick start

```bash
pip install -r requirements.txt
python train.py                          # uses configs/default.yaml
python train.py --data /path/to/scans    # override data root
python train.py --epochs 200 --batch 8   # CLI overrides
```

## Data format

Place `.nii.gz` volumes in `data/` organised by class:

```
data/
├── train/
│   ├── 0/          # class 0 (e.g. no tumour)
│   │   ├── scan01.nii.gz
│   │   └── scan02.nii.gz
│   └── 1/          # class 1 (e.g. low grade)
│       └── scan03.nii.gz
├── val/
└── test/
```

Class labels are auto-inferred from directory names. The `train.py` split logic
will repartition a flat `data/` dir into `train/val/test/` if subdirectories
don't exist yet.

## Config

Edit `configs/default.yaml` for hyperparameters. CLI args override config.

## Model

`src/model.py` ships a lightweight 3D ResNet (~few M params). Swap in
`DenseNet3D` or `ConvNeXt3D` from MONAI when validation plateaus.

## Outputs

Checkpoints saved to `outputs/`. Best model is `outputs/best.pt`.
