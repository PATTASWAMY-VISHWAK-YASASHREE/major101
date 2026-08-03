"""3D CNN models for brain tumour classification."""

import torch
import torch.nn as nn


class BasicBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class ResNet3D(nn.Module):
    """
    Lightweight 3D ResNet for volumetric MRI/CT.
    # ponytail: shallow depth (18 blocks) for fast iteration; switch to DenseNet3D/ConvNeXt3D
    when validation accuracy plateaus.
    """

    def __init__(
        self,
        block=BasicBlock3D,
        layers: list[int] = [2, 2, 2, 2],
        input_ch: int = 1,
        num_classes: int = 4,
        base_width: int = 32,
    ):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(input_ch, base_width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2),
        )
        # Stage widths double each time
        widths = [base_width * (2 ** i) for i in range(len(layers))]
        self.layers = nn.ModuleList()
        prev_ch = base_width
        for i, (l, out_ch) in enumerate(zip(layers, widths)):
            self.layers.append(self._make_layer(block, prev_ch, out_ch, l, i == 0))
            prev_ch = out_ch

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(prev_ch, num_classes),
        )

    def _make_layer(self, block, in_ch: int, out_ch: int, count: int, first: bool) -> nn.Sequential:
        strides = [1 if first else 2] + [1] * (count - 1)
        return nn.Sequential(*[
            block(in_ch if i == 0 else out_ch, out_ch, strides[i])
            for i in range(count)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def build_model(cfg: dict) -> nn.Module:
    model_type = cfg.get("type", "resnet3d").lower()
    if model_type == "resnet3d":
        return ResNet3D(
            layers=cfg.get("layers", [2, 2, 2, 2]),
            input_ch=cfg.get("input_channels", 1),
            num_classes=cfg.get("num_classes", 4),
            base_width=cfg.get("base_width", 32),
        )
    raise ValueError(f"Unknown model type: {model_type}")
