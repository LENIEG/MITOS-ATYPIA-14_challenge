"""
Model definition for mitosis heatmap detection.

Uses an EfficientNet backbone and a lightweight upsampling decoder that
predicts a dense 1-channel mitosis heatmap.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from Mitosis.config import ModelConfig


class MitosisHeatmapModel(nn.Module):
    """EfficientNet backbone + upsampling heatmap head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        if config.backbone == "efficientnet_b3":
            backbone = models.efficientnet_b3(
                weights="IMAGENET1K_V1" if config.pretrained else None
            )
            backbone_out = 1536
        else:
            raise ValueError(f"Unsupported backbone: {config.backbone}")

        self.backbone = backbone.features

        self.decoder = nn.Sequential(
            nn.Conv2d(backbone_out, config.decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.decoder_channels, config.decoder_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.decoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.decoder_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, input_size, input_size)
        Returns:
            logits: (batch, 1, input_size / output_stride, input_size / output_stride)
        """
        features = self.backbone(x)
        logits_small = self.decoder(features)

        out_h = self.config.input_size // self.config.output_stride
        out_w = self.config.input_size // self.config.output_stride
        logits = F.interpolate(logits_small, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return logits


def create_model(config: ModelConfig, device: str = "cuda") -> MitosisHeatmapModel:
    """Instantiate the mitosis model and move it to the requested device."""
    model = MitosisHeatmapModel(config)
    return model.to(device)
