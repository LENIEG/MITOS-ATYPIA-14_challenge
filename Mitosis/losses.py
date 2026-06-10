"""
Loss functions for mitosis heatmap detection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedHeatmapBCELoss(nn.Module):
    """
Weighted BCE-with-logits loss for sparse detection heatmaps.
"""

    def __init__(self, pos_weight: float = 15.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted BCE loss."""
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
        )


def get_loss_fn(pos_weight: float = 15.0) -> nn.Module:
    """Factory for loss instantiation."""
    return WeightedHeatmapBCELoss(pos_weight=pos_weight)
