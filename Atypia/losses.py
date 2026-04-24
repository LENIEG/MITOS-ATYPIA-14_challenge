"""
Loss functions for ordinal atypia classification.

Implements CORN (Cumulative Output Regression Network) loss for ordinal targets,
and weighted cross-entropy as fallback.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CORNLoss(nn.Module):
    """
    Cumulative Output Regression Network (CORN) loss.
    
    For ordinal regression with K classes, uses K-1 output units (thresholds).
    Encourages monotonic cumulative probabilities: P(y≥1) ≤ P(y≥2) ≤ P(y≥3).
    
    Reference: Cao et al. "Rank Consistent Ordinal Regression for Neural Networks"
    """
    
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_logits = num_classes - 1
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, K-1) unnormalized ordinal thresholds
            targets: (batch,) integer labels in {0, 1, 2} for 3-class
        
        Returns:
            scalar loss
        """
        # Convert targets (0, 1, 2) to binary task targets
        # For 3 classes: target=0 → set1=[0, 0], target=1 → set1=[1, 0], target=2 → set1=[1, 1]
        set_labels = self._ordinal_targets_to_binary(targets)  # (batch, K-1)
        
        # Binary cross-entropy on each threshold
        loss = F.binary_cross_entropy_with_logits(logits, set_labels.float())
        return loss
    
    def _ordinal_targets_to_binary(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Convert ordinal targets to binary task labels.
        
        For 3 classes (0, 1, 2):
          target=0 (class 1): [0, 0] (y < 2 and y < 3)
          target=1 (class 2): [1, 0] (y >= 2 and y < 3)
          target=2 (class 3): [1, 1] (y >= 2 and y >= 3)
        """
        batch_size = targets.size(0)
        set_labels = torch.zeros(batch_size, self.num_logits, device=targets.device)
        
        for i in range(self.num_logits):
            # For threshold i: set_labels[:, i] = 1 if target > i else 0
            set_labels[:, i] = (targets > i).long()
        
        return set_labels


class WeightedCELoss(nn.Module):
    """
    Weighted cross-entropy loss with class weights.
    
    Penalizes off-by-2 errors harder (e.g., predicting 0 when target is 2).
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        class_weights: list[float] | None = None,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        
        if class_weights is None:
            class_weights = [1.0] * num_classes
        
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32)
        )
    
    def forward(
        self,
        logits_3class: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits_3class: (batch, 3) regular softmax logits (NOT ordinal)
            targets: (batch,) integer labels in {0, 1, 2}
        
        Returns:
            scalar loss
        """
        # Standard weighted cross-entropy with label smoothing
        loss = F.cross_entropy(
            logits_3class,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        return loss


def get_loss_fn(
    loss_type: str = "ordinal",
    num_classes: int = 3,
    class_weights: list[float] | None = None,
    label_smoothing: float = 0.1,
) -> nn.Module:
    """
    Factory function to create the loss function.
    
    Args:
        loss_type: "ordinal" (CORN) or "weighted_ce"
        num_classes: number of classes (3 for atypia)
        class_weights: per-class weights (if None, uniform)
        label_smoothing: for CE loss
    
    Returns:
        Loss module
    """
    if loss_type == "ordinal":
        return CORNLoss(num_classes=num_classes)
    elif loss_type == "weighted_ce":
        return WeightedCELoss(
            num_classes=num_classes,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
