"""
Model definition for atypia classification.

Wraps a pretrained EfficientNet backbone with an ordinal regression head.
The head outputs cumulative logits for ordinal classification (score 1/2/3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

from Atypia.config import ModelConfig


class OrdinalHead(nn.Module):
    """
    Ordinal regression head for 3-class ordinal target (1, 2, 3).
    
    Outputs K-1=2 logits representing cumulative thresholds:
      P(y >= 2) = sigmoid(logits[0])
      P(y >= 3) = sigmoid(logits[1])
    From these, we derive P(y=1), P(y=2), P(y=3).
    """
    
    def __init__(self, in_features: int, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.num_logits = num_classes - 1  # K-1 = 2 logits for 3 classes
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, self.num_logits)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_features)
        Returns:
            (batch, num_logits) ordinal logits
        """
        x = self.dropout(x)
        return self.fc(x)


class AtypiaModel(nn.Module):
    """
    EfficientNet backbone + ordinal regression head for atypia scoring.
    
    Predicts 3-way ordinal classification (low/moderate/high atypia).
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load pretrained backbone
        if config.backbone == "efficientnet_b3":
            self.backbone = models.efficientnet_b3(
                weights="IMAGENET1K_V1" if config.pretrained else None
            )
            backbone_out_features = 1536  # EfficientNet-B3 output channels
        else:
            raise ValueError(f"Unsupported backbone: {config.backbone}")
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Ordinal head
        self.head = OrdinalHead(
            in_features=backbone_out_features,
            num_classes=config.num_classes,
            dropout=config.dropout_rate
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, H, W) RGB image tensor
        Returns:
            (batch, 2) ordinal logits for 3-class classification
        """
        # Backbone
        features = self.backbone.features(x)          # (batch, 1536, h, w)
        
        # Global pooling
        pooled = self.global_pool(features)           # (batch, 1536, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)      # (batch, 1536)
        
        # Ordinal head
        logits = self.head(pooled)                    # (batch, 2)
        
        return logits


def create_model(config: ModelConfig, device: str = "cuda") -> AtypiaModel:
    """
    Instantiate the atypia model.
    
    Args:
        config: Model configuration
        device: "cuda" or "cpu"
    
    Returns:
        Model on the specified device
    """
    model = AtypiaModel(config)
    model = model.to(device)
    return model
