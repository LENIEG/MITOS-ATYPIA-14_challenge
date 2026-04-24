"""
Atypia — training pipeline for nuclear atypia classification task.

Public modules:
  - config: Hyperparameter configuration
  - models: EfficientNet backbone + ordinal head
  - losses: CORN ordinal loss and weighted cross-entropy
  - metrics: Challenge-specific evaluation metrics
  - train: Main training loop with k-fold cross-validation
"""

from Atypia.config import Config, get_default_config
from Atypia.models import AtypiaModel, create_model
from Atypia.losses import CORNLoss, WeightedCELoss, get_loss_fn
from Atypia.metrics import AtypiaMetrics, ordinal_logits_to_predictions
from Atypia.train import main, train_fold

__all__ = [
    "Config",
    "get_default_config",
    "AtypiaModel",
    "create_model",
    "CORNLoss",
    "WeightedCELoss",
    "get_loss_fn",
    "AtypiaMetrics",
    "ordinal_logits_to_predictions",
    "main",
    "train_fold",
]
