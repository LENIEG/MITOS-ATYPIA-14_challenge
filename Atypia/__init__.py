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
from Atypia.fold_merge import build_and_save_merged_model
from Atypia.class_balance import compute_class_weights, get_sample_weights, get_class_counts


def main(*args, **kwargs):
  """Lazy proxy to avoid importing Atypia.train at package import time."""
  from Atypia.train import main as _main
  return _main(*args, **kwargs)


def train_fold(*args, **kwargs):
  """Lazy proxy to avoid importing Atypia.train at package import time."""
  from Atypia.train import train_fold as _train_fold
  return _train_fold(*args, **kwargs)

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
    "build_and_save_merged_model",
    "compute_class_weights",
    "get_sample_weights",
    "get_class_counts",
    "main",
    "train_fold",
]
