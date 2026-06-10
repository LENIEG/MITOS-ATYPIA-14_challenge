"""
Mitosis - training pipeline for mitosis centroid detection task.
"""

from Mitosis.config import Config, get_default_config
from Mitosis.models import MitosisHeatmapModel, create_model
from Mitosis.losses import WeightedHeatmapBCELoss, get_loss_fn
from Mitosis.metrics import MitosisMetrics, match_detections
from Mitosis.heatmap import build_target_heatmaps, decode_heatmaps
from Mitosis.ensemble import (
    load_models_from_checkpoints,
    parse_checkpoint_paths,
    predict_ensemble_logits,
)


def main(*args, **kwargs):
    """Lazy proxy to avoid importing Mitosis.train at package import time."""
    from Mitosis.train import main as _main

    return _main(*args, **kwargs)


def train_fold(*args, **kwargs):
    """Lazy proxy to avoid importing Mitosis.train at package import time."""
    from Mitosis.train import train_fold as _train_fold

    return _train_fold(*args, **kwargs)


def visualize_inference(*args, **kwargs):
    """Lazy proxy to avoid importing visualization module at package import time."""
    from Mitosis.visualize_inference import run as _run

    return _run(*args, **kwargs)


__all__ = [
    "Config",
    "get_default_config",
    "MitosisHeatmapModel",
    "create_model",
    "WeightedHeatmapBCELoss",
    "get_loss_fn",
    "MitosisMetrics",
    "match_detections",
    "build_target_heatmaps",
    "decode_heatmaps",
    "load_models_from_checkpoints",
    "parse_checkpoint_paths",
    "predict_ensemble_logits",
    "main",
    "train_fold",
    "visualize_inference",
]
