"""
Heatmap ensembling helpers for mitosis inference and evaluation.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from Mitosis.config import Config
from Mitosis.models import create_model


def parse_checkpoint_paths(
    checkpoint: str | None = None,
    checkpoints: str | None = None,
) -> list[Path]:
    """
    Parse checkpoint arguments into a validated list of paths.

    Priority is given to --checkpoints when provided.
    """
    paths: list[Path] = []

    if checkpoints:
        for raw in checkpoints.split(","):
            p = Path(raw.strip())
            if raw.strip():
                paths.append(p)
    elif checkpoint:
        paths.append(Path(checkpoint))

    if not paths:
        raise ValueError("No checkpoint paths provided.")

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Checkpoint(s) not found: {missing}")

    return paths


def load_models_from_checkpoints(
    cfg: Config,
    checkpoint_paths: list[Path],
) -> list[nn.Module]:
    """Instantiate one model per checkpoint and load state dicts."""
    models: list[nn.Module] = []
    for ckpt in checkpoint_paths:
        model = create_model(cfg.model, device=cfg.device)
        state = torch.load(ckpt, map_location=cfg.device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


def predict_ensemble_logits(
    models: list[nn.Module],
    images: torch.Tensor,
) -> torch.Tensor:
    """
    Average per-model logits to form ensemble logits.
    """
    if not models:
        raise ValueError("Ensemble requires at least one model.")

    logits_sum = None
    for model in models:
        logits = model(images)
        logits_sum = logits if logits_sum is None else (logits_sum + logits)
    return logits_sum / float(len(models))
