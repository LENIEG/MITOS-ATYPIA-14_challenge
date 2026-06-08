"""
Class-imbalance utilities for atypia training.
"""

from __future__ import annotations

import numpy as np
import torch

from CommonRoutines import AtypiaDataset


def compute_class_weights(dataset: AtypiaDataset, num_classes: int = 3) -> np.ndarray:
    """
    Compute inverse-frequency class weights from dataset labels.

    weight[i] = 1 / count[i], normalized so sum(weights) == num_classes.
    """
    labels = np.array([sample["label"] for sample in dataset._samples])
    class_counts = np.bincount(labels, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)  # avoid division by zero

    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(weights)
    return weights


def get_sample_weights(dataset: AtypiaDataset, class_weights: np.ndarray) -> torch.Tensor:
    """
    Map class weights to a per-sample tensor used by weighted samplers.
    """
    labels = np.array([sample["label"] for sample in dataset._samples])
    sample_weights = class_weights[labels]
    return torch.from_numpy(sample_weights).float()


def get_class_counts(dataset: AtypiaDataset, num_classes: int = 3) -> np.ndarray:
    """
    Return sample count per class for logging/debugging.
    """
    labels = np.array([sample["label"] for sample in dataset._samples])
    return np.bincount(labels, minlength=num_classes)
