"""
Evaluation metrics for atypia task.

Computes accuracy, balanced accuracy, and the challenge-specific weighted score
where off-by-2 errors are penalized with -1 point.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


class AtypiaMetrics:
    """
    Compute metrics for atypia classification.
    
    Challenge scoring:
      +1 point for correct prediction
      0 points for off-by-1 error
      -1 point for off-by-2 error
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Clear accumulated predictions and targets."""
        self.predictions: list[int] = []
        self.targets: list[int] = []
    
    def update(
        self,
        predictions: np.ndarray | list[int],
        targets: np.ndarray | list[int],
    ) -> None:
        """
        Accumulate predictions and targets.
        
        Args:
            predictions: predicted class indices (0, 1, 2)
            targets: ground truth class indices (0, 1, 2)
        """
        self.predictions.extend(np.array(predictions).flatten().tolist())
        self.targets.extend(np.array(targets).flatten().tolist())
    
    def accuracy(self) -> float:
        """Compute standard accuracy."""
        return accuracy_score(self.targets, self.predictions)
    
    def balanced_accuracy(self) -> float:
        """Compute balanced accuracy (macro recall)."""
        return balanced_accuracy_score(self.targets, self.predictions)
    
    def challenge_score(self) -> float:
        """
        Compute the MITOS-ATYPIA-14 challenge score.
        
        Scoring per sample:
          +1 if prediction == target
          0 if |prediction - target| == 1
          -1 if |prediction - target| == 2
        
        Return average score per sample.
        """
        scores = []
        for pred, target in zip(self.predictions, self.targets):
            diff = abs(pred - target)
            if diff == 0:
                scores.append(1)
            elif diff == 1:
                scores.append(0)
            else:  # diff == 2
                scores.append(-1)
        return float(np.mean(scores))
    
    def confusion_matrix(self) -> np.ndarray:
        """Return (3, 3) confusion matrix."""
        return confusion_matrix(self.targets, self.predictions, labels=[0, 1, 2])
    
    def per_class_accuracy(self) -> dict[str, float]:
        """Compute per-class recall (sensitivity)."""
        cm = self.confusion_matrix()
        class_names = ["Low (1)", "Moderate (2)", "High (3)"]
        acc = {}
        
        for i, name in enumerate(class_names):
            row_sum = cm[i].sum()
            if row_sum > 0:
                acc[name] = cm[i, i] / row_sum
            else:
                acc[name] = 0.0
        
        return acc
    
    def summary(self) -> dict[str, float]:
        """Return all metrics as a dictionary."""
        return {
            "accuracy": self.accuracy(),
            "balanced_accuracy": self.balanced_accuracy(),
            "challenge_score": self.challenge_score(),
            **{f"recall_{k}": v for k, v in self.per_class_accuracy().items()},
        }


def ordinal_logits_to_predictions(logits: np.ndarray) -> np.ndarray:
    """
    Convert ordinal logits (batch, 2) to class predictions (batch,).
    
    For 3-class ordinal:
      logits[:, 0] = log-odds of y >= 2
      logits[:, 1] = log-odds of y >= 3
    
    Predictions:
      y = 0 (class 1) if P(y>=2) < 0.5
      y = 1 (class 2) if P(y>=2) >= 0.5 and P(y>=3) < 0.5
      y = 2 (class 3) if P(y>=3) >= 0.5
    """
    batch_size = logits.shape[0]
    preds = np.zeros(batch_size, dtype=np.int64)
    
    # Convert logits to probabilities
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    
    # Decision thresholds
    for i in range(batch_size):
        if probs[i, 1] >= 0.5:
            preds[i] = 2  # class 3
        elif probs[i, 0] >= 0.5:
            preds[i] = 1  # class 2
        else:
            preds[i] = 0  # class 1
    
    return preds
