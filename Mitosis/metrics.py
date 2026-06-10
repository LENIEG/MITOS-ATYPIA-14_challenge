"""
Evaluation metrics for mitosis centroid detection.
"""

from __future__ import annotations

import math


def match_detections(
    predictions: list[tuple[float, float, float]],
    targets: list[tuple[float, float]],
    radius_px: float,
) -> tuple[int, int, int]:
    """
    One-to-one centroid matching within a fixed radius.

    Matching strategy: iterate predictions by descending confidence and greedily
    assign each prediction to the nearest unmatched target within radius.
    """
    if not predictions and not targets:
        return 0, 0, 0

    preds_sorted = sorted(predictions, key=lambda p: p[2], reverse=True)
    matched_targets: set[int] = set()
    tp = 0

    radius_sq = radius_px * radius_px
    for px, py, _ in preds_sorted:
        best_j = None
        best_d2 = None
        for j, (tx, ty) in enumerate(targets):
            if j in matched_targets:
                continue
            dx = px - tx
            dy = py - ty
            d2 = dx * dx + dy * dy
            if d2 <= radius_sq and (best_d2 is None or d2 < best_d2):
                best_d2 = d2
                best_j = j

        if best_j is not None:
            matched_targets.add(best_j)
            tp += 1

    fp = max(len(preds_sorted) - tp, 0)
    fn = max(len(targets) - tp, 0)
    return tp, fp, fn


class MitosisMetrics:
    """Accumulate TP/FP/FN and report precision, recall, and F1."""

    def __init__(self, radius_px: float = 8.0):
        self.radius_px = radius_px
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(
        self,
        predictions: list[tuple[float, float, float]],
        targets: list[tuple[float, float]],
    ) -> None:
        tp, fp, fn = match_detections(predictions, targets, radius_px=self.radius_px)
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def precision(self) -> float:
        denom = self.tp + self.fp
        return float(self.tp / denom) if denom > 0 else 0.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return float(self.tp / denom) if denom > 0 else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        denom = p + r
        return float(2.0 * p * r / denom) if denom > 0 else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "tp": float(self.tp),
            "fp": float(self.fp),
            "fn": float(self.fn),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
        }
