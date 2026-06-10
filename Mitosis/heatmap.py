"""
Heatmap utilities for mitosis centroid detection.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def build_target_heatmaps(
    centroids_batch: list[list[tuple[float, float]]],
    image_size: int,
    output_stride: int,
    sigma: float,
    device: torch.device | str,
) -> torch.Tensor:
    """
    Convert centroid lists into gaussian heatmaps.

    Returns:
        Tensor with shape (B, 1, Hm, Wm), where Hm = image_size / output_stride.
    """
    heatmap_size = image_size // output_stride
    batch_size = len(centroids_batch)

    yy, xx = torch.meshgrid(
        torch.arange(heatmap_size, device=device, dtype=torch.float32),
        torch.arange(heatmap_size, device=device, dtype=torch.float32),
        indexing="ij",
    )

    out = torch.zeros((batch_size, 1, heatmap_size, heatmap_size), device=device, dtype=torch.float32)
    denom = max(2.0 * sigma * sigma, 1e-6)

    for b_idx, centroids in enumerate(centroids_batch):
        if not centroids:
            continue

        hm = torch.zeros((heatmap_size, heatmap_size), device=device, dtype=torch.float32)
        for x, y in centroids:
            gx = float(x) / float(output_stride)
            gy = float(y) / float(output_stride)

            gauss = torch.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / denom)
            hm = torch.maximum(hm, gauss)

        out[b_idx, 0] = hm.clamp_(0.0, 1.0)

    return out


def _extract_local_maxima(
    prob_map: torch.Tensor,
    threshold: float,
    nms_kernel: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return local maxima coordinates and scores from a single probability map."""
    if nms_kernel % 2 == 0:
        nms_kernel += 1
    pad = nms_kernel // 2
    pooled = F.max_pool2d(prob_map[None, None], kernel_size=nms_kernel, stride=1, padding=pad)[0, 0]
    keep = (prob_map >= threshold) & (prob_map == pooled)
    ys, xs = torch.nonzero(keep, as_tuple=True)
    scores = prob_map[ys, xs]
    return xs, ys, scores


def decode_heatmaps(
    logits: torch.Tensor,
    output_stride: int,
    threshold: float,
    nms_kernel: int,
    max_detections: int,
) -> list[list[tuple[float, float, float]]]:
    """
    Decode heatmap logits into centroid detections.

    Returns one detection list per batch item: [(x, y, score), ...].
    Coordinates are in resized image pixels.
    """
    probs = torch.sigmoid(logits)
    batch = probs.size(0)
    out: list[list[tuple[float, float, float]]] = []

    for b_idx in range(batch):
        prob_map = probs[b_idx, 0]
        xs, ys, scores = _extract_local_maxima(prob_map, threshold=threshold, nms_kernel=nms_kernel)

        if scores.numel() == 0:
            out.append([])
            continue

        order = torch.argsort(scores, descending=True)
        if max_detections > 0:
            order = order[:max_detections]

        detections: list[tuple[float, float, float]] = []
        for idx in order.tolist():
            x = (float(xs[idx]) + 0.5) * float(output_stride)
            y = (float(ys[idx]) + 0.5) * float(output_stride)
            score = float(scores[idx])
            detections.append((x, y, score))

        out.append(detections)

    return out


def strip_scores(
    detections: Iterable[tuple[float, float, float]],
) -> list[tuple[float, float]]:
    """Drop confidence scores from decoded detections."""
    return [(x, y) for x, y, _ in detections]
