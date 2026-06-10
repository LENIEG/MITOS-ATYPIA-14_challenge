"""
Visualize mitosis inference predictions against ground truth.

Reads a predictions CSV from Mitosis.infer and creates per-frame PNG overlays:
  - GT centroids in green
  - Predicted centroids in red

Usage example:
  python -m Mitosis.visualize_inference \
    --predictions outputs/mitosis/inference_ensemble.csv \
    --data-split testing \
    --out-dir outputs/mitosis/inference_visuals \
    --max-frames 60
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from CommonRoutines import load_image_rgb, read_mitosis_csv, slide_id_to_paths


@dataclass
class PredPoint:
    x: float
    y: float
    confidence: float | None


def _to_float_or_none(value: str) -> float | None:
    text = (value or "").strip()
    if text == "":
        return None
    return float(text)


def load_predictions_by_frame(predictions_csv: Path) -> dict[tuple[str, str], list[PredPoint]]:
    """Load prediction CSV and group detections by (slide_id, frame_id)."""
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

    grouped: dict[tuple[str, str], list[PredPoint]] = {}

    with predictions_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slide_id = row["slide_id"].strip()
            frame_id = row["frame_id"].strip()
            key = (slide_id, frame_id)
            grouped.setdefault(key, [])

            x = _to_float_or_none(row.get("pred_x", ""))
            y = _to_float_or_none(row.get("pred_y", ""))
            c = _to_float_or_none(row.get("confidence", ""))
            if x is None or y is None:
                continue

            grouped[key].append(PredPoint(x=x, y=y, confidence=c))

    return grouped


def gt_centroids_for_frame(extract_root: Path, data_split: str, slide_id: str, frame_id: str) -> list[tuple[float, float]]:
    """Load GT mitosis centroids for a frame from *_mitosis.csv."""
    paths = slide_id_to_paths(slide_id, extract_root=extract_root, split=data_split)
    pos_csv = paths["mitosis_dir"] / f"{frame_id}_mitosis.csv"
    entries = read_mitosis_csv(pos_csv)
    return [(x, y) for x, y, _ in entries]


def frame_image_path(extract_root: Path, data_split: str, slide_id: str, frame_id: str, magnification: str = "x40") -> Path:
    paths = slide_id_to_paths(slide_id, extract_root=extract_root, split=data_split)
    return paths["frames_dir"] / magnification / f"{frame_id}.tiff"


def draw_overlay(
    image: np.ndarray,
    gt_points: list[tuple[float, float]],
    pred_points: list[PredPoint],
    title: str,
    output_path: Path,
) -> None:
    """Render and save one overlay image."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(image)

    if gt_points:
        gx = [p[0] for p in gt_points]
        gy = [p[1] for p in gt_points]
        ax.scatter(gx, gy, s=24, c="lime", marker="o", edgecolors="black", linewidths=0.6, label="GT")

    if pred_points:
        px = [p.x for p in pred_points]
        py = [p.y for p in pred_points]
        ax.scatter(px, py, s=24, c="deepskyblue", marker="x", linewidths=1.4, label="Pred")

    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    predictions_csv = Path(args.predictions)
    extract_root = Path(args.extract_root)
    out_dir = Path(args.out_dir)

    grouped = load_predictions_by_frame(predictions_csv)
    items = sorted(grouped.items())
    if args.max_frames > 0:
        items = items[: args.max_frames]

    if not items:
        raise RuntimeError("No frames found in predictions CSV.")

    n_done = 0
    n_missing = 0

    for (slide_id, frame_id), pred_points in items:
        image_path = frame_image_path(
            extract_root=extract_root,
            data_split=args.data_split,
            slide_id=slide_id,
            frame_id=frame_id,
            magnification=args.magnification,
        )
        if not image_path.exists():
            n_missing += 1
            continue

        image = load_image_rgb(image_path)
        gt_points = gt_centroids_for_frame(
            extract_root=extract_root,
            data_split=args.data_split,
            slide_id=slide_id,
            frame_id=frame_id,
        )

        title = (
            f"{slide_id}/{frame_id}  "
            f"GT={len(gt_points)}  Pred={len(pred_points)}"
        )
        output_path = out_dir / slide_id / f"{frame_id}.png"
        draw_overlay(
            image=image,
            gt_points=gt_points,
            pred_points=pred_points,
            title=title,
            output_path=output_path,
        )
        n_done += 1

    print("Visualization complete")
    print(f"  predictions: {predictions_csv}")
    print(f"  output_dir: {out_dir}")
    print(f"  rendered_frames: {n_done}")
    print(f"  missing_frames: {n_missing}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize mitosis predictions vs GT")
    parser.add_argument(
        "--predictions",
        type=str,
        default="outputs/mitosis/inference_ensemble.csv",
        help="Path to inference CSV from Mitosis.infer",
    )
    parser.add_argument(
        "--extract-root",
        type=str,
        default="data/extracted",
        help="Root path of extracted dataset",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        choices=["training", "testing"],
        default="testing",
        help="Dataset split used by predictions",
    )
    parser.add_argument(
        "--magnification",
        type=str,
        default="x40",
        help="Frame magnification folder to visualize",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/mitosis/inference_visuals",
        help="Directory to write PNG overlays",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=80,
        help="Maximum number of frames to render (<=0 means all)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
