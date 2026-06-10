"""
Frame-level inference for mitosis centroid detection.

Writes one CSV row per predicted centroid. If a frame has no prediction,
one placeholder row is written with empty centroid fields.

Usage examples:
    python -m Mitosis.infer
    python -m Mitosis.infer --data-split training --max-slides 2 --max-frames-per-slide 20
    python -m Mitosis.infer --slides A06,H06 --output outputs/mitosis/inference_some_data.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from CommonRoutines import (
    TEST_SLIDE_IDS,
    TRAIN_SLIDE_IDS,
    MacenkoNormalizer,
    load_image_rgb,
    read_mitosis_csv,
    slide_id_to_paths,
)
from CommonRoutines.augmentation import get_mitosis_augmentation
from Mitosis.config import get_default_config
from Mitosis.ensemble import (
    load_models_from_checkpoints,
    parse_checkpoint_paths,
    predict_ensemble_logits,
)
from Mitosis.heatmap import decode_heatmaps


@dataclass
class InferenceSample:
    image_path: Path
    slide_id: str
    frame_id: str
    scanner: str
    true_centroids: list[tuple[float, float]]


class MitosisInferenceDataset(Dataset):
    """Inference dataset for frame-level mitosis detection."""

    def __init__(
        self,
        samples: list[InferenceSample],
        normalizers: dict[str, MacenkoNormalizer | None],
        magnification: str,
    ) -> None:
        self.samples = samples
        self.normalizers = normalizers
        self.transform = get_mitosis_augmentation(magnification=magnification, split="val")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = load_image_rgb(sample.image_path)

        norm = self.normalizers.get(sample.scanner)
        if norm is not None and norm.is_fitted:
            image = norm.transform(image)

        tensor = self.transform(image=image, keypoints=[])["image"]
        return tensor, sample.slide_id, sample.frame_id, sample.scanner, len(sample.true_centroids)


def resolve_device(preferred: str) -> str:
    """Resolve runtime device with fallback to CPU if CUDA is unavailable."""
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return preferred


def load_stain_normalizers(norm_dir: Path) -> dict[str, MacenkoNormalizer | None]:
    """Load scanner-specific stain normalizers when available."""
    normalizers = {"A": None, "H": None}
    for scanner, fname in [("A", "aperio_macenko.npz"), ("H", "hamamatsu_macenko.npz")]:
        p = norm_dir / fname
        if p.exists():
            normalizers[scanner] = MacenkoNormalizer.load(p)
    return normalizers


def parse_slides(arg_slides: str | None, data_split: str, max_slides: int) -> list[str]:
    """Resolve target slide IDs from CLI args."""
    if arg_slides:
        slides = [s.strip() for s in arg_slides.split(",") if s.strip()]
        if not slides:
            raise ValueError("--slides was provided but no valid IDs were parsed")
        return slides

    pool = TRAIN_SLIDE_IDS if data_split == "training" else TEST_SLIDE_IDS
    return list(pool[:max_slides])


def collect_samples(
    extract_root: Path,
    data_split: str,
    slide_ids: Iterable[str],
    magnification: str,
    max_frames_per_slide: int,
) -> list[InferenceSample]:
    """Collect frame paths and optional true centroid annotations."""
    samples: list[InferenceSample] = []

    for slide_id in slide_ids:
        paths = slide_id_to_paths(slide_id, extract_root=extract_root, split=data_split)
        frames_dir = paths["frames_dir"] / magnification
        mitosis_dir = paths["mitosis_dir"]
        scanner = paths["scanner"]

        if not frames_dir.exists():
            print(f"Skipping {slide_id}: missing frames directory {frames_dir}")
            continue

        frame_files = sorted(frames_dir.glob("*.tiff"))[:max_frames_per_slide]
        for frame_file in frame_files:
            frame_id = frame_file.stem
            pos_csv = mitosis_dir / f"{frame_id}_mitosis.csv"
            entries = read_mitosis_csv(pos_csv)
            true_centroids = [(x, y) for x, y, _ in entries]
            samples.append(
                InferenceSample(
                    image_path=frame_file,
                    slide_id=slide_id,
                    frame_id=frame_id,
                    scanner=scanner,
                    true_centroids=true_centroids,
                )
            )

    return samples


def run_inference(args: argparse.Namespace) -> Path:
    """Run inference and write CSV detections."""
    cfg = get_default_config()
    cfg.device = resolve_device(cfg.device)

    checkpoint_paths = parse_checkpoint_paths(
        checkpoint=args.checkpoint,
        checkpoints=args.checkpoints,
    )

    slide_ids = parse_slides(args.slides, args.data_split, args.max_slides)
    samples = collect_samples(
        extract_root=cfg.data.extract_root,
        data_split=args.data_split,
        slide_ids=slide_ids,
        magnification=cfg.data.magnification,
        max_frames_per_slide=args.max_frames_per_slide,
    )
    if not samples:
        raise RuntimeError("No samples found for inference. Check slide IDs and data paths.")

    normalizers = load_stain_normalizers(cfg.data.norm_dir)
    dataset = MitosisInferenceDataset(samples, normalizers, cfg.data.magnification)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    models = load_models_from_checkpoints(cfg, checkpoint_paths)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_frames = 0
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "slide_id",
                "frame_id",
                "scanner",
                "pred_x",
                "pred_y",
                "confidence",
                "true_mitosis_count_if_available",
            ]
        )

        with torch.no_grad():
            for images, slide_ids_b, frame_ids_b, scanners_b, true_counts_b in tqdm(
                loader, desc="Inference", leave=False
            ):
                images = images.to(cfg.device)
                logits = predict_ensemble_logits(models, images)
                detections_batch = decode_heatmaps(
                    logits,
                    output_stride=cfg.model.output_stride,
                    threshold=cfg.training.decode_threshold,
                    nms_kernel=cfg.training.decode_nms_kernel,
                    max_detections=cfg.training.decode_max_detections,
                )

                for i, detections in enumerate(detections_batch):
                    n_frames += 1
                    if detections:
                        for x, y, score in detections:
                            writer.writerow(
                                [
                                    slide_ids_b[i],
                                    frame_ids_b[i],
                                    scanners_b[i],
                                    f"{x:.3f}",
                                    f"{y:.3f}",
                                    f"{score:.6f}",
                                    int(true_counts_b[i]),
                                ]
                            )
                            n_rows += 1
                    else:
                        writer.writerow(
                            [
                                slide_ids_b[i],
                                frame_ids_b[i],
                                scanners_b[i],
                                "",
                                "",
                                "",
                                int(true_counts_b[i]),
                            ]
                        )
                        n_rows += 1

    print("Inference completed")
    print(f"  checkpoints: {[str(p) for p in checkpoint_paths]}")
    print(f"  ensemble_size: {len(checkpoint_paths)}")
    print(f"  data_split: {args.data_split}")
    print(f"  slides: {slide_ids}")
    print(f"  frames: {n_frames}")
    print(f"  rows: {n_rows}")
    print(f"  output: {output_path}")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run mitosis inference on a subset of slides")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/mitosis/checkpoints/fold0_best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help=(
            "Comma-separated checkpoint paths for heatmap ensembling. "
            "When provided, overrides --checkpoint."
        ),
    )
    parser.add_argument(
        "--data-split",
        type=str,
        choices=["training", "testing"],
        default="testing",
        help="Dataset split under data/extracted",
    )
    parser.add_argument(
        "--slides",
        type=str,
        default=None,
        help="Comma-separated slide IDs (e.g., A06,H06). If omitted, uses first N slides.",
    )
    parser.add_argument(
        "--max-slides",
        type=int,
        default=2,
        help="Number of slides to use when --slides is not provided",
    )
    parser.add_argument(
        "--max-frames-per-slide",
        type=int,
        default=20,
        help="Maximum frames to process per slide",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/mitosis/inference_some_data.csv",
        help="Output CSV path",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
