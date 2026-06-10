"""
Frame-level inference for atypia scoring.

Runs inference with a trained checkpoint on a subset of slides and writes
predictions to CSV.

Usage examples:
    python -m Atypia.infer
    python -m Atypia.infer --data-split training --max-slides 4 --max-frames-per-slide 30
    python -m Atypia.infer --slides A06,H06 --output outputs/atypia/inference_some_data.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from CommonRoutines import (
    TEST_SLIDE_IDS,
    TRAIN_SLIDE_IDS,
    MacenkoNormalizer,
    load_image_rgb,
    read_atypia_label,
    slide_id_to_paths,
)
from CommonRoutines.augmentation import get_atypia_augmentation
from Atypia.config import get_default_config
from Atypia.models import create_model
from Atypia.metrics import ordinal_logits_to_predictions


@dataclass
class InferenceSample:
    image_path: Path
    slide_id: str
    frame_id: str
    scanner: str
    true_label: int | None


class AtypiaInferenceDataset(Dataset):
    """Inference dataset for frame-level atypia prediction."""

    def __init__(
        self,
        samples: list[InferenceSample],
        normalizers: dict[str, MacenkoNormalizer | None],
        magnification: str,
    ) -> None:
        self.samples = samples
        self.normalizers = normalizers
        self.transform = get_atypia_augmentation(magnification=magnification, split="val")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = load_image_rgb(sample.image_path)

        norm = self.normalizers.get(sample.scanner)
        if norm is not None and norm.is_fitted:
            image = norm.transform(image)

        tensor = self.transform(image=image)["image"]
        return tensor, sample.slide_id, sample.frame_id, sample.scanner, sample.true_label


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
    """Collect frame paths and optional labels for inference."""
    samples: list[InferenceSample] = []

    for slide_id in slide_ids:
        paths = slide_id_to_paths(slide_id, extract_root=extract_root, split=data_split)
        frames_dir = paths["frames_dir"] / magnification
        atypia_x20 = paths["atypia_dir"] / "x20"
        scanner = paths["scanner"]

        if not frames_dir.exists():
            print(f"Skipping {slide_id}: missing frames directory {frames_dir}")
            continue

        frame_files = sorted(frames_dir.glob("*.tiff"))[:max_frames_per_slide]
        for frame_file in frame_files:
            frame_id = frame_file.stem
            true_label: int | None = None

            # Labels are expected in training split. For testing they may be absent.
            label_csv = atypia_x20 / f"{frame_id}_cna_score_decision.csv"
            if magnification == "x40" and not label_csv.exists() and len(frame_id) > 0:
                parent_stem = frame_id[:-1]
                label_csv = atypia_x20 / f"{parent_stem}_cna_score_decision.csv"

            if label_csv.exists():
                try:
                    true_label = read_atypia_label(label_csv)
                except Exception:
                    true_label = None

            samples.append(
                InferenceSample(
                    image_path=frame_file,
                    slide_id=slide_id,
                    frame_id=frame_id,
                    scanner=scanner,
                    true_label=true_label,
                )
            )

    return samples


def ordinal_logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """
    Convert ordinal logits (batch, 2) to class probabilities (batch, 3).
    """
    ge2 = 1.0 / (1.0 + np.exp(-logits[:, 0]))
    ge3 = 1.0 / (1.0 + np.exp(-logits[:, 1]))

    p1 = 1.0 - ge2
    p2 = ge2 - ge3
    p3 = ge3

    probs = np.stack([p1, p2, p3], axis=1)
    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def save_annotated_inference_image(
    image_path: Path,
    output_path: Path,
    pred_score: int,
    true_score: int | None,
    probs: np.ndarray,
) -> None:
    """Save an image preview annotated with GT and prediction."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    gt_text = str(true_score) if true_score is not None else "NA"
    line1 = f"GT={gt_text}  Pred={pred_score}"
    line2 = f"P(low/mod/high)={probs[0]:.2f}/{probs[1]:.2f}/{probs[2]:.2f}"

    # Draw a simple header background for readability.
    draw.rectangle([(0, 0), (image.width, 40)], fill=(0, 0, 0))
    draw.text((8, 4), line1, fill=(255, 255, 255))
    draw.text((8, 22), line2, fill=(255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def run_inference(args: argparse.Namespace) -> Path:
    """Run inference and write CSV predictions."""
    cfg = get_default_config()
    cfg.device = resolve_device(cfg.device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
    dataset = AtypiaInferenceDataset(samples, normalizers, cfg.data.magnification)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    model = create_model(cfg.model, device=cfg.device)
    state = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(state)
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir)

    sample_lookup = {(s.slide_id, s.frame_id): s for s in samples}

    n_rows = 0
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "slide_id",
                "frame_id",
                "scanner",
                "pred_class_index",
                "pred_score_1to3",
                "prob_low",
                "prob_moderate",
                "prob_high",
                "true_class_index_if_available",
                "true_score_1to3_if_available",
            ]
        )

        with torch.no_grad():
            for images, slide_ids_b, frame_ids_b, scanners_b, true_labels_b in tqdm(
                loader, desc="Inference", leave=False
            ):
                images = images.to(cfg.device)
                logits = model(images)
                logits_np = logits.cpu().numpy()

                preds = ordinal_logits_to_predictions(logits_np)
                probs = ordinal_logits_to_probs(logits_np)

                for i in range(len(preds)):
                    pred_idx = int(preds[i])
                    pred_score = pred_idx + 1

                    true_label_obj = true_labels_b[i]
                    true_idx: int | None
                    true_score: int | None
                    if isinstance(true_label_obj, torch.Tensor):
                        true_idx = int(true_label_obj.item()) if true_label_obj.numel() > 0 else None
                    elif true_label_obj is None:
                        true_idx = None
                    else:
                        try:
                            true_idx = int(true_label_obj)
                        except Exception:
                            true_idx = None
                    true_score = (true_idx + 1) if true_idx is not None else None

                    writer.writerow(
                        [
                            slide_ids_b[i],
                            frame_ids_b[i],
                            scanners_b[i],
                            pred_idx,
                            pred_score,
                            float(probs[i, 0]),
                            float(probs[i, 1]),
                            float(probs[i, 2]),
                            true_idx,
                            true_score,
                        ]
                    )

                    if args.save_images:
                        sample = sample_lookup[(slide_ids_b[i], frame_ids_b[i])]
                        image_out = images_dir / slide_ids_b[i] / f"{frame_ids_b[i]}.png"
                        save_annotated_inference_image(
                            image_path=sample.image_path,
                            output_path=image_out,
                            pred_score=pred_score,
                            true_score=true_score,
                            probs=probs[i],
                        )
                    n_rows += 1

    print("Inference completed")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  data_split: {args.data_split}")
    print(f"  slides: {slide_ids}")
    print(f"  samples: {n_rows}")
    print(f"  output: {output_path}")
    if args.save_images:
        print(f"  annotated images: {images_dir}")

    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run atypia inference on a subset of slides")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/atypia/checkpoints/final_merged_finetuned.pt",
        help="Path to model checkpoint",
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
        default="outputs/atypia/inference_some_data.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save annotated per-frame PNG images with GT and prediction",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="outputs/atypia/inference_images",
        help="Directory for annotated inference images",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
