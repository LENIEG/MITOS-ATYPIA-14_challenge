"""
Save sample preprocessed patches for visual inspection.

Saves a grid image showing raw vs. stain-normalized vs. augmented versions
of a small number of frames from each task and scanner.

Usage
-----
    python -m CommonRoutines.preview

    # or call from code:
    from CommonRoutines.preview import save_preview_grid
    save_preview_grid(out_dir="data/previews", n_samples=4)

Output
------
    data/previews/
      atypia_aperio_preview.png
      atypia_hamamatsu_preview.png
      mitosis_aperio_preview.png
      mitosis_hamamatsu_preview.png
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Support running this file directly (Play button) where project root
# might not be on sys.path yet.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from CommonRoutines.dataset import (
    AtypiaDataset,
    MitosisDataset,
    load_image_rgb,
    read_atypia_label,
    read_mitosis_csv,
    slide_id_to_paths,
)
from CommonRoutines.stain_norm import MacenkoNormalizer
from CommonRoutines.splits import TRAIN_SLIDE_IDS

EXTRACT_ROOT = Path(__file__).parent.parent / "data" / "extracted"
NORM_DIR     = Path(__file__).parent.parent / "data" / "norms"

ATYPIA_LABEL_NAMES = {0: "Low (1)", 1: "Moderate (2)", 2: "High (3)"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tensor_to_uint8(tensor) -> np.ndarray:
    """Convert a normalised float tensor (C,H,W) back to uint8 (H,W,3)."""
    import torch
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = tensor.permute(1, 2, 0).cpu().numpy()
    arr  = arr * std + mean
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def _annotate(img: np.ndarray, text: str) -> np.ndarray:
    """Burn a text annotation into the top-left corner of an image."""
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, pil.width, 18], fill=(0, 0, 0))
    draw.text((2, 2), text, fill=(255, 255, 255))
    return np.array(pil)


def _make_grid(images: list[np.ndarray], ncols: int = 4) -> np.ndarray:
    """Stack a list of (H, W, 3) images (same size) into a grid."""
    nrows = math.ceil(len(images) / ncols)
    h, w  = images[0].shape[:2]
    pad   = np.zeros((h, w, 3), dtype=np.uint8)
    cells = images + [pad] * (nrows * ncols - len(images))
    rows  = [np.hstack(cells[r * ncols:(r + 1) * ncols]) for r in range(nrows)]
    return np.vstack(rows)


def _resize_to(img: np.ndarray, size: int = 256) -> np.ndarray:
    """Resize to (size, size) for compact preview grid."""
    pil = Image.fromarray(img)
    return np.array(pil.resize((size, size), Image.BILINEAR))


# ---------------------------------------------------------------------------
# Stain normalizer loading with graceful fallback
# ---------------------------------------------------------------------------

def _load_normalizers() -> dict[str, MacenkoNormalizer | None]:
    norms: dict[str, MacenkoNormalizer | None] = {"A": None, "H": None}
    for scanner, fname in [("A", "aperio_macenko.npz"), ("H", "hamamatsu_macenko.npz")]:
        p = NORM_DIR / fname
        if p.exists():
            norms[scanner] = MacenkoNormalizer.load(p)
    return norms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_preview_grid(
    out_dir: str | Path = "data/previews",
    n_samples: int = 4,
    thumb_size: int = 256,
) -> None:
    """
    Save four preview grid images (PNG) showing preprocessing results.

    For each sample the grid contains three columns:
      [raw image] | [stain-normalized] | [augmented]

    Parameters
    ----------
    out_dir    : directory to write PNGs into (created if missing)
    n_samples  : number of samples per scanner / task combination
    thumb_size : side length of each thumbnail in the grid
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    norms = _load_normalizers()

    # --- collect samples per (task, scanner) ---
    configs = [
        ("atypia",  "x20", "A", [s for s in TRAIN_SLIDE_IDS if s.startswith("A")]),
        ("atypia",  "x20", "H", [s for s in TRAIN_SLIDE_IDS if s.startswith("H")]),
        ("mitosis", "x40", "A", [s for s in TRAIN_SLIDE_IDS if s.startswith("A")]),
        ("mitosis", "x40", "H", [s for s in TRAIN_SLIDE_IDS if s.startswith("H")]),
    ]

    for task, mag, scanner_char, slide_ids in configs:
        cells: list[np.ndarray] = []
        count = 0

        for slide_id in slide_ids:
            if count >= n_samples:
                break
            paths      = slide_id_to_paths(slide_id, EXTRACT_ROOT, "training")
            frames_dir = paths["frames_dir"] / mag
            if not frames_dir.exists():
                continue

            for frame_file in sorted(frames_dir.glob("*.tiff")):
                if count >= n_samples:
                    break

                raw = load_image_rgb(frame_file)

                # stain normalized
                norm = norms.get(scanner_char)
                normalized = norm.transform(raw) if (norm and norm.is_fitted) else raw

                # augmented (use "train" split augmentation)
                if task == "atypia":
                    from CommonRoutines.augmentation import get_atypia_augmentation
                    tfm = get_atypia_augmentation(mag, "train")
                    aug_result = tfm(image=normalized)
                else:
                    from CommonRoutines.augmentation import get_mitosis_augmentation
                    tfm = get_mitosis_augmentation(mag, "train")
                    aug_result = tfm(image=normalized, keypoints=[])

                aug_arr = _tensor_to_uint8(aug_result["image"])

                # build label string
                if task == "atypia":
                    atypia_csv = (paths["atypia_dir"] / "x20" /
                                  f"{frame_file.stem}_cna_score_decision.csv")
                    label_str = ""
                    if atypia_csv.exists():
                        lbl = read_atypia_label(atypia_csv)
                        label_str = ATYPIA_LABEL_NAMES[lbl]
                else:
                    pos_csv   = paths["mitosis_dir"] / f"{frame_file.stem}_mitosis.csv"
                    centroids = read_mitosis_csv(pos_csv) if pos_csv.exists() else []
                    label_str = f"mitoses: {len(centroids)}"

                # annotate and resize
                cells.append(_annotate(_resize_to(raw,        thumb_size), f"RAW  {label_str}"))
                cells.append(_annotate(_resize_to(normalized, thumb_size), "NORM"))
                cells.append(_annotate(_resize_to(aug_arr,    thumb_size), "AUG"))
                count += 1

        if not cells:
            print(f"[PREVIEW] No frames found for {task}/{scanner_char}, skipping.")
            continue

        grid = _make_grid(cells, ncols=3)
        out_path = out_dir / f"{task}_{('aperio' if scanner_char == 'A' else 'hamamatsu')}_preview.png"
        Image.fromarray(grid).save(out_path)
        print(f"[PREVIEW] Saved {out_path}  ({len(cells)//3} samples)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Save preprocessing preview grids.")
    parser.add_argument("--out_dir",   default="data/previews")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--thumb_size",type=int, default=256)
    args = parser.parse_args()

    save_preview_grid(
        out_dir   = args.out_dir,
        n_samples = args.n_samples,
        thumb_size= args.thumb_size,
    )
