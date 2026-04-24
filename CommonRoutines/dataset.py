"""
PyTorch Dataset classes for the MITOS-ATYPIA-14 challenge.

Two independent datasets:
  - AtypiaDataset   — frame-level ordinal classification (score 1/2/3) at X20
  - MitosisDataset  — centroid detection at X40 using pre-extracted patches

Shared utilities
----------------
  - slide_id_to_paths()   — resolve slide ID → (scanner, frames_dir, …)
  - read_atypia_label()   — parse `*_cna_score_decision.csv` → int label
  - read_mitosis_csv()    — parse `*_mitosis.csv` → list of (x, y, score)

Usage
-----
    from CommonRoutines.dataset import AtypiaDataset, MitosisDataset
    from CommonRoutines.augmentation import get_atypia_augmentation
    from CommonRoutines.splits import get_fixed_split
    from CommonRoutines.stain_norm import MacenkoNormalizer

    train_ids, val_ids = get_fixed_split()

    norm_a = MacenkoNormalizer.load("data/norms/aperio_macenko.npz")
    norm_h = MacenkoNormalizer.load("data/norms/hamamatsu_macenko.npz")
    normalizers = {"A": norm_a, "H": norm_h}

    ds = AtypiaDataset(
        slide_ids   = train_ids,
        extract_root= "data/extracted",
        magnification = "x20",
        split       = "train",
        normalizers = normalizers,   # optional; pass None to skip
    )
    image, label, meta = ds[0]
    # meta = {"slide_id": "A03", "scanner": "A", "frame_id": "A03_00A", …}
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from CommonRoutines.augmentation import get_atypia_augmentation, get_mitosis_augmentation
from CommonRoutines.stain_norm import MacenkoNormalizer


# ---------------------------------------------------------------------------
# Path / annotation helpers  (shared between both datasets)
# ---------------------------------------------------------------------------

EXTRACT_ROOT_DEFAULT = Path(__file__).parent.parent / "data" / "extracted"


def slide_id_to_paths(
    slide_id: str,
    extract_root: str | Path = EXTRACT_ROOT_DEFAULT,
    split: str = "training",
) -> dict[str, Path]:
    """
    Resolve a slide ID (e.g. "A03", "H03") to its directory paths.

    Returns a dict with keys:
      slide_dir, frames_dir, atypia_dir, mitosis_dir, scanner
    """
    scanner_char = slide_id[0].upper()   # "A" or "H"
    scanner_name = {"A": "aperio", "H": "hamamatsu"}[scanner_char]

    root      = Path(extract_root)
    slide_dir = root / split / scanner_name / slide_id

    return {
        "scanner"    : scanner_char,
        "slide_dir"  : slide_dir,
        "frames_dir" : slide_dir / "frames",
        "atypia_dir" : slide_dir / "atypia",
        "mitosis_dir": slide_dir / "mitosis",
    }


def read_atypia_label(decision_csv: Path) -> int:
    """
    Read the majority-vote atypia score from a `*_cna_score_decision.csv`.
    File contains a single integer: 1, 2, or 3.
    Returns the score as a 0-indexed class label (0 / 1 / 2).
    """
    text = decision_csv.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty atypia label file: {decision_csv}")
    score = int(text)
    if score not in (1, 2, 3):
        raise ValueError(f"Unexpected atypia score {score} in {decision_csv}")
    return score - 1   # 0-indexed for CrossEntropyLoss


def read_mitosis_csv(csv_path: Path) -> list[tuple[float, float, float]]:
    """
    Parse a `*_mitosis.csv` or `*_not_mitosis.csv` file.
    Each row: x, y, confidence_score
    Returns a list of (x, y, score) float tuples.
    """
    entries: list[tuple[float, float, float]] = []
    if not csv_path.exists():
        return entries
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                x, y = float(row[0]), float(row[1])
                score = float(row[2]) if len(row) >= 3 else 1.0
                entries.append((x, y, score))
    return entries


def load_image_rgb(path: Path) -> np.ndarray:
    """Load any image file as a uint8 RGB numpy array (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Dataset A — Nuclear Atypia
# ---------------------------------------------------------------------------

class AtypiaDataset(Dataset):
    """
    Frame-level dataset for nuclear atypia scoring.

    Each sample is one X20 (or X40/X10) frame with its majority-vote score.

    Parameters
    ----------
    slide_ids : list[str]
        Slide IDs to include, e.g. ["A03", "H03", "A04", "H04"].
    extract_root : str | Path
        Root of the extracted data tree (contains training/ and testing/).
    magnification : {"x10", "x20", "x40"}
        Which frames subfolder to use.  Atypia labels live in atypia/x20/.
        If you use x40, labels are inherited from the parent x20 frame.
    data_split : {"training", "testing"}
        Whether to read from training/ or testing/ sub-tree.
    split : {"train", "val"}
        Controls data augmentation.
    normalizers : dict[str, MacenkoNormalizer] | None
        {"A": aperio_norm, "H": hamamatsu_norm}.  Pass None to skip.
    """

    def __init__(
        self,
        slide_ids: list[str],
        extract_root: str | Path = EXTRACT_ROOT_DEFAULT,
        magnification: str = "x20",
        data_split: str = "training",
        split: str = "train",
        normalizers: dict[str, MacenkoNormalizer] | None = None,
    ) -> None:
        self.magnification = magnification
        self.normalizers   = normalizers or {}
        self.transform     = get_atypia_augmentation(magnification, split)

        self._samples: list[dict[str, Any]] = []
        self._build_index(slide_ids, extract_root, data_split)

    # ------------------------------------------------------------------
    def _build_index(
        self,
        slide_ids: list[str],
        extract_root: str | Path,
        data_split: str,
    ) -> None:
        for slide_id in slide_ids:
            paths = slide_id_to_paths(slide_id, extract_root, data_split)
            frames_dir  = paths["frames_dir"] / self.magnification
            atypia_x20  = paths["atypia_dir"] / "x20"
            scanner     = paths["scanner"]

            if not frames_dir.exists():
                continue

            for frame_file in sorted(frames_dir.glob("*.tiff")):
                frame_stem = frame_file.stem          # e.g. "A03_00A"
                label_csv  = atypia_x20 / f"{frame_stem}_cna_score_decision.csv"

                # For x40 frames (e.g. "A03_00Aa"), the label lives at the
                # parent x20 level (strip the lowercase suffix letter)
                if self.magnification == "x40" and not label_csv.exists():
                    parent_stem = frame_stem[:-1]     # "A03_00Aa" → "A03_00A"
                    label_csv = atypia_x20 / f"{parent_stem}_cna_score_decision.csv"

                if not label_csv.exists():
                    continue

                try:
                    label = read_atypia_label(label_csv)
                except ValueError:
                    continue

                self._samples.append({
                    "frame_path": frame_file,
                    "label_csv" : label_csv,
                    "label"     : label,
                    "frame_id"  : frame_stem,
                    "slide_id"  : slide_id,
                    "scanner"   : scanner,
                })

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict]:
        sample = self._samples[idx]
        image  = load_image_rgb(sample["frame_path"])
        label  = sample["label"]

        # stain normalization (scanner-specific)
        norm = self.normalizers.get(sample["scanner"])
        if norm is not None and norm.is_fitted:
            image = norm.transform(image)

        result = self.transform(image=image)
        tensor = result["image"]

        meta = {
            "slide_id" : sample["slide_id"],
            "scanner"  : sample["scanner"],
            "frame_id" : sample["frame_id"],
        }
        return tensor, label, meta


# ---------------------------------------------------------------------------
# Dataset B — Mitosis Detection
# ---------------------------------------------------------------------------

class MitosisDataset(Dataset):
    """
    Frame-level dataset for mitosis centroid detection.

    Each sample is one X40 frame with:
      - the full-resolution image tensor
      - a list of (x, y) mitosis centroids (ground truth positive only)

    The pre-extracted patch JPGs (*_mitosis.jpg / *_not_mitosis.jpg) are
    NOT used here — the full X40 frame is loaded so that the detection head
    can predict centroid heatmaps over the entire field.

    Parameters
    ----------
    slide_ids : list[str]
    extract_root : str | Path
    magnification : {"x40"}
        Only X40 is meaningful for mitosis annotation.
    data_split : {"training", "testing"}
    split : {"train", "val"}
    normalizers : dict[str, MacenkoNormalizer] | None
    """

    def __init__(
        self,
        slide_ids: list[str],
        extract_root: str | Path = EXTRACT_ROOT_DEFAULT,
        magnification: str = "x40",
        data_split: str = "training",
        split: str = "train",
        normalizers: dict[str, MacenkoNormalizer] | None = None,
    ) -> None:
        self.magnification = magnification
        self.normalizers   = normalizers or {}
        self.transform     = get_mitosis_augmentation(magnification, split)

        self._samples: list[dict[str, Any]] = []
        self._build_index(slide_ids, extract_root, data_split)

    # ------------------------------------------------------------------
    def _build_index(
        self,
        slide_ids: list[str],
        extract_root: str | Path,
        data_split: str,
    ) -> None:
        for slide_id in slide_ids:
            paths       = slide_id_to_paths(slide_id, extract_root, data_split)
            frames_dir  = paths["frames_dir"] / self.magnification
            mitosis_dir = paths["mitosis_dir"]
            scanner     = paths["scanner"]

            if not frames_dir.exists():
                continue

            for frame_file in sorted(frames_dir.glob("*.tiff")):
                frame_stem = frame_file.stem    # e.g. "A03_00Aa"
                pos_csv    = mitosis_dir / f"{frame_stem}_mitosis.csv"

                self._samples.append({
                    "frame_path"  : frame_file,
                    "pos_csv"     : pos_csv,
                    "frame_id"    : frame_stem,
                    "slide_id"    : slide_id,
                    "scanner"     : scanner,
                })

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, list[tuple[float, float]], dict]:
        sample = self._samples[idx]
        image  = load_image_rgb(sample["frame_path"])

        # Ground-truth centroids (positive mitosis only)
        entries    = read_mitosis_csv(sample["pos_csv"])
        centroids  = [(x, y) for x, y, _ in entries]   # drop confidence

        # Stain normalization
        norm = self.normalizers.get(sample["scanner"])
        if norm is not None and norm.is_fitted:
            image = norm.transform(image)

        result    = self.transform(image=image, keypoints=centroids)
        tensor    = result["image"]
        aug_kpts  = result["keypoints"]   # list of (x, y) after augmentation

        meta = {
            "slide_id" : sample["slide_id"],
            "scanner"  : sample["scanner"],
            "frame_id" : sample["frame_id"],
        }
        return tensor, aug_kpts, meta
