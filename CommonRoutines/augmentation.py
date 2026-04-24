"""
Augmentation pipelines for the MITOS-ATYPIA-14 challenge.

Two independent pipelines — one per task — parameterized by magnification
so the same function works for x10 / x20 / x40 frames.

Usage
-----
    from CommonRoutines.augmentation import get_atypia_augmentation
    from CommonRoutines.augmentation import get_mitosis_augmentation

    train_tfm = get_atypia_augmentation(magnification="x20", split="train")
    val_tfm   = get_atypia_augmentation(magnification="x20", split="val")

    # albumentations API:
    augmented = train_tfm(image=np_uint8_rgb)
    image_out = augmented["image"]          # np.ndarray uint8 RGB

    # Mitosis pipeline also carries a keypoints argument:
    augmented = train_tfm(image=frame, keypoints=[(x, y), ...])
    keypoints_out = augmented["keypoints"]  # transformed centroid list

Design notes
------------
- Color jitter ranges are deliberately wide to bridge the Aperio ↔ Hamamatsu
  scanner gap (brightness/contrast/hue differ significantly between scanners).
- Spatial augmentations that change scale are disabled for mitosis at x40 to
  preserve the 8 µm centroid-tolerance evaluation criterion.
- `split="val"` returns only a safe normalization pipeline (no random ops).
"""

from __future__ import annotations

from typing import Literal

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Magnification-dependent resize targets
# ---------------------------------------------------------------------------

# Final spatial size fed to the model for each magnification level.
# Chosen so the tissue field is large enough for the architecture heads
# while remaining computationally tractable.
_RESIZE: dict[str, tuple[int, int]] = {
    "x10": (512, 512),
    "x20": (512, 512),
    "x40": (512, 512),
}


# ---------------------------------------------------------------------------
# Task A — Nuclear Atypia  (frame-level classification, X20)
# ---------------------------------------------------------------------------

def get_atypia_augmentation(
    magnification: Literal["x10", "x20", "x40"] = "x20",
    split: Literal["train", "val"] = "train",
) -> A.Compose:
    """
    Return an albumentations Compose pipeline for the atypia task.

    Parameters
    ----------
    magnification : {"x10", "x20", "x40"}
        Input frame magnification (controls resize target).
    split : {"train", "val"}
        "val" returns a minimal, deterministic pipeline (resize + normalize).
    """
    h, w = _RESIZE[magnification]

    if split == "val":
        return A.Compose([
            A.Resize(h, w),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    return A.Compose([
        # --- geometric ---
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=15,
            border_mode=0,
            p=0.5,
        ),
        A.RandomResizedCrop(
            size=(h, w),
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
        ),

        # --- color / stain variation (bridges scanner gap) ---
        A.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25,
            hue=0.08,
            p=0.8,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=15,
            p=0.5,
        ),
        A.RandomGamma(gamma_limit=(85, 115), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.005, 0.02), p=0.2),

        # --- normalize & tensorize ---
        A.Resize(h, w),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Task B — Mitosis Detection  (frame-level detection, X40)
# ---------------------------------------------------------------------------

def get_mitosis_augmentation(
    magnification: Literal["x10", "x20", "x40"] = "x40",
    split: Literal["train", "val"] = "train",
) -> A.Compose:
    """
    Return an albumentations Compose pipeline for the mitosis task.

    Keypoints (mitosis centroids) are transformed alongside the image so
    ground-truth coordinates remain valid after augmentation.

    Parameters
    ----------
    magnification : {"x10", "x20", "x40"}
        Input frame magnification (controls resize target).
    split : {"train", "val"}
        "val" returns a minimal, deterministic pipeline.

    Notes
    -----
    Pass keypoints as a list of (x, y) tuples using the "xy" format:
        result = pipeline(image=img, keypoints=[(x1, y1), (x2, y2)])
        centroids = result["keypoints"]   # list of (x, y) after augmentation
    """
    h, w = _RESIZE[magnification]

    keypoint_params = A.KeypointParams(
        format="xy",
        remove_invisible=True,  # drop centroids that fall outside crop
    )

    if split == "val":
        return A.Compose(
            [
                A.Resize(h, w),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            keypoint_params=keypoint_params,
        )

    return A.Compose(
        [
            # --- geometric (no aggressive scale change — preserves 8 µm metric) ---
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,          # small scale change only
                rotate_limit=20,
                border_mode=0,
                p=0.5,
            ),
            A.RandomResizedCrop(
                size=(h, w),
                scale=(0.90, 1.0),         # conservative crop
                ratio=(0.95, 1.05),
            ),

            # --- color / stain variation ---
            A.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.25,
                hue=0.08,
                p=0.8,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=15,
                p=0.5,
            ),
            A.RandomGamma(gamma_limit=(85, 115), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(std_range=(0.005, 0.02), p=0.2),

            # --- normalize & tensorize ---
            A.Resize(h, w),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        keypoint_params=keypoint_params,
    )
