"""
Macenko stain normalization for H&E histology images.

Usage
-----
# Fit a normalizer on a reference image (run once per scanner):
    norm = MacenkoNormalizer()
    norm.fit(reference_image_rgb_uint8)
    norm.save("aperio_norm.npz")

# Apply at runtime:
    norm = MacenkoNormalizer.load("aperio_norm.npz")
    normalized = norm.transform(image_rgb_uint8)

Notes
-----
- All inputs/outputs are uint8 RGB numpy arrays (H, W, 3).
- The algorithm follows Macenko et al. (2009) as implemented in
  "A Method for Normalizing Histology Slides for Quantitative Analysis".
- Pure numpy — no staintools dependency required.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_LUMINOSITY_THRESHOLD = 0.8   # pixels brighter than this (OD space) are background
_ANGULAR_PERCENTILE  = 99     # percentile used to find stain vectors


def _rgb_to_od(image: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB image to optical-density (OD) space."""
    img = image.astype(np.float64)
    img = np.clip(img, 1, 255)          # avoid log(0)
    return -np.log(img / 255.0)


def _od_to_rgb(od: np.ndarray) -> np.ndarray:
    """Convert OD array back to uint8 RGB."""
    rgb = np.exp(-od) * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _get_tissue_mask(od: np.ndarray) -> np.ndarray:
    """Boolean mask selecting non-background pixels (OD norm > threshold)."""
    return np.linalg.norm(od, axis=1) > _LUMINOSITY_THRESHOLD


def _get_stain_matrix(od: np.ndarray) -> np.ndarray:
    """
    Return a (2, 3) stain matrix [H-stain; E-stain] via SVD + angular method.
    od: (N, 3) array of tissue-pixel OD values.
    """
    _, _, Vt = np.linalg.svd(od, full_matrices=False)
    plane = Vt[:2].T                       # principal plane (3, 2)
    proj  = od @ plane                     # project pixels onto plane (N, 2)

    angles = np.arctan2(proj[:, 1], proj[:, 0])
    phi1   = np.percentile(angles, 100 - _ANGULAR_PERCENTILE)
    phi2   = np.percentile(angles, _ANGULAR_PERCENTILE)

    v1 = plane @ np.array([np.cos(phi1), np.sin(phi1)])
    v2 = plane @ np.array([np.cos(phi2), np.sin(phi2)])

    # Ensure H-stain (more blue) comes first by checking the blue channel
    stains = np.array([v1 / np.linalg.norm(v1),
                       v2 / np.linalg.norm(v2)])
    if stains[0, 2] < stains[1, 2]:
        stains = stains[[1, 0]]
    return stains


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MacenkoNormalizer:
    """
    Macenko H&E stain normalizer.

    Parameters
    ----------
    percentile : int
        Percentile (per stain channel) used to determine the reference
        concentration scale.  Default 99.
    """

    def __init__(self, percentile: int = 99) -> None:
        self.percentile  = percentile
        self._stain_mat: np.ndarray | None = None   # (2, 3)
        self._max_conc:  np.ndarray | None = None   # (2,)

    # ------------------------------------------------------------------
    def fit(self, image: np.ndarray) -> "MacenkoNormalizer":
        """
        Fit the normalizer on a representative reference image.

        Parameters
        ----------
        image : np.ndarray  shape (H, W, 3) dtype uint8
        """
        h, w = image.shape[:2]
        od   = _rgb_to_od(image).reshape(-1, 3)
        mask = _get_tissue_mask(od)
        tissue_od = od[mask]

        if tissue_od.shape[0] < 100:
            raise ValueError("Too few tissue pixels in the reference image — "
                             "ensure the image contains H&E-stained tissue.")

        self._stain_mat = _get_stain_matrix(tissue_od)          # (2, 3)
        conc = tissue_od @ np.linalg.pinv(self._stain_mat).T    # (N, 2)
        self._max_conc = np.percentile(conc, self.percentile, axis=0)  # (2,)
        return self

    # ------------------------------------------------------------------
    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize a new image to match the fitted reference stain.

        Parameters
        ----------
        image : np.ndarray  shape (H, W, 3) dtype uint8

        Returns
        -------
        np.ndarray  shape (H, W, 3) dtype uint8
        """
        if self._stain_mat is None:
            raise RuntimeError("Call fit() or load() before transform().")

        h, w = image.shape[:2]
        od   = _rgb_to_od(image).reshape(-1, 3)

        # Decompose source image into concentrations
        src_conc = od @ np.linalg.pinv(self._stain_mat).T   # (N, 2)
        src_max  = np.percentile(src_conc, self.percentile, axis=0)

        # Rescale concentrations so they match the reference scale
        src_conc_norm = src_conc / (src_max + 1e-8) * self._max_conc

        # Reconstruct OD and convert back to RGB
        od_norm = src_conc_norm @ self._stain_mat
        return _od_to_rgb(od_norm).reshape(h, w, 3)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save fitted parameters to a .npz file."""
        if self._stain_mat is None:
            raise RuntimeError("Normalizer is not fitted — nothing to save.")
        np.savez(str(path),
                 stain_mat=self._stain_mat,
                 max_conc=self._max_conc,
                 percentile=np.array(self.percentile))

    @classmethod
    def load(cls, path: str | Path) -> "MacenkoNormalizer":
        """Load a previously saved normalizer."""
        data = np.load(str(path))
        norm = cls(percentile=int(data["percentile"]))
        norm._stain_mat = data["stain_mat"]
        norm._max_conc  = data["max_conc"]
        return norm

    @property
    def is_fitted(self) -> bool:
        return self._stain_mat is not None


# ---------------------------------------------------------------------------
# Convenience: fit both scanners from a single function call
# ---------------------------------------------------------------------------

def fit_scanner_normalizers(
    aperio_ref_image: np.ndarray,
    hamamatsu_ref_image: np.ndarray,
    save_dir: str | Path | None = None,
) -> tuple[MacenkoNormalizer, MacenkoNormalizer]:
    """
    Fit and optionally save two normalizers (one per scanner).

    Returns
    -------
    (aperio_norm, hamamatsu_norm)
    """
    aperio_norm    = MacenkoNormalizer().fit(aperio_ref_image)
    hamamatsu_norm = MacenkoNormalizer().fit(hamamatsu_ref_image)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        aperio_norm.save(save_dir / "aperio_macenko.npz")
        hamamatsu_norm.save(save_dir / "hamamatsu_macenko.npz")

    return aperio_norm, hamamatsu_norm
