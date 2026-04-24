"""
CommonRoutines — shared preprocessing, dataset, and split utilities
for the MITOS-ATYPIA-14 challenge.
"""

from CommonRoutines.stain_norm   import MacenkoNormalizer, fit_scanner_normalizers
from CommonRoutines.augmentation import get_atypia_augmentation, get_mitosis_augmentation
from CommonRoutines.splits       import (
    get_kfold_splits,
    get_fixed_split,
    leave_one_block_out,
    TRAIN_SLIDE_IDS,
    TEST_SLIDE_IDS,
    TRAIN_BLOCKS,
    TEST_BLOCKS,
)
from CommonRoutines.dataset      import (
    AtypiaDataset,
    MitosisDataset,
    slide_id_to_paths,
    read_atypia_label,
    read_mitosis_csv,
    load_image_rgb,
)

__all__ = [
    "MacenkoNormalizer",
    "fit_scanner_normalizers",
    "get_atypia_augmentation",
    "get_mitosis_augmentation",
    "get_kfold_splits",
    "get_fixed_split",
    "leave_one_block_out",
    "TRAIN_SLIDE_IDS",
    "TEST_SLIDE_IDS",
    "TRAIN_BLOCKS",
    "TEST_BLOCKS",
    "AtypiaDataset",
    "MitosisDataset",
    "slide_id_to_paths",
    "read_atypia_label",
    "read_mitosis_csv",
    "load_image_rgb",
]
