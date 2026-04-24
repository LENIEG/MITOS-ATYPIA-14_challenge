"""
Block-stratified train / validation splits for the MITOS-ATYPIA-14 challenge.

Key rule
--------
The Aperio and Hamamatsu slides are the SAME tissue blocks scanned twice
(A03 ↔ H03, A04 ↔ H04, …).  A block number must NEVER appear in both
the training fold and the validation fold — even across scanner types —
otherwise the split leaks tissue-block information.

Correct split:
    fold-out block 03  →  hold out A03 AND H03 together.

Usage
-----
    from CommonRoutines.splits import get_kfold_splits, TRAIN_BLOCKS

    # 5-fold cross-validation over training blocks (11 blocks):
    for fold_idx, (train_ids, val_ids) in enumerate(get_kfold_splits(n_splits=5)):
        print(fold_idx, train_ids, val_ids)
        # train_ids / val_ids are lists of slide IDs, e.g. ["A03", "H03", "A04", …]

    # Fixed leave-one-block-out split by block number:
    from CommonRoutines.splits import leave_one_block_out
    for block, train_ids, val_ids in leave_one_block_out():
        ...

    # Just get the final train and test slide IDs:
    from CommonRoutines.splits import TRAIN_SLIDE_IDS, TEST_SLIDE_IDS
"""

from __future__ import annotations

import random
from typing import Iterator

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

# Block numbers present in the training set
TRAIN_BLOCKS: list[str] = ["03", "04", "05", "07", "10", "11", "12", "14", "15", "17", "18"]

# Block numbers present in the test set (held out by the challenge organisers)
TEST_BLOCKS: list[str] = ["06", "08", "09", "13", "16"]

SCANNERS: tuple[str, str] = ("A", "H")   # Aperio prefix, Hamamatsu prefix


def _block_to_slide_ids(blocks: list[str]) -> list[str]:
    """Convert a list of block numbers to all slide IDs for both scanners."""
    ids: list[str] = []
    for b in blocks:
        for s in SCANNERS:
            ids.append(f"{s}{b}")
    return ids


TRAIN_SLIDE_IDS: list[str] = _block_to_slide_ids(TRAIN_BLOCKS)   # 22 ids
TEST_SLIDE_IDS:  list[str] = _block_to_slide_ids(TEST_BLOCKS)     # 10 ids


# ---------------------------------------------------------------------------
# K-fold: stratified by block (not by slide)
# ---------------------------------------------------------------------------

def get_kfold_splits(
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 42,
) -> list[tuple[list[str], list[str]]]:
    """
    Return n_splits (train_slide_ids, val_slide_ids) pairs.

    Splitting is performed at the BLOCK level so that both the Aperio and
    Hamamatsu scans of the same tissue always land in the same fold.

    Parameters
    ----------
    n_splits : int
        Number of folds.  Must satisfy 2 ≤ n_splits ≤ len(TRAIN_BLOCKS).
    shuffle : bool
        Shuffle block order before assigning to folds.
    seed : int
        Random seed (only used when shuffle=True).

    Returns
    -------
    List of (train_ids, val_ids) tuples, one per fold.
    Each ID is a slide-level string such as "A03" or "H03".
    """
    if not (2 <= n_splits <= len(TRAIN_BLOCKS)):
        raise ValueError(
            f"n_splits must be between 2 and {len(TRAIN_BLOCKS)}, got {n_splits}."
        )

    blocks = list(TRAIN_BLOCKS)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(blocks)

    # Assign blocks to folds in round-robin order
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for i, block in enumerate(blocks):
        folds[i % n_splits].append(block)

    splits: list[tuple[list[str], list[str]]] = []
    for fold_idx in range(n_splits):
        val_blocks   = folds[fold_idx]
        train_blocks = [b for i, fold_blocks in enumerate(folds)
                        for b in fold_blocks if i != fold_idx]
        splits.append(
            (_block_to_slide_ids(train_blocks),
             _block_to_slide_ids(val_blocks))
        )
    return splits


# ---------------------------------------------------------------------------
# Leave-one-block-out
# ---------------------------------------------------------------------------

def leave_one_block_out() -> Iterator[tuple[str, list[str], list[str]]]:
    """
    Yield (held_out_block, train_slide_ids, val_slide_ids) for every block
    in the training set.  Useful for small-data analysis.
    """
    for block in TRAIN_BLOCKS:
        val_blocks   = [block]
        train_blocks = [b for b in TRAIN_BLOCKS if b != block]
        yield (
            block,
            _block_to_slide_ids(train_blocks),
            _block_to_slide_ids(val_blocks),
        )


# ---------------------------------------------------------------------------
# Simple helper: fixed split — first N blocks for val
# ---------------------------------------------------------------------------

def get_fixed_split(
    n_val_blocks: int = 2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Return a single (train_slide_ids, val_slide_ids) split where
    `n_val_blocks` randomly chosen blocks are held out for validation.
    """
    blocks = list(TRAIN_BLOCKS)
    rng = random.Random(seed)
    rng.shuffle(blocks)
    val_blocks   = blocks[:n_val_blocks]
    train_blocks = blocks[n_val_blocks:]
    return _block_to_slide_ids(train_blocks), _block_to_slide_ids(val_blocks)
