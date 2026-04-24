"""
Main training loop for atypia classification using k-fold cross-validation.

Orchestrates: data loading, model training, validation, checkpointing, and evaluation.
This is the entry point for training. Modify config.py to adjust hyperparameters.

Usage:
    python -m Atypia.train
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# Add project root to path (handles both module and direct execution)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CommonRoutines import (
    AtypiaDataset,
    get_kfold_splits,
    MacenkoNormalizer,
)
from Atypia.config import Config, get_default_config
from Atypia.models import create_model
from Atypia.losses import get_loss_fn
from Atypia.metrics import AtypiaMetrics, ordinal_logits_to_predictions


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------

def load_stain_normalizers(cfg: Config) -> dict[str, MacenkoNormalizer | None]:
    """
    Load pre-fitted stain normalizers for each scanner.
    
    If files don't exist, returns None values and training proceeds
    without stain normalization.
    """
    norms = {"A": None, "H": None}
    for scanner, fname in [("A", "aperio_macenko.npz"), ("H", "hamamatsu_macenko.npz")]:
        p = cfg.data.norm_dir / fname
        if p.exists():
            norms[scanner] = MacenkoNormalizer.load(p)
    return norms


def create_dataloaders(
    cfg: Config,
    train_ids: list[str],
    val_ids: list[str],
    normalizers: dict[str, MacenkoNormalizer | None],
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.
    
    Applies augmentation and stain normalization per dataset split.
    """
    train_ds = AtypiaDataset(
        slide_ids=train_ids,
        extract_root=cfg.data.extract_root,
        magnification=cfg.data.magnification,
        data_split="training",
        split="train",
        normalizers=normalizers if cfg.stain.enabled else None,
    )
    
    val_ds = AtypiaDataset(
        slide_ids=val_ids,
        extract_root=cfg.data.extract_root,
        magnification=cfg.data.magnification,
        data_split="training",
        split="val",
        normalizers=normalizers if cfg.stain.enabled else None,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Run one training epoch.
    
    Returns average loss over all batches.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch_idx, (images, labels, meta) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": total_loss / n_batches})
    
    return total_loss / n_batches


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, dict[str, float]]:
    """
    Run validation on the entire validation set.
    
    Returns (avg loss, metrics dict).
    """
    model.eval()
    total_loss = 0.0
    metrics = AtypiaMetrics()
    n_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Val", leave=False)
        for images, labels, meta in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Convert ordinal logits to predictions
            logits_np = logits.cpu().numpy()
            labels_np = labels.cpu().numpy()
            preds = ordinal_logits_to_predictions(logits_np)
            
            metrics.update(preds, labels_np)
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": total_loss / n_batches})
    
    summary = metrics.summary()
    return total_loss / n_batches, summary


def train_fold(
    cfg: Config,
    fold_idx: int,
    train_ids: list[str],
    val_ids: list[str],
) -> dict[str, Any]:
    """
    Train model on one fold (train_ids vs val_ids split).
    
    Returns: dict with metrics and best checkpoint path.
    """
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}/{cfg.data.n_folds}")
    print(f"  Train slides: {len(train_ids)} | Val slides: {len(val_ids)}")
    print(f"{'='*60}")
    
    device = cfg.device
    
    # Load normalizers
    normalizers = load_stain_normalizers(cfg)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        cfg, train_ids, val_ids, normalizers
    )
    
    # Create model
    model = create_model(cfg.model, device=device)
    
    # Loss and optimizer
    criterion = get_loss_fn(
        loss_type=cfg.training.loss_type,
        class_weights=cfg.training.class_weights,
        label_smoothing=cfg.training.label_smoothing,
    )
    criterion = criterion.to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.training.num_epochs // 2,
        T_mult=1,
        eta_min=1e-6,
    )
    
    # Training loop
    best_score = -float("inf")
    patience_counter = 0
    fold_results = {"best_val_metrics": {}}
    
    for epoch in range(cfg.training.num_epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            max_grad_norm=cfg.training.max_grad_norm
        )
        
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        val_score = val_metrics[cfg.training.early_stopping_metric]
        
        print(f"Epoch {epoch+1:3d} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_score={val_score:.4f}")
        
        scheduler.step()
        
        # Early stopping
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            fold_results["best_val_metrics"] = val_metrics
            
            # Save checkpoint
            ckpt_path = cfg.checkpoint_dir / f"fold{fold_idx}_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Saved checkpoint: {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                print(f"  → Early stopping (patience={cfg.training.early_stopping_patience})")
                break
    
    fold_results["checkpoint_path"] = ckpt_path
    return fold_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: Config | None = None):
    """
    Run full k-fold training pipeline.
    
    Trains and validates the model on each fold, then saves results.
    """
    if cfg is None:
        cfg = get_default_config()
    
    print("\n" + "="*60)
    print("ATYPIA TRAINING PIPELINE")
    print("="*60)
    print(f"Device: {cfg.device}")
    print(f"Model: {cfg.model.backbone}")
    print(f"Dataset: {cfg.data.extract_root}")
    print(f"K-fold: {cfg.data.n_folds}")
    
    # Set random seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # Get folds
    folds = get_kfold_splits(
        n_splits=cfg.data.n_folds,
        shuffle=True,
        seed=cfg.seed,
    )
    
    all_fold_results = []
    
    # Train each fold
    for fold_idx, (train_ids, val_ids) in enumerate(folds):
        result = train_fold(cfg, fold_idx, train_ids, val_ids)
        all_fold_results.append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    for fold_idx, result in enumerate(all_fold_results):
        metrics = result["best_val_metrics"]
        print(f"Fold {fold_idx}: "
              f"challenge_score={metrics.get('challenge_score', 0):.4f}, "
              f"accuracy={metrics.get('accuracy', 0):.4f}")
    
    # Save summary
    summary_path = cfg.output_dir / "training_summary.txt"
    with open(summary_path, "w") as f:
        for fold_idx, result in enumerate(all_fold_results):
            f.write(f"Fold {fold_idx}\n")
            f.write(f"  Checkpoint: {result['checkpoint_path']}\n")
            f.write(f"  Metrics: {result['best_val_metrics']}\n\n")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
