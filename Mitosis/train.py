"""
Main training loop for mitosis detection using k-fold cross-validation.

Usage:
    python -m Mitosis.train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CommonRoutines import MitosisDataset, MacenkoNormalizer, get_kfold_splits
from Mitosis.config import Config, get_default_config
from Mitosis.heatmap import build_target_heatmaps, decode_heatmaps
from Mitosis.losses import get_loss_fn
from Mitosis.metrics import MitosisMetrics
from Mitosis.models import create_model


def resolve_device(preferred: str) -> str:
    """Resolve runtime device with safe fallback to CPU."""
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return preferred


def load_stain_normalizers(cfg: Config) -> dict[str, MacenkoNormalizer | None]:
    """Load pre-fitted scanner-specific stain normalizers if present."""
    norms = {"A": None, "H": None}
    for scanner, fname in [("A", "aperio_macenko.npz"), ("H", "hamamatsu_macenko.npz")]:
        p = cfg.data.norm_dir / fname
        if p.exists():
            norms[scanner] = MacenkoNormalizer.load(p)
    return norms


def mitosis_collate_fn(batch):
    """Custom collate for variable-length centroid targets."""
    images = torch.stack([item[0] for item in batch], dim=0)
    centroids = [item[1] for item in batch]
    meta = [item[2] for item in batch]
    return images, centroids, meta


def create_dataloaders(
    cfg: Config,
    train_ids: list[str],
    val_ids: list[str],
    normalizers: dict[str, MacenkoNormalizer | None],
) -> tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders for one fold."""
    train_ds = MitosisDataset(
        slide_ids=train_ids,
        extract_root=cfg.data.extract_root,
        magnification=cfg.data.magnification,
        data_split="training",
        split="train",
        normalizers=normalizers if cfg.stain.enabled else None,
    )
    val_ds = MitosisDataset(
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
        collate_fn=mitosis_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=mitosis_collate_fn,
    )
    return train_loader, val_loader


def train_epoch_with_optimizer(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
) -> float:
    """Run one training epoch with optimizer steps."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    device = cfg.device

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, centroids_batch, _ in pbar:
        images = images.to(device)
        targets = build_target_heatmaps(
            centroids_batch,
            image_size=cfg.model.input_size,
            output_stride=cfg.model.output_stride,
            sigma=cfg.training.gaussian_sigma,
            device=device,
        )

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        pbar.set_postfix({"loss": total_loss / n_batches})

    return total_loss / max(n_batches, 1)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    cfg: Config,
) -> tuple[float, dict[str, float]]:
    """Validate on one fold and return loss and detection metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    metrics = MitosisMetrics(radius_px=cfg.training.match_radius_px)

    with torch.no_grad():
        pbar = tqdm(loader, desc="Val", leave=False)
        for images, centroids_batch, _ in pbar:
            images = images.to(cfg.device)
            targets = build_target_heatmaps(
                centroids_batch,
                image_size=cfg.model.input_size,
                output_stride=cfg.model.output_stride,
                sigma=cfg.training.gaussian_sigma,
                device=cfg.device,
            )

            logits = model(images)
            loss = criterion(logits, targets)

            detections_batch = decode_heatmaps(
                logits,
                output_stride=cfg.model.output_stride,
                threshold=cfg.training.decode_threshold,
                nms_kernel=cfg.training.decode_nms_kernel,
                max_detections=cfg.training.decode_max_detections,
            )
            for detections, gt_centroids in zip(detections_batch, centroids_batch):
                metrics.update(detections, list(gt_centroids))

            total_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix({"loss": total_loss / n_batches})

    return total_loss / max(n_batches, 1), metrics.summary()


def train_fold(
    cfg: Config,
    fold_idx: int,
    train_ids: list[str],
    val_ids: list[str],
) -> dict[str, Any]:
    """Train one fold and save best checkpoint by validation F1."""
    print(f"\n{'=' * 60}")
    print(f"Fold {fold_idx + 1}/{cfg.data.n_folds}")
    print(f"  Train slides: {len(train_ids)} | Val slides: {len(val_ids)}")
    print(f"{'=' * 60}")

    normalizers = load_stain_normalizers(cfg)
    train_loader, val_loader = create_dataloaders(cfg, train_ids, val_ids, normalizers)

    model = create_model(cfg.model, device=cfg.device)
    criterion = get_loss_fn(pos_weight=cfg.training.pos_weight).to(cfg.device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(cfg.training.num_epochs // 2, 1),
        T_mult=1,
        eta_min=1e-6,
    )

    best_score = -float("inf")
    patience_counter = 0
    fold_results: dict[str, Any] = {"best_val_metrics": {}}
    ckpt_path = cfg.checkpoint_dir / f"fold{fold_idx}_best.pt"

    for epoch in range(cfg.training.num_epochs):
        train_loss = train_epoch_with_optimizer(model, train_loader, criterion, optimizer, cfg)
        val_loss, val_metrics = validate(model, val_loader, criterion, cfg)
        val_score = float(val_metrics[cfg.training.early_stopping_metric])

        print(
            f"Epoch {epoch + 1:3d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_f1={val_metrics['f1']:.4f} | "
            f"precision={val_metrics['precision']:.4f} | recall={val_metrics['recall']:.4f}"
        )
        scheduler.step()

        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            fold_results["best_val_metrics"] = val_metrics
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved checkpoint: {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                print(f"  -> Early stopping (patience={cfg.training.early_stopping_patience})")
                break

    fold_results["checkpoint_path"] = ckpt_path
    return fold_results


def main(cfg: Config | None = None):
    """Run full k-fold mitosis training pipeline."""
    if cfg is None:
        cfg = get_default_config()
    cfg.device = resolve_device(cfg.device)

    print("\n" + "=" * 60)
    print("MITOSIS TRAINING PIPELINE")
    print("=" * 60)
    print(f"Device: {cfg.device}")
    print(f"Model: {cfg.model.backbone}")
    print(f"Dataset: {cfg.data.extract_root}")
    print(f"K-fold: {cfg.data.n_folds}")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    folds = get_kfold_splits(
        n_splits=cfg.data.n_folds,
        shuffle=True,
        seed=cfg.seed,
    )

    all_fold_results = []
    for fold_idx, (train_ids, val_ids) in enumerate(folds):
        result = train_fold(cfg, fold_idx, train_ids, val_ids)
        all_fold_results.append(result)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for fold_idx, result in enumerate(all_fold_results):
        metrics = result["best_val_metrics"]
        print(
            f"Fold {fold_idx}: "
            f"f1={metrics.get('f1', 0.0):.4f}, "
            f"precision={metrics.get('precision', 0.0):.4f}, "
            f"recall={metrics.get('recall', 0.0):.4f}"
        )

    summary_path = cfg.output_dir / "training_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for fold_idx, result in enumerate(all_fold_results):
            f.write(f"Fold {fold_idx}\n")
            f.write(f"  Checkpoint: {result['checkpoint_path']}\n")
            f.write(f"  Metrics: {result['best_val_metrics']}\n\n")

    print(f"Summary saved to: {summary_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for training overrides."""
    parser = argparse.ArgumentParser(description="Run Mitosis k-fold training")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--n-folds", type=int, default=None, help="Override number of folds")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda or cpu")
    parser.add_argument(
        "--disable-stain-norm",
        action="store_true",
        help="Disable stain normalization during training",
    )
    return parser


def main_cli(args: argparse.Namespace) -> None:
    """CLI wrapper for configurable training runs."""
    cfg = get_default_config()
    if args.num_epochs is not None:
        cfg.training.num_epochs = args.num_epochs
    if args.n_folds is not None:
        cfg.data.n_folds = args.n_folds
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.device is not None:
        cfg.device = args.device
    if args.disable_stain_norm:
        cfg.stain.enabled = False

    main(cfg)


if __name__ == "__main__":
    parser = build_arg_parser()
    main_cli(parser.parse_args())
