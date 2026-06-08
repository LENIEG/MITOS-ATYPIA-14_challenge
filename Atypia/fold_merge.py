"""
Utilities to merge k-fold checkpoints into a single deployable model.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from CommonRoutines import AtypiaDataset, MacenkoNormalizer
from Atypia.config import Config
from Atypia.losses import get_loss_fn
from Atypia.models import create_model
from Atypia.class_balance import compute_class_weights, get_sample_weights


def compute_fold_ensemble_weights(
    all_fold_results: list[dict[str, Any]],
    score_key: str = "challenge_score",
    alpha: float = 8.0,
) -> list[float]:
    """
    Convert fold validation scores into softmax-normalized merge weights.
    """
    scores = [float(result["best_val_metrics"].get(score_key, 0.0)) for result in all_fold_results]

    scores_np = np.array(scores, dtype=np.float64)
    logits = alpha * scores_np
    logits = logits - logits.max()  # numerical stability
    weights = np.exp(logits)
    weights = weights / weights.sum()
    return weights.tolist()


def merge_fold_checkpoints(
    cfg: Config,
    all_fold_results: list[dict[str, Any]],
    merge_weights: list[float],
) -> nn.Module:
    """
    Create one merged model by weighted-averaging fold checkpoint parameters.
    """
    merged_model = create_model(cfg.model, device=cfg.device)
    weighted_state: OrderedDict[str, torch.Tensor] = OrderedDict()

    for fold_idx, (result, weight) in enumerate(zip(all_fold_results, merge_weights)):
        ckpt_path = Path(result["checkpoint_path"])
        state = torch.load(ckpt_path, map_location=cfg.device)

        for key, value in state.items():
            value_weighted = value.detach().to(cfg.device) * float(weight)
            if key not in weighted_state:
                weighted_state[key] = value_weighted
            else:
                weighted_state[key] += value_weighted

        score = result["best_val_metrics"].get("challenge_score", 0.0)
        print(f"  Fold {fold_idx}: weight={weight:.4f}, val_challenge_score={score:.4f}")

    merged_model.load_state_dict(weighted_state)
    return merged_model


def recalibrate_batchnorm_stats(
    model: nn.Module,
    cfg: Config,
    normalizers: dict[str, MacenkoNormalizer | None],
    train_ids: list[str],
) -> None:
    """
    Recompute BatchNorm running stats on full training data after weight merge.
    """
    ds = AtypiaDataset(
        slide_ids=train_ids,
        extract_root=cfg.data.extract_root,
        magnification=cfg.data.magnification,
        data_split="training",
        split="val",  # deterministic pipeline for BN stat pass
        normalizers=normalizers if cfg.stain.enabled else None,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    model.train()
    with torch.no_grad():
        for images, _, _ in tqdm(loader, desc="Recalibrate BN", leave=False):
            images = images.to(cfg.device)
            _ = model(images)


def fine_tune_merged_model(
    model: nn.Module,
    cfg: Config,
    normalizers: dict[str, MacenkoNormalizer | None],
    train_ids: list[str],
    num_epochs: int = 5,
    lr_scale: float = 0.1,
) -> float:
    """
    Run a short conservative fine-tuning stage on full training data.

    Returns the best training loss observed.
    """
    ds = AtypiaDataset(
        slide_ids=train_ids,
        extract_root=cfg.data.extract_root,
        magnification=cfg.data.magnification,
        data_split="training",
        split="train",
        normalizers=normalizers if cfg.stain.enabled else None,
    )

    class_weights = compute_class_weights(ds)
    sample_weights = get_sample_weights(ds, class_weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(ds),
        replacement=True,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    criterion = get_loss_fn(
        loss_type=cfg.training.loss_type,
        class_weights=class_weights.tolist(),
        label_smoothing=cfg.training.label_smoothing,
    ).to(cfg.device)

    fine_tune_lr = max(cfg.training.learning_rate * lr_scale, 1e-6)
    optimizer = AdamW(
        model.parameters(),
        lr=fine_tune_lr,
        weight_decay=cfg.training.weight_decay,
    )

    print("\nShort full-data fine-tuning")
    print(f"  epochs={num_epochs}, lr={fine_tune_lr:.2e}")

    best_loss = float("inf")
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Fine-tune {epoch+1}/{num_epochs}", leave=False)
        for images, labels, _ in pbar:
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix({"loss": total_loss / n_batches})

        epoch_loss = total_loss / max(n_batches, 1)
        print(f"  Fine-tune epoch {epoch + 1}: train_loss={epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_loss


def build_and_save_merged_model(
    cfg: Config,
    all_fold_results: list[dict[str, Any]],
    folds: list[tuple[list[str], list[str]]],
    normalizers: dict[str, MacenkoNormalizer | None],
    summary_path: Path,
    alpha: float = 8.0,
) -> Path:
    """
    Build, recalibrate, and save the merged final model from fold checkpoints.
    """
    print("\n" + "=" * 60)
    print("BUILDING MERGED FINAL MODEL")
    print("=" * 60)

    merge_weights = compute_fold_ensemble_weights(
        all_fold_results,
        score_key=cfg.training.early_stopping_metric,
        alpha=alpha,
    )
    merged_model = merge_fold_checkpoints(cfg, all_fold_results, merge_weights)

    all_train_ids = sorted({sid for train_ids, val_ids in folds for sid in (train_ids + val_ids)})
    recalibrate_batchnorm_stats(merged_model, cfg, normalizers, all_train_ids)

    merged_ckpt_path = cfg.checkpoint_dir / "final_merged_from_folds.pt"
    torch.save(merged_model.state_dict(), merged_ckpt_path)
    print(f"Merged final checkpoint saved to: {merged_ckpt_path}")

    # Conservative post-merge refinement on all available training data.
    best_finetune_loss = fine_tune_merged_model(
        model=merged_model,
        cfg=cfg,
        normalizers=normalizers,
        train_ids=all_train_ids,
        num_epochs=5,
        lr_scale=0.1,
    )

    finetuned_ckpt_path = cfg.checkpoint_dir / "final_merged_finetuned.pt"
    torch.save(merged_model.state_dict(), finetuned_ckpt_path)
    print(f"Fine-tuned merged checkpoint saved to: {finetuned_ckpt_path}")

    with open(summary_path, "a") as f:
        f.write("Merged Final Model\n")
        f.write(f"  Checkpoint: {merged_ckpt_path}\n")
        f.write(f"  Fold merge weights: {merge_weights}\n\n")
        f.write("Merged Final Model (Fine-tuned)\n")
        f.write(f"  Checkpoint: {finetuned_ckpt_path}\n")
        f.write("  Fine-tune config: epochs=5, lr_scale=0.1\n")
        f.write(f"  Best fine-tune train_loss: {best_finetune_loss:.6f}\n\n")

    return finetuned_ckpt_path
