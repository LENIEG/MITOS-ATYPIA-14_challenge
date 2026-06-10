"""
Evaluate and compare atypia checkpoints on consistent k-fold validation splits.

Usage:
    python -m Atypia.evaluate_models
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from CommonRoutines import AtypiaDataset, MacenkoNormalizer, get_kfold_splits
from Atypia.config import Config, get_default_config
from Atypia.metrics import AtypiaMetrics, ordinal_logits_to_predictions
from Atypia.models import create_model


@dataclass
class ModelEvalResult:
    name: str
    checkpoint_path: Path
    per_fold_metrics: list[dict[str, float]]


def resolve_device(preferred: str) -> str:
    """Resolve runtime device with safe fallback to CPU."""
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return preferred


def load_stain_normalizers(cfg: Config) -> dict[str, MacenkoNormalizer | None]:
    """Load pre-fitted stain normalizers if they exist."""
    norms = {"A": None, "H": None}
    for scanner, fname in [("A", "aperio_macenko.npz"), ("H", "hamamatsu_macenko.npz")]:
        path = cfg.data.norm_dir / fname
        if path.exists():
            norms[scanner] = MacenkoNormalizer.load(path)
    return norms


def create_val_loader(
    cfg: Config,
    val_ids: list[str],
    normalizers: dict[str, MacenkoNormalizer | None],
) -> DataLoader:
    """Build validation dataloader for a list of slide ids."""
    val_ds = AtypiaDataset(
        slide_ids=val_ids,
        extract_root=cfg.data.extract_root,
        magnification=cfg.data.magnification,
        data_split="training",
        split="val",
        normalizers=normalizers if cfg.stain.enabled else None,
    )
    return DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )


def evaluate_checkpoint_on_fold(
    cfg: Config,
    checkpoint_path: Path,
    val_ids: list[str],
    normalizers: dict[str, MacenkoNormalizer | None],
) -> dict[str, float]:
    """Evaluate one checkpoint on one validation fold and return metrics."""
    model = create_model(cfg.model, device=cfg.device)
    state = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(state)
    model.eval()

    loader = create_val_loader(cfg, val_ids, normalizers)
    metrics = AtypiaMetrics()

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc=f"Eval {checkpoint_path.name}", leave=False):
            images = images.to(cfg.device)
            logits = model(images)
            preds = ordinal_logits_to_predictions(logits.cpu().numpy())
            metrics.update(preds, labels.numpy())

    summary = metrics.summary()
    summary["n_samples"] = float(len(loader.dataset))
    return summary


def aggregate_fold_metrics(per_fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-fold metrics into mean and std summaries."""
    if not per_fold_metrics:
        return {}

    metric_keys = [k for k in per_fold_metrics[0].keys() if k != "n_samples"]
    out: dict[str, float] = {}

    for key in metric_keys:
        vals = np.array([m[key] for m in per_fold_metrics], dtype=np.float64)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std(ddof=0))

    out["n_folds"] = float(len(per_fold_metrics))
    out["n_samples_total"] = float(sum(m.get("n_samples", 0.0) for m in per_fold_metrics))
    return out


def discover_candidates(checkpoint_dir: Path) -> tuple[dict[int, Path], list[tuple[str, Path]]]:
    """Discover fold checkpoints and merged candidates from checkpoint directory."""
    fold_ckpts: dict[int, Path] = {}
    fold_pattern = re.compile(r"fold(\d+)_best\.pt$")

    for ckpt in sorted(checkpoint_dir.glob("fold*_best.pt")):
        m = fold_pattern.search(ckpt.name)
        if m:
            fold_ckpts[int(m.group(1))] = ckpt

    merged_candidates: list[tuple[str, Path]] = []
    merged = checkpoint_dir / "final_merged_from_folds.pt"
    finetuned = checkpoint_dir / "final_merged_finetuned.pt"
    if merged.exists():
        merged_candidates.append(("merged", merged))
    if finetuned.exists():
        merged_candidates.append(("merged_finetuned", finetuned))

    return fold_ckpts, merged_candidates


def format_ranking_rows(rows: list[dict[str, Any]]) -> str:
    """Format a compact ranking table."""
    header = (
        "Rank | Model | Folds | Challenge(mean+-std) | BalancedAcc(mean) | Acc(mean)\n"
        "-----|-------|-------|----------------------|-------------------|---------"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"{row['rank']:>4} | {row['model']} | {row['folds']} | "
            f"{row['challenge_score_mean']:.4f}+-{row['challenge_score_std']:.4f} | "
            f"{row['balanced_accuracy_mean']:.4f} | {row['accuracy_mean']:.4f}"
        )
    return "\n".join(lines)


def main(cfg: Config | None = None) -> None:
    """Run checkpoint comparison and write a persistent report."""
    if cfg is None:
        cfg = get_default_config()

    cfg.device = resolve_device(cfg.device)
    checkpoint_dir = cfg.checkpoint_dir
    report_path = cfg.output_dir / "model_comparison.txt"

    folds = get_kfold_splits(
        n_splits=cfg.data.n_folds,
        shuffle=True,
        seed=cfg.seed,
    )
    normalizers = load_stain_normalizers(cfg)

    fold_ckpts, merged_candidates = discover_candidates(checkpoint_dir)

    if not fold_ckpts and not merged_candidates:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoint_dir}. Expected fold or merged checkpoints."
        )

    all_results: list[ModelEvalResult] = []

    # Evaluate each fold checkpoint on its own validation fold.
    for fold_idx, ckpt_path in sorted(fold_ckpts.items()):
        if fold_idx >= len(folds):
            print(f"Skipping {ckpt_path.name}: fold index {fold_idx} out of range")
            continue
        _, val_ids = folds[fold_idx]
        metrics = evaluate_checkpoint_on_fold(cfg, ckpt_path, val_ids, normalizers)
        all_results.append(
            ModelEvalResult(
                name=f"fold{fold_idx}",
                checkpoint_path=ckpt_path,
                per_fold_metrics=[metrics],
            )
        )

    # Evaluate merged variants across all fold validation sets.
    for name, ckpt_path in merged_candidates:
        per_fold_metrics: list[dict[str, float]] = []
        for fold_idx, (_, val_ids) in enumerate(folds):
            fold_metrics = evaluate_checkpoint_on_fold(cfg, ckpt_path, val_ids, normalizers)
            per_fold_metrics.append(fold_metrics)

        all_results.append(
            ModelEvalResult(
                name=name,
                checkpoint_path=ckpt_path,
                per_fold_metrics=per_fold_metrics,
            )
        )

    ranked_rows: list[dict[str, Any]] = []
    details_lines: list[str] = []

    for result in all_results:
        agg = aggregate_fold_metrics(result.per_fold_metrics)
        if not agg:
            continue

        ranked_rows.append(
            {
                "model": result.name,
                "checkpoint": str(result.checkpoint_path),
                "folds": int(agg["n_folds"]),
                "challenge_score_mean": agg.get("challenge_score_mean", -1.0),
                "challenge_score_std": agg.get("challenge_score_std", 0.0),
                "balanced_accuracy_mean": agg.get("balanced_accuracy_mean", -1.0),
                "accuracy_mean": agg.get("accuracy_mean", -1.0),
            }
        )

        details_lines.append(f"Model: {result.name}")
        details_lines.append(f"  Checkpoint: {result.checkpoint_path}")
        details_lines.append(f"  Folds evaluated: {int(agg['n_folds'])}")
        details_lines.append(
            f"  challenge_score mean/std: {agg.get('challenge_score_mean', 0.0):.6f} / "
            f"{agg.get('challenge_score_std', 0.0):.6f}"
        )
        details_lines.append(
            f"  accuracy mean/std: {agg.get('accuracy_mean', 0.0):.6f} / "
            f"{agg.get('accuracy_std', 0.0):.6f}"
        )
        details_lines.append(
            f"  balanced_accuracy mean/std: {agg.get('balanced_accuracy_mean', 0.0):.6f} / "
            f"{agg.get('balanced_accuracy_std', 0.0):.6f}"
        )
        details_lines.append(
            f"  recall_Low mean/std: {agg.get('recall_Low (1)_mean', 0.0):.6f} / "
            f"{agg.get('recall_Low (1)_std', 0.0):.6f}"
        )
        details_lines.append(
            f"  recall_Moderate mean/std: {agg.get('recall_Moderate (2)_mean', 0.0):.6f} / "
            f"{agg.get('recall_Moderate (2)_std', 0.0):.6f}"
        )
        details_lines.append(
            f"  recall_High mean/std: {agg.get('recall_High (3)_mean', 0.0):.6f} / "
            f"{agg.get('recall_High (3)_std', 0.0):.6f}"
        )
        details_lines.append("")

    ranked_rows.sort(key=lambda x: x["challenge_score_mean"], reverse=True)
    for i, row in enumerate(ranked_rows, start=1):
        row["rank"] = i

    ranking_table = format_ranking_rows(ranked_rows)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "Atypia Model Comparison Report",
        f"Generated: {timestamp}",
        f"Device: {cfg.device}",
        f"Seed: {cfg.seed}",
        f"Checkpoint directory: {checkpoint_dir}",
        "",
        "Ranking",
        ranking_table,
        "",
        "Details",
        *details_lines,
    ]

    report_text = "\n".join(report_lines)
    print("\n" + ranking_table)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nComparison report saved to: {report_path}")


if __name__ == "__main__":
    main()
