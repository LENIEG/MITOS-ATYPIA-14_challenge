"""
Evaluate and compare mitosis checkpoints on consistent k-fold validation splits.

Usage:
    python -m Mitosis.evaluate_models
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from CommonRoutines import MitosisDataset, MacenkoNormalizer, get_kfold_splits
from Mitosis.config import Config, get_default_config
from Mitosis.ensemble import (
    load_models_from_checkpoints,
    parse_checkpoint_paths,
    predict_ensemble_logits,
)
from Mitosis.heatmap import decode_heatmaps
from Mitosis.metrics import MitosisMetrics
from Mitosis.train import mitosis_collate_fn


@dataclass
class ModelEvalResult:
    name: str
    checkpoint_path: Path
    per_fold_metrics: list[dict[str, float]]


def resolve_device(preferred: str) -> str:
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return preferred


def load_stain_normalizers(cfg: Config) -> dict[str, MacenkoNormalizer | None]:
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
    ds = MitosisDataset(
        slide_ids=val_ids,
        extract_root=cfg.data.extract_root,
        magnification=cfg.data.magnification,
        data_split="training",
        split="val",
        normalizers=normalizers if cfg.stain.enabled else None,
    )
    return DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=mitosis_collate_fn,
    )


def evaluate_checkpoint_on_fold(
    cfg: Config,
    checkpoint_paths: list[Path],
    val_ids: list[str],
    normalizers: dict[str, MacenkoNormalizer | None],
) -> dict[str, float]:
    models = load_models_from_checkpoints(cfg, checkpoint_paths)

    loader = create_val_loader(cfg, val_ids, normalizers)
    metrics = MitosisMetrics(radius_px=cfg.training.match_radius_px)

    with torch.no_grad():
        model_label = (
            checkpoint_paths[0].name
            if len(checkpoint_paths) == 1
            else f"ensemble({len(checkpoint_paths)})"
        )
        for images, centroids_batch, _ in tqdm(loader, desc=f"Eval {model_label}", leave=False):
            images = images.to(cfg.device)
            logits = predict_ensemble_logits(models, images)
            detections_batch = decode_heatmaps(
                logits,
                output_stride=cfg.model.output_stride,
                threshold=cfg.training.decode_threshold,
                nms_kernel=cfg.training.decode_nms_kernel,
                max_detections=cfg.training.decode_max_detections,
            )
            for detections, gt in zip(detections_batch, centroids_batch):
                metrics.update(detections, list(gt))

    summary = metrics.summary()
    summary["n_samples"] = float(len(loader.dataset))
    return summary


def aggregate_fold_metrics(per_fold_metrics: list[dict[str, float]]) -> dict[str, float]:
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


def discover_candidates(checkpoint_dir: Path) -> dict[int, Path]:
    fold_ckpts: dict[int, Path] = {}
    fold_pattern = re.compile(r"fold(\d+)_best\.pt$")

    for ckpt in sorted(checkpoint_dir.glob("fold*_best.pt")):
        m = fold_pattern.search(ckpt.name)
        if m:
            fold_ckpts[int(m.group(1))] = ckpt
    return fold_ckpts


def format_ranking_rows(rows: list[dict[str, Any]]) -> str:
    header = (
        "Rank | Model | Folds | F1(mean+-std) | Precision(mean) | Recall(mean)\n"
        "-----|-------|-------|--------------|-----------------|-------------"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"{row['rank']:>4} | {row['model']} | {row['folds']} | "
            f"{row['f1_mean']:.4f}+-{row['f1_std']:.4f} | "
            f"{row['precision_mean']:.4f} | {row['recall_mean']:.4f}"
        )
    return "\n".join(lines)


def main(
    cfg: Config | None = None,
    ensemble_all_folds: bool = False,
    explicit_ensemble_checkpoints: list[Path] | None = None,
) -> None:
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
    fold_ckpts = discover_candidates(checkpoint_dir)

    if not fold_ckpts:
        raise FileNotFoundError(f"No fold checkpoints found in {checkpoint_dir}")

    all_results: list[ModelEvalResult] = []
    for fold_idx, ckpt_path in sorted(fold_ckpts.items()):
        if fold_idx >= len(folds):
            print(f"Skipping {ckpt_path.name}: fold index {fold_idx} out of range")
            continue
        _, val_ids = folds[fold_idx]
        metrics = evaluate_checkpoint_on_fold(cfg, [ckpt_path], val_ids, normalizers)
        all_results.append(
            ModelEvalResult(
                name=f"fold{fold_idx}",
                checkpoint_path=ckpt_path,
                per_fold_metrics=[metrics],
            )
        )

    ensemble_paths: list[Path] = []
    ensemble_name: str | None = None
    if explicit_ensemble_checkpoints:
        ensemble_paths = explicit_ensemble_checkpoints
        ensemble_name = "ensemble_custom"
    elif ensemble_all_folds and len(fold_ckpts) >= 2:
        ensemble_paths = [p for _, p in sorted(fold_ckpts.items())]
        ensemble_name = "ensemble_all_folds"

    if ensemble_paths and ensemble_name is not None:
        per_fold_metrics: list[dict[str, float]] = []
        for _, val_ids in folds:
            fold_metrics = evaluate_checkpoint_on_fold(cfg, ensemble_paths, val_ids, normalizers)
            per_fold_metrics.append(fold_metrics)

        all_results.append(
            ModelEvalResult(
                name=ensemble_name,
                checkpoint_path=ensemble_paths[0],
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
                "f1_mean": agg.get("f1_mean", -1.0),
                "f1_std": agg.get("f1_std", 0.0),
                "precision_mean": agg.get("precision_mean", -1.0),
                "recall_mean": agg.get("recall_mean", -1.0),
            }
        )

        details_lines.append(f"Model: {result.name}")
        details_lines.append(f"  Checkpoint: {result.checkpoint_path}")
        details_lines.append(f"  Folds evaluated: {int(agg['n_folds'])}")
        details_lines.append(
            f"  f1 mean/std: {agg.get('f1_mean', 0.0):.6f} / {agg.get('f1_std', 0.0):.6f}"
        )
        details_lines.append(
            f"  precision mean/std: {agg.get('precision_mean', 0.0):.6f} / {agg.get('precision_std', 0.0):.6f}"
        )
        details_lines.append(
            f"  recall mean/std: {agg.get('recall_mean', 0.0):.6f} / {agg.get('recall_std', 0.0):.6f}"
        )
        details_lines.append(
            f"  TP mean/std: {agg.get('tp_mean', 0.0):.3f} / {agg.get('tp_std', 0.0):.3f}"
        )
        details_lines.append(
            f"  FP mean/std: {agg.get('fp_mean', 0.0):.3f} / {agg.get('fp_std', 0.0):.3f}"
        )
        details_lines.append(
            f"  FN mean/std: {agg.get('fn_mean', 0.0):.3f} / {agg.get('fn_std', 0.0):.3f}"
        )
        details_lines.append("")

    ranked_rows.sort(key=lambda x: x["f1_mean"], reverse=True)
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "MITOSIS CHECKPOINT COMPARISON",
        f"Generated at: {timestamp}",
        f"Checkpoint dir: {checkpoint_dir}",
        "",
        "Ranking:",
        format_ranking_rows(ranked_rows),
        "",
        "Details:",
        *details_lines,
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print("MITOSIS MODEL COMPARISON")
    print("=" * 60)
    print(format_ranking_rows(ranked_rows))
    print(f"\nReport saved to: {report_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for evaluation overrides."""
    parser = argparse.ArgumentParser(description="Evaluate Mitosis checkpoints across folds")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda or cpu")
    parser.add_argument("--n-folds", type=int, default=None, help="Override number of folds")
    parser.add_argument(
        "--disable-stain-norm",
        action="store_true",
        help="Disable stain normalization for evaluation",
    )
    parser.add_argument(
        "--ensemble-all-folds",
        action="store_true",
        help="Evaluate heatmap ensemble that averages all discovered fold checkpoints.",
    )
    parser.add_argument(
        "--ensemble-checkpoints",
        type=str,
        default=None,
        help=(
            "Comma-separated checkpoint paths for custom heatmap ensemble. "
            "When provided, this ensemble is evaluated across all folds."
        ),
    )
    return parser


def main_cli(args: argparse.Namespace) -> None:
    """CLI wrapper for configurable evaluation runs."""
    cfg = get_default_config()
    if args.device is not None:
        cfg.device = args.device
    if args.n_folds is not None:
        cfg.data.n_folds = args.n_folds
    if args.disable_stain_norm:
        cfg.stain.enabled = False
    explicit_ensemble = None
    if args.ensemble_checkpoints:
        explicit_ensemble = parse_checkpoint_paths(checkpoints=args.ensemble_checkpoints)

    main(
        cfg,
        ensemble_all_folds=args.ensemble_all_folds,
        explicit_ensemble_checkpoints=explicit_ensemble,
    )


if __name__ == "__main__":
    parser = build_arg_parser()
    main_cli(parser.parse_args())
