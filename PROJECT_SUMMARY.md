# MITOS-ATYPIA-14 Project Summary

## 1. Repository Pipeline Structure

### Atypia Pipeline
- Configuration: Atypia/config.py
- Model: Atypia/models.py (EfficientNet-B3 + ordinal head)
- Losses: Atypia/losses.py (CORN / weighted CE)
- Metrics: Atypia/metrics.py (challenge score, accuracy, balanced accuracy)
- Training entry point: Atypia/train.py
- Fold checkpoint merge and fine-tune: Atypia/fold_merge.py
- Inference entry point: Atypia/infer.py
- Model evaluation/comparison: Atypia/evaluate_models.py

### Mitosis Pipeline
- Configuration: Mitosis/config.py
- Model: Mitosis/models.py (EfficientNet-B3 backbone + heatmap decoder)
- Heatmap utilities: Mitosis/heatmap.py
- Loss: Mitosis/losses.py (weighted BCE on heatmap)
- Metrics: Mitosis/metrics.py (TP/FP/FN, precision, recall, F1)
- Training entry point: Mitosis/train.py
- Inference entry point: Mitosis/infer.py
- Evaluation/comparison: Mitosis/evaluate_models.py
- Heatmap ensembling helpers: Mitosis/ensemble.py
- Prediction vs GT visualization: Mitosis/visualize_inference.py

### Shared Components
- Common dataset loaders and annotation readers: CommonRoutines/dataset.py
- Split logic (block-wise k-fold): CommonRoutines/splits.py
- Augmentation pipelines: CommonRoutines/augmentation.py
- Stain normalization: CommonRoutines/stain_norm.py

## 2. Configuration Summary

### Atypia Defaults
- Input magnification: x20
- K-folds: 5
- Batch size: 12
- Epochs: 50
- Learning rate: 1e-4
- Loss: ordinal (CORN)
- Early stopping metric: challenge_score
- Output location: outputs/atypia

### Mitosis Defaults
- Input magnification: x40
- K-folds: 5
- Batch size: 4
- Epochs: 40
- Learning rate: 1e-4
- Heatmap sigma: 1.8
- Decode threshold: 0.35
- NMS kernel: 3
- Match radius: 8.0 px
- Pos weight: 15.0
- Early stopping metric: f1
- Output location: outputs/mitosis

## 3. Current Results (Short)

### Atypia (from outputs/atypia/model_comparison.txt)
- Best single fold: fold1
- Challenge score: 0.7143
- Accuracy: 0.7286
- Best merged model variant: merged_finetuned
- 5-fold challenge mean: 0.7058 +- 0.1860
- 5-fold balanced accuracy mean: 0.8419

### Mitosis (from outputs/mitosis/model_comparison.txt)
- Best single fold: fold4
- F1: 0.1538
- Precision: 0.1082
- Recall: 0.2662
- 5-fold ensemble_all_folds:
- F1 mean: 0.1366 +- 0.0206
- Precision mean: 0.0935 +- 0.0171
- Recall mean: 0.2617 +- 0.0407

## 4. Conclusion

- The project has two complete and runnable pipelines: Atypia and Mitosis.
- Atypia performance is moderate-to-strong relative to this dataset size and currently the more mature pipeline.
- Mitosis pipeline is operational end-to-end (training, evaluation, ensembling, inference, and visualization), but absolute detection performance remains low and not yet at target quality for robust practical use.
- Heatmap ensembling improved stability and cross-fold robustness, but did not fully solve high false-positive behavior.

## 5. Future Needed Work

1. Improve Mitosis precision via decode/post-processing tuning (threshold, NMS behavior, duplicate suppression, confidence calibration).
2. Run structured hyperparameter sweeps for Mitosis (sigma, pos_weight, LR, scheduler settings) with fixed evaluation protocol.
3. Add stronger hard-negative handling for mitosis-like nuclei and scanner-specific error analysis.
4. Evaluate optional two-stage design (candidate proposal + classifier refinement) to reduce false positives.
5. Consider fold-ensemble weighting instead of uniform averaging for Mitosis.
6. Add experiment tracking table for reproducibility (config hash, commit hash, metrics).
7. Re-run final reporting after optimization and document selected production/default inference settings.
