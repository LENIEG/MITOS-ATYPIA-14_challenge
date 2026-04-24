# Atypia Training Pipeline

Complete training system for nuclear atypia classification task.

## File Structure

| File | Purpose |
|------|---------|
| `config.py` | Centralized hyperparameter configuration |
| `models.py` | EfficientNet-B3 backbone + ordinal regression head |
| `losses.py` | CORN loss (ordinal) + weighted cross-entropy |
| `metrics.py` | Challenge-specific scoring (with -1 penalty for off-by-2) |
| `train.py` | Main training loop with k-fold cross-validation |
| `__init__.py` | Package exports |

## Quick Start

### 1. Setup stain normalizers (one-time)

```python
from CommonRoutines import fit_scanner_normalizers, load_image_rgb
from pathlib import Path

# Load reference images
aperio_ref = load_image_rgb("data/extracted/training/aperio/A03/frames/x20/A03_00A.tiff")
hama_ref   = load_image_rgb("data/extracted/training/hamamatsu/H03/frames/x20/H03_00A.tiff")

# Fit and save
fit_scanner_normalizers(aperio_ref, hama_ref, save_dir="data/norms")
```

### 2. Train model (from terminal)

```bash
cd c:\MLProjectChallenge\MITOS-ATYPIA-14_challenge
python -m Atypia.train
```

Or from Python:
```python
from Atypia import main, get_default_config

config = get_default_config()
# Modify config as needed:
# config.model.backbone = "efficientnet_b4"  # larger model
# config.training.num_epochs = 100
# config.training.learning_rate = 5e-5

main(config)
```

## Configuration

Edit `Atypia/config.py` to modify:
- **Model**: backbone, input size, dropout
- **Training**: epochs, LR, loss type (CORN vs weighted CE)
- **Data**: batch size, k-fold splits, stain normalization
- **Augmentation**: CutMix/MixUp alpha values

## Output

Training saves:
- Checkpoints: `outputs/atypia/checkpoints/fold{i}_best.pt`
- Summary: `outputs/atypia/training_summary.txt`

## Loss Functions

### CORN (Cumulative Output Regression Network)
- Ordinal-aware: enforces monotonic cumulative probabilities
- Recommended for small datasets
- Outputs K-1 = 2 logits for 3 classes

### Weighted CE
- Standard cross-entropy with class weights
- Can add label smoothing
- Simpler, but ignores ordinal structure

## Metrics

Challenge score per sample:
- **+1** if prediction == target (correct)
- **0** if |prediction - target| == 1 (off-by-1)
- **-1** if |prediction - target| == 2 (off-by-2, heavily penalized)

## Key Design Decisions

1. **EfficientNet-B3**: Fits 8 GB VRAM at batch 12, good accuracy/parameter tradeoff
2. **Ordinal head**: 2 logits for 3 classes (monotonic cumulative probs)
3. **K-fold CV**: More reliable evaluation with limited data
4. **Block stratification**: A03/H03 always in same fold (no tissue leakage)
5. **Stain normalization**: Scanner-specific Macenko normalization
6. **Early stopping**: On challenge_score metric
7. **Freeze-then-unfreeze**: Train head first 5 epochs, then all layers
