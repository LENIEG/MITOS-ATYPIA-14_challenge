"""
Hyperparameter configuration for atypia training.

Centralized settings for model, training, and data pipeline.
Modify here to tune the training behavior.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data loading and preprocessing settings."""
    
    extract_root: Path = Path(__file__).parent.parent / "data" / "extracted"
    norm_dir: Path = Path(__file__).parent.parent / "data" / "norms"
    
    magnification: str = "x20"           # atypia uses x20 frames
    batch_size: int = 12
    num_workers: int = 4
    
    train_split_ratio: float = 0.8      # or use k-fold instead
    n_folds: int = 5
    use_kfold: bool = True              # if False, use fixed train/val split


@dataclass
class ModelConfig:
    """Model architecture settings."""
    
    backbone: str = "efficientnet_b3"   # pretrained from torchvision
    pretrained: bool = True
    dropout_rate: float = 0.3
    input_size: int = 512
    num_classes: int = 3                # atypia: 1, 2, 3 (0-indexed: 0, 1, 2)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5              # warmup before cosine annealing
    
    # Loss function
    loss_type: str = "ordinal"          # "ordinal" (CORN) or "weighted_ce"
    label_smoothing: float = 0.1
    
    # Class weights for imbalance (can be adjusted based on data)
    class_weights: list[float] = None   # if None, computed from data
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "challenge_score"  # maximize this
    
    # Gradient clipping
    max_grad_norm: float = 1.0


@dataclass
class AugmentationConfig:
    """Data augmentation settings."""
    
    enable_augmentation: bool = True
    cutmix_alpha: float = 0.0           # 0 = disabled; 0.2–1.0 if enabled
    mixup_alpha: float = 0.0            # 0 = disabled
    # (rest handled by albumentations in CommonRoutines.augmentation)


@dataclass
class StainNormConfig:
    """Stain normalization settings."""
    
    enabled: bool = True
    percentile: int = 99


# --- Consolidated Config Object ---

@dataclass
class Config:
    """Master config combining all sub-configs."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    stain: StainNormConfig = field(default_factory=StainNormConfig)
    
    # Paths
    output_dir: Path = Path(__file__).parent.parent / "outputs" / "atypia"
    checkpoint_dir: Path = None         # set to output_dir/checkpoints if None
    
    # Logging
    seed: int = 42
    device: str = "cuda"                # or "cpu"
    
    def __post_init__(self):
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.output_dir / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def get_default_config() -> Config:
    """Return default configuration."""
    return Config()
