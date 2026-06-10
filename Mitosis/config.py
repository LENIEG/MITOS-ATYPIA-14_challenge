"""
Hyperparameter configuration for mitosis detection.

Mirrors Atypia config structure to keep task pipelines consistent.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data loading and preprocessing settings."""

    extract_root: Path = Path(__file__).parent.parent / "data" / "extracted"
    norm_dir: Path = Path(__file__).parent.parent / "data" / "norms"

    magnification: str = "x40"
    batch_size: int = 4
    num_workers: int = 4

    n_folds: int = 5


@dataclass
class ModelConfig:
    """Model architecture settings."""

    backbone: str = "efficientnet_b3"
    pretrained: bool = True
    decoder_channels: int = 256
    input_size: int = 512
    output_stride: int = 4


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    num_epochs: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Heatmap target generation.
    gaussian_sigma: float = 1.8

    # Prediction decoding + validation matching.
    decode_threshold: float = 0.35
    decode_nms_kernel: int = 3
    decode_max_detections: int = 256
    match_radius_px: float = 8.0

    # BCE positive weighting (mitosis is sparse).
    pos_weight: float = 15.0

    # Early stopping on validation F1.
    early_stopping_patience: int = 8
    early_stopping_metric: str = "f1"

    max_grad_norm: float = 1.0


@dataclass
class StainNormConfig:
    """Stain normalization settings."""

    enabled: bool = True


@dataclass
class Config:
    """Master config combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    stain: StainNormConfig = field(default_factory=StainNormConfig)

    output_dir: Path = Path(__file__).parent.parent / "outputs" / "mitosis"
    checkpoint_dir: Path = None

    seed: int = 42
    device: str = "cuda"

    def __post_init__(self):
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.output_dir / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def get_default_config() -> Config:
    """Return default configuration."""
    return Config()
