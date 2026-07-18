from dataclasses import dataclass
import torch

@dataclass
class PreprocessingConfig:
    data_path: str = "data/LittleShakespeare.txt"
    block_size: int = 512
    num_merges: int = 256
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    # test_fraction is implied: 1 - train_fraction - val_fraction

@dataclass
class ModelConfig:
    d_model: int = 512
    num_heads: int = 16
    ff_hidden_dim: int = 2048
    num_layers: int = 6
    dropout_rate: float = 0.1
    max_pos_encoding_len: int = 5000

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads}) "
            "— attention splits d_model evenly across heads."
        )

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 300
    weight_decay: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    patience: int = 5
    mixed_precision: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True

@dataclass
class GenerationConfig:
    max_length: int = 1000
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.85
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def validate_configs(model_config: ModelConfig, preprocessing_config: PreprocessingConfig) -> None:
    """Cross-config invariant neither dataclass can check alone."""
    assert preprocessing_config.block_size <= model_config.max_pos_encoding_len, (
        f"block_size ({preprocessing_config.block_size}) must not exceed "
        f"max_pos_encoding_len ({model_config.max_pos_encoding_len}) — positions beyond "
        "max_pos_encoding_len have no positional encoding to add."
    )