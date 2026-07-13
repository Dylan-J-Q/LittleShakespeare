from dataclasses import dataclass
import torch

@dataclass
class PreprocessingConfig:
    data_path: str = "data/LittleShakespeare.txt"
    block_size: int = 512
    num_merges: int = 256

@dataclass
class ModelConfig:
    embedding_dim: int = 512
    d_model: int = 512
    num_heads: int = 16
    ff_hidden_dim: int = 2048
    num_layers: int = 6
    dropout_rate: float = 0.1
    max_pos_encoding_len: int = 5000

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