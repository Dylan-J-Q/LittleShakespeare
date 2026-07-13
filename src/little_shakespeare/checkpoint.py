"""Checkpoint save/load.

Merges the former ``Trainer.save_checkpoint`` and ``utils.load_model`` into one
place. ``save_checkpoint`` stores the configs as plain dicts (via ``asdict``)
rather than live dataclass instances, so a checkpoint no longer pickles a
reference to a specific module path.
"""
import sys
from dataclasses import asdict
from typing import Optional, Tuple

import torch

from little_shakespeare import config as _config_module
from little_shakespeare.config import ModelConfig, TrainingConfig
from little_shakespeare.model.transformer import TransformerModel

# Backward-compat: checkpoints saved before the src/ migration pickled live
# ModelConfig/TrainingConfig instances under the top-level module name "config".
# Alias it so torch.load (weights_only=False) can still unpickle those two runs.
sys.modules.setdefault("config", _config_module)


def save_checkpoint(path, model: TransformerModel, model_config: ModelConfig,
                    training_config: TrainingConfig, epoch: int, val_loss: float) -> None:
    """Save model weights plus config snapshots and metadata to ``path``."""
    checkpoint = {
        'epoch': epoch,
        'val_loss': val_loss,
        'config': {
            'model_config': asdict(model_config),
            'training_config': asdict(training_config),
        },
        'model_state_dict': model.state_dict(),
    }
    torch.save(checkpoint, path)


def load_model(checkpoint_path: str = "best_model.pt", device: str = "cpu",
               vocab_size: Optional[int] = None) -> Tuple[TransformerModel, ModelConfig, TrainingConfig]:
    """
    Loads the best model from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to load the model onto.
        vocab_size: The vocabulary size. If None, it will try to infer it from the model state dict.

    Returns:
        A tuple of (TransformerModel, ModelConfig, TrainingConfig).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint['config']

    if isinstance(config_dict['model_config'], dict):
        model_config = ModelConfig(**config_dict['model_config'])
    else:
        model_config = config_dict['model_config']

    if isinstance(config_dict['training_config'], dict):
        training_config = TrainingConfig(**config_dict['training_config'])
    else:
        training_config = config_dict['training_config']

    if vocab_size is None:
        state_dict = checkpoint['model_state_dict']
        for key, value in state_dict.items():
            if "output_layer.weight" in key:
                vocab_size = value.shape[0]
                break

    # We need to determine the number of layers from the state dict
    num_layers = 0
    for key in checkpoint['model_state_dict'].keys():
        if "transformer_blocks." in key:
            parts = key.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                num_layers = max(num_layers, int(parts[1]) + 1)

    if num_layers == 0:
        num_layers = model_config.num_layers
    else:
        model_config.num_layers = num_layers

    model = TransformerModel(
        vocab_size=vocab_size,
        config=model_config
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model, model_config, training_config
