import torch
from model_components import TransformerModel
from config import ModelConfig, TrainingConfig

def load_model(checkpoint_path: str = "best_model.pt", device: str = "cpu", vocab_size: int = None) -> tuple[TransformerModel, ModelConfig, TrainingConfig]:
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