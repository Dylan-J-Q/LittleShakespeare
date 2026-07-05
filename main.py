import logging
import os
import json
from dataclasses import asdict
from preprocessing import BPETokenizer, ShakespeareDataset
from model_components import TransformerModel
from training import Trainer
from torch.utils.data import DataLoader
from config import ModelConfig, TrainingConfig, PreprocessingConfig
from utils import load_model

def get_next_model_id():
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    ids = []
    for f in os.listdir(models_dir):
        if f.isdigit():
            ids.append(int(f))
    
    if not ids:
        return 0
    return max(ids) + 1

def main():
    model_id = get_next_model_id()
    model_dir = f"models/{model_id}"
    os.makedirs(model_dir, exist_ok=True)

    model_config = ModelConfig()
    training_config = TrainingConfig()
    preprocessing_config = PreprocessingConfig()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(model_dir, "training.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    with open(preprocessing_config.data_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Split text into training, validation, and test sets
    train_size = int(len(raw_text) * 0.8)
    val_size = int(len(raw_text) * 0.1)
    
    train_text = raw_text[:train_size]
    val_text = raw_text[train_size:train_size + val_size]
    test_text = raw_text[train_size + val_size:]

    tokenizer = BPETokenizer(raw_text, preprocessing_config, logger=logger)

    train_dataset = ShakespeareDataset(train_text, tokenizer, preprocessing_config, logger=logger)
    val_dataset = ShakespeareDataset(val_text, tokenizer, preprocessing_config, logger=logger)
    test_dataset = ShakespeareDataset(test_text, tokenizer, preprocessing_config, logger=logger)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        persistent_workers=training_config.persistent_workers if training_config.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        persistent_workers=training_config.persistent_workers if training_config.num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        persistent_workers=training_config.persistent_workers if training_config.num_workers > 0 else False
    )

    logger.info(f"Data Preprocessing Complete. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    model = TransformerModel(
        vocab_size=tokenizer.get_vocab_size(),
        config=model_config
    )

    # Save configuration
    config_data = {
        "preprocessing_config": asdict(preprocessing_config),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config)
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config_data, f, indent=4)

    trainer = Trainer(
        model_config, 
        training_config, 
        model, 
        model_dir=model_dir,
        logger=logger
    )
    trainer.train(train_loader, val_loader)
    trainer.plot_losses()
    
    best_model, _, _ = load_model(checkpoint_path=os.path.join(model_dir, "model.pt"), device=training_config.device)
    trainer.set_model(best_model)
    test_loss = trainer.evaluate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}")

    logger.info("Training Complete.")


if __name__ == "__main__":
    main()