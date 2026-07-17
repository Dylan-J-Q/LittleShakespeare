"""End-to-end training pipeline.

The former body of ``main.py`` as a callable ``run_training(...)`` — data split,
tokenizer, loaders, model, Trainer wiring, and final test evaluation — so it can
be driven from a thin CLI (``scripts/train.py``) or exercised in isolation.
"""
import json
import logging
from dataclasses import asdict
from logging import Logger
from typing import Optional

from torch.utils.data import DataLoader

from little_shakespeare.config import ModelConfig, TrainingConfig, PreprocessingConfig
from little_shakespeare.data.tokenizer import BPETokenizer
from little_shakespeare.data.dataset import ShakespeareDataset
from little_shakespeare.data.splits import split_text
from little_shakespeare.model.transformer import TransformerModel
from little_shakespeare.training.trainer import Trainer
from little_shakespeare.training import reporting
from little_shakespeare.checkpoint import load_model
from little_shakespeare.run_dir import (
    get_next_model_id,
    model_dir as model_dir_for,
    CONFIG_FILENAME,
    LOG_FILENAME,
    CHECKPOINT_FILENAME,
)


def _build_loader(dataset, training_config: TrainingConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=shuffle,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        persistent_workers=training_config.persistent_workers if training_config.num_workers > 0 else False,
    )


def run_training(model_config: Optional[ModelConfig] = None,
                 training_config: Optional[TrainingConfig] = None,
                 preprocessing_config: Optional[PreprocessingConfig] = None,
                 logger: Optional[Logger] = None) -> int:
    """Run a full training pass and return the new run's model id.

    If ``logger`` is None, default file+stream logging is configured into the
    new run directory (preserving the original ``main.py`` behavior).
    """
    model_config = model_config or ModelConfig()
    training_config = training_config or TrainingConfig()
    preprocessing_config = preprocessing_config or PreprocessingConfig()

    model_id = get_next_model_id()
    model_dir = model_dir_for(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)

    if logger is None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(model_dir / LOG_FILENAME),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)

    with open(preprocessing_config.data_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    train_text, val_text, test_text = split_text(raw_text, preprocessing_config)

    tokenizer = BPETokenizer(raw_text, preprocessing_config, logger=logger)

    train_dataset = ShakespeareDataset(train_text, tokenizer, preprocessing_config, logger=logger)
    val_dataset = ShakespeareDataset(val_text, tokenizer, preprocessing_config, logger=logger)
    test_dataset = ShakespeareDataset(test_text, tokenizer, preprocessing_config, logger=logger)

    train_loader = _build_loader(train_dataset, training_config, shuffle=True)
    val_loader = _build_loader(val_dataset, training_config, shuffle=False)
    test_loader = _build_loader(test_dataset, training_config, shuffle=False)

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
    with open(model_dir / CONFIG_FILENAME, "w") as f:
        json.dump(config_data, f, indent=4)

    trainer = Trainer(
        model_config,
        training_config,
        model,
        model_dir=model_dir,
        tokenizer=tokenizer,
        logger=logger
    )
    trainer.train(train_loader, val_loader)
    reporting.plot_losses(trainer.train_losses, trainer.val_losses, model_dir, logger=logger)

    best_model, _, _ = load_model(checkpoint_path=str(model_dir / CHECKPOINT_FILENAME), device=training_config.device)
    trainer.set_model(best_model)
    test_metrics = trainer.evaluate(test_loader)
    logger.info(
        f"Test Loss: {test_metrics.loss:.4f} | "
        f"Test Perplexity: {test_metrics.perplexity:.2f} | Test bpc: {test_metrics.bpc:.4f}"
    )

    logger.info("Training Complete.")
    return model_id
