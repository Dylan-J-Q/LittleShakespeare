"""End-to-end training pipeline.

The former body of ``main.py`` as a callable ``run_training(...)`` — data split,
tokenizer, loaders, model, Trainer wiring, and final test evaluation — so it can
be driven from a thin CLI (``scripts/train.py``) or exercised in isolation.
"""
import json
import logging
import subprocess
from dataclasses import asdict
from datetime import datetime
from logging import Logger
from typing import Optional

from torch.utils.data import DataLoader

from little_shakespeare.config import ModelConfig, TrainingConfig, PreprocessingConfig
from little_shakespeare.data.tokenizer import BPETokenizer
from little_shakespeare.data.dataset import ShakespeareDataset
from little_shakespeare.data.splits import split_text
from little_shakespeare.eval.perplexity import accumulate_nll, metrics_report
from little_shakespeare.eval.leaderboard import build_leaderboard, write_csv as write_leaderboard_csv, write_markdown as write_leaderboard_markdown
from little_shakespeare.model.transformer import TransformerModel
from little_shakespeare.training.trainer import Trainer
from little_shakespeare.training import reporting
from little_shakespeare.checkpoint import load_model
from little_shakespeare.run_dir import (
    get_next_model_id,
    model_dir as model_dir_for,
    BENCHMARKS_ROOT,
    CONFIG_FILENAME,
    METRICS_FILENAME,
    LOG_FILENAME,
    CHECKPOINT_FILENAME,
)


def _current_git_commit() -> Optional[str]:
    """Short commit hash for metrics.json's provenance, or None outside a repo /
    if git isn't on PATH — never worth failing a training run over."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


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
                 logger: Optional[Logger] = None,
                 note: Optional[str] = None) -> int:
    """Run a full training pass and return the new run's model id.

    If ``logger`` is None, default file+stream logging is configured into the
    new run directory (preserving the original ``main.py`` behavior).

    ``note`` is a freeform hypothesis string ("doubled batch size to test VRAM
    headroom") written into metrics.json — writing it down before training
    starts is the point, not documentation after the fact.
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
        "training_config": asdict(training_config),
        "vocab_size": tokenizer.get_vocab_size(),
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

    # Full reports (not just Trainer.evaluate()'s loss/perplexity/bpc) on the
    # BEST checkpoint specifically — training may run patience epochs past
    # the best one before stopping, so the last logged CSV row isn't it.
    val_stats = accumulate_nll(trainer.model, val_loader, tokenizer, trainer.device)
    test_stats = accumulate_nll(trainer.model, test_loader, tokenizer, trainer.device)
    val_report = metrics_report(val_stats)
    test_report = metrics_report(test_stats)
    logger.info(
        f"Val  | Loss: {val_report['loss']:.4f} | Perplexity: {val_report['perplexity']:.2f} | bpc: {val_report['bpc']:.4f}"
    )
    logger.info(
        f"Test | Loss: {test_report['loss']:.4f} | Perplexity: {test_report['perplexity']:.2f} | bpc: {test_report['bpc']:.4f}"
    )

    metrics_data = {
        "note": note,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _current_git_commit(),
        "final_epoch": trainer.final_epoch,
        "early_stopped": trainer.early_stopped,
        "best_val_loss": trainer.best_val_loss,
        "total_train_time_seconds": trainer.total_train_time,
        "val": val_report,
        "test": test_report,
    }
    with open(model_dir / METRICS_FILENAME, "w") as f:
        json.dump(metrics_data, f, indent=4)

    # Cheap (no model loading) — keep benchmarks/leaderboard.* current on every
    # run rather than requiring a separate manual step someone has to remember.
    leaderboard_rows = build_leaderboard()
    BENCHMARKS_ROOT.mkdir(exist_ok=True)
    write_leaderboard_markdown(leaderboard_rows, BENCHMARKS_ROOT / "leaderboard.md")
    write_leaderboard_csv(leaderboard_rows, BENCHMARKS_ROOT / "leaderboard.csv")
    logger.info(f"Leaderboard updated: benchmarks/leaderboard.md ({len(leaderboard_rows)} runs)")

    logger.info("Training Complete.")
    return model_id
