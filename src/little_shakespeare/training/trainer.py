import logging
import time
from pathlib import Path
from typing import List, NamedTuple
from logging import Logger

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from little_shakespeare.config import TrainingConfig, ModelConfig
from little_shakespeare.data.tokenizer import BaseTokenizer
from little_shakespeare.model.transformer import TransformerModel
from little_shakespeare import checkpoint
from little_shakespeare.eval.perplexity import (
    accumulate_nll,
    perplexity as compute_perplexity,
    bits_per_char as compute_bpc,
)
from little_shakespeare.training import reporting
from little_shakespeare.run_dir import CHECKPOINT_FILENAME, CSV_LOG_FILENAME


class EvalMetrics(NamedTuple):
    """Result of one Trainer.evaluate() pass — loss plus its two derived metrics."""
    loss: float
    perplexity: float
    bpc: float


class Trainer:
    """
    A trainer class for training and evaluating a Transformer model.
    """
    def __init__(self,
                  model_config: ModelConfig,
                  training_config: TrainingConfig,
                  model: TransformerModel,
                  model_dir: str,
                  tokenizer: BaseTokenizer,
                  logger: Logger = logging.getLogger(__name__)):
        """
        Initialize the Trainer.

        Args:
            model_config: Configuration for the model.
            training_config: Configuration for the training process.
            model: The Transformer model to train.
            model_dir: Directory where logs, plots, and models will be saved.
            tokenizer: Tokenizer used to decode target tokens for bits-per-character.
            logger: Logger for training logs.
        """
        self.model_config = model_config
        self.training_config = training_config
        self.tokenizer = tokenizer
        self.device = training_config.device
        self.criterion = nn.CrossEntropyLoss()
        self.set_model(model)
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.model_dir = Path(model_dir)
        self.log_file = self.model_dir / CSV_LOG_FILENAME
        self.best_val_loss = float('inf')
        self.patience = self.training_config.patience
        self.patience_counter = 0
        self.logger = logger

        # Initialize GradScaler for mixed precision
        self.scaler = None
        if self.training_config.mixed_precision and self.device == "cuda":
            self.scaler = torch.amp.GradScaler()

        # Initialize CSV file with headers if it doesn't exist
        reporting.init_csv_log(self.log_file)

    def set_model(self, model: TransformerModel):
        """
        Sets the model for the Trainer.

        Args:
            model: The Transformer model to set.
        """
        self.model = model
        self.model.to(self.device)
        # Re-initialize optimizer if the model changed
        self.optimiser = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )

    def _run_batch(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        with torch.amp.autocast(enabled=self.training_config.mixed_precision and self.device == "cuda", device_type=self.device):
            outputs = self.model(inputs)
            loss = self.criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
        return loss

    def _run_epoch(self, loader: DataLoader) -> float:
        """
        Internal method to run a single training epoch.

        Note: this averages per-batch means (mean-of-means), which is only a
        rough training-progress indicator — fine here since nothing derives
        perplexity/bpc from it. Evaluation uses accumulate_nll's token-weighted
        sum instead, since that precision matters once loss is exponentiated.

        Args:
            loader: The DataLoader to iterate over.

        Returns:
            The average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(loader):
            self.optimiser.zero_grad(set_to_none=True)
            loss = self._run_batch(inputs, targets)

            if self.training_config.mixed_precision and self.device == "cuda":
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimiser)
                self.scaler.update()
            else:
                loss.backward()
                self.optimiser.step()

            running_loss += loss.item()
            if batch_idx % 10 == 0:
                self.logger.debug(f"Batch {batch_idx} | Loss: {loss.item():.4f}")
        return running_loss / len(loader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Train the model for a specified number of epochs with early stopping.

        Args:
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
        """
        self.logger.info("Starting Initial Evaluation...")
        initial_metrics = self.evaluate(val_loader)
        self.val_losses.append(initial_metrics.loss)

        total_start_time = time.perf_counter()

        for epoch in range(self.training_config.epochs):
            epoch_start_time = time.perf_counter()
            avg_train_loss = self._run_epoch(train_loader)
            self.train_losses.append(avg_train_loss)

            epoch_duration = time.perf_counter() - epoch_start_time
            self.logger.info(f"--- Epoch {epoch + 1}/{self.training_config.epochs} Complete. Average Train Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f}s ---")

            self.logger.info(f"Epoch {epoch + 1} Evaluation:")
            val_metrics = self.evaluate(val_loader)
            self.val_losses.append(val_metrics.loss)
            self.logger.info(f"Average Val Loss: {val_metrics.loss:.4f}")

            # Save best model
            if val_metrics.loss < self.best_val_loss:
                self.best_val_loss = val_metrics.loss
                self.save_checkpoint(epoch + 1, val_metrics.loss)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                self.logger.info(f"No improvement in validation loss for {self.patience_counter} epoch(s).")

            # Log to CSV
            reporting.append_csv_row(self.log_file, epoch + 1, avg_train_loss,
                                      val_metrics.loss, val_metrics.perplexity, val_metrics.bpc)

            if self.patience_counter >= self.patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        total_duration = time.perf_counter() - total_start_time
        self.logger.info(f"Total training time: {total_duration:.2f}s")

    def evaluate(self, loader: DataLoader) -> EvalMetrics:
        """
        Evaluate the model on a given loader.

        Uses a single token-weighted pass (accumulate_nll) rather than
        averaging per-batch means, so loss/perplexity/bpc all derive from the
        same correctly-scoped totals.

        Args:
            loader: The DataLoader for the evaluation set.

        Returns:
            EvalMetrics(loss, perplexity, bpc).
        """
        stats = accumulate_nll(self.model, loader, self.tokenizer, self.device)
        metrics = EvalMetrics(
            loss=stats.total_nll / stats.total_tokens,
            perplexity=compute_perplexity(stats),
            bpc=compute_bpc(stats),
        )
        self.logger.info(
            f"Evaluation Complete. Loss: {metrics.loss:.4f} | "
            f"Perplexity: {metrics.perplexity:.2f} | bpc: {metrics.bpc:.4f}"
        )
        return metrics

    def save_checkpoint(self, epoch: int, val_loss: float):
        """
        Saves the model state dict along with metadata.

        Args:
            epoch: The current epoch number.
            val_loss: The validation loss at the current epoch.
        """
        path = self.model_dir / CHECKPOINT_FILENAME
        checkpoint.save_checkpoint(path, self.model, self.model_config, self.training_config, epoch, val_loss)
        self.logger.info(f"Checkpoint saved to '{path}' at epoch {epoch} with val_loss {val_loss:.4f}")
