import logging
import time
from pathlib import Path
from typing import List
from logging import Logger

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from little_shakespeare.config import TrainingConfig, ModelConfig
from little_shakespeare.model.transformer import TransformerModel
from little_shakespeare import checkpoint
from little_shakespeare.training import reporting
from little_shakespeare.run_dir import CHECKPOINT_FILENAME, CSV_LOG_FILENAME

class Trainer:
    """
    A trainer class for training and evaluating a Transformer model.
    """
    def __init__(self,
                  model_config: ModelConfig,
                  training_config: TrainingConfig,
                  model: TransformerModel,
                  model_dir: str,
                  logger: Logger = logging.getLogger(__name__)):
        """
        Initialize the Trainer.

        Args:
            model_config: Configuration for the model.
            training_config: Configuration for the training process.
            model: The Transformer model to train.
            model_dir: Directory where logs, plots, and models will be saved.
            logger: Logger for training logs.
        """
        self.model_config = model_config
        self.training_config = training_config
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

    def _run_epoch(self,
                    loader: DataLoader,
                    mode: str = 'train') -> float:
        """
        Internal method to run a single epoch of training or evaluation.

        Args:
            loader: The DataLoader to iterate over.
            mode: The mode to run in ('train' or 'eval').

        Returns:
            The average loss for the epoch.
        """
        if mode == 'train':
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
        else:
            self.model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for inputs, targets in loader:
                    loss = self._run_batch(inputs, targets)
                    total_loss += loss.item()
            return total_loss / len(loader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Train the model for a specified number of epochs with early stopping.

        Args:
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
        """
        self.logger.info("Starting Initial Evaluation...")
        initial_val_loss = self.evaluate(val_loader)
        self.val_losses.append(initial_val_loss)

        total_start_time = time.perf_counter()

        for epoch in range(self.training_config.epochs):
            epoch_start_time = time.perf_counter()
            avg_train_loss = self._run_epoch(train_loader, mode='train')
            self.train_losses.append(avg_train_loss)

            epoch_duration = time.perf_counter() - epoch_start_time
            self.logger.info(f"--- Epoch {epoch + 1}/{self.training_config.epochs} Complete. Average Train Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f}s ---")

            self.logger.info(f"Epoch {epoch + 1} Evaluation:")
            avg_val_loss = self.evaluate(val_loader)
            self.val_losses.append(avg_val_loss)
            self.logger.info(f"Average Val Loss: {avg_val_loss:.4f}")

            # Save best model
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_checkpoint(epoch + 1, avg_val_loss)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                self.logger.info(f"No improvement in validation loss for {self.patience_counter} epoch(s).")

            # Log to CSV
            reporting.append_csv_row(self.log_file, epoch + 1, avg_train_loss, avg_val_loss)

            if self.patience_counter >= self.patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        total_duration = time.perf_counter() - total_start_time
        self.logger.info(f"Total training time: {total_duration:.2f}s")

    def evaluate(self, loader: DataLoader) -> float:
        """
        Evaluate the model on a given loader.

        Args:
            loader: The DataLoader for the evaluation set.

        Returns:
            The average loss for the evaluation.
        """
        avg_loss = self._run_epoch(loader, mode='eval')
        self.logger.info(f"Evaluation Complete. Average Loss: {avg_loss:.4f}")
        return avg_loss

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
