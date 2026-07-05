import torch
import torch.nn as nn
import torch.cuda.amp as amp
import csv
import os
import logging
import matplotlib.pyplot as plt
import time
from typing import List
from torch.utils.data import DataLoader
from model_components import TransformerModel
from config import TrainingConfig, ModelConfig
from logging import Logger

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
        self.log_file = os.path.join(model_dir, "training_log.csv")
        self.model_dir = model_dir
        self.best_val_loss = float('inf')
        self.patience = self.training_config.patience
        self.patience_counter = 0
        self.logger = logger

        # Initialize GradScaler for mixed precision
        self.scaler = None
        if self.training_config.mixed_precision and self.device == "cuda":
            self.scaler = torch.amp.GradScaler()

        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            self._log_to_csv(0, 0.0, 0.0, header_only=True)

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

            if self.patience_counter >= self.patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                break
            
            # Log to CSV
            self._log_to_csv(epoch + 1, avg_train_loss, avg_val_loss)

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

    def _log_to_csv(self, epoch: int, train_loss: float, val_loss: float, header_only: bool = False):
        """
        Helper method to log training metrics to a CSV file.
        """
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if header_only:
                writer.writerow(["epoch", "train_loss", "val_loss"])
            else:
                writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}"])

    def save_checkpoint(self, epoch: int, val_loss: float):
        """
        Saves the model state dict along with metadata.

        Args:
            epoch: The current epoch number.
            val_loss: The validation loss at the current epoch.
        """
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'config': {
                'model_config': self.model_config,
                'training_config': self.training_config
            },
            'model_state_dict': self.model.state_dict(),
        }
        torch.save(checkpoint, os.path.join(self.model_dir, "model.pt"))
        self.logger.info(f"Checkpoint saved to '{os.path.join(self.model_dir, 'model.pt')}' at epoch {epoch} with val_loss {val_loss:.4f}")

    def plot_losses(self):
        """
        Plots the training and validation losses in linear and log scales.
        """
        num_epochs = len(self.train_losses)
        # self.val_losses contains an initial evaluation at index 0.
        # We want to plot the losses corresponding to the training loop.
        plot_val_losses = self.val_losses[-num_epochs:] if len(self.val_losses) > num_epochs else self.val_losses
        
        epochs = range(1, num_epochs + 1)

        for scaleType in ["linear", "log"]:
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, self.train_losses, label='Training Loss')
            plt.plot(epochs, plot_val_losses, label='Validation Loss')
            plt.yscale(scaleType)
            plt.title(f'Training and Validation Loss ({scaleType} scale)')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.model_dir, f'loss_curves_{scaleType}.png'))
            plt.close()    
        
        self.logger.info(f"Loss curves saved as 'loss_curves_linear.png' and 'loss_curves_log.png' for {num_epochs} epochs.")