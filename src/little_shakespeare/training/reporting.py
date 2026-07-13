"""Reporting side of training: CSV metric logging and loss-curve plots.

Kept separate from the training loop so the loop itself has no matplotlib/CSV
concerns and can be exercised (or benchmarked) without producing artifacts.
"""
import csv
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
from logging import Logger

CSV_HEADER = ["epoch", "train_loss", "val_loss"]


def init_csv_log(log_file) -> None:
    """Write the CSV header once, only if the file does not already exist."""
    log_file = Path(log_file)
    if not log_file.exists():
        with open(log_file, mode='w', newline='') as f:
            csv.writer(f).writerow(CSV_HEADER)


def append_csv_row(log_file, epoch: int, train_loss: float, val_loss: float) -> None:
    with open(log_file, mode='a', newline='') as f:
        csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}"])


def plot_losses(train_losses: Sequence[float], val_losses: Sequence[float],
                model_dir, logger: Optional[Logger] = None) -> None:
    """
    Plots the training and validation losses in linear and log scales.
    """
    model_dir = Path(model_dir)
    num_epochs = len(train_losses)
    # val_losses contains an initial evaluation at index 0; align to the training loop.
    plot_val_losses = val_losses[-num_epochs:] if len(val_losses) > num_epochs else val_losses

    epochs = range(1, num_epochs + 1)

    for scaleType in ["linear", "log"]:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, plot_val_losses, label='Validation Loss')
        plt.yscale(scaleType)
        plt.title(f'Training and Validation Loss ({scaleType} scale)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(model_dir / f'loss_curves_{scaleType}.png')
        plt.close()

    if logger:
        logger.info(f"Loss curves saved as 'loss_curves_linear.png' and 'loss_curves_log.png' for {num_epochs} epochs.")
