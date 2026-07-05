import torch
import torch.nn as nn
import csv
import os
from dataclasses import dataclass
from torch.utils.data import DataLoader
from model_components import TransformerModel

@dataclass
class Config:
    d_model: int = 512
    num_heads: int = 8
    ff_hidden_dim: int = 2048
    
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    weight_decay: float = 1e-6
    device: str = "cpu"

class Trainer:
    def __init__(self, config: Config, model: TransformerModel, log_file: str = "training_log.csv"):
        self.config = config
        self.model = model
        self.device = config.device
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimiser = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        self.train_losses = []
        self.val_losses = []
        self.log_file = log_file

        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss"])

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        # Initial evaluation
        print("Initial Evaluation:")
        self.evaluate(val_loader)
        self.val_losses.append(self.evaluate_loss(val_loader))

        for epoch in range(self.config.epochs):
            self.model.train()
            running_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                self.optimiser.zero_grad()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
                
                loss.backward()
                self.optimiser.step()
                
                running_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            
            avg_train_loss = running_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            print(f"--- Epoch {epoch + 1}/{self.config.epochs} Complete. Average Train Loss: {avg_train_loss:.4f} ---")
            
            # Evaluation after each epoch
            print(f"Epoch {epoch + 1} Evaluation:")
            avg_val_loss = self.evaluate_loss(val_loader)
            self.val_losses.append(avg_val_loss)
            print(f"Average Val Loss: {avg_val_loss:.4f}")
            
            # Log to CSV
            with open(self.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, f"{avg_train_loss:.4f}", f"{avg_val_loss:.4f}"])

    def evaluate(self, val_loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        print(f"Evaluation Complete. Average Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate_loss(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
                total_loss += loss.item()
        return total_loss / len(val_loader)
