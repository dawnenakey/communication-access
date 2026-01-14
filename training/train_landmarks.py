#!/usr/bin/env python3
"""
Landmark-Based SLR Training Script
===================================
Train a lightweight model on extracted hand/pose landmarks.

This approach is much more efficient than training on raw video:
- ~500K parameters vs ~14M for 3D CNN
- Faster training (minutes vs hours)
- Works better with limited data
- More robust to lighting/background variations

Architecture:
  Landmarks (T, 165) -> LSTM/Transformer -> Classification

Usage:
    python training/train_landmarks.py \
        --data ./landmark_data \
        --output ./models \
        --epochs 100

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class TrainingConfig:
    """Training configuration for landmark-based model."""
    # Data
    data_dir: str = "./landmark_data"
    output_dir: str = "./models"

    # Model architecture
    model_type: str = "lstm"  # lstm, transformer, tcn
    input_dim: int = 165  # Will be updated from data
    hidden_size: int = 256
    num_layers: int = 2
    num_classes: int = 100  # Will be updated from data
    dropout: float = 0.3
    bidirectional: bool = True

    # Training
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    max_lr: float = 5e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    # Sequence handling
    max_seq_length: int = 64
    pad_value: float = 0.0

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# MODELS
# ==============================================================================

class LandmarkLSTM(nn.Module):
    """
    LSTM-based model for landmark sequence classification.
    Much lighter than 3D CNN (~500K params vs ~14M).
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )

        # Output dimension
        lstm_out_dim = config.hidden_size * (2 if config.bidirectional else 1)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes)
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, T, input_dim)
            lengths: Sequence lengths for masking (B,)

        Returns:
            logits: (B, num_classes)
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)  # (B, T, hidden)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, hidden*2)

        # Attention pooling
        attn_weights = self.attention(lstm_out)  # (B, T, 1)

        # Mask padding if lengths provided
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device).expand(batch_size, -1)
            mask = mask >= lengths.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1), float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (B, hidden*2)

        # Classify
        logits = self.classifier(context)

        return logits


class LandmarkTransformer(nn.Module):
    """
    Transformer-based model for landmark sequence classification.
    Better for capturing long-range dependencies.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_size)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.max_seq_length, config.hidden_size) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=8,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes)
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, T, input_dim)
            lengths: Sequence lengths for masking (B,)

        Returns:
            logits: (B, num_classes)
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)  # (B, T, hidden)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, hidden)

        # Create attention mask if lengths provided
        mask = None
        if lengths is not None:
            # Account for CLS token
            mask = torch.arange(seq_len + 1, device=x.device).expand(batch_size, -1)
            mask = mask > lengths.unsqueeze(1)  # True = masked

        # Transformer
        x = self.transformer(x, src_key_padding_mask=mask)

        # Use CLS token for classification
        cls_output = x[:, 0, :]

        # Classify
        logits = self.classifier(cls_output)

        return logits


# ==============================================================================
# DATASET
# ==============================================================================

class LandmarkDataset(Dataset):
    """Dataset for landmark sequences."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        max_seq_length: int = 64,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_seq_length = max_seq_length
        self.augment = augment and split == "train"

        # Load metadata
        metadata_path = self.data_dir / "landmarks_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.feature_dim = metadata["feature_dim"]
        self.num_classes = metadata["num_classes"]
        self.label_map = metadata["label_map"]

        # Filter samples by split
        self.samples = [s for s in metadata["samples"] if s["split"] == split]

        logger.info(f"Loaded {len(self.samples)} {split} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load landmarks
        landmarks_path = self.data_dir / sample["landmarks_path"]
        landmarks = np.load(landmarks_path)

        # Data augmentation
        if self.augment:
            landmarks = self._augment(landmarks)

        # Pad/truncate to max_seq_length
        seq_len = len(landmarks)
        if seq_len > self.max_seq_length:
            # Uniform sampling
            indices = np.linspace(0, seq_len - 1, self.max_seq_length, dtype=int)
            landmarks = landmarks[indices]
            seq_len = self.max_seq_length
        elif seq_len < self.max_seq_length:
            # Pad with zeros
            padding = np.zeros((self.max_seq_length - seq_len, landmarks.shape[1]), dtype=np.float32)
            landmarks = np.concatenate([landmarks, padding], axis=0)

        return {
            "landmarks": torch.from_numpy(landmarks).float(),
            "label": sample["label"],
            "length": seq_len,
            "gloss": sample["gloss"]
        }

    def _augment(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply data augmentation to landmarks."""
        # Random time scaling (speed variation)
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.8, 1.2)
            new_len = max(1, int(len(landmarks) * scale))
            indices = np.linspace(0, len(landmarks) - 1, new_len, dtype=int)
            landmarks = landmarks[indices]

        # Random noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, landmarks.shape).astype(np.float32)
            landmarks = landmarks + noise

        # Random horizontal flip (mirror)
        if np.random.random() < 0.5:
            # Flip x coordinates (swap left/right hands)
            landmarks = landmarks.copy()
            # Flip x coordinates (assuming x is at indices 0, 3, 6, ...)
            for i in range(0, landmarks.shape[1], 3):
                landmarks[:, i] = -landmarks[:, i]

        # Random temporal crop
        if np.random.random() < 0.3 and len(landmarks) > 10:
            start = np.random.randint(0, len(landmarks) // 4)
            end = len(landmarks) - np.random.randint(0, len(landmarks) // 4)
            landmarks = landmarks[start:end]

        return landmarks


# ==============================================================================
# TRAINING
# ==============================================================================

class Trainer:
    """Trainer for landmark-based model."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        logger.info(f"Using device: {self.device}")

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load datasets
        self.train_dataset = LandmarkDataset(
            config.data_dir, "train",
            max_seq_length=config.max_seq_length
        )
        self.val_dataset = LandmarkDataset(
            config.data_dir, "val",
            max_seq_length=config.max_seq_length,
            augment=False
        )

        # Update config from data
        config.input_dim = self.train_dataset.feature_dim
        config.num_classes = self.train_dataset.num_classes

        logger.info(f"Input dim: {config.input_dim}, Classes: {config.num_classes}")

        # Create model
        if config.model_type == "transformer":
            self.model = LandmarkTransformer(config).to(self.device)
        else:
            self.model = LandmarkLSTM(config).to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {num_params:,}")

        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.max_lr,
            epochs=config.epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=config.warmup_epochs / config.epochs,
            div_factor=config.max_lr / config.learning_rate,
            final_div_factor=100
        )

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Training state
        self.best_val_acc = 0
        self.training_history = []

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")

        for batch in pbar:
            landmarks = batch["landmarks"].to(self.device)
            labels = batch["label"].to(self.device)
            lengths = batch["length"].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(landmarks, lengths)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100. * correct / total:.1f}%"
            })

        return {
            "loss": total_loss / len(self.train_loader),
            "accuracy": 100. * correct / total
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            landmarks = batch["landmarks"].to(self.device)
            labels = batch["label"].to(self.device)
            lengths = batch["length"].to(self.device)

            logits = self.model(landmarks, lengths)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return {
            "loss": total_loss / len(self.val_loader),
            "accuracy": 100. * correct / total
        }

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "best_val_acc": self.best_val_acc,
            "label_map": self.train_dataset.label_map
        }

        path = self.output_dir / filename
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.output_dir / "best_landmark_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"  Saved best model (acc={self.best_val_acc:.1f}%)")

    def train(self):
        """Main training loop."""
        logger.info("=" * 60)
        logger.info("Starting landmark-based training")
        logger.info("=" * 60)
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")

        start_time = time.time()

        for epoch in range(self.config.epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Log
            logger.info(
                f"Epoch {epoch + 1}: "
                f"Train Loss={train_metrics['loss']:.4f}, "
                f"Train Acc={train_metrics['accuracy']:.1f}%, "
                f"Val Acc={val_metrics['accuracy']:.1f}%"
            )

            # Save history
            self.training_history.append({
                "epoch": epoch + 1,
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })

            # Save checkpoint
            is_best = val_metrics["accuracy"] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics["accuracy"]

            if is_best or (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"landmark_checkpoint_{epoch + 1}.pt", is_best=is_best)

        # Training complete
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Training complete in {elapsed / 60:.1f} minutes")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.1f}%")

        # Save final model
        self.save_checkpoint("landmark_model_final.pt")

        # Save training history
        with open(self.output_dir / "landmark_training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train landmark-based SLR model')

    parser.add_argument('--data', type=str, default='./landmark_data',
                       help='Path to landmark data')
    parser.add_argument('--output', type=str, default='./models',
                       help='Output directory')
    parser.add_argument('--model', type=str, default='lstm',
                       choices=['lstm', 'transformer'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=256,
                       help='Hidden size')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of layers')

    args = parser.parse_args()

    config = TrainingConfig(
        data_dir=args.data,
        output_dir=args.output,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    )

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
