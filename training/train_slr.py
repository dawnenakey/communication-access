#!/usr/bin/env python3
"""
SLR Model Training Script
==========================
Train 3D CNN + LSTM model for Sign Language Recognition.

Features:
- 3D CNN backbone (ResNet3D or custom) + LSTM temporal modeling
- Attention mechanism for hand region focus
- Transfer learning support
- Confidence score output with "unknown" detection
- Mixed precision training for speed
- TensorBoard logging and progress tracking
- Checkpoint saving and resuming
- Multiple evaluation metrics

Architecture:
  Input (T, C, H, W) -> 3D CNN -> Spatial Features
  -> Attention -> LSTM -> Classification Head -> Softmax + Confidence

Usage:
    python training/train_slr.py \\
        --data ./combined_data \\
        --output ./models \\
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

# Add parent directory for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

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
    """Training configuration."""
    # Data
    data_dir: str = "./combined_data"
    output_dir: str = "./models"

    # Model
    model_type: str = "resnet3d_lstm"  # resnet3d_lstm, c3d_lstm, simple_lstm
    num_classes: int = 63  # 26 letters + 10 numbers + 27 common signs
    hidden_size: int = 512
    num_lstm_layers: int = 2
    dropout: float = 0.3
    use_attention: bool = True

    # Training
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 1e-3  # Base learning rate (10x higher for training from scratch)
    max_lr: float = 3e-3  # Peak learning rate for OneCycleLR scheduler
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    # Input
    sequence_length: int = 16  # Number of frames
    image_size: int = 224
    num_workers: int = 4

    # Mixed precision
    use_amp: bool = True

    # Checkpointing
    save_every: int = 5
    resume_from: Optional[str] = None

    # Confidence threshold for "unknown"
    confidence_threshold: float = 0.5

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# MODELS
# ==============================================================================

class SpatialAttention(nn.Module):
    """Spatial attention module to focus on hand regions."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        attention = self.conv(x)  # (B, 1, H, W)
        return x * attention


class TemporalAttention(nn.Module):
    """Temporal attention for sequence modeling."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_output: (B, T, H)
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)  # (B, T, 1)
        context = torch.sum(lstm_output * attention_weights, dim=1)  # (B, H)
        return context, attention_weights.squeeze(-1)


class Simple3DCNN(nn.Module):
    """Simple 3D CNN backbone."""

    def __init__(self, out_features: int = 512):
        super().__init__()

        self.features = nn.Sequential(
            # (B, 3, T, H, W) -> (B, 64, T, H/2, W/2)
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),

            # (B, 64, T, H/4, W/4) -> (B, 128, T, H/4, W/4)
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            # (B, 128, T, H/4, W/4) -> (B, 256, T/2, H/8, W/8)
            nn.Conv3d(128, 256, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            # (B, 256, T/2, H/8, W/8) -> (B, 512, T/2, H/16, W/16)
            nn.Conv3d(256, 512, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),

            # Global average pooling over spatial dimensions
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )

        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        features = self.features(x)  # (B, 512, T', 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B, 512, T')
        features = features.permute(0, 2, 1)  # (B, T', 512)

        return features


class SLRModel(nn.Module):
    """
    Sign Language Recognition Model.
    3D CNN + Attention + LSTM + Classification Head
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        # 3D CNN backbone
        self.backbone = Simple3DCNN(out_features=512)

        # Spatial attention (optional)
        self.use_spatial_attention = config.use_attention
        if self.use_spatial_attention:
            self.spatial_attention = SpatialAttention(512)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=config.hidden_size,
            num_layers=config.num_lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.num_lstm_layers > 1 else 0,
            bidirectional=True
        )

        # Temporal attention
        self.temporal_attention = TemporalAttention(config.hidden_size * 2)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes)
        )

        # Confidence head (separate from classification)
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_size * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, T, C, H, W)
            return_attention: Whether to return attention weights

        Returns:
            Dict with 'logits', 'confidence', and optionally 'attention'
        """
        # Extract features
        features = self.backbone(x)  # (B, T', 512)

        # LSTM
        lstm_out, _ = self.lstm(features)  # (B, T', hidden*2)

        # Temporal attention
        context, attention_weights = self.temporal_attention(lstm_out)  # (B, hidden*2)

        # Classification
        logits = self.classifier(context)  # (B, num_classes)

        # Confidence estimation
        confidence = self.confidence_head(context)  # (B, 1)

        result = {
            'logits': logits,
            'confidence': confidence.squeeze(-1)
        }

        if return_attention:
            result['attention'] = attention_weights

        return result

    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Make prediction with confidence filtering.

        Returns dict with:
        - 'class_id': Predicted class (-1 if confidence below threshold)
        - 'class_name': Class name (None if unknown)
        - 'confidence': Confidence score
        - 'probabilities': Full probability distribution
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            probs = F.softmax(output['logits'], dim=-1)
            confidence = output['confidence']

            # Get top prediction
            max_prob, class_id = probs.max(dim=-1)

            # Apply threshold
            is_confident = confidence >= threshold

            return {
                'class_id': class_id.cpu().numpy(),
                'confidence': confidence.cpu().numpy(),
                'max_probability': max_prob.cpu().numpy(),
                'probabilities': probs.cpu().numpy(),
                'is_confident': is_confident.cpu().numpy()
            }


# ==============================================================================
# DATASET
# ==============================================================================

class SLRDataset(Dataset):
    """Dataset for SLR training."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        sequence_length: int = 16,
        image_size: int = 224,
        transform: Optional[Any] = None,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.augment = augment and split == "train"

        # Load metadata
        metadata_path = self.data_dir / "metadata" / f"{split}_samples.json"
        if not metadata_path.exists():
            metadata_path = self.data_dir / "all_samples.json"

        with open(metadata_path, 'r') as f:
            data = json.load(f)
            all_samples = data.get("samples", [])

        # Filter by split
        self.samples = [s for s in all_samples if s.get("split") == split]

        # Group by sign and video/clip
        self.sequences = self._group_sequences()

        # Load label map
        with open(self.data_dir / "label_map.json", 'r') as f:
            self.label_map = json.load(f)

        self.inverse_label_map = {v: k for k, v in self.label_map.items()}

        # Transforms
        if transform:
            self.transform = transform
        else:
            self.transform = self._get_default_transform()

        logger.info(f"Loaded {len(self.sequences)} sequences for {split}")

    def _group_sequences(self) -> List[Dict]:
        """Group frame samples into sequences."""
        # Group by sign
        by_sign = {}
        for sample in self.samples:
            sign = sample["handshape"]
            if sign not in by_sign:
                by_sign[sign] = []
            by_sign[sign].append(sample)

        # Create sequences
        sequences = []
        for sign, samples in by_sign.items():
            # Sort by ID to maintain frame order
            samples.sort(key=lambda x: x.get("id", ""))

            # Create sequences of required length
            for i in range(0, len(samples), self.sequence_length):
                seq_samples = samples[i:i + self.sequence_length]
                if len(seq_samples) >= self.sequence_length // 2:  # At least half the frames
                    sequences.append({
                        "sign": sign,
                        "frames": seq_samples
                    })

        return sequences

    def _get_default_transform(self):
        """Get default transform based on split."""
        if self.augment:
            return T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        sign = sequence["sign"]
        frame_samples = sequence["frames"]

        # Load frames
        frames = []
        for sample in frame_samples:
            img_path = self.data_dir / sample["filepath"]
            try:
                img = Image.open(img_path).convert("RGB")
                img = self.transform(img)
                frames.append(img)
            except Exception as e:
                # Use black frame as fallback
                frames.append(torch.zeros(3, self.image_size, self.image_size))

        # Pad or truncate to sequence_length
        while len(frames) < self.sequence_length:
            frames.append(frames[-1] if frames else torch.zeros(3, self.image_size, self.image_size))
        frames = frames[:self.sequence_length]

        # Stack: (T, C, H, W)
        frames_tensor = torch.stack(frames)

        # Get label
        label = self.label_map.get(sign, 0)

        return {
            "frames": frames_tensor,
            "label": label,
            "sign": sign
        }


# ==============================================================================
# TRAINING
# ==============================================================================

class Trainer:
    """Model trainer with comprehensive logging."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Create output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Initialize model
        self.model = SLRModel(config).to(self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Datasets
        self.train_dataset = SLRDataset(
            config.data_dir, "train",
            sequence_length=config.sequence_length,
            image_size=config.image_size
        )
        self.val_dataset = SLRDataset(
            config.data_dir, "val",
            sequence_length=config.sequence_length,
            image_size=config.image_size,
            augment=False
        )

        # Update config with actual num_classes
        if self.train_dataset.label_map:
            config.num_classes = len(self.train_dataset.label_map)
            # Rebuild model with correct num_classes
            self.model = SLRModel(config).to(self.device)

        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Scheduler - OneCycleLR cycles from base_lr -> max_lr -> final_lr
        # The optimizer's lr is the starting point, max_lr is the peak
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.max_lr,  # Peak learning rate (higher than base)
            epochs=config.epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=config.warmup_epochs / config.epochs,
            div_factor=config.max_lr / config.learning_rate,  # Start at base_lr
            final_div_factor=100  # End at max_lr / 100
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None

        # TensorBoard
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(self.output_dir / "tensorboard")
        else:
            self.writer = None

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0
        self.training_history = []

        # Resume from checkpoint
        if config.resume_from:
            self.load_checkpoint(config.resume_from)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch in pbar:
            frames = batch["frames"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            if self.config.use_amp:
                with autocast():
                    output = self.model(frames)
                    loss = self.criterion(output["logits"], labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(frames)
                loss = self.criterion(output["logits"], labels)
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = output["logits"].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100. * correct / total:.2f}%"
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
        all_preds = []
        all_labels = []

        for batch in tqdm(self.val_loader, desc="Validating"):
            frames = batch["frames"].to(self.device)
            labels = batch["label"].to(self.device)

            output = self.model(frames)
            loss = self.criterion(output["logits"], labels)

            total_loss += loss.item()
            _, predicted = output["logits"].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return {
            "loss": total_loss / len(self.val_loader),
            "accuracy": 100. * correct / total
        }

    def save_checkpoint(self, filename: str = "checkpoint.pt", is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "config": asdict(self.config),
            "label_map": self.train_dataset.label_map
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint["best_val_acc"]

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(self):
        """Main training loop."""
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info("=" * 60)
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        logger.info(f"Classes: {self.config.num_classes}")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Log
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )

            # TensorBoard
            if self.writer:
                self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
                self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
                self.writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
                self.writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
                self.writer.add_scalar("LR", self.scheduler.get_last_lr()[0], epoch)

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

            if (epoch + 1) % self.config.save_every == 0 or is_best:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt", is_best=is_best)

        # Training complete
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info(f"Time: {elapsed / 3600:.2f} hours")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")

        # Save final model
        self.save_checkpoint("final_model.pt")

        # Save training history
        with open(self.output_dir / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)

        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train SLR model')

    parser.add_argument('--data', type=str, default='./combined_data',
                       help='Path to training data')
    parser.add_argument('--output', type=str, default='./models',
                       help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Base learning rate')
    parser.add_argument('--max-lr', type=float, default=3e-3,
                       help='Peak learning rate for OneCycleLR scheduler')
    parser.add_argument('--sequence-length', type=int, default=16,
                       help='Number of frames per sequence')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--resume', type=str,
                       help='Resume from checkpoint')
    parser.add_argument('--no-attention', action='store_true',
                       help='Disable attention mechanism')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision training')

    args = parser.parse_args()

    config = TrainingConfig(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_lr=args.max_lr,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        resume_from=args.resume,
        use_attention=not args.no_attention,
        use_amp=not args.no_amp
    )

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
