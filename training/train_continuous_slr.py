#!/usr/bin/env python3
"""
Continuous Sign Language Recognition Training
==============================================
CTC-based model for sentence-level sign language recognition.

Unlike isolated sign recognition, this handles continuous signing
without requiring pauses between signs.

Architecture:
  Video frames → 3D CNN → BiLSTM → CTC Loss → Sign sequence

Key Features:
- CTC (Connectionist Temporal Classification) for variable-length output
- Handles coarticulation (signs blending together)
- Beam search decoding for inference
- Supports sentence-level training data

Usage:
    python training/train_continuous_slr.py \
        --data ./sentence_data \
        --output ./models/continuous

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class ContinuousConfig:
    """Configuration for continuous SLR training."""
    # Data
    data_dir: str = "./sentence_data"
    output_dir: str = "./models/continuous"

    # Vocabulary (signs + blank for CTC)
    vocab_size: int = 64  # 63 signs + 1 blank token
    blank_idx: int = 0    # CTC blank token index

    # Model architecture
    cnn_features: int = 512
    hidden_size: int = 512
    num_lstm_layers: int = 3
    dropout: float = 0.3
    bidirectional: bool = True

    # Training
    batch_size: int = 8
    epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0

    # Input
    max_frames: int = 300      # Max frames per sentence (~10 seconds at 30fps)
    min_frames: int = 30       # Min frames
    image_size: int = 224

    # Decoding
    beam_width: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# CTC MODEL
# ==============================================================================

class VisualEncoder(nn.Module):
    """3D CNN for extracting visual features from video frames."""

    def __init__(self, out_features: int = 512):
        super().__init__()

        # Efficient 3D convolutions with temporal stride
        self.conv_layers = nn.Sequential(
            # (B, C, T, H, W) -> (B, 64, T, H/2, W/2)
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),

            # -> (B, 128, T, H/4, W/4)
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            # -> (B, 256, T/2, H/8, W/8)
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            # -> (B, 512, T/4, H/16, W/16)
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),

            # Global spatial pooling, keep temporal
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )

        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C, H, W) video frames

        Returns:
            features: (B, T', out_features) where T' = T/4 due to temporal pooling
            lengths: actual feature lengths for each batch item
        """
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        features = self.conv_layers(x)  # (B, 512, T', 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B, 512, T')
        features = features.permute(0, 2, 1)  # (B, T', 512)

        return features


class ContinuousSLRModel(nn.Module):
    """
    CTC-based Continuous Sign Language Recognition Model.

    Architecture:
        Video → 3D CNN → BiLSTM → Linear → CTC
    """

    def __init__(self, config: ContinuousConfig):
        super().__init__()
        self.config = config

        # Visual encoder
        self.encoder = VisualEncoder(config.cnn_features)

        # Temporal modeling with BiLSTM
        lstm_hidden = config.hidden_size
        self.lstm = nn.LSTM(
            input_size=config.cnn_features,
            hidden_size=lstm_hidden,
            num_layers=config.num_lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.num_lstm_layers > 1 else 0,
            bidirectional=config.bidirectional
        )

        # Output projection
        lstm_output_size = lstm_hidden * 2 if config.bidirectional else lstm_hidden
        self.output_proj = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(lstm_output_size, config.vocab_size)
        )

        # CTC loss
        self.ctc_loss = nn.CTCLoss(blank=config.blank_idx, reduction='mean', zero_infinity=True)

    def forward(
        self,
        frames: torch.Tensor,
        frame_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            frames: (B, T, C, H, W) input video frames
            frame_lengths: (B,) actual frame counts (for variable length batches)

        Returns:
            log_probs: (T', B, vocab_size) log probabilities for CTC
        """
        # Encode visual features
        features = self.encoder(frames)  # (B, T', features)

        # Temporal modeling
        lstm_out, _ = self.lstm(features)  # (B, T', hidden*2)

        # Project to vocabulary
        logits = self.output_proj(lstm_out)  # (B, T', vocab_size)

        # Log softmax for CTC (needs T, B, C format)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.permute(1, 0, 2)  # (T', B, vocab_size)

        return log_probs

    def compute_loss(
        self,
        frames: torch.Tensor,
        targets: torch.Tensor,
        frame_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CTC loss.

        Args:
            frames: (B, T, C, H, W) input frames
            targets: (B, S) target sign sequences (padded)
            frame_lengths: (B,) actual frame counts
            target_lengths: (B,) actual target lengths

        Returns:
            CTC loss value
        """
        log_probs = self.forward(frames, frame_lengths)  # (T', B, vocab)

        # Calculate output lengths (T' = T/4 due to temporal pooling)
        input_lengths = (frame_lengths / 4).long().clamp(min=1)

        # Flatten targets for CTC
        targets_flat = targets[targets != -1]  # Remove padding

        loss = self.ctc_loss(log_probs, targets_flat, input_lengths, target_lengths)

        return loss

    @torch.no_grad()
    def decode_greedy(self, frames: torch.Tensor) -> List[List[int]]:
        """
        Greedy decoding (fast but suboptimal).

        Returns list of predicted sign sequences.
        """
        log_probs = self.forward(frames)  # (T', B, vocab)
        log_probs = log_probs.permute(1, 0, 2)  # (B, T', vocab)

        # Greedy: take argmax at each timestep
        predictions = log_probs.argmax(dim=-1)  # (B, T')

        # Collapse repeats and remove blanks
        decoded = []
        for pred in predictions:
            collapsed = self._ctc_collapse(pred.cpu().numpy())
            decoded.append(collapsed)

        return decoded

    @torch.no_grad()
    def decode_beam(
        self,
        frames: torch.Tensor,
        beam_width: int = 10
    ) -> List[List[int]]:
        """
        Beam search decoding (slower but better).

        Returns list of predicted sign sequences.
        """
        log_probs = self.forward(frames)  # (T', B, vocab)
        log_probs = log_probs.permute(1, 0, 2)  # (B, T', vocab)

        decoded = []
        for b in range(log_probs.size(0)):
            sequence = self._beam_search(log_probs[b], beam_width)
            decoded.append(sequence)

        return decoded

    def _ctc_collapse(self, sequence: np.ndarray) -> List[int]:
        """Collapse CTC output: remove blanks and merge repeats."""
        result = []
        prev = -1
        for idx in sequence:
            if idx != self.config.blank_idx and idx != prev:
                result.append(int(idx))
            prev = idx
        return result

    def _beam_search(
        self,
        log_probs: torch.Tensor,
        beam_width: int
    ) -> List[int]:
        """
        Beam search decoding for single sequence.

        Args:
            log_probs: (T, vocab) log probabilities
            beam_width: number of beams to keep

        Returns:
            Best decoded sequence
        """
        T, V = log_probs.shape
        blank = self.config.blank_idx

        # Beam: (prefix, log_prob_blank, log_prob_non_blank)
        beams = [(tuple(), 0.0, float('-inf'))]

        for t in range(T):
            new_beams = {}
            probs_t = log_probs[t].cpu().numpy()

            for prefix, pb, pnb in beams:
                # Extend with blank
                new_pb = np.logaddexp(pb, pnb) + probs_t[blank]
                key = prefix
                if key in new_beams:
                    old_pb, old_pnb = new_beams[key]
                    new_beams[key] = (np.logaddexp(old_pb, new_pb), old_pnb)
                else:
                    new_beams[key] = (new_pb, float('-inf'))

                # Extend with non-blank
                for c in range(V):
                    if c == blank:
                        continue

                    if len(prefix) > 0 and prefix[-1] == c:
                        # Same as last char: only from blank
                        new_pnb = pb + probs_t[c]
                    else:
                        # Different char: from both
                        new_pnb = np.logaddexp(pb, pnb) + probs_t[c]

                    new_prefix = prefix + (c,)
                    if new_prefix in new_beams:
                        old_pb, old_pnb = new_beams[new_prefix]
                        new_beams[new_prefix] = (old_pb, np.logaddexp(old_pnb, new_pnb))
                    else:
                        new_beams[new_prefix] = (float('-inf'), new_pnb)

            # Prune to beam_width
            beams = sorted(
                [(k, pb, pnb) for k, (pb, pnb) in new_beams.items()],
                key=lambda x: np.logaddexp(x[1], x[2]),
                reverse=True
            )[:beam_width]

        # Return best beam
        if beams:
            best_prefix = beams[0][0]
            return list(best_prefix)
        return []


# ==============================================================================
# SENTENCE DATASET
# ==============================================================================

class SentenceDataset(Dataset):
    """
    Dataset for continuous sign language recognition.

    Expects data in format:
    {
        "sentences": [
            {
                "id": "sent_001",
                "signs": ["HELLO", "HOW", "ARE", "YOU"],
                "frames_dir": "sentences/sent_001/",
                "num_frames": 120
            },
            ...
        ]
    }
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
        config: ContinuousConfig,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.config = config
        self.augment = augment and split == "train"

        # Load sentences
        metadata_path = self.data_dir / f"{split}_sentences.json"
        if not metadata_path.exists():
            metadata_path = self.data_dir / "sentences.json"

        with open(metadata_path, 'r') as f:
            data = json.load(f)

        self.sentences = data.get("sentences", [])

        # Filter by split if needed
        if "split" in self.sentences[0] if self.sentences else False:
            self.sentences = [s for s in self.sentences if s.get("split") == split]

        # Load vocabulary
        vocab_path = self.data_dir / "vocabulary.json"
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)

        self.sign_to_idx = vocab_data["sign_to_idx"]
        self.idx_to_sign = {v: k for k, v in self.sign_to_idx.items()}

        # Transforms
        self.transform = self._get_transform()

        logger.info(f"Loaded {len(self.sentences)} sentences for {split}")
        logger.info(f"Vocabulary size: {len(self.sign_to_idx)}")

    def _get_transform(self):
        if self.augment:
            return T.Compose([
                T.Resize((self.config.image_size, self.config.image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize((self.config.image_size, self.config.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sentence = self.sentences[idx]

        # Load frames
        frames_dir = self.data_dir / sentence["frames_dir"]
        num_frames = sentence["num_frames"]

        # Sample frames if too many
        if num_frames > self.config.max_frames:
            # Uniform sampling
            indices = np.linspace(0, num_frames - 1, self.config.max_frames, dtype=int)
        else:
            indices = range(num_frames)

        frames = []
        for i in indices:
            frame_path = frames_dir / f"frame_{i:05d}.png"
            if frame_path.exists():
                img = Image.open(frame_path).convert("RGB")
                img = self.transform(img)
                frames.append(img)
            else:
                # Black frame fallback
                frames.append(torch.zeros(3, self.config.image_size, self.config.image_size))

        # Pad to min_frames if needed
        while len(frames) < self.config.min_frames:
            frames.append(frames[-1] if frames else torch.zeros(3, self.config.image_size, self.config.image_size))

        frames_tensor = torch.stack(frames)  # (T, C, H, W)

        # Convert signs to indices
        target = [self.sign_to_idx.get(sign, 0) for sign in sentence["signs"]]
        target_tensor = torch.tensor(target, dtype=torch.long)

        return {
            "frames": frames_tensor,
            "frame_length": len(frames),
            "target": target_tensor,
            "target_length": len(target),
            "signs": sentence["signs"],
            "id": sentence.get("id", str(idx))
        }


def collate_sentences(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length sentences."""

    # Pad frames to max length in batch
    max_frames = max(item["frame_length"] for item in batch)
    padded_frames = []

    for item in batch:
        frames = item["frames"]
        if frames.size(0) < max_frames:
            padding = torch.zeros(max_frames - frames.size(0), *frames.shape[1:])
            frames = torch.cat([frames, padding], dim=0)
        padded_frames.append(frames)

    # Pad targets
    max_target = max(item["target_length"] for item in batch)
    padded_targets = []

    for item in batch:
        target = item["target"]
        if target.size(0) < max_target:
            padding = torch.full((max_target - target.size(0),), -1, dtype=torch.long)
            target = torch.cat([target, padding])
        padded_targets.append(target)

    return {
        "frames": torch.stack(padded_frames),
        "frame_lengths": torch.tensor([item["frame_length"] for item in batch]),
        "targets": torch.stack(padded_targets),
        "target_lengths": torch.tensor([item["target_length"] for item in batch]),
        "signs": [item["signs"] for item in batch],
        "ids": [item["id"] for item in batch]
    }


# ==============================================================================
# TRAINER
# ==============================================================================

class ContinuousTrainer:
    """Trainer for continuous SLR model."""

    def __init__(self, config: ContinuousConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model
        self.model = ContinuousSLRModel(config).to(self.device)
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {param_count:,}")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Training state
        self.current_epoch = 0
        self.best_wer = float('inf')

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch in pbar:
            frames = batch["frames"].to(self.device)
            targets = batch["targets"].to(self.device)
            frame_lengths = batch["frame_lengths"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)

            self.optimizer.zero_grad()

            loss = self.model.compute_loss(frames, targets, frame_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN/Inf loss detected, skipping batch")
                continue

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return {"loss": total_loss / len(train_loader)}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model and compute WER."""
        self.model.eval()
        total_loss = 0

        all_predictions = []
        all_targets = []

        for batch in tqdm(val_loader, desc="Validating"):
            frames = batch["frames"].to(self.device)
            targets = batch["targets"].to(self.device)
            frame_lengths = batch["frame_lengths"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)

            loss = self.model.compute_loss(frames, targets, frame_lengths, target_lengths)
            total_loss += loss.item()

            # Decode predictions
            predictions = self.model.decode_greedy(frames)

            # Collect for WER
            for pred, tgt, tgt_len in zip(predictions, targets, target_lengths):
                actual_target = tgt[:tgt_len].cpu().tolist()
                all_predictions.append(pred)
                all_targets.append(actual_target)

        # Calculate WER (Word Error Rate)
        wer = self._calculate_wer(all_predictions, all_targets)

        return {
            "loss": total_loss / len(val_loader),
            "wer": wer
        }

    def _calculate_wer(
        self,
        predictions: List[List[int]],
        targets: List[List[int]]
    ) -> float:
        """Calculate Word Error Rate using Levenshtein distance."""
        total_errors = 0
        total_words = 0

        for pred, target in zip(predictions, targets):
            errors = self._levenshtein_distance(pred, target)
            total_errors += errors
            total_words += len(target)

        return total_errors / max(total_words, 1)

    def _levenshtein_distance(self, s1: List[int], s2: List[int]) -> int:
        """Calculate Levenshtein distance between two sequences."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_wer": self.best_wer,
            "config": asdict(self.config)
        }

        path = self.output_dir / filename
        torch.save(checkpoint, path)

        if is_best:
            torch.save(checkpoint, self.output_dir / "best_model.pt")

        logger.info(f"Saved checkpoint: {path}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        logger.info("=" * 60)
        logger.info("Starting Continuous SLR Training")
        logger.info("=" * 60)

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step(val_metrics["loss"])

            # Log
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"WER: {val_metrics['wer']:.2%}"
            )

            # Save checkpoint
            is_best = val_metrics["wer"] < self.best_wer
            if is_best:
                self.best_wer = val_metrics["wer"]

            if (epoch + 1) % 5 == 0 or is_best:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt", is_best)

        logger.info("Training complete!")
        logger.info(f"Best WER: {self.best_wer:.2%}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Continuous SLR Model')

    parser.add_argument('--data', type=str, default='./sentence_data',
                       help='Path to sentence-level training data')
    parser.add_argument('--output', type=str, default='./models/continuous',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')

    args = parser.parse_args()

    config = ContinuousConfig(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    # Create datasets
    train_dataset = SentenceDataset(config.data_dir, "train", config)
    val_dataset = SentenceDataset(config.data_dir, "val", config, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_sentences,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_sentences,
        pin_memory=True
    )

    # Train
    trainer = ContinuousTrainer(config)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
