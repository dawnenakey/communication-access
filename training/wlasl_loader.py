#!/usr/bin/env python3
"""
WLASL Dataset Loader and Training Script
=========================================
Load and train on WLASL (Word-Level American Sign Language) dataset.

WLASL Format:
- WLASL_v0.3.json: Annotations with video IDs, glosses, splits
- videos/: Video files (mp4/webm)

Usage:
    # Prepare data
    python training/wlasl_loader.py --prepare \
        --json /path/to/WLASL_v0.3.json \
        --videos /path/to/videos \
        --output ./wlasl_prepared

    # Train model
    python training/wlasl_loader.py --train \
        --data ./wlasl_prepared \
        --subset 100 \
        --epochs 50

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import argparse
import json
import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# WLASL JSON PARSER
# ==============================================================================

def parse_wlasl_json(json_path: str, subset: int = 100) -> Dict[str, Any]:
    """
    Parse WLASL JSON annotation file.

    WLASL JSON format:
    [
        {
            "gloss": "book",
            "instances": [
                {
                    "video_id": "00001",
                    "split": "train",
                    "bbox": [...],
                    ...
                }
            ]
        },
        ...
    ]

    Args:
        json_path: Path to WLASL_v0.3.json
        subset: Number of glosses to use (100, 300, 1000, 2000)

    Returns:
        Dict with glosses, videos, and mappings
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Take top N glosses by number of instances
    glosses_sorted = sorted(data, key=lambda x: len(x.get('instances', [])), reverse=True)
    glosses_subset = glosses_sorted[:subset]

    # Build mappings
    gloss_to_idx = {}
    idx_to_gloss = {}
    videos = {'train': [], 'val': [], 'test': []}

    for idx, item in enumerate(glosses_subset):
        gloss = item['gloss'].upper()
        gloss_to_idx[gloss] = idx
        idx_to_gloss[idx] = gloss

        for instance in item.get('instances', []):
            video_id = instance.get('video_id', '')
            split = instance.get('split', 'train')

            if split not in videos:
                split = 'train'

            videos[split].append({
                'video_id': video_id,
                'gloss': gloss,
                'label': idx,
                'bbox': instance.get('bbox', None),
                'fps': instance.get('fps', 30),
                'frame_start': instance.get('frame_start', 0),
                'frame_end': instance.get('frame_end', -1),
            })

    logger.info(f"Parsed {len(gloss_to_idx)} glosses (subset={subset})")
    logger.info(f"  Train: {len(videos['train'])} videos")
    logger.info(f"  Val: {len(videos['val'])} videos")
    logger.info(f"  Test: {len(videos['test'])} videos")

    return {
        'gloss_to_idx': gloss_to_idx,
        'idx_to_gloss': idx_to_gloss,
        'videos': videos,
        'num_classes': len(gloss_to_idx)
    }


def find_video_file(video_id: str, videos_dir: Path) -> Optional[Path]:
    """Find video file by ID, checking multiple extensions."""
    extensions = ['.mp4', '.webm', '.avi', '.mov', '.mkv']

    for ext in extensions:
        path = videos_dir / f"{video_id}{ext}"
        if path.exists():
            return path

    # Try with leading zeros
    for ext in extensions:
        path = videos_dir / f"{video_id.zfill(5)}{ext}"
        if path.exists():
            return path

    return None


# ==============================================================================
# DATA PREPARATION
# ==============================================================================

def prepare_wlasl_data(
    json_path: str,
    videos_dir: str,
    output_dir: str,
    subset: int = 100,
    frames_per_video: int = 32
):
    """
    Prepare WLASL data for training.

    Extracts frames from videos and creates structured dataset.
    """
    json_path = Path(json_path)
    videos_dir = Path(videos_dir)
    output_dir = Path(output_dir)

    # Parse annotations
    parsed = parse_wlasl_json(str(json_path), subset)

    # Create output structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'frames').mkdir(exist_ok=True)

    # Process videos
    processed = {'train': [], 'val': [], 'test': []}
    missing_videos = []

    for split in ['train', 'val', 'test']:
        logger.info(f"Processing {split} split...")

        for video_info in tqdm(parsed['videos'][split], desc=split):
            video_id = video_info['video_id']
            video_path = find_video_file(video_id, videos_dir)

            if video_path is None:
                missing_videos.append(video_id)
                continue

            # Extract frames
            frames_dir = output_dir / 'frames' / video_id
            frames_dir.mkdir(parents=True, exist_ok=True)

            try:
                num_frames = extract_video_frames(
                    str(video_path),
                    str(frames_dir),
                    frames_per_video,
                    video_info.get('frame_start', 0),
                    video_info.get('frame_end', -1)
                )

                if num_frames > 0:
                    processed[split].append({
                        'video_id': video_id,
                        'gloss': video_info['gloss'],
                        'label': video_info['label'],
                        'frames_dir': f'frames/{video_id}',
                        'num_frames': num_frames
                    })
            except Exception as e:
                logger.warning(f"Failed to process {video_id}: {e}")

    # Save metadata
    metadata = {
        'gloss_to_idx': parsed['gloss_to_idx'],
        'idx_to_gloss': parsed['idx_to_gloss'],
        'num_classes': parsed['num_classes'],
        'frames_per_video': frames_per_video,
        'train': processed['train'],
        'val': processed['val'],
        'test': processed['test']
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Report
    logger.info(f"\nData preparation complete!")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Train: {len(processed['train'])} videos")
    logger.info(f"  Val: {len(processed['val'])} videos")
    logger.info(f"  Test: {len(processed['test'])} videos")

    if missing_videos:
        logger.warning(f"  Missing videos: {len(missing_videos)}")
        with open(output_dir / 'missing_videos.txt', 'w') as f:
            f.write('\n'.join(missing_videos))


def extract_video_frames(
    video_path: str,
    output_dir: str,
    num_frames: int = 32,
    frame_start: int = 0,
    frame_end: int = -1
) -> int:
    """Extract frames from video file."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_end == -1:
        frame_end = total_frames

    # Calculate frame indices to extract
    available_frames = frame_end - frame_start
    if available_frames <= num_frames:
        indices = list(range(frame_start, frame_end))
    else:
        indices = np.linspace(frame_start, frame_end - 1, num_frames, dtype=int)

    # Extract frames
    extracted = 0
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            # Resize to standard size
            frame = cv2.resize(frame, (224, 224))
            output_path = os.path.join(output_dir, f'frame_{i:04d}.jpg')
            cv2.imwrite(output_path, frame)
            extracted += 1

    cap.release()
    return extracted


# ==============================================================================
# DATASET
# ==============================================================================

class WLASLDataset(Dataset):
    """PyTorch dataset for WLASL."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        sequence_length: int = 16,
        transform: Optional[Any] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length

        # Load metadata
        with open(self.data_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        self.samples = metadata[split]
        self.gloss_to_idx = metadata['gloss_to_idx']
        self.idx_to_gloss = {int(k): v for k, v in metadata['idx_to_gloss'].items()}
        self.num_classes = metadata['num_classes']

        # Transforms
        if transform:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        logger.info(f"Loaded {len(self.samples)} samples for {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load frames
        frames_dir = self.data_dir / sample['frames_dir']
        frame_files = sorted(frames_dir.glob('frame_*.jpg'))

        # Sample frames
        if len(frame_files) >= self.sequence_length:
            indices = np.linspace(0, len(frame_files) - 1, self.sequence_length, dtype=int)
            frame_files = [frame_files[i] for i in indices]

        # Load and transform frames
        frames = []
        for frame_path in frame_files:
            img = Image.open(frame_path).convert('RGB')
            img = self.transform(img)
            frames.append(img)

        # Pad if needed
        while len(frames) < self.sequence_length:
            frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))

        frames = torch.stack(frames[:self.sequence_length])  # (T, C, H, W)

        return {
            'frames': frames,
            'label': sample['label'],
            'gloss': sample['gloss']
        }


# ==============================================================================
# MODEL (I3D-style)
# ==============================================================================

class IsolatedSignModel(nn.Module):
    """3D CNN model for isolated sign recognition."""

    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()

        # 3D CNN backbone
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),

            # Block 2
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Global pooling
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) video frames
        Returns:
            logits: (B, num_classes)
        """
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        features = self.features(x)  # (B, 512, 1, 1, 1)
        features = features.view(B, -1)  # (B, 512)

        logits = self.classifier(features)
        return logits

    def predict(self, x: torch.Tensor, top_k: int = 5) -> Dict[str, Any]:
        """Make prediction with confidence scores."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)

            top_probs, top_indices = probs.topk(top_k, dim=-1)

            return {
                'top_indices': top_indices.cpu().numpy(),
                'top_probs': top_probs.cpu().numpy(),
                'all_probs': probs.cpu().numpy()
            }


# ==============================================================================
# TRAINING
# ==============================================================================

def train_wlasl(
    data_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    sequence_length: int = 16
):
    """Train isolated sign model on WLASL data."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create datasets
    train_dataset = WLASLDataset(data_dir, 'train', sequence_length)
    val_dataset = WLASLDataset(data_dir, 'val', sequence_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    model = IsolatedSignModel(num_classes=train_dataset.num_classes).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_acc = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(frames)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.1f}%'
            })

        scheduler.step()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                frames = batch['frames'].to(device)
                labels = batch['label'].to(device)

                logits = model(frames)
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        train_acc = 100. * train_correct / train_total

        logger.info(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss/len(train_loader):.4f}, "
            f"Train Acc={train_acc:.1f}%, "
            f"Val Acc={val_acc:.1f}%"
        )

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': train_dataset.num_classes,
                'gloss_to_idx': train_dataset.gloss_to_idx,
                'idx_to_gloss': train_dataset.idx_to_gloss
            }, output_dir / 'best_model.pt')
            logger.info(f"  Saved best model (acc={val_acc:.1f}%)")

    logger.info(f"\nTraining complete! Best validation accuracy: {best_acc:.1f}%")
    return best_acc


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='WLASL Dataset Loader and Training')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Prepare command
    prep_parser = subparsers.add_parser('prepare', help='Prepare WLASL data')
    prep_parser.add_argument('--json', type=str, required=True,
                            help='Path to WLASL_v0.3.json')
    prep_parser.add_argument('--videos', type=str, required=True,
                            help='Path to videos directory')
    prep_parser.add_argument('--output', type=str, default='./wlasl_prepared',
                            help='Output directory')
    prep_parser.add_argument('--subset', type=int, default=100,
                            choices=[100, 300, 1000, 2000],
                            help='Number of glosses to use')
    prep_parser.add_argument('--frames', type=int, default=32,
                            help='Frames per video')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train on WLASL')
    train_parser.add_argument('--data', type=str, required=True,
                             help='Path to prepared data')
    train_parser.add_argument('--output', type=str, default='./models/wlasl',
                             help='Output directory for model')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=16,
                             help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-4,
                             help='Learning rate')

    args = parser.parse_args()

    if args.command == 'prepare':
        prepare_wlasl_data(
            args.json,
            args.videos,
            args.output,
            args.subset,
            args.frames
        )
    elif args.command == 'train':
        train_wlasl(
            args.data,
            args.output,
            args.epochs,
            args.batch_size,
            args.lr
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
