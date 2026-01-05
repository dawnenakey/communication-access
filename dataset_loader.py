#!/usr/bin/env python3
"""
Synthetic ASL Dataset Loader for SLR Training
==============================================
Load and combine synthetic data with real SignCut annotations
for training sign language recognition models.

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import json
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator, Any, Union
from dataclasses import dataclass
import logging

# PyTorch imports (install with: pip install torch torchvision)
try:
    import torch
    from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch torchvision")

# OpenCV for additional augmentation
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SampleMetadata:
    """Metadata for a single sample."""
    id: int
    handshape: str
    label: int
    filepath: Optional[str] = None
    image_base64: Optional[str] = None
    keypoints_2d: Optional[List[List[float]]] = None
    keypoints_3d: Optional[List[List[float]]] = None
    skin_tone_idx: Optional[int] = None
    is_synthetic: bool = True


class SyntheticASLDataset(Dataset):
    """PyTorch Dataset for synthetic ASL images."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        transform: Optional[Any] = None,
        include_keypoints: bool = False,
        target_handshapes: Optional[List[str]] = None,
        max_samples_per_class: Optional[int] = None
    ):
        """
        Initialize the synthetic dataset.
        
        Args:
            data_dir: Path to synthetic data directory
            transform: Torchvision transforms to apply
            include_keypoints: Whether to return keypoints with samples
            target_handshapes: Filter to specific handshapes
            max_samples_per_class: Limit samples per handshape class
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch torchvision")
        
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.include_keypoints = include_keypoints
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load label mapping
        label_map_path = self.data_dir / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
        else:
            # Create label map from unique handshapes
            handshapes = sorted(set(s['handshape'] for s in self.metadata['samples']))
            self.label_map = {h: i for i, h in enumerate(handshapes)}
        
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # Filter and limit samples
        self.samples = []
        class_counts = {}
        
        for sample in self.metadata['samples']:
            handshape = sample['handshape']
            
            # Filter by target handshapes
            if target_handshapes and handshape not in target_handshapes:
                continue
            
            # Limit samples per class
            if max_samples_per_class:
                if class_counts.get(handshape, 0) >= max_samples_per_class:
                    continue
                class_counts[handshape] = class_counts.get(handshape, 0) + 1
            
            self.samples.append(SampleMetadata(
                id=sample['id'],
                handshape=handshape,
                label=self.label_map[handshape],
                filepath=sample.get('filepath'),
                image_base64=sample.get('image_base64'),
                keypoints_2d=sample.get('keypoints_2d'),
                keypoints_3d=sample.get('keypoints_3d'),
                skin_tone_idx=sample.get('skin_tone_idx'),
                is_synthetic=True
            ))
        
        logger.info(f"Loaded {len(self.samples)} synthetic samples")
        logger.info(f"Classes: {len(self.label_map)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load image
        if sample.image_base64:
            img_data = base64.b64decode(sample.image_base64)
            image = Image.open(BytesIO(img_data)).convert('RGB')
        elif sample.filepath:
            img_path = self.data_dir / sample.filepath
            image = Image.open(img_path).convert('RGB')
        else:
            raise ValueError(f"No image data for sample {sample.id}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)
        
        result = {
            'image': image,
            'label': sample.label,
            'handshape': sample.handshape,
            'is_synthetic': sample.is_synthetic,
            'sample_id': sample.id
        }
        
        # Include keypoints if requested
        if self.include_keypoints and sample.keypoints_2d:
            result['keypoints_2d'] = torch.tensor(sample.keypoints_2d, dtype=torch.float32)
        if self.include_keypoints and sample.keypoints_3d:
            result['keypoints_3d'] = torch.tensor(sample.keypoints_3d, dtype=torch.float32)
        
        return result
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced sampling."""
        class_counts = {}
        for sample in self.samples:
            class_counts[sample.label] = class_counts.get(sample.label, 0) + 1
        
        total = len(self.samples)
        num_classes = len(self.label_map)
        
        weights = torch.zeros(num_classes)
        for label, count in class_counts.items():
            weights[label] = total / (num_classes * count)
        
        return weights
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        return torch.tensor([class_weights[s.label] for s in self.samples])


class RealASLDataset(Dataset):
    """
    PyTorch Dataset for real ASL data (SignCut format).
    Adapt this to match your actual SignCut annotation format.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        annotation_file: str = "annotations.json",
        transform: Optional[Any] = None,
        label_map: Optional[Dict[str, int]] = None
    ):
        """
        Initialize real dataset.
        
        Args:
            data_dir: Path to real data directory
            annotation_file: Name of annotation JSON file
            transform: Torchvision transforms
            label_map: Mapping from handshape names to integer labels
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.label_map = label_map or {}
        
        # Load annotations - ADAPT THIS TO YOUR FORMAT
        ann_path = self.data_dir / annotation_file
        if ann_path.exists():
            with open(ann_path, 'r') as f:
                self.annotations = json.load(f)
        else:
            # Try to infer from directory structure
            self.annotations = self._infer_annotations()
        
        self.samples = []
        for ann in self.annotations:
            handshape = ann.get('handshape', ann.get('label', 'unknown'))
            
            # Skip if handshape not in label map
            if self.label_map and handshape not in self.label_map:
                continue
            
            label = self.label_map.get(handshape, len(self.label_map))
            if handshape not in self.label_map:
                self.label_map[handshape] = label
            
            self.samples.append(SampleMetadata(
                id=ann.get('id', len(self.samples)),
                handshape=handshape,
                label=label,
                filepath=ann.get('filepath', ann.get('image_path')),
                keypoints_2d=ann.get('keypoints_2d'),
                keypoints_3d=ann.get('keypoints_3d'),
                is_synthetic=False
            ))
        
        logger.info(f"Loaded {len(self.samples)} real samples")
    
    def _infer_annotations(self) -> List[Dict]:
        """Infer annotations from directory structure (class/image.png)."""
        annotations = []
        
        for class_dir in self.data_dir.iterdir():
            if class_dir.is_dir() and not class_dir.name.startswith('.'):
                handshape = class_dir.name.upper()
                
                for img_file in class_dir.glob('*.png'):
                    annotations.append({
                        'handshape': handshape,
                        'filepath': str(img_file.relative_to(self.data_dir))
                    })
                for img_file in class_dir.glob('*.jpg'):
                    annotations.append({
                        'handshape': handshape,
                        'filepath': str(img_file.relative_to(self.data_dir))
                    })
        
        return annotations
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        img_path = self.data_dir / sample.filepath
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)
        
        return {
            'image': image,
            'label': sample.label,
            'handshape': sample.handshape,
            'is_synthetic': False,
            'sample_id': sample.id
        }


class CombinedASLDataset(Dataset):
    """
    Combined dataset that merges synthetic and real data
    with configurable mixing ratios.
    """
    
    def __init__(
        self,
        synthetic_dataset: SyntheticASLDataset,
        real_dataset: RealASLDataset,
        synthetic_ratio: float = 0.5,
        curriculum_phase: str = "mixed"  # "synthetic_only", "mixed", "real_heavy"
    ):
        """
        Initialize combined dataset.
        
        Args:
            synthetic_dataset: Synthetic data
            real_dataset: Real data
            synthetic_ratio: Ratio of synthetic samples (0.0-1.0)
            curriculum_phase: Training curriculum phase
        """
        self.synthetic = synthetic_dataset
        self.real = real_dataset
        self.synthetic_ratio = synthetic_ratio
        self.curriculum_phase = curriculum_phase
        
        # Ensure label maps are aligned
        self._align_label_maps()
        
        # Build combined sample list based on curriculum
        self._build_sample_list()
    
    def _align_label_maps(self):
        """Ensure both datasets use the same label mapping."""
        # Use synthetic label map as base
        combined_map = dict(self.synthetic.label_map)
        
        # Add any missing labels from real data
        next_label = max(combined_map.values()) + 1 if combined_map else 0
        for handshape in self.real.label_map:
            if handshape not in combined_map:
                combined_map[handshape] = next_label
                next_label += 1
        
        # Update both datasets
        self.label_map = combined_map
        self.synthetic.label_map = combined_map
        self.real.label_map = combined_map
        
        # Remap sample labels
        for sample in self.synthetic.samples:
            sample.label = combined_map[sample.handshape]
        for sample in self.real.samples:
            sample.label = combined_map[sample.handshape]
    
    def _build_sample_list(self):
        """Build the combined sample list based on curriculum phase."""
        if self.curriculum_phase == "synthetic_only":
            # Phase 1: Train only on synthetic
            self.samples = list(self.synthetic.samples)
            self.source_dataset = [self.synthetic] * len(self.samples)
            
        elif self.curriculum_phase == "mixed":
            # Phase 2: Mixed training
            # Subsample synthetic to match ratio
            n_synthetic = int(len(self.real) * self.synthetic_ratio / (1 - self.synthetic_ratio))
            n_synthetic = min(n_synthetic, len(self.synthetic))
            
            # Random subsample synthetic
            indices = np.random.choice(len(self.synthetic), n_synthetic, replace=False)
            synthetic_samples = [self.synthetic.samples[i] for i in indices]
            
            self.samples = synthetic_samples + list(self.real.samples)
            self.source_dataset = (
                [self.synthetic] * len(synthetic_samples) + 
                [self.real] * len(self.real)
            )
            
        elif self.curriculum_phase == "real_heavy":
            # Phase 3: Mostly real, some synthetic for regularization
            n_synthetic = int(len(self.real) * 0.1)  # 10% synthetic
            n_synthetic = min(n_synthetic, len(self.synthetic))
            
            indices = np.random.choice(len(self.synthetic), n_synthetic, replace=False)
            synthetic_samples = [self.synthetic.samples[i] for i in indices]
            
            self.samples = synthetic_samples + list(self.real.samples)
            self.source_dataset = (
                [self.synthetic] * len(synthetic_samples) + 
                [self.real] * len(self.real)
            )
        
        # Shuffle
        combined = list(zip(self.samples, self.source_dataset))
        np.random.shuffle(combined)
        self.samples, self.source_dataset = zip(*combined)
        self.samples = list(self.samples)
        self.source_dataset = list(self.source_dataset)
        
        logger.info(f"Combined dataset: {len(self.samples)} samples "
                   f"({sum(1 for s in self.samples if s.is_synthetic)} synthetic, "
                   f"{sum(1 for s in self.samples if not s.is_synthetic)} real)")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        dataset = self.source_dataset[idx]
        
        # Find original index in source dataset
        orig_idx = dataset.samples.index(sample)
        return dataset[orig_idx]
    
    def set_curriculum_phase(self, phase: str):
        """Change curriculum phase and rebuild sample list."""
        self.curriculum_phase = phase
        self._build_sample_list()


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def get_training_transforms(image_size: int = 224) -> T.Compose:
    """Get standard training augmentations."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_validation_transforms(image_size: int = 224) -> T.Compose:
    """Get validation/test transforms (no augmentation)."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ============================================================================
# DATA LOADERS
# ============================================================================

def create_dataloaders(
    synthetic_dir: str,
    real_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    synthetic_ratio: float = 0.5,
    curriculum_phase: str = "mixed"
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders.
    
    Args:
        synthetic_dir: Path to synthetic data
        real_dir: Path to real data (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size
        synthetic_ratio: Ratio of synthetic data in combined dataset
        curriculum_phase: Curriculum learning phase
    
    Returns:
        (train_loader, val_loader) tuple
    """
    train_transform = get_training_transforms(image_size)
    val_transform = get_validation_transforms(image_size)
    
    # Load synthetic data
    synthetic_train = SyntheticASLDataset(
        synthetic_dir, 
        transform=train_transform,
        include_keypoints=True
    )
    
    if real_dir:
        # Combined training
        real_train = RealASLDataset(
            real_dir,
            transform=train_transform,
            label_map=synthetic_train.label_map
        )
        
        train_dataset = CombinedASLDataset(
            synthetic_train,
            real_train,
            synthetic_ratio=synthetic_ratio,
            curriculum_phase=curriculum_phase
        )
        
        # Create validation set from real data only
        # (In practice, split real data into train/val)
        val_dataset = None  # TODO: Add real validation split
    else:
        # Synthetic only training
        train_dataset = synthetic_train
        val_dataset = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def visualize_batch(batch: Dict[str, Any], save_path: Optional[str] = None):
    """Visualize a batch of samples."""
    import matplotlib.pyplot as plt
    
    images = batch['image']
    labels = batch['label']
    handshapes = batch['handshape']
    is_synthetic = batch['is_synthetic']
    
    n = min(16, len(images))
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for i in range(n):
        img = images[i] * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        source = "S" if is_synthetic[i] else "R"
        axes[i].set_title(f"{handshapes[i]} [{source}]")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compute_dataset_statistics(dataset: Dataset) -> Dict[str, Any]:
    """Compute statistics about a dataset."""
    class_counts = {}
    synthetic_counts = {}
    
    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample['label']
        handshape = sample['handshape']
        is_syn = sample['is_synthetic']
        
        class_counts[handshape] = class_counts.get(handshape, 0) + 1
        if is_syn:
            synthetic_counts[handshape] = synthetic_counts.get(handshape, 0) + 1
    
    return {
        'total_samples': len(dataset),
        'num_classes': len(class_counts),
        'class_counts': class_counts,
        'synthetic_counts': synthetic_counts,
        'samples_per_class_mean': np.mean(list(class_counts.values())),
        'samples_per_class_std': np.std(list(class_counts.values())),
    }


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test synthetic ASL dataset loader')
    parser.add_argument('--synthetic-dir', type=str, required=True, help='Synthetic data directory')
    parser.add_argument('--real-dir', type=str, help='Real data directory (optional)')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--visualize', action='store_true', help='Visualize a batch')
    
    args = parser.parse_args()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        synthetic_dir=args.synthetic_dir,
        real_dir=args.real_dir,
        batch_size=args.batch_size
    )
    
    # Print statistics
    print("\n" + "="*50)
    print("Dataset Statistics")
    print("="*50)
    stats = compute_dataset_statistics(train_loader.dataset)
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in sorted(value.items()):
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  images: {batch['image'].shape}")
    print(f"  labels: {batch['label'].shape}")
    
    if args.visualize:
        visualize_batch(batch, save_path="batch_visualization.png")
