#!/usr/bin/env python3
"""
Data Merger for SLR Training
=============================
Combines synthetic (Blender) and real (webcam) ASL data into a unified
training dataset with configurable mixing ratios.

Features:
- Merges synthetic and real datasets
- Configurable ratio (default: 80% synthetic, 20% real)
- Automatic class balancing
- Creates unified train/val/test splits
- Generates combined metadata and label maps

Usage:
    python scripts/merge_data.py \\
        --synthetic ./synthetic_data \\
        --real ./real_data \\
        --output ./combined_data \\
        --synthetic-ratio 0.8

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import random
from collections import defaultdict

# Add parent directory for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class MergeConfig:
    """Configuration for data merging."""
    synthetic_dir: str
    real_dir: str
    output_dir: str
    synthetic_ratio: float = 0.8  # 80% synthetic, 20% real

    # Class balancing
    balance_classes: bool = True
    max_samples_per_class: Optional[int] = None

    # Copying vs linking
    copy_files: bool = True  # If False, create symlinks

    # Random seed for reproducibility
    seed: int = 42


class DataMerger:
    """Merge synthetic and real ASL datasets."""

    def __init__(self, config: MergeConfig):
        self.config = config
        self.synthetic_dir = Path(config.synthetic_dir)
        self.real_dir = Path(config.real_dir)
        self.output_dir = Path(config.output_dir)

        random.seed(config.seed)

        # Data containers
        self.synthetic_samples: List[Dict] = []
        self.real_samples: List[Dict] = []
        self.combined_samples: List[Dict] = []

        # Statistics
        self.stats = {
            "synthetic": defaultdict(lambda: defaultdict(int)),
            "real": defaultdict(lambda: defaultdict(int)),
            "combined": defaultdict(lambda: defaultdict(int))
        }

    def load_synthetic_data(self):
        """Load synthetic dataset metadata."""
        print("Loading synthetic data...")

        # Try different metadata file locations
        metadata_paths = [
            self.synthetic_dir / "metadata" / "all_samples.json",
            self.synthetic_dir / "metadata.json",
            self.synthetic_dir / "all_samples.json"
        ]

        for path in metadata_paths:
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.synthetic_samples = data.get("samples", [])
                    print(f"  Loaded {len(self.synthetic_samples)} synthetic samples")
                    break
        else:
            # Infer from directory structure
            print("  No metadata file found, inferring from directory structure...")
            self.synthetic_samples = self._infer_samples(self.synthetic_dir, is_synthetic=True)
            print(f"  Found {len(self.synthetic_samples)} synthetic samples")

        # Update stats
        for sample in self.synthetic_samples:
            split = sample.get("split", "train")
            sign = sample.get("handshape", "unknown")
            self.stats["synthetic"][split][sign] += 1

    def load_real_data(self):
        """Load real dataset metadata."""
        print("Loading real data...")

        if not self.real_dir.exists():
            print("  Real data directory not found, skipping...")
            return

        metadata_paths = [
            self.real_dir / "all_samples.json",
            self.real_dir / "metadata" / "all_samples.json",
        ]

        for path in metadata_paths:
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.real_samples = data.get("samples", [])
                    print(f"  Loaded {len(self.real_samples)} real samples")
                    break
        else:
            # Infer from directory structure
            print("  No metadata file found, inferring from directory structure...")
            self.real_samples = self._infer_samples(self.real_dir, is_synthetic=False)
            print(f"  Found {len(self.real_samples)} real samples")

        # Update stats
        for sample in self.real_samples:
            split = sample.get("split", "train")
            sign = sample.get("handshape", "unknown")
            self.stats["real"][split][sign] += 1

    def _infer_samples(self, data_dir: Path, is_synthetic: bool) -> List[Dict]:
        """Infer samples from directory structure (split/sign/image.png)."""
        samples = []

        for split in ["train", "val", "test"]:
            split_dir = data_dir / split
            if not split_dir.exists():
                continue

            for sign_dir in split_dir.iterdir():
                if not sign_dir.is_dir():
                    continue

                sign_name = sign_dir.name.upper()

                for img_file in sign_dir.glob("*.png"):
                    samples.append({
                        "id": img_file.stem,
                        "handshape": sign_name,
                        "filepath": str(img_file.relative_to(data_dir)),
                        "split": split,
                        "is_synthetic": is_synthetic
                    })

                for img_file in sign_dir.glob("*.jpg"):
                    samples.append({
                        "id": img_file.stem,
                        "handshape": sign_name,
                        "filepath": str(img_file.relative_to(data_dir)),
                        "split": split,
                        "is_synthetic": is_synthetic
                    })

        return samples

    def merge_datasets(self):
        """Merge synthetic and real datasets with configured ratio."""
        print("\nMerging datasets...")

        # Get all unique signs
        synthetic_signs = set(s["handshape"] for s in self.synthetic_samples)
        real_signs = set(s["handshape"] for s in self.real_samples) if self.real_samples else set()
        all_signs = synthetic_signs | real_signs

        print(f"  Synthetic signs: {len(synthetic_signs)}")
        print(f"  Real signs: {len(real_signs)}")
        print(f"  Combined signs: {len(all_signs)}")

        # Group samples by sign and split
        synthetic_by_sign = defaultdict(lambda: defaultdict(list))
        real_by_sign = defaultdict(lambda: defaultdict(list))

        for sample in self.synthetic_samples:
            sign = sample["handshape"]
            split = sample.get("split", "train")
            synthetic_by_sign[sign][split].append(sample)

        for sample in self.real_samples:
            sign = sample["handshape"]
            split = sample.get("split", "train")
            real_by_sign[sign][split].append(sample)

        # Merge with configured ratio
        for split in ["train", "val", "test"]:
            print(f"\n  Processing {split} split...")

            for sign in sorted(all_signs):
                syn_samples = synthetic_by_sign[sign][split]
                real_samples_sign = real_by_sign[sign][split]

                # Calculate target counts based on ratio
                total_real = len(real_samples_sign)
                if total_real > 0 and self.config.synthetic_ratio < 1.0:
                    # We have real data, calculate synthetic to match ratio
                    target_synthetic = int(total_real * self.config.synthetic_ratio / (1 - self.config.synthetic_ratio))
                else:
                    # No real data or 100% synthetic
                    target_synthetic = len(syn_samples)

                # Apply max samples per class limit
                if self.config.max_samples_per_class:
                    target_synthetic = min(target_synthetic, self.config.max_samples_per_class)
                    total_real = min(total_real, self.config.max_samples_per_class)

                # Sample synthetic data
                if len(syn_samples) > target_synthetic:
                    selected_synthetic = random.sample(syn_samples, target_synthetic)
                else:
                    selected_synthetic = syn_samples

                # Sample real data
                if len(real_samples_sign) > total_real:
                    selected_real = random.sample(real_samples_sign, total_real)
                else:
                    selected_real = real_samples_sign

                # Add to combined
                for sample in selected_synthetic:
                    new_sample = sample.copy()
                    new_sample["source"] = "synthetic"
                    new_sample["split"] = split
                    self.combined_samples.append(new_sample)
                    self.stats["combined"][split][sign] += 1

                for sample in selected_real:
                    new_sample = sample.copy()
                    new_sample["source"] = "real"
                    new_sample["split"] = split
                    self.combined_samples.append(new_sample)
                    self.stats["combined"][split][sign] += 1

        print(f"\n  Total combined samples: {len(self.combined_samples)}")

    def create_output_structure(self):
        """Create output directory structure and copy/link files."""
        print("\nCreating output structure...")

        # Create directories
        for split in ["train", "val", "test"]:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metadata").mkdir(parents=True, exist_ok=True)

        # Process each sample
        for i, sample in enumerate(self.combined_samples):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(self.combined_samples)} samples...")

            split = sample["split"]
            sign = sample["handshape"]
            source = sample["source"]

            # Determine source and destination paths
            if source == "synthetic":
                src_dir = self.synthetic_dir
            else:
                src_dir = self.real_dir

            src_path = src_dir / sample["filepath"]
            dst_dir = self.output_dir / split / sign
            dst_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique filename
            dst_filename = f"{source}_{sample['id']}.png"
            dst_path = dst_dir / dst_filename

            # Copy or link
            if src_path.exists():
                if self.config.copy_files:
                    shutil.copy2(src_path, dst_path)
                else:
                    # Create relative symlink
                    rel_path = os.path.relpath(src_path, dst_dir)
                    dst_path.symlink_to(rel_path)

                # Update sample filepath
                sample["filepath"] = str(dst_path.relative_to(self.output_dir))
            else:
                print(f"  Warning: Source file not found: {src_path}")

    def save_metadata(self):
        """Save combined metadata and statistics."""
        print("\nSaving metadata...")

        # Combined metadata
        with open(self.output_dir / "all_samples.json", 'w') as f:
            json.dump({
                "version": "2.0",
                "generator": "SonZo AI Data Merger",
                "config": asdict(self.config),
                "total_samples": len(self.combined_samples),
                "samples": self.combined_samples
            }, f, indent=2)

        # Split-specific metadata
        for split in ["train", "val", "test"]:
            split_samples = [s for s in self.combined_samples if s["split"] == split]
            with open(self.output_dir / "metadata" / f"{split}_samples.json", 'w') as f:
                json.dump({
                    "split": split,
                    "total_samples": len(split_samples),
                    "samples": split_samples
                }, f, indent=2)

        # Label map
        signs = sorted(set(s["handshape"] for s in self.combined_samples))
        label_map = {sign: idx for idx, sign in enumerate(signs)}
        with open(self.output_dir / "label_map.json", 'w') as f:
            json.dump(label_map, f, indent=2)

        # Statistics
        with open(self.output_dir / "metadata" / "statistics.json", 'w') as f:
            json.dump({
                "synthetic": dict(self.stats["synthetic"]),
                "real": dict(self.stats["real"]),
                "combined": dict(self.stats["combined"])
            }, f, indent=2)

    def print_summary(self):
        """Print merge summary."""
        print("\n" + "=" * 60)
        print("MERGE COMPLETE")
        print("=" * 60)

        print(f"\nOutput directory: {self.output_dir}")
        print(f"Total samples: {len(self.combined_samples)}")

        # Count by source
        n_synthetic = sum(1 for s in self.combined_samples if s["source"] == "synthetic")
        n_real = sum(1 for s in self.combined_samples if s["source"] == "real")

        print(f"\nSource distribution:")
        print(f"  Synthetic: {n_synthetic} ({100*n_synthetic/len(self.combined_samples):.1f}%)")
        print(f"  Real: {n_real} ({100*n_real/len(self.combined_samples):.1f}%)")

        print(f"\nSplit distribution:")
        for split in ["train", "val", "test"]:
            count = sum(1 for s in self.combined_samples if s["split"] == split)
            print(f"  {split}: {count}")

        # Signs coverage
        signs = set(s["handshape"] for s in self.combined_samples)
        print(f"\nTotal signs: {len(signs)}")

    def run(self):
        """Execute the merge process."""
        print("=" * 60)
        print("SonZo AI - Data Merger")
        print("=" * 60)
        print(f"Synthetic ratio: {self.config.synthetic_ratio * 100:.0f}%")
        print(f"Real ratio: {(1 - self.config.synthetic_ratio) * 100:.0f}%")

        self.load_synthetic_data()
        self.load_real_data()
        self.merge_datasets()
        self.create_output_structure()
        self.save_metadata()
        self.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description='Merge synthetic and real ASL datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge with default 80% synthetic, 20% real
  python scripts/merge_data.py --synthetic ./synthetic_data --real ./real_data --output ./combined

  # Custom ratio (90% synthetic, 10% real)
  python scripts/merge_data.py --synthetic ./syn --real ./real --output ./out --synthetic-ratio 0.9

  # Use symlinks instead of copying
  python scripts/merge_data.py --synthetic ./syn --real ./real --output ./out --link
"""
    )

    parser.add_argument('--synthetic', type=str, required=True,
                       help='Path to synthetic data directory')
    parser.add_argument('--real', type=str, default='',
                       help='Path to real data directory (optional)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for combined dataset')
    parser.add_argument('--synthetic-ratio', type=float, default=0.8,
                       help='Ratio of synthetic data (default: 0.8)')
    parser.add_argument('--max-per-class', type=int,
                       help='Maximum samples per class')
    parser.add_argument('--link', action='store_true',
                       help='Create symlinks instead of copying files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    config = MergeConfig(
        synthetic_dir=args.synthetic,
        real_dir=args.real,
        output_dir=args.output,
        synthetic_ratio=args.synthetic_ratio,
        max_samples_per_class=args.max_per_class,
        copy_files=not args.link,
        seed=args.seed
    )

    merger = DataMerger(config)
    merger.run()


if __name__ == "__main__":
    main()
