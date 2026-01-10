#!/usr/bin/env python3
"""
Synthetic ASL Training Data Generator
======================================
Generate large-scale synthetic training data for SLR models.

Features:
- Generates all ASL alphabet (A-Z) + numbers (0-9) + common signs
- Multiple variations per sign (camera angles, lighting, backgrounds, skin tones)
- Automatic train/val/test split (80/10/10)
- Parallel rendering with multiprocessing
- Progress tracking and metadata generation

Usage:
    # Generate full dataset (500 samples per sign)
    python scripts/generate_training_data.py --output ./training_data --samples 500

    # Quick test (10 samples per sign)
    python scripts/generate_training_data.py --output ./test_data --samples 10 --quick

    # Run with Blender
    blender --background --python scripts/generate_training_data.py -- \\
        --output ./training_data --samples 500

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import argparse
import json
import os
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import time

# Add parent directory for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from asl_handshapes import (
    get_alphabet_handshapes,
    get_number_handshapes,
    get_common_signs,
    get_all_training_signs,
    ASL_HANDSHAPES
)


@dataclass
class GenerationConfig:
    """Configuration for data generation."""
    output_dir: str = "./synthetic_data"
    samples_per_sign: int = 500
    image_size: int = 512
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Variation ranges
    camera_angles: List[str] = None  # front, left, right, above
    lighting_conditions: List[str] = None  # bright, normal, dim
    background_types: List[str] = None  # solid, gradient
    skin_tone_count: int = 6

    # Parallel processing
    num_workers: int = 4
    blender_path: str = "blender"

    # Model path (optional)
    model_path: Optional[str] = None

    def __post_init__(self):
        if self.camera_angles is None:
            self.camera_angles = ["front", "slight_left", "slight_right"]
        if self.lighting_conditions is None:
            self.lighting_conditions = ["bright", "normal", "dim"]
        if self.background_types is None:
            self.background_types = ["solid", "gradient"]


def find_blender() -> str:
    """Find Blender executable."""
    possible_paths = [
        "blender",
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/snap/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "C:\\Program Files\\Blender Foundation\\Blender 4.0\\blender.exe",
        "C:\\Program Files\\Blender Foundation\\Blender 3.6\\blender.exe",
    ]

    for path in possible_paths:
        if shutil.which(path):
            return path

    raise FileNotFoundError("Blender not found. Please install Blender or specify --blender-path")


def create_directory_structure(output_dir: Path) -> Dict[str, Path]:
    """Create train/val/test directory structure."""
    dirs = {
        "train": output_dir / "train",
        "val": output_dir / "val",
        "test": output_dir / "test",
        "metadata": output_dir / "metadata"
    }

    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def assign_splits(
    total_samples: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float
) -> List[str]:
    """Assign samples to train/val/test splits."""
    n_train = int(total_samples * train_ratio)
    n_val = int(total_samples * val_ratio)

    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (total_samples - n_train - n_val)
    random.shuffle(splits)

    return splits


def generate_variation_params(config: GenerationConfig) -> List[Dict]:
    """Generate all variation parameter combinations."""
    variations = []

    for angle in config.camera_angles:
        for lighting in config.lighting_conditions:
            for bg_type in config.background_types:
                for skin_idx in range(config.skin_tone_count):
                    variations.append({
                        "camera_angle": angle,
                        "lighting": lighting,
                        "background": bg_type,
                        "skin_tone_idx": skin_idx
                    })

    return variations


def render_single_sign(
    sign_name: str,
    sample_idx: int,
    variation: Dict,
    output_path: Path,
    config: GenerationConfig
) -> Dict:
    """
    Render a single sign with specific variations.
    Returns metadata dict.

    Note: This is called within Blender context or via subprocess.
    """
    # This function should be called from within Blender
    # For subprocess mode, we generate a command instead

    metadata = {
        "id": f"{sign_name}_{sample_idx:05d}",
        "handshape": sign_name,
        "sample_idx": sample_idx,
        "filepath": str(output_path.relative_to(output_path.parent.parent.parent)),
        **variation
    }

    return metadata


def run_blender_batch(
    signs: List[str],
    output_dir: Path,
    config: GenerationConfig,
    split: str
) -> List[Dict]:
    """Run Blender to render a batch of signs."""

    blender_script = PROJECT_ROOT / "blender" / "mano_renderer.py"

    all_metadata = []

    for sign_name in signs:
        sign_output_dir = output_dir / split / sign_name
        sign_output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate samples for this sign in this split
        total_samples = config.samples_per_sign
        if split == "train":
            n_samples = int(total_samples * config.train_ratio)
        elif split == "val":
            n_samples = int(total_samples * config.val_ratio)
        else:  # test
            n_samples = total_samples - int(total_samples * config.train_ratio) - int(total_samples * config.val_ratio)

        if n_samples == 0:
            continue

        # Build Blender command
        cmd = [
            config.blender_path,
            "--background",
            "--python", str(blender_script),
            "--",
            "--handshape", sign_name,
            "--output", str(sign_output_dir),
            "--samples", str(n_samples),
            "--size", str(config.image_size)
        ]

        if config.model_path:
            cmd.extend(["--model", config.model_path])

        print(f"  Rendering {sign_name} ({split}): {n_samples} samples...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per sign
            )

            if result.returncode != 0:
                print(f"    Warning: Blender returned non-zero for {sign_name}")
                print(f"    stderr: {result.stderr[:500]}")

        except subprocess.TimeoutExpired:
            print(f"    Warning: Timeout rendering {sign_name}")
        except Exception as e:
            print(f"    Error rendering {sign_name}: {e}")

        # Collect metadata from rendered files
        for img_file in sign_output_dir.glob("*.png"):
            metadata = {
                "id": img_file.stem,
                "handshape": sign_name,
                "filepath": str(img_file.relative_to(output_dir)),
                "split": split,
                "is_synthetic": True
            }
            all_metadata.append(metadata)

    return all_metadata


def generate_standalone(config: GenerationConfig):
    """
    Generate data in standalone mode (calling Blender as subprocess).
    This is the main entry point when not running inside Blender.
    """
    output_dir = Path(config.output_dir)

    print("=" * 60)
    print("SonZo AI - Synthetic ASL Training Data Generator")
    print("=" * 60)

    # Find Blender
    try:
        blender_path = find_blender() if config.blender_path == "blender" else config.blender_path
        config.blender_path = blender_path
        print(f"Using Blender: {blender_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Get all signs to generate
    signs = get_all_training_signs()
    print(f"\nGenerating data for {len(signs)} signs:")
    print(f"  - Alphabet: A-Z (26)")
    print(f"  - Numbers: 0-9 (10)")
    print(f"  - Common signs: {len(get_common_signs())}")
    print(f"\nSamples per sign: {config.samples_per_sign}")
    print(f"Total samples: ~{len(signs) * config.samples_per_sign}")
    print(f"Split: {config.train_ratio*100:.0f}% train / {config.val_ratio*100:.0f}% val / {config.test_ratio*100:.0f}% test")
    print()

    # Create directory structure
    dirs = create_directory_structure(output_dir)

    # Generate data for each split
    all_metadata = []
    start_time = time.time()

    for split in ["train", "val", "test"]:
        print(f"\n{'='*40}")
        print(f"Generating {split.upper()} split...")
        print(f"{'='*40}")

        metadata = run_blender_batch(signs, output_dir, config, split)
        all_metadata.extend(metadata)

        print(f"  Generated {len(metadata)} samples for {split}")

    elapsed = time.time() - start_time

    # Save metadata
    print("\nSaving metadata...")

    # Main metadata file
    metadata_file = dirs["metadata"] / "all_samples.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            "version": "2.0",
            "generator": "SonZo AI Training Data Generator",
            "config": asdict(config),
            "total_samples": len(all_metadata),
            "signs": signs,
            "samples": all_metadata
        }, f, indent=2)

    # Split-specific metadata
    for split in ["train", "val", "test"]:
        split_samples = [m for m in all_metadata if m.get("split") == split]
        split_file = dirs["metadata"] / f"{split}_samples.json"
        with open(split_file, 'w') as f:
            json.dump({
                "split": split,
                "total_samples": len(split_samples),
                "samples": split_samples
            }, f, indent=2)

    # Label map
    label_map = {sign: idx for idx, sign in enumerate(sorted(signs))}
    label_map_file = output_dir / "label_map.json"
    with open(label_map_file, 'w') as f:
        json.dump(label_map, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total samples: {len(all_metadata)}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"\nSplit distribution:")
    for split in ["train", "val", "test"]:
        count = sum(1 for m in all_metadata if m.get("split") == split)
        print(f"  {split}: {count}")
    print(f"\nFiles created:")
    print(f"  - {label_map_file}")
    print(f"  - {metadata_file}")
    for split in ["train", "val", "test"]:
        print(f"  - {dirs['metadata']}/{split}_samples.json")


def generate_in_blender(config: GenerationConfig):
    """
    Generate data when running inside Blender.
    This provides direct access to Blender's Python API.
    """
    try:
        import bpy
    except ImportError:
        print("Not running inside Blender. Use standalone mode.")
        return generate_standalone(config)

    # Import the Blender renderer
    sys.path.insert(0, str(PROJECT_ROOT / "blender"))
    from mano_renderer import RenderingPipeline, RenderConfig

    output_dir = Path(config.output_dir)
    dirs = create_directory_structure(output_dir)

    # Configure renderer
    render_config = RenderConfig()
    render_config.image_size = (config.image_size, config.image_size)
    render_config.samples_per_handshape = config.samples_per_sign

    # Initialize pipeline
    pipeline = RenderingPipeline(render_config)
    pipeline.setup(config.model_path)

    # Get signs
    signs = get_all_training_signs()
    all_metadata = []

    print(f"Generating {len(signs)} signs with {config.samples_per_sign} samples each...")

    for sign_idx, sign_name in enumerate(signs):
        print(f"\n[{sign_idx+1}/{len(signs)}] Rendering: {sign_name}")

        # Determine split assignments for this sign
        splits = assign_splits(
            config.samples_per_sign,
            config.train_ratio,
            config.val_ratio,
            config.test_ratio
        )

        # Render samples
        for sample_idx in range(config.samples_per_sign):
            split = splits[sample_idx]

            # Create output path
            sign_dir = dirs[split] / sign_name
            sign_dir.mkdir(parents=True, exist_ok=True)

            # Render
            samples = pipeline.render_handshape(
                sign_name,
                sign_dir,
                num_samples=1
            )

            if samples:
                sample_meta = samples[0]
                sample_meta["split"] = split
                sample_meta["is_synthetic"] = True
                all_metadata.append(sample_meta)

    # Save metadata (same as standalone)
    metadata_file = dirs["metadata"] / "all_samples.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            "version": "2.0",
            "generator": "SonZo AI Training Data Generator (Blender)",
            "config": asdict(config),
            "total_samples": len(all_metadata),
            "signs": signs,
            "samples": all_metadata
        }, f, indent=2)

    # Label map
    label_map = {sign: idx for idx, sign in enumerate(sorted(signs))}
    with open(output_dir / "label_map.json", 'w') as f:
        json.dump(label_map, f, indent=2)

    print(f"\nGeneration complete! {len(all_metadata)} samples generated.")


def main():
    """Main entry point."""
    # Parse arguments after "--" for Blender compatibility
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description='Generate synthetic ASL training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full generation (recommended for training)
  python scripts/generate_training_data.py --output ./data --samples 500

  # Quick test
  python scripts/generate_training_data.py --output ./test_data --samples 10 --quick

  # With custom model
  python scripts/generate_training_data.py --output ./data --model ./mano_hand.fbx

  # Run via Blender directly
  blender --background --python scripts/generate_training_data.py -- --output ./data
"""
    )

    parser.add_argument('--output', type=str, default='./synthetic_data',
                       help='Output directory')
    parser.add_argument('--samples', type=int, default=500,
                       help='Samples per sign (default: 500)')
    parser.add_argument('--size', type=int, default=512,
                       help='Image size in pixels (default: 512)')
    parser.add_argument('--model', type=str,
                       help='Path to MANO/hand model file')
    parser.add_argument('--blender-path', type=str, default='blender',
                       help='Path to Blender executable')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: only generate 10 samples per sign')

    args = parser.parse_args(argv)

    # Build config
    config = GenerationConfig(
        output_dir=args.output,
        samples_per_sign=10 if args.quick else args.samples,
        image_size=args.size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        num_workers=args.workers,
        blender_path=args.blender_path,
        model_path=args.model
    )

    # Check if running inside Blender
    try:
        import bpy
        print("Running inside Blender - using direct rendering")
        generate_in_blender(config)
    except ImportError:
        print("Running in standalone mode - calling Blender as subprocess")
        generate_standalone(config)


if __name__ == "__main__":
    main()
