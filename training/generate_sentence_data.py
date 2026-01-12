#!/usr/bin/env python3
"""
Sentence-Level Sign Language Data Generator
============================================
Generate synthetic training data for continuous sign language recognition.

Key Features:
- Realistic sentence structures following ASL grammar
- Coarticulation simulation (hand shape blending between signs)
- Natural timing variation (prosody)
- Transition frame generation

This generates data that looks like natural continuous signing,
not just concatenated isolated signs.

Usage:
    python training/generate_sentence_data.py \
        --isolated-data ./combined_data \
        --output ./sentence_data \
        --num-sentences 10000

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import argparse
import json
import os
import sys
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# ASL SENTENCE PATTERNS
# ==============================================================================

# ASL grammar differs from English:
# - Topic-Comment structure (often OSV instead of SVO)
# - Questions use facial markers, word order changes
# - Time references come first
# - Pronouns can be spatial pointing

SENTENCE_TEMPLATES = {
    "greeting": [
        ["HELLO"],
        ["HELLO", "HOW", "YOU"],
        ["HELLO", "NICE", "MEET", "YOU"],
        ["GOODBYE"],
        ["GOODBYE", "SEE", "YOU"],
    ],
    "question_wh": [
        ["WHAT", "NAME", "YOU"],
        ["WHERE", "YOU", "LIVE"],
        ["HOW", "YOU"],
        ["WHAT", "YOU", "WANT"],
        ["WHO", "THAT"],
        ["WHEN", "YOU", "GO"],
        ["WHY", "YOU", "SAD"],
    ],
    "question_yn": [
        ["YOU", "UNDERSTAND"],
        ["YOU", "WANT", "HELP"],
        ["YOU", "HUNGRY"],
        ["YOU", "FINISH"],
        ["YOU", "LIKE"],
    ],
    "statement": [
        ["I", "UNDERSTAND"],
        ["I", "LOVE", "YOU"],
        ["I", "WANT", "LEARN"],
        ["I", "NEED", "HELP"],
        ["I", "HAPPY"],
        ["I", "SAD"],
        ["THANK_YOU"],
        ["PLEASE", "HELP"],
        ["SORRY"],
    ],
    "response": [
        ["YES"],
        ["NO"],
        ["MAYBE"],
        ["I", "KNOW"],
        ["I", "NOT", "KNOW"],
        ["I", "UNDERSTAND"],
        ["I", "NOT", "UNDERSTAND"],
    ],
    "instruction": [
        ["PLEASE", "WAIT"],
        ["PLEASE", "HELP"],
        ["FINISH"],
        ["AGAIN"],
        ["SLOW"],
    ],
}

# Sign categories for generating varied sentences
SIGN_CATEGORIES = {
    "pronouns": ["I", "YOU", "HE", "SHE", "WE", "THEY"],
    "greetings": ["HELLO", "GOODBYE", "THANK_YOU", "PLEASE", "SORRY"],
    "questions": ["WHAT", "WHERE", "WHO", "WHY", "HOW", "WHEN"],
    "feelings": ["HAPPY", "SAD", "LOVE", "LIKE", "WANT", "NEED"],
    "actions": ["UNDERSTAND", "KNOW", "LEARN", "HELP", "FINISH", "WAIT", "GO", "SEE", "MEET"],
    "modifiers": ["NOT", "AGAIN", "SLOW", "NICE"],
    "responses": ["YES", "NO", "MAYBE"],
}


# ==============================================================================
# COARTICULATION MODEL
# ==============================================================================

@dataclass
class HandConfig:
    """Configuration of hand shape/position for a sign."""
    # Simplified hand state (in real system, would be detailed joint angles)
    openness: float = 0.5  # 0=closed fist, 1=open hand
    thumb_out: bool = False
    index_extended: bool = False
    position_x: float = 0.5  # Relative position in frame
    position_y: float = 0.5
    orientation: float = 0.0  # Rotation angle


# Simplified hand configurations for common signs
# In production, these would come from motion capture or detailed annotation
SIGN_HAND_CONFIGS = {
    "HELLO": HandConfig(openness=1.0, thumb_out=True, position_x=0.7, position_y=0.3),
    "GOODBYE": HandConfig(openness=0.8, thumb_out=True, position_x=0.6, position_y=0.4),
    "YES": HandConfig(openness=0.0, thumb_out=False, position_x=0.5, position_y=0.5),
    "NO": HandConfig(openness=0.3, thumb_out=True, index_extended=True, position_x=0.5, position_y=0.5),
    "THANK_YOU": HandConfig(openness=0.7, position_x=0.5, position_y=0.6),
    "PLEASE": HandConfig(openness=0.8, position_x=0.5, position_y=0.7),
    "I": HandConfig(openness=0.0, index_extended=True, position_x=0.5, position_y=0.7),
    "YOU": HandConfig(openness=0.0, index_extended=True, position_x=0.6, position_y=0.5),
    "LOVE": HandConfig(openness=0.0, position_x=0.5, position_y=0.7),
    "UNDERSTAND": HandConfig(openness=0.0, index_extended=True, position_x=0.7, position_y=0.3),
    "WHAT": HandConfig(openness=0.8, position_x=0.5, position_y=0.5),
    "WHERE": HandConfig(openness=0.0, index_extended=True, position_x=0.6, position_y=0.4),
    "WHO": HandConfig(openness=0.3, index_extended=True, position_x=0.5, position_y=0.5),
    "HOW": HandConfig(openness=0.5, position_x=0.5, position_y=0.5),
    # Default for unknown signs
    "_DEFAULT": HandConfig(openness=0.5, position_x=0.5, position_y=0.5),
}


def get_hand_config(sign: str) -> HandConfig:
    """Get hand configuration for a sign."""
    return SIGN_HAND_CONFIGS.get(sign, SIGN_HAND_CONFIGS["_DEFAULT"])


def interpolate_hand_configs(
    config1: HandConfig,
    config2: HandConfig,
    t: float
) -> HandConfig:
    """
    Interpolate between two hand configurations.

    This simulates coarticulation - the hand transitioning between signs
    without returning to a neutral position.

    Args:
        config1: Starting hand configuration
        config2: Ending hand configuration
        t: Interpolation parameter (0=config1, 1=config2)

    Returns:
        Interpolated hand configuration
    """
    # Use ease-in-out for natural motion
    t = 3 * t**2 - 2 * t**3  # Smoothstep

    return HandConfig(
        openness=config1.openness * (1 - t) + config2.openness * t,
        thumb_out=config1.thumb_out if t < 0.5 else config2.thumb_out,
        index_extended=config1.index_extended if t < 0.5 else config2.index_extended,
        position_x=config1.position_x * (1 - t) + config2.position_x * t,
        position_y=config1.position_y * (1 - t) + config2.position_y * t,
        orientation=config1.orientation * (1 - t) + config2.orientation * t,
    )


# ==============================================================================
# SENTENCE DATA GENERATOR
# ==============================================================================

class SentenceDataGenerator:
    """
    Generate sentence-level training data from isolated sign data.

    Creates realistic continuous signing by:
    1. Concatenating isolated sign clips
    2. Generating transition frames (coarticulation)
    3. Adding timing variation (prosody)
    4. Applying augmentation
    """

    def __init__(
        self,
        isolated_data_dir: Path,
        output_dir: Path,
        fps: int = 30,
        transition_frames: int = 5,  # Frames for coarticulation
        hold_variation: Tuple[int, int] = (0, 3),  # Random frames to add/remove
    ):
        self.isolated_data_dir = Path(isolated_data_dir)
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.transition_frames = transition_frames
        self.hold_variation = hold_variation

        # Load isolated sign data
        self._load_isolated_data()

        # Create output structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "sentences").mkdir(exist_ok=True)

    def _load_isolated_data(self):
        """Load metadata about isolated sign clips."""
        # Load label map
        label_path = self.isolated_data_dir / "label_map.json"
        with open(label_path, 'r') as f:
            self.label_map = json.load(f)

        self.inverse_label_map = {v: k for k, v in self.label_map.items()}

        # Find all frame directories for each sign
        self.sign_frames = {}  # sign -> list of frame paths

        for sign in self.label_map.keys():
            sign_dir = self.isolated_data_dir / sign
            if sign_dir.exists():
                frames = sorted(sign_dir.glob("*.png")) + sorted(sign_dir.glob("*.jpg"))
                if frames:
                    self.sign_frames[sign] = frames

        # Also check for grouped frame files
        samples_path = self.isolated_data_dir / "metadata" / "train_samples.json"
        if samples_path.exists():
            with open(samples_path, 'r') as f:
                data = json.load(f)

            for sample in data.get("samples", []):
                sign = sample.get("handshape", "")
                if sign and sign not in self.sign_frames:
                    self.sign_frames[sign] = []

                if sign:
                    frame_path = self.isolated_data_dir / sample.get("filepath", "")
                    if frame_path.exists():
                        self.sign_frames[sign].append(frame_path)

        logger.info(f"Loaded {len(self.sign_frames)} signs with frame data")

    def get_sign_frames(self, sign: str, num_frames: int = 16) -> List[Path]:
        """
        Get frame paths for a sign.

        Args:
            sign: Sign name
            num_frames: Target number of frames

        Returns:
            List of frame paths
        """
        if sign not in self.sign_frames or not self.sign_frames[sign]:
            logger.warning(f"No frames for sign: {sign}")
            return []

        frames = self.sign_frames[sign]

        # Sample or repeat to get target frames
        if len(frames) >= num_frames:
            # Uniform sampling
            indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            return [frames[i] for i in indices]
        else:
            # Repeat frames
            repeated = []
            while len(repeated) < num_frames:
                repeated.extend(frames)
            return repeated[:num_frames]

    def generate_transition_frames(
        self,
        end_frame: np.ndarray,
        start_frame: np.ndarray,
        sign_from: str,
        sign_to: str,
        num_frames: int = 5
    ) -> List[np.ndarray]:
        """
        Generate transition frames between two signs.

        This simulates coarticulation by blending:
        1. Image content (cross-fade)
        2. Hand position (interpolation)
        3. Motion blur (optional)

        Args:
            end_frame: Last frame of previous sign
            start_frame: First frame of next sign
            sign_from: Previous sign
            sign_to: Next sign
            num_frames: Number of transition frames

        Returns:
            List of transition frame images
        """
        transition_frames = []

        # Get hand configurations for coarticulation modeling
        config_from = get_hand_config(sign_from)
        config_to = get_hand_config(sign_to)

        for i in range(num_frames):
            t = (i + 1) / (num_frames + 1)  # Progress 0 -> 1

            # Interpolate hand config (for future use with synthetic hands)
            _ = interpolate_hand_configs(config_from, config_to, t)

            # Simple cross-fade for now
            # In production, would use optical flow or warping
            alpha = t
            blended = cv2.addWeighted(
                end_frame.astype(np.float32),
                1 - alpha,
                start_frame.astype(np.float32),
                alpha,
                0
            ).astype(np.uint8)

            # Add slight motion blur for natural look
            if i == num_frames // 2:
                kernel_size = 3
                blended = cv2.GaussianBlur(blended, (kernel_size, kernel_size), 0)

            transition_frames.append(blended)

        return transition_frames

    def generate_sentence(
        self,
        signs: List[str],
        sentence_id: str,
        frames_per_sign: int = 20
    ) -> Dict[str, Any]:
        """
        Generate a complete sentence video from sign list.

        Args:
            signs: List of sign names
            sentence_id: Unique identifier
            frames_per_sign: Base frames per sign

        Returns:
            Sentence metadata dict
        """
        output_dir = self.output_dir / "sentences" / sentence_id
        output_dir.mkdir(parents=True, exist_ok=True)

        all_frames = []
        frame_annotations = []  # Track which sign each frame belongs to

        prev_frame = None
        prev_sign = None

        for sign_idx, sign in enumerate(signs):
            # Add timing variation (prosody)
            variation = random.randint(*self.hold_variation)
            actual_frames = max(8, frames_per_sign + variation)

            # Get frames for this sign
            sign_frame_paths = self.get_sign_frames(sign, actual_frames)

            if not sign_frame_paths:
                logger.warning(f"Skipping sign with no frames: {sign}")
                continue

            # Load sign frames
            sign_frames = []
            for path in sign_frame_paths:
                img = cv2.imread(str(path))
                if img is not None:
                    sign_frames.append(img)

            if not sign_frames:
                continue

            # Generate transition from previous sign
            if prev_frame is not None and prev_sign is not None:
                transitions = self.generate_transition_frames(
                    prev_frame,
                    sign_frames[0],
                    prev_sign,
                    sign,
                    self.transition_frames
                )

                # Add transition frames (labeled as "transition")
                for t_frame in transitions:
                    all_frames.append(t_frame)
                    frame_annotations.append({
                        "type": "transition",
                        "from": prev_sign,
                        "to": sign
                    })

            # Add sign frames
            for frame in sign_frames:
                all_frames.append(frame)
                frame_annotations.append({
                    "type": "sign",
                    "sign": sign,
                    "sign_idx": sign_idx
                })

            prev_frame = sign_frames[-1]
            prev_sign = sign

        # Save frames
        for i, frame in enumerate(all_frames):
            frame_path = output_dir / f"frame_{i:05d}.png"
            cv2.imwrite(str(frame_path), frame)

        # Create metadata
        metadata = {
            "id": sentence_id,
            "signs": signs,
            "frames_dir": f"sentences/{sentence_id}/",
            "num_frames": len(all_frames),
            "fps": self.fps,
            "duration_sec": len(all_frames) / self.fps,
            "frame_annotations": frame_annotations
        }

        return metadata

    def generate_dataset(
        self,
        num_sentences: int = 1000,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Dict[str, Any]:
        """
        Generate complete sentence dataset.

        Args:
            num_sentences: Total sentences to generate
            train_ratio: Proportion for training
            val_ratio: Proportion for validation

        Returns:
            Dataset statistics
        """
        all_sentences = []

        # Generate sentences from templates
        templates_flat = []
        for category, templates in SENTENCE_TEMPLATES.items():
            for template in templates:
                templates_flat.append((category, template))

        logger.info(f"Generating {num_sentences} sentences...")

        for i in tqdm(range(num_sentences)):
            # Randomly select template or generate random
            if random.random() < 0.7:
                # Use template
                category, template = random.choice(templates_flat)
                signs = list(template)
            else:
                # Generate random sentence
                signs = self._generate_random_sentence()

            # Filter to available signs
            signs = [s for s in signs if s in self.sign_frames]

            if not signs:
                continue

            # Generate sentence
            sentence_id = f"sent_{i:06d}"
            try:
                metadata = self.generate_sentence(signs, sentence_id)
                all_sentences.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to generate {sentence_id}: {e}")

        # Split into train/val/test
        random.shuffle(all_sentences)
        n_train = int(len(all_sentences) * train_ratio)
        n_val = int(len(all_sentences) * val_ratio)

        train_sentences = all_sentences[:n_train]
        val_sentences = all_sentences[n_train:n_train + n_val]
        test_sentences = all_sentences[n_train + n_val:]

        # Add split labels
        for s in train_sentences:
            s["split"] = "train"
        for s in val_sentences:
            s["split"] = "val"
        for s in test_sentences:
            s["split"] = "test"

        # Save metadata
        self._save_metadata(train_sentences, "train_sentences.json")
        self._save_metadata(val_sentences, "val_sentences.json")
        self._save_metadata(test_sentences, "test_sentences.json")

        # Save vocabulary
        self._save_vocabulary()

        stats = {
            "total_sentences": len(all_sentences),
            "train": len(train_sentences),
            "val": len(val_sentences),
            "test": len(test_sentences),
            "vocab_size": len(self.sign_frames)
        }

        logger.info(f"Generated dataset: {stats}")
        return stats

    def _generate_random_sentence(self, min_signs: int = 2, max_signs: int = 5) -> List[str]:
        """Generate a random but plausible sentence."""
        length = random.randint(min_signs, max_signs)
        sentence = []

        # Start with subject or greeting
        if random.random() < 0.3:
            sentence.append(random.choice(SIGN_CATEGORIES["greetings"]))
        else:
            sentence.append(random.choice(SIGN_CATEGORIES["pronouns"]))

        # Add content
        for _ in range(length - 1):
            category = random.choice(["feelings", "actions", "modifiers"])
            sign = random.choice(SIGN_CATEGORIES[category])
            if sign not in sentence:  # Avoid immediate repeats
                sentence.append(sign)

        return sentence

    def _save_metadata(self, sentences: List[Dict], filename: str):
        """Save sentence metadata to JSON."""
        path = self.output_dir / filename
        with open(path, 'w') as f:
            json.dump({"sentences": sentences}, f, indent=2)
        logger.info(f"Saved {len(sentences)} sentences to {path}")

    def _save_vocabulary(self):
        """Save vocabulary mapping."""
        # Index 0 reserved for CTC blank
        sign_to_idx = {"<blank>": 0}
        for i, sign in enumerate(sorted(self.sign_frames.keys())):
            sign_to_idx[sign] = i + 1

        vocab_path = self.output_dir / "vocabulary.json"
        with open(vocab_path, 'w') as f:
            json.dump({
                "sign_to_idx": sign_to_idx,
                "idx_to_sign": {v: k for k, v in sign_to_idx.items()},
                "vocab_size": len(sign_to_idx)
            }, f, indent=2)
        logger.info(f"Saved vocabulary ({len(sign_to_idx)} signs) to {vocab_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate sentence-level SLR training data')

    parser.add_argument('--isolated-data', type=str, default='./combined_data',
                       help='Path to isolated sign data')
    parser.add_argument('--output', type=str, default='./sentence_data',
                       help='Output directory')
    parser.add_argument('--num-sentences', type=int, default=1000,
                       help='Number of sentences to generate')
    parser.add_argument('--frames-per-sign', type=int, default=20,
                       help='Target frames per sign')
    parser.add_argument('--transition-frames', type=int, default=5,
                       help='Frames for coarticulation transitions')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio')

    args = parser.parse_args()

    generator = SentenceDataGenerator(
        isolated_data_dir=args.isolated_data,
        output_dir=args.output,
        transition_frames=args.transition_frames
    )

    stats = generator.generate_dataset(
        num_sentences=args.num_sentences,
        train_ratio=args.train_ratio
    )

    print("\n" + "=" * 60)
    print("Dataset Generation Complete")
    print("=" * 60)
    print(f"Total sentences: {stats['total_sentences']}")
    print(f"  Train: {stats['train']}")
    print(f"  Val: {stats['val']}")
    print(f"  Test: {stats['test']}")
    print(f"Vocabulary: {stats['vocab_size']} signs")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
