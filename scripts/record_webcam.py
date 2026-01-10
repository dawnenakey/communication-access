#!/usr/bin/env python3
"""
Webcam Sign Language Recorder
==============================
Record real ASL signs via webcam with live preview and instant labeling.

Features:
- Live webcam preview
- Key-press labeling (press 'A' for letter A, etc.)
- Records 2-3 second clips per sign
- Extracts frames in same format as Blender output
- Applies data augmentation (flip, rotate, brightness)
- Saves to matching folder structure

Controls:
- A-Z: Record that letter sign
- 0-9: Record that number
- Space: Start/stop generic recording
- Q: Quit
- H: Show help overlay

Usage:
    python scripts/record_webcam.py --output ./real_data

    # Record specific signs only
    python scripts/record_webcam.py --output ./real_data --signs A,B,C,HELLO,THANK_YOU

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
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
import queue

# Add parent directory for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: OpenCV and NumPy are required.")
    print("Install with: pip install opencv-python numpy")
    sys.exit(1)

from asl_handshapes import get_all_training_signs, ASL_HANDSHAPES


@dataclass
class RecordingConfig:
    """Configuration for recording."""
    output_dir: str = "./real_data"
    clip_duration: float = 2.5  # seconds
    fps: int = 30
    frame_size: Tuple[int, int] = (512, 512)
    camera_id: int = 0

    # Augmentation settings
    enable_augmentation: bool = True
    augmentation_factor: int = 3  # Generate 3x augmented versions

    # Frame extraction
    frames_per_clip: int = 16  # Extract N frames per clip

    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


class DataAugmentor:
    """Apply augmentations to recorded frames."""

    @staticmethod
    def horizontal_flip(image: np.ndarray) -> np.ndarray:
        """Mirror flip the image."""
        return cv2.flip(image, 1)

    @staticmethod
    def rotate(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle degrees."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))

    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust brightness by factor (1.0 = no change)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust contrast by factor."""
        mean = np.mean(image)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

    @staticmethod
    def add_noise(image: np.ndarray, sigma: float = 10) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, sigma, image.shape).astype(np.int16)
        return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def random_crop_pad(image: np.ndarray, max_shift: int = 20) -> np.ndarray:
        """Random translation via crop and pad."""
        h, w = image.shape[:2]
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)

        # Create output image
        result = np.zeros_like(image)

        # Calculate source and destination regions
        src_x1 = max(0, dx)
        src_y1 = max(0, dy)
        src_x2 = min(w, w + dx)
        src_y2 = min(h, h + dy)

        dst_x1 = max(0, -dx)
        dst_y1 = max(0, -dy)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        result[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
        return result

    def augment(self, image: np.ndarray, aug_type: str = "random") -> np.ndarray:
        """Apply augmentation based on type."""
        if aug_type == "flip":
            return self.horizontal_flip(image)
        elif aug_type == "rotate_left":
            return self.rotate(image, random.uniform(5, 15))
        elif aug_type == "rotate_right":
            return self.rotate(image, random.uniform(-15, -5))
        elif aug_type == "bright":
            return self.adjust_brightness(image, random.uniform(1.1, 1.3))
        elif aug_type == "dark":
            return self.adjust_brightness(image, random.uniform(0.7, 0.9))
        elif aug_type == "contrast":
            return self.adjust_contrast(image, random.uniform(1.1, 1.3))
        elif aug_type == "noise":
            return self.add_noise(image, random.uniform(5, 15))
        elif aug_type == "shift":
            return self.random_crop_pad(image)
        elif aug_type == "random":
            # Apply random combination
            img = image.copy()
            if random.random() > 0.5:
                img = self.rotate(img, random.uniform(-10, 10))
            if random.random() > 0.5:
                img = self.adjust_brightness(img, random.uniform(0.8, 1.2))
            if random.random() > 0.3:
                img = self.random_crop_pad(img, 10)
            return img
        return image


class WebcamRecorder:
    """Main webcam recording class."""

    def __init__(self, config: RecordingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.augmentor = DataAugmentor()

        # Create output directories
        self.dirs = self._create_directories()

        # Recording state
        self.is_recording = False
        self.current_sign = None
        self.recorded_frames: List[np.ndarray] = []
        self.recording_start_time = 0

        # Session metadata
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.all_metadata: List[Dict] = []

        # Key mappings
        self.key_to_sign = self._build_key_mapping()

        # Camera
        self.cap = None

    def _create_directories(self) -> Dict[str, Path]:
        """Create output directory structure."""
        dirs = {
            "train": self.output_dir / "train",
            "val": self.output_dir / "val",
            "test": self.output_dir / "test",
            "clips": self.output_dir / "clips",  # Raw video clips
            "metadata": self.output_dir / "metadata"
        }

        for name, path in dirs.items():
            path.mkdir(parents=True, exist_ok=True)

        return dirs

    def _build_key_mapping(self) -> Dict[int, str]:
        """Build keyboard key to sign mapping."""
        mapping = {}

        # Letters A-Z (ASCII 97-122 for lowercase, 65-90 for uppercase)
        for i in range(26):
            letter = chr(ord('A') + i)
            mapping[ord('a') + i] = letter  # lowercase
            mapping[ord('A') + i] = letter  # uppercase

        # Numbers 0-9
        for i in range(10):
            mapping[ord('0') + i] = str(i)

        # Special signs via function keys or shortcuts
        # F1-F12 for common signs (handled separately)

        return mapping

    def _assign_split(self) -> str:
        """Randomly assign a sample to train/val/test split."""
        r = random.random()
        if r < self.config.train_ratio:
            return "train"
        elif r < self.config.train_ratio + self.config.val_ratio:
            return "val"
        else:
            return "test"

    def _extract_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Extract evenly spaced frames from a clip."""
        if len(frames) == 0:
            return []

        n_frames = min(self.config.frames_per_clip, len(frames))
        indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)

        return [frames[i] for i in indices]

    def _save_frames(
        self,
        frames: List[np.ndarray],
        sign_name: str,
        sample_id: str,
        split: str,
        is_augmented: bool = False
    ) -> List[Dict]:
        """Save frames to disk and return metadata."""
        metadata = []

        sign_dir = self.dirs[split] / sign_name
        sign_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            # Resize to target size
            frame_resized = cv2.resize(frame, self.config.frame_size)

            # Generate filename
            aug_suffix = "_aug" if is_augmented else ""
            filename = f"{sample_id}{aug_suffix}_f{i:03d}.png"
            filepath = sign_dir / filename

            # Save
            cv2.imwrite(str(filepath), frame_resized)

            # Metadata
            meta = {
                "id": f"{sample_id}{aug_suffix}_f{i:03d}",
                "handshape": sign_name,
                "filepath": str(filepath.relative_to(self.output_dir)),
                "split": split,
                "frame_idx": i,
                "is_synthetic": False,
                "is_augmented": is_augmented,
                "session_id": self.session_id
            }
            metadata.append(meta)

        return metadata

    def _process_recording(self, sign_name: str, frames: List[np.ndarray]):
        """Process a completed recording."""
        if len(frames) < 10:
            print(f"Recording too short ({len(frames)} frames), discarding...")
            return

        sample_id = f"{sign_name}_{self.session_id}_{len(self.all_metadata):04d}"
        split = self._assign_split()

        print(f"\nProcessing {sign_name} recording ({len(frames)} frames)...")
        print(f"  Split: {split}")

        # Extract key frames
        key_frames = self._extract_frames(frames)
        print(f"  Extracted {len(key_frames)} key frames")

        # Save original frames
        metadata = self._save_frames(key_frames, sign_name, sample_id, split)
        self.all_metadata.extend(metadata)
        print(f"  Saved {len(metadata)} original frames")

        # Apply augmentations
        if self.config.enable_augmentation:
            aug_types = ["random", "flip", "bright", "dark"]

            for aug_idx in range(self.config.augmentation_factor):
                aug_type = random.choice(aug_types)
                aug_frames = [self.augmentor.augment(f, aug_type) for f in key_frames]

                aug_sample_id = f"{sample_id}_a{aug_idx}"
                aug_metadata = self._save_frames(
                    aug_frames, sign_name, aug_sample_id, split, is_augmented=True
                )
                self.all_metadata.extend(aug_metadata)

            print(f"  Generated {self.config.augmentation_factor} augmented versions")

        # Save raw clip (optional)
        clip_path = self.dirs["clips"] / f"{sample_id}.mp4"
        self._save_clip(frames, clip_path)

        print(f"  Total frames saved: {len(metadata) * (1 + self.config.augmentation_factor)}")

    def _save_clip(self, frames: List[np.ndarray], output_path: Path):
        """Save frames as video clip."""
        if len(frames) == 0:
            return

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, self.config.fps, (w, h))

        for frame in frames:
            writer.write(frame)

        writer.release()

    def _draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI overlay on frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Recording indicator
        if self.is_recording:
            elapsed = time.time() - self.recording_start_time
            remaining = self.config.clip_duration - elapsed

            # Red recording dot
            cv2.circle(overlay, (30, 30), 15, (0, 0, 255), -1)

            # Sign name
            cv2.putText(overlay, f"Recording: {self.current_sign}",
                       (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Progress bar
            progress = elapsed / self.config.clip_duration
            bar_width = int(w * 0.8)
            bar_x = int(w * 0.1)
            bar_y = h - 40

            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20),
                         (100, 100, 100), -1)
            cv2.rectangle(overlay, (bar_x, bar_y),
                         (bar_x + int(bar_width * progress), bar_y + 20),
                         (0, 255, 0), -1)

            cv2.putText(overlay, f"{remaining:.1f}s",
                       (bar_x + bar_width + 10, bar_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        else:
            # Instructions
            cv2.putText(overlay, "Press A-Z for letters, 0-9 for numbers",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(overlay, "Press Q to quit, H for help",
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Sample count
        cv2.putText(overlay, f"Samples: {len(self.all_metadata)}",
                   (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return overlay

    def _draw_help(self, frame: np.ndarray) -> np.ndarray:
        """Draw help overlay."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent background
        cv2.rectangle(overlay, (50, 50), (w - 50, h - 50), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

        # Help text
        help_text = [
            "=== WEBCAM SIGN RECORDER ===",
            "",
            "CONTROLS:",
            "  A-Z    : Record letter sign",
            "  0-9    : Record number sign",
            "  SPACE  : Generic recording",
            "  Q      : Quit and save",
            "  H      : Toggle this help",
            "",
            "TIPS:",
            "- Position hand in center of frame",
            "- Good lighting improves recognition",
            "- Recording auto-stops after 2.5 seconds",
            "",
            f"Session: {self.session_id}",
            f"Samples recorded: {len(self.all_metadata)}"
        ]

        y = 80
        for line in help_text:
            cv2.putText(frame, line, (70, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 25

        return frame

    def run(self):
        """Main recording loop."""
        print("=" * 60)
        print("SonZo AI - Webcam Sign Recorder")
        print("=" * 60)
        print(f"Session ID: {self.session_id}")
        print(f"Output directory: {self.output_dir}")
        print("\nOpening camera...")

        # Open camera
        self.cap = cv2.VideoCapture(self.config.camera_id)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        print("Camera opened successfully!")
        print("\nPress A-Z to record letters, 0-9 for numbers")
        print("Press Q to quit\n")

        show_help = False

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                # Check if recording should auto-stop
                if self.is_recording:
                    elapsed = time.time() - self.recording_start_time
                    self.recorded_frames.append(frame.copy())

                    if elapsed >= self.config.clip_duration:
                        # Stop recording and process
                        self.is_recording = False
                        self._process_recording(self.current_sign, self.recorded_frames)
                        self.recorded_frames = []
                        self.current_sign = None

                # Draw UI
                if show_help:
                    display_frame = self._draw_help(frame)
                else:
                    display_frame = self._draw_ui(frame)

                # Show frame
                cv2.imshow('Sign Recorder', display_frame)

                # Handle key press
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('h') or key == ord('H'):
                    show_help = not show_help
                elif key in self.key_to_sign and not self.is_recording:
                    # Start recording
                    self.current_sign = self.key_to_sign[key]
                    self.is_recording = True
                    self.recording_start_time = time.time()
                    self.recorded_frames = [frame.copy()]
                    print(f"Recording: {self.current_sign}")
                elif key == ord(' ') and not self.is_recording:
                    # Generic recording (will prompt for label)
                    self.current_sign = "UNKNOWN"
                    self.is_recording = True
                    self.recording_start_time = time.time()
                    self.recorded_frames = [frame.copy()]
                    print("Recording: UNKNOWN (will prompt for label)")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

        # Save session metadata
        self._save_metadata()

    def _save_metadata(self):
        """Save all metadata to disk."""
        if len(self.all_metadata) == 0:
            print("No samples recorded.")
            return

        print("\nSaving metadata...")

        # Main metadata file
        metadata_file = self.dirs["metadata"] / f"session_{self.session_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                "session_id": self.session_id,
                "config": asdict(self.config),
                "total_samples": len(self.all_metadata),
                "samples": self.all_metadata
            }, f, indent=2)

        # Aggregate with existing metadata
        all_sessions_file = self.output_dir / "all_samples.json"
        existing_samples = []

        if all_sessions_file.exists():
            with open(all_sessions_file, 'r') as f:
                data = json.load(f)
                existing_samples = data.get("samples", [])

        all_samples = existing_samples + self.all_metadata

        with open(all_sessions_file, 'w') as f:
            json.dump({
                "total_samples": len(all_samples),
                "samples": all_samples
            }, f, indent=2)

        # Label map
        signs = list(set(m["handshape"] for m in all_samples))
        label_map = {sign: idx for idx, sign in enumerate(sorted(signs))}

        with open(self.output_dir / "label_map.json", 'w') as f:
            json.dump(label_map, f, indent=2)

        print(f"\nSession complete!")
        print(f"  Samples recorded: {len(self.all_metadata)}")
        print(f"  Total samples: {len(all_samples)}")
        print(f"  Signs: {len(signs)}")
        print(f"\nMetadata saved to: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Record ASL signs via webcam',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start recording
  python scripts/record_webcam.py --output ./real_data

  # Record with more augmentation
  python scripts/record_webcam.py --output ./real_data --augment 5

  # Use different camera
  python scripts/record_webcam.py --output ./real_data --camera 1
"""
    )

    parser.add_argument('--output', type=str, default='./real_data',
                       help='Output directory')
    parser.add_argument('--duration', type=float, default=2.5,
                       help='Recording duration in seconds (default: 2.5)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Recording FPS (default: 30)')
    parser.add_argument('--size', type=int, default=512,
                       help='Output frame size (default: 512)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--augment', type=int, default=3,
                       help='Augmentation factor (default: 3)')
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable augmentation')
    parser.add_argument('--frames', type=int, default=16,
                       help='Frames to extract per clip (default: 16)')

    args = parser.parse_args()

    config = RecordingConfig(
        output_dir=args.output,
        clip_duration=args.duration,
        fps=args.fps,
        frame_size=(args.size, args.size),
        camera_id=args.camera,
        enable_augmentation=not args.no_augment,
        augmentation_factor=args.augment,
        frames_per_clip=args.frames
    )

    recorder = WebcamRecorder(config)
    recorder.run()


if __name__ == "__main__":
    main()
