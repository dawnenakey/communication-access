#!/usr/bin/env python3
"""
Landmark Extraction Script
===========================
Extract hand/pose landmarks from videos using MediaPipe.
Converts video data to structured landmark sequences for training.

This approach is inspired by modern SLR systems that train on
structured pose data rather than raw pixels.

Usage:
    python training/extract_landmarks.py \
        --input ~/wlasl-prepared \
        --output ./landmark_data

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

import cv2
import numpy as np
from tqdm import tqdm

# Try to import MediaPipe - handle both old and new API
MEDIAPIPE_AVAILABLE = False
MEDIAPIPE_LEGACY = False

try:
    import mediapipe as mp
    # Check if legacy solutions API is available
    if hasattr(mp, 'solutions'):
        MEDIAPIPE_LEGACY = True
        MEDIAPIPE_AVAILABLE = True
    else:
        # Try new tasks API
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe import solutions as mp_solutions
            MEDIAPIPE_AVAILABLE = True
            MEDIAPIPE_LEGACY = True  # solutions still available as import
        except ImportError:
            # Try direct solutions import
            try:
                from mediapipe import solutions as mp_solutions
                MEDIAPIPE_AVAILABLE = True
                MEDIAPIPE_LEGACY = True
            except ImportError:
                pass
except ImportError:
    pass

if not MEDIAPIPE_AVAILABLE:
    print("ERROR: MediaPipe not installed or incompatible version.")
    print("Try: pip install mediapipe==0.10.14")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LandmarkConfig:
    """Configuration for landmark extraction."""
    # MediaPipe settings
    static_image_mode: bool = False
    max_num_hands: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Extraction settings
    use_holistic: bool = True  # Use Holistic (hands + pose + face) vs just Hands
    include_pose: bool = True
    include_face: bool = False  # Face landmarks are less relevant for SLR

    # Video processing
    max_frames: int = 64  # Max frames per video
    target_fps: int = 15  # Target FPS for extraction

    # Output
    normalize: bool = True  # Normalize coordinates to [-1, 1]


class LandmarkExtractor:
    """Extract landmarks from videos using MediaPipe."""

    # Landmark counts
    HAND_LANDMARKS = 21  # MediaPipe hand landmarks
    POSE_LANDMARKS = 33  # MediaPipe pose landmarks
    FACE_LANDMARKS = 468  # MediaPipe face landmarks (we'll use subset)

    def __init__(self, config: LandmarkConfig):
        self.config = config

        # Get the solutions module (works with both old and new API)
        if hasattr(mp, 'solutions'):
            solutions = mp.solutions
        else:
            solutions = mp_solutions

        if config.use_holistic:
            self.mp_holistic = solutions.holistic
            self.detector = self.mp_holistic.Holistic(
                static_image_mode=config.static_image_mode,
                min_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence
            )
        else:
            self.mp_hands = solutions.hands
            self.detector = self.mp_hands.Hands(
                static_image_mode=config.static_image_mode,
                max_num_hands=config.max_num_hands,
                min_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence
            )

        # Calculate feature dimension
        self.feature_dim = self._calculate_feature_dim()
        logger.info(f"Feature dimension per frame: {self.feature_dim}")

    def _calculate_feature_dim(self) -> int:
        """Calculate total feature dimension per frame."""
        dim = 0

        # Two hands (left + right), each with 21 landmarks, 3 coords (x, y, z)
        dim += 2 * self.HAND_LANDMARKS * 3

        if self.config.include_pose and self.config.use_holistic:
            # Upper body pose (we use 13 key points)
            dim += 13 * 3

        if self.config.include_face and self.config.use_holistic:
            # Face subset (10 key points for expression)
            dim += 10 * 3

        return dim

    def extract_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extract landmarks from a video file.

        Returns:
            numpy array of shape (T, feature_dim) or None if extraction fails
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return None

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            logger.warning(f"Invalid video properties: {video_path}")
            cap.release()
            return None

        # Calculate frame sampling
        frame_skip = max(1, int(fps / self.config.target_fps))

        landmarks_sequence = []
        frame_idx = 0

        while len(landmarks_sequence) < self.config.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames to match target FPS
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            # Extract landmarks from frame
            frame_landmarks = self._extract_frame_landmarks(frame)
            landmarks_sequence.append(frame_landmarks)

            frame_idx += 1

        cap.release()

        if len(landmarks_sequence) == 0:
            return None

        return np.array(landmarks_sequence, dtype=np.float32)

    def extract_from_frames(self, frame_paths: List[str]) -> Optional[np.ndarray]:
        """
        Extract landmarks from a list of frame images.

        Returns:
            numpy array of shape (T, feature_dim) or None if extraction fails
        """
        landmarks_sequence = []

        for frame_path in frame_paths[:self.config.max_frames]:
            frame = cv2.imread(frame_path)
            if frame is None:
                # Use zero landmarks for missing frames
                landmarks_sequence.append(np.zeros(self.feature_dim, dtype=np.float32))
                continue

            frame_landmarks = self._extract_frame_landmarks(frame)
            landmarks_sequence.append(frame_landmarks)

        if len(landmarks_sequence) == 0:
            return None

        return np.array(landmarks_sequence, dtype=np.float32)

    def _extract_frame_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Extract landmarks from a single frame."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = self.detector.process(rgb_frame)

        # Initialize feature vector
        features = np.zeros(self.feature_dim, dtype=np.float32)
        idx = 0

        if self.config.use_holistic:
            # Extract left hand
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    features[idx:idx+3] = [lm.x, lm.y, lm.z]
                    idx += 3
            else:
                idx += self.HAND_LANDMARKS * 3

            # Extract right hand
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    features[idx:idx+3] = [lm.x, lm.y, lm.z]
                    idx += 3
            else:
                idx += self.HAND_LANDMARKS * 3

            # Extract pose (upper body keypoints)
            if self.config.include_pose and results.pose_landmarks:
                # Upper body indices: shoulders, elbows, wrists, hips, etc.
                upper_body_indices = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24]
                for i in upper_body_indices:
                    lm = results.pose_landmarks.landmark[i]
                    features[idx:idx+3] = [lm.x, lm.y, lm.z]
                    idx += 3
            elif self.config.include_pose:
                idx += 13 * 3
        else:
            # Hands only mode
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                    hand_offset = hand_idx * self.HAND_LANDMARKS * 3
                    for lm_idx, lm in enumerate(hand_landmarks.landmark):
                        features[hand_offset + lm_idx*3:hand_offset + lm_idx*3 + 3] = [lm.x, lm.y, lm.z]

        # Normalize coordinates
        if self.config.normalize:
            # Convert from [0, 1] to [-1, 1]
            features = features * 2 - 1
            # Handle zero values (no detection)
            features = np.where(features == -1, 0, features)

        return features

    def close(self):
        """Release resources."""
        self.detector.close()


def extract_wlasl_landmarks(
    input_dir: Path,
    output_dir: Path,
    config: LandmarkConfig
) -> Dict:
    """
    Extract landmarks from WLASL-prepared dataset.

    Args:
        input_dir: Path to wlasl-prepared directory (with frames/ and metadata.json)
        output_dir: Output directory for landmark data

    Returns:
        Statistics dictionary
    """
    # Load metadata
    metadata_path = input_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Create extractor
    extractor = LandmarkExtractor(config)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    landmarks_dir = output_dir / "landmarks"
    landmarks_dir.mkdir(exist_ok=True)

    # Process each video/gloss
    stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "glosses": {}
    }

    # Group by gloss
    videos_by_gloss = {}
    for video_id, video_info in metadata.items():
        gloss = video_info.get("gloss", "unknown")
        if gloss not in videos_by_gloss:
            videos_by_gloss[gloss] = []
        videos_by_gloss[gloss].append((video_id, video_info))

    logger.info(f"Processing {len(metadata)} videos across {len(videos_by_gloss)} glosses")

    # Build label map
    label_map = {gloss: idx for idx, gloss in enumerate(sorted(videos_by_gloss.keys()))}

    # Process videos
    all_samples = []

    for gloss, videos in tqdm(videos_by_gloss.items(), desc="Processing glosses"):
        gloss_dir = landmarks_dir / gloss
        gloss_dir.mkdir(exist_ok=True)

        for video_id, video_info in videos:
            stats["total"] += 1

            # Find frames for this video
            frames_dir = input_dir / "frames" / video_id
            if not frames_dir.exists():
                stats["failed"] += 1
                continue

            # Get sorted frame paths
            frame_files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
            if not frame_files:
                stats["failed"] += 1
                continue

            frame_paths = [str(f) for f in frame_files]

            # Extract landmarks
            landmarks = extractor.extract_from_frames(frame_paths)

            if landmarks is None or len(landmarks) == 0:
                stats["failed"] += 1
                continue

            # Save landmarks
            output_path = gloss_dir / f"{video_id}.npy"
            np.save(output_path, landmarks)

            # Record sample
            all_samples.append({
                "video_id": video_id,
                "gloss": gloss,
                "label": label_map[gloss],
                "landmarks_path": str(output_path.relative_to(output_dir)),
                "num_frames": len(landmarks),
                "feature_dim": extractor.feature_dim
            })

            stats["success"] += 1
            stats["glosses"][gloss] = stats["glosses"].get(gloss, 0) + 1

    extractor.close()

    # Split into train/val/test
    np.random.seed(42)
    np.random.shuffle(all_samples)

    n_total = len(all_samples)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train+n_val]
    test_samples = all_samples[n_train+n_val:]

    # Assign splits
    for s in train_samples:
        s["split"] = "train"
    for s in val_samples:
        s["split"] = "val"
    for s in test_samples:
        s["split"] = "test"

    all_samples = train_samples + val_samples + test_samples

    # Save metadata
    output_metadata = {
        "feature_dim": extractor.feature_dim,
        "num_classes": len(label_map),
        "label_map": label_map,
        "config": asdict(config),
        "samples": all_samples,
        "stats": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples)
        }
    }

    with open(output_dir / "landmarks_metadata.json", 'w') as f:
        json.dump(output_metadata, f, indent=2)

    # Save label map separately
    with open(output_dir / "label_map.json", 'w') as f:
        json.dump(label_map, f, indent=2)

    logger.info(f"\nExtraction complete!")
    logger.info(f"  Success: {stats['success']}/{stats['total']}")
    logger.info(f"  Classes: {len(label_map)}")
    logger.info(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    return stats


def extract_video_landmarks(
    input_dir: Path,
    output_dir: Path,
    config: LandmarkConfig
) -> Dict:
    """
    Extract landmarks from video files directly.

    Args:
        input_dir: Directory containing .mp4 files organized by class
        output_dir: Output directory for landmark data

    Returns:
        Statistics dictionary
    """
    extractor = LandmarkExtractor(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    landmarks_dir = output_dir / "landmarks"
    landmarks_dir.mkdir(exist_ok=True)

    stats = {"total": 0, "success": 0, "failed": 0}
    all_samples = []
    label_map = {}

    # Find all video files
    video_files = list(input_dir.rglob("*.mp4"))
    logger.info(f"Found {len(video_files)} video files")

    for video_path in tqdm(video_files, desc="Extracting landmarks"):
        stats["total"] += 1

        # Get class from parent directory or filename
        gloss = video_path.parent.name
        if gloss == input_dir.name:
            # Videos are flat, use filename pattern
            gloss = video_path.stem.split("_")[0] if "_" in video_path.stem else "unknown"

        # Update label map
        if gloss not in label_map:
            label_map[gloss] = len(label_map)

        # Extract landmarks
        landmarks = extractor.extract_from_video(str(video_path))

        if landmarks is None or len(landmarks) == 0:
            stats["failed"] += 1
            continue

        # Save landmarks
        gloss_dir = landmarks_dir / gloss
        gloss_dir.mkdir(exist_ok=True)
        output_path = gloss_dir / f"{video_path.stem}.npy"
        np.save(output_path, landmarks)

        all_samples.append({
            "video_id": video_path.stem,
            "gloss": gloss,
            "label": label_map[gloss],
            "landmarks_path": str(output_path.relative_to(output_dir)),
            "num_frames": len(landmarks),
            "feature_dim": extractor.feature_dim
        })

        stats["success"] += 1

    extractor.close()

    # Split and save metadata (same as above)
    np.random.seed(42)
    np.random.shuffle(all_samples)

    n_total = len(all_samples)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    for i, s in enumerate(all_samples):
        if i < n_train:
            s["split"] = "train"
        elif i < n_train + n_val:
            s["split"] = "val"
        else:
            s["split"] = "test"

    output_metadata = {
        "feature_dim": extractor.feature_dim,
        "num_classes": len(label_map),
        "label_map": label_map,
        "config": asdict(config),
        "samples": all_samples,
        "stats": {
            "train": len([s for s in all_samples if s["split"] == "train"]),
            "val": len([s for s in all_samples if s["split"] == "val"]),
            "test": len([s for s in all_samples if s["split"] == "test"])
        }
    }

    with open(output_dir / "landmarks_metadata.json", 'w') as f:
        json.dump(output_metadata, f, indent=2)

    with open(output_dir / "label_map.json", 'w') as f:
        json.dump(label_map, f, indent=2)

    logger.info(f"\nExtraction complete! Success: {stats['success']}/{stats['total']}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Extract landmarks from video data')

    parser.add_argument('--input', type=str, required=True,
                       help='Input directory (wlasl-prepared or video folder)')
    parser.add_argument('--output', type=str, default='./landmark_data',
                       help='Output directory for landmarks')
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['auto', 'wlasl', 'videos'],
                       help='Extraction mode')
    parser.add_argument('--max-frames', type=int, default=64,
                       help='Maximum frames per video')
    parser.add_argument('--target-fps', type=int, default=15,
                       help='Target FPS for extraction')
    parser.add_argument('--no-pose', action='store_true',
                       help='Exclude pose landmarks (hands only)')

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    config = LandmarkConfig(
        max_frames=args.max_frames,
        target_fps=args.target_fps,
        include_pose=not args.no_pose
    )

    # Auto-detect mode
    if args.mode == 'auto':
        if (input_dir / "metadata.json").exists() and (input_dir / "frames").exists():
            mode = 'wlasl'
        else:
            mode = 'videos'
    else:
        mode = args.mode

    logger.info(f"Extraction mode: {mode}")

    if mode == 'wlasl':
        extract_wlasl_landmarks(input_dir, output_dir, config)
    else:
        extract_video_landmarks(input_dir, output_dir, config)


if __name__ == "__main__":
    main()
