#!/usr/bin/env python3
"""
Face Swap Module
=================
Swap faces in sign language videos to create personalized avatars.

Uses InsightFace's inswapper model for high-quality face swapping.

Features:
- Single image to video face swap
- Batch processing
- Quality enhancement with GFPGAN
- Temporal consistency for video

Usage:
    from face_swap import FaceSwapper

    swapper = FaceSwapper()
    output_video = swapper.swap_video("source_face.jpg", "signer_video.mp4")

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import tempfile
import subprocess
import time

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

try:
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


@dataclass
class SwapConfig:
    """Face swap configuration."""
    # Model settings
    model_path: str = "inswapper_128.onnx"

    # Quality settings
    use_enhancer: bool = True
    enhancer_model: str = "gfpgan"  # gfpgan, codeformer, or none

    # Video settings
    output_quality: int = 23  # FFmpeg CRF (lower = better, 18-28 typical)
    output_fps: Optional[int] = None  # None = match source

    # Processing
    batch_size: int = 1
    use_gpu: bool = True


class FaceSwapper:
    """Face swapping for avatar video generation."""

    def __init__(self, config: Optional[SwapConfig] = None):
        """Initialize face swapper."""
        self.config = config or SwapConfig()

        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace required. Install with: pip install insightface")

        # Initialize face analyzer
        providers = self._get_providers()

        self.face_analyzer = FaceAnalysis(name='buffalo_l', providers=providers)
        self.face_analyzer.prepare(ctx_id=0 if self.config.use_gpu else -1)

        # Load swapper model
        self.swapper = self._load_swapper_model()

        # Load enhancer if enabled
        self.enhancer = None
        if self.config.use_enhancer:
            self.enhancer = self._load_enhancer()

        print("FaceSwapper initialized")

    def _get_providers(self) -> List[str]:
        """Get ONNX execution providers."""
        if self.config.use_gpu:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']

    def _load_swapper_model(self):
        """Load the face swapper model."""
        model_path = Path(self.config.model_path)

        # Try to find model in common locations
        search_paths = [
            model_path,
            Path.home() / ".insightface" / "models" / "inswapper_128.onnx",
            Path(__file__).parent / "models" / "inswapper_128.onnx",
        ]

        for path in search_paths:
            if path.exists():
                print(f"Loading swapper model from: {path}")
                return insightface.model_zoo.get_model(str(path), providers=self._get_providers())

        # Try to download
        print("Swapper model not found. Attempting to download...")
        try:
            model = insightface.model_zoo.get_model('inswapper_128.onnx', download=True)
            return model
        except Exception as e:
            print(f"Could not load swapper model: {e}")
            print("Download manually from: https://huggingface.co/deepinsight/inswapper/tree/main")
            return None

    def _load_enhancer(self):
        """Load face enhancement model (GFPGAN or CodeFormer)."""
        try:
            if self.config.enhancer_model == "gfpgan":
                from gfpgan import GFPGANer
                return GFPGANer(
                    model_path='GFPGANv1.4.pth',
                    upscale=1,
                    arch='clean',
                    channel_multiplier=2
                )
            elif self.config.enhancer_model == "codeformer":
                # CodeFormer integration
                pass
        except ImportError:
            print("Face enhancer not available. Install with: pip install gfpgan")
        except Exception as e:
            print(f"Could not load enhancer: {e}")

        return None

    def swap_face(
        self,
        source_face: np.ndarray,
        target_image: np.ndarray,
        source_face_data: Any = None
    ) -> np.ndarray:
        """
        Swap face in a single image.

        Args:
            source_face: Source face image (the face to put in)
            target_image: Target image (contains face to replace)
            source_face_data: Pre-extracted face data (optional)

        Returns:
            Result image with swapped face
        """
        if self.swapper is None:
            raise RuntimeError("Swapper model not loaded")

        # Get source face embedding
        if source_face_data is None:
            source_faces = self.face_analyzer.get(source_face)
            if len(source_faces) == 0:
                raise ValueError("No face detected in source image")
            source_face_data = source_faces[0]

        # Detect faces in target
        target_faces = self.face_analyzer.get(target_image)
        if len(target_faces) == 0:
            return target_image  # No face to swap

        # Swap each face in target
        result = target_image.copy()
        for target_face in target_faces:
            result = self.swapper.get(result, target_face, source_face_data, paste_back=True)

        # Apply enhancement
        if self.enhancer is not None:
            result = self._enhance_face(result)

        return result

    def swap_video(
        self,
        source_face: Any,
        video_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """
        Swap face throughout a video.

        Args:
            source_face: Source face image path, numpy array, or FaceData
            video_path: Path to target video
            output_path: Output video path (auto-generated if None)
            progress_callback: Called with progress (0.0-1.0)

        Returns:
            Path to output video
        """
        # Load source face
        if isinstance(source_face, str):
            source_img = cv2.imread(source_face)
        elif isinstance(source_face, np.ndarray):
            source_img = source_face
        else:
            source_img = source_face.face_image

        source_faces = self.face_analyzer.get(source_img)
        if len(source_faces) == 0:
            raise ValueError("No face detected in source image")
        source_face_data = source_faces[0]

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output path
        if output_path is None:
            output_path = str(Path(video_path).stem) + "_swapped.mp4"

        # Create temporary file for frames
        temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_fps = self.config.output_fps or fps
        writer = cv2.VideoWriter(temp_output, fourcc, output_fps, (width, height))

        print(f"Processing {total_frames} frames...")
        start_time = time.time()

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Swap face
            try:
                swapped = self.swap_face(source_img, frame, source_face_data)
            except Exception as e:
                # Use original frame if swap fails
                swapped = frame

            writer.write(swapped)

            # Progress
            frame_idx += 1
            if progress_callback:
                progress_callback(frame_idx / total_frames)

            if frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_idx / elapsed
                eta = (total_frames - frame_idx) / fps_actual
                print(f"  Frame {frame_idx}/{total_frames} ({fps_actual:.1f} fps, ETA: {eta:.0f}s)")

        cap.release()
        writer.release()

        # Re-encode with FFmpeg for better quality and audio
        self._reencode_video(temp_output, video_path, output_path)

        # Clean up temp file
        Path(temp_output).unlink(missing_ok=True)

        elapsed = time.time() - start_time
        print(f"Done! Processed {total_frames} frames in {elapsed:.1f}s ({total_frames/elapsed:.1f} fps)")
        print(f"Output: {output_path}")

        return output_path

    def _enhance_face(self, image: np.ndarray) -> np.ndarray:
        """Enhance face quality using GFPGAN."""
        if self.enhancer is None:
            return image

        try:
            _, _, enhanced = self.enhancer.enhance(
                image,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
            return enhanced
        except Exception as e:
            print(f"Enhancement failed: {e}")
            return image

    def _reencode_video(self, temp_path: str, original_path: str, output_path: str):
        """Re-encode video with FFmpeg, preserving audio."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_path,
                '-i', original_path,
                '-map', '0:v',  # Video from temp
                '-map', '1:a?',  # Audio from original (if exists)
                '-c:v', 'libx264',
                '-crf', str(self.config.output_quality),
                '-preset', 'medium',
                '-c:a', 'aac',
                output_path
            ]

            subprocess.run(cmd, capture_output=True, check=True)
        except FileNotFoundError:
            print("FFmpeg not found, using OpenCV output")
            import shutil
            shutil.move(temp_path, output_path)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            import shutil
            shutil.move(temp_path, output_path)


class BatchSwapper:
    """Batch process multiple videos with same source face."""

    def __init__(self, swapper: FaceSwapper):
        self.swapper = swapper

    def process_library(
        self,
        source_face: Any,
        video_dir: str,
        output_dir: str,
        phrases: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Process entire video library.

        Args:
            source_face: Source face to swap in
            video_dir: Directory with sign videos
            output_dir: Output directory
            phrases: Specific phrases to process (None = all)

        Returns:
            Dict mapping phrase -> output video path
        """
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Find videos
        video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.mov"))

        if phrases:
            video_files = [v for v in video_files if v.stem.upper() in [p.upper() for p in phrases]]

        print(f"Processing {len(video_files)} videos...")

        for i, video_path in enumerate(video_files):
            phrase = video_path.stem.upper()
            output_path = output_dir / f"{phrase}.mp4"

            print(f"\n[{i+1}/{len(video_files)}] {phrase}")

            try:
                self.swapper.swap_video(
                    source_face,
                    str(video_path),
                    str(output_path)
                )
                results[phrase] = str(output_path)
            except Exception as e:
                print(f"  Error: {e}")
                results[phrase] = None

        return results


def main():
    """Test face swapper."""
    import argparse

    parser = argparse.ArgumentParser(description='Swap faces in video')
    parser.add_argument('--source', type=str, required=True, help='Source face image')
    parser.add_argument('--video', type=str, required=True, help='Target video')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--no-enhance', action='store_true', help='Disable face enhancement')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')

    args = parser.parse_args()

    config = SwapConfig(
        use_enhancer=not args.no_enhance,
        use_gpu=not args.no_gpu
    )

    swapper = FaceSwapper(config)
    output = swapper.swap_video(args.source, args.video, args.output)

    print(f"\nOutput saved to: {output}")


if __name__ == "__main__":
    main()
