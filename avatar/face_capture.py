#!/usr/bin/env python3
"""
Face Capture Module
====================
Extract and process face from photos/webcam for avatar generation.

Features:
- Face detection and alignment
- Face embedding extraction
- Quality validation
- Multiple face handling

Usage:
    from face_capture import FaceCapture

    capture = FaceCapture()
    face_data = capture.extract_face("selfie.jpg")

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import base64
import io

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: InsightFace not available. Install with: pip install insightface onnxruntime")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class FaceData:
    """Extracted face data."""
    face_image: np.ndarray  # Aligned face crop
    embedding: np.ndarray   # Face embedding vector (512-dim)
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    landmarks: np.ndarray   # 5 key facial landmarks
    age: Optional[int] = None
    gender: Optional[str] = None
    quality_score: float = 0.0


class FaceCapture:
    """Face detection and extraction for avatar generation."""

    def __init__(
        self,
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        use_gpu: bool = True
    ):
        """
        Initialize face capture.

        Args:
            det_size: Detection input size
            det_thresh: Detection confidence threshold
            use_gpu: Use GPU acceleration if available
        """
        self.det_size = det_size
        self.det_thresh = det_thresh

        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace required. Install with: pip install insightface onnxruntime-gpu")

        # Initialize InsightFace
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

        self.app = FaceAnalysis(
            name='buffalo_l',  # High quality model
            providers=providers
        )
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=det_size)

        print(f"FaceCapture initialized (GPU: {use_gpu})")

    def extract_face(
        self,
        image_source: Any,
        return_all: bool = False
    ) -> Optional[FaceData]:
        """
        Extract face from image.

        Args:
            image_source: File path, numpy array, base64 string, or PIL Image
            return_all: If True, return all detected faces

        Returns:
            FaceData object or None if no face detected
        """
        # Load image
        img = self._load_image(image_source)
        if img is None:
            return None

        # Detect faces
        faces = self.app.get(img)

        if len(faces) == 0:
            print("No face detected")
            return None

        # Process faces
        face_data_list = []
        for face in faces:
            # Extract aligned face crop
            face_crop = self._align_face(img, face)

            # Calculate quality score
            quality = self._calculate_quality(face, img.shape)

            face_data = FaceData(
                face_image=face_crop,
                embedding=face.embedding,
                bbox=tuple(face.bbox.astype(int)),
                landmarks=face.kps,
                age=int(face.age) if hasattr(face, 'age') else None,
                gender='M' if hasattr(face, 'gender') and face.gender == 1 else 'F',
                quality_score=quality
            )
            face_data_list.append(face_data)

        if return_all:
            return face_data_list

        # Return best quality face
        best_face = max(face_data_list, key=lambda f: f.quality_score)
        return best_face

    def extract_from_webcam(self, timeout: float = 30.0) -> Optional[FaceData]:
        """
        Capture face from webcam with live preview.

        Args:
            timeout: Maximum time to wait for good face

        Returns:
            FaceData or None
        """
        import time

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Position your face in the frame. Press SPACE to capture, Q to quit.")

        start_time = time.time()
        best_face = None
        best_score = 0

        try:
            while time.time() - start_time < timeout:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect face
                faces = self.app.get(frame)

                display = frame.copy()

                if len(faces) > 0:
                    face = faces[0]
                    bbox = face.bbox.astype(int)
                    quality = self._calculate_quality(face, frame.shape)

                    # Draw bounding box
                    color = (0, 255, 0) if quality > 0.7 else (0, 165, 255)
                    cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                    # Quality indicator
                    cv2.putText(display, f"Quality: {quality:.0%}",
                               (bbox[0], bbox[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Track best face
                    if quality > best_score:
                        best_score = quality
                        best_face = frame.copy()

                # Instructions
                cv2.putText(display, "SPACE: Capture | Q: Quit",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Face Capture", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' ') and best_face is not None:
                    # Capture current best
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        if best_face is not None:
            return self.extract_face(best_face)

        return None

    def validate_face(self, face_data: FaceData) -> Dict[str, Any]:
        """
        Validate face quality for avatar generation.

        Returns dict with validation results and recommendations.
        """
        issues = []
        recommendations = []

        # Check quality score
        if face_data.quality_score < 0.5:
            issues.append("Low overall quality")
            recommendations.append("Use better lighting or higher resolution photo")

        if face_data.quality_score < 0.7:
            issues.append("Moderate quality - results may vary")

        # Check face size (should be substantial portion of image)
        bbox = face_data.bbox
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        img_area = face_data.face_image.shape[0] * face_data.face_image.shape[1]

        if face_area < img_area * 0.1:
            issues.append("Face too small in frame")
            recommendations.append("Move closer to camera or crop photo")

        # Check landmark confidence (simplified)
        if face_data.landmarks is None:
            issues.append("Could not detect facial landmarks")

        return {
            "is_valid": len(issues) == 0 or face_data.quality_score >= 0.5,
            "quality_score": face_data.quality_score,
            "issues": issues,
            "recommendations": recommendations
        }

    def _load_image(self, source: Any) -> Optional[np.ndarray]:
        """Load image from various sources."""
        if isinstance(source, np.ndarray):
            return source

        if isinstance(source, str):
            # Check if base64
            if source.startswith('data:image'):
                # Data URL
                source = source.split(',')[1]

            if len(source) > 260:  # Likely base64
                try:
                    img_data = base64.b64decode(source)
                    img_array = np.frombuffer(img_data, np.uint8)
                    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                except:
                    pass

            # File path
            path = Path(source)
            if path.exists():
                return cv2.imread(str(path))

        if PIL_AVAILABLE and hasattr(source, 'convert'):
            # PIL Image
            return cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)

        return None

    def _align_face(self, img: np.ndarray, face: Any, size: int = 512) -> np.ndarray:
        """Align and crop face."""
        # Use InsightFace's alignment
        from insightface.utils import face_align

        aligned = face_align.norm_crop(img, face.kps, image_size=size)
        return aligned

    def _calculate_quality(self, face: Any, img_shape: Tuple) -> float:
        """
        Calculate face quality score.

        Factors:
        - Detection confidence
        - Face size relative to image
        - Pose (frontal is better)
        - Blur detection
        """
        score = 0.0

        # Detection confidence (det_score)
        if hasattr(face, 'det_score'):
            score += face.det_score * 0.3

        # Face size ratio
        bbox = face.bbox
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        img_area = img_shape[0] * img_shape[1]
        size_ratio = min(face_area / img_area, 0.5) * 2  # Normalize to 0-1
        score += size_ratio * 0.3

        # Pose estimation (if available)
        if hasattr(face, 'pose'):
            # pose is [pitch, yaw, roll] in degrees
            yaw = abs(face.pose[1])
            pitch = abs(face.pose[0])

            # Penalize non-frontal poses
            pose_score = max(0, 1 - (yaw / 45) - (pitch / 30))
            score += pose_score * 0.2

        # Landmark quality (simple check)
        if face.kps is not None and len(face.kps) == 5:
            score += 0.2

        return min(score, 1.0)

    def save_face(self, face_data: FaceData, output_path: str):
        """Save extracted face to file."""
        cv2.imwrite(output_path, face_data.face_image)
        print(f"Saved face to: {output_path}")

    def to_base64(self, face_data: FaceData) -> str:
        """Convert face image to base64."""
        _, buffer = cv2.imencode('.png', face_data.face_image)
        return base64.b64encode(buffer).decode('utf-8')


def main():
    """Test face capture."""
    import argparse

    parser = argparse.ArgumentParser(description='Extract face from image')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--webcam', action='store_true', help='Capture from webcam')
    parser.add_argument('--output', type=str, default='face_output.png', help='Output path')

    args = parser.parse_args()

    capture = FaceCapture()

    if args.webcam:
        face_data = capture.extract_from_webcam()
    elif args.image:
        face_data = capture.extract_face(args.image)
    else:
        print("Specify --image or --webcam")
        return

    if face_data:
        validation = capture.validate_face(face_data)
        print(f"\nFace extracted successfully!")
        print(f"Quality score: {face_data.quality_score:.2%}")
        print(f"Validation: {validation}")

        capture.save_face(face_data, args.output)
    else:
        print("No face detected")


if __name__ == "__main__":
    main()
