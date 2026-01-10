#!/usr/bin/env python3
"""
Real-Time ASL Recognition Demo
===============================
Live webcam demo with real-time sign language recognition.

Features:
- Live webcam feed with prediction overlay
- Top 3 predicted signs with confidence percentages
- Running text accumulation (like subtitles)
- Clear button to reset
- Recording for investor pitch demos
- <500ms latency optimized

Usage:
    python demo/realtime_demo.py --model ./models/best_model.pt

    # With recording enabled
    python demo/realtime_demo.py --model ./models/best_model.pt --record

Controls:
    Space: Clear accumulated text
    R: Start/stop recording
    Q: Quit

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading
import queue

# Add parent directory for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")


class FrameBuffer:
    """Thread-safe frame buffer for maintaining sequence."""

    def __init__(self, max_size: int = 16):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add(self, frame: np.ndarray):
        """Add frame to buffer."""
        with self.lock:
            self.buffer.append(frame)

    def get_sequence(self) -> List[np.ndarray]:
        """Get current frame sequence."""
        with self.lock:
            return list(self.buffer)

    def clear(self):
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class PredictionSmoother:
    """Smooth predictions over time to reduce jitter."""

    def __init__(self, window_size: int = 5, threshold: float = 0.6):
        self.window_size = window_size
        self.threshold = threshold
        self.predictions = deque(maxlen=window_size)

    def add(self, prediction: Dict):
        """Add a prediction to the smoother."""
        self.predictions.append(prediction)

    def get_smoothed(self) -> Optional[Dict]:
        """Get smoothed prediction using voting."""
        if len(self.predictions) < 2:
            return None

        # Count class votes
        votes = {}
        confidences = {}

        for pred in self.predictions:
            class_id = pred.get("class_id")
            conf = pred.get("confidence", 0)

            if class_id is not None:
                votes[class_id] = votes.get(class_id, 0) + 1
                if class_id not in confidences:
                    confidences[class_id] = []
                confidences[class_id].append(conf)

        if not votes:
            return None

        # Get most common prediction
        top_class = max(votes, key=votes.get)
        vote_ratio = votes[top_class] / len(self.predictions)
        avg_confidence = np.mean(confidences[top_class])

        # Only return if consistent
        if vote_ratio >= self.threshold and avg_confidence >= self.threshold:
            return {
                "class_id": top_class,
                "confidence": avg_confidence,
                "vote_ratio": vote_ratio
            }

        return None

    def clear(self):
        """Clear predictions."""
        self.predictions.clear()


class RealTimeDemo:
    """Real-time ASL recognition demo application."""

    def __init__(
        self,
        model_path: str,
        camera_id: int = 0,
        target_fps: int = 30,
        sequence_length: int = 16,
        image_size: int = 224,
        confidence_threshold: float = 0.5
    ):
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.label_map = self._load_model(model_path)
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}

        # Frame buffer
        self.frame_buffer = FrameBuffer(max_size=sequence_length)

        # Prediction smoother
        self.smoother = PredictionSmoother(window_size=5, threshold=0.6)

        # Transforms
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # UI state
        self.accumulated_text = []
        self.current_prediction = None
        self.last_added_sign = None
        self.prediction_history: List[Dict] = []

        # Recording
        self.is_recording = False
        self.video_writer = None
        self.recording_path = None

        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()

        # Latency tracking
        self.latency_history = deque(maxlen=100)

    def _load_model(self, model_path: str) -> Tuple[Any, Dict]:
        """Load trained model and label map."""
        print(f"Loading model from: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # Get config and label map
        config_dict = checkpoint.get("config", {})
        label_map = checkpoint.get("label_map", {})

        # Import and initialize model
        from train_slr import SLRModel, TrainingConfig

        config = TrainingConfig(**config_dict)
        config.num_classes = len(label_map)

        model = SLRModel(config).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print(f"Model loaded. Classes: {len(label_map)}")
        print(f"Device: {self.device}")

        return model, label_map

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single frame."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(frame_rgb)

    @torch.no_grad()
    def predict(self, frames: List[np.ndarray]) -> Dict:
        """Run prediction on frame sequence."""
        start_time = time.time()

        # Preprocess frames
        tensors = [self.preprocess_frame(f) for f in frames]

        # Pad if needed
        while len(tensors) < self.sequence_length:
            tensors.append(tensors[-1] if tensors else torch.zeros(3, self.image_size, self.image_size))
        tensors = tensors[:self.sequence_length]

        # Stack and add batch dimension
        sequence = torch.stack(tensors).unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        # Predict
        output = self.model(sequence, return_attention=True)
        probs = F.softmax(output["logits"], dim=-1)[0]
        confidence = output["confidence"][0].item()

        # Get top 3 predictions
        top_probs, top_indices = probs.topk(3)
        top_predictions = []

        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            sign_name = self.inverse_label_map.get(idx, "UNKNOWN")
            top_predictions.append({
                "sign": sign_name,
                "probability": float(prob),
                "class_id": int(idx)
            })

        # Track latency
        latency = (time.time() - start_time) * 1000
        self.latency_history.append(latency)

        return {
            "top_predictions": top_predictions,
            "confidence": confidence,
            "class_id": top_predictions[0]["class_id"] if top_predictions else None,
            "latency_ms": latency
        }

    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI overlay on frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent panel on right side
        panel_width = 300
        cv2.rectangle(overlay, (w - panel_width, 0), (w, h), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        x_offset = w - panel_width + 10
        y_offset = 30

        # Title
        cv2.putText(frame, "SonZo AI", (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        y_offset += 25
        cv2.putText(frame, "Sign Language Recognition", (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 40

        # Current prediction
        if self.current_prediction:
            top_preds = self.current_prediction.get("top_predictions", [])

            cv2.putText(frame, "DETECTED:", (x_offset, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            y_offset += 25

            for i, pred in enumerate(top_preds[:3]):
                sign = pred["sign"]
                prob = pred["probability"] * 100
                conf = self.current_prediction.get("confidence", 0) * 100

                # Color based on rank
                if i == 0:
                    color = (0, 255, 0) if prob > 70 else (0, 200, 200)
                    font_scale = 0.9
                else:
                    color = (150, 150, 150)
                    font_scale = 0.5

                cv2.putText(frame, f"{sign}", (x_offset, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2 if i == 0 else 1)

                # Probability bar
                bar_width = int((panel_width - 80) * (prob / 100))
                bar_y = y_offset - 5
                cv2.rectangle(frame, (x_offset + 120, bar_y - 10),
                             (x_offset + 120 + bar_width, bar_y + 5), color, -1)

                cv2.putText(frame, f"{prob:.0f}%", (x_offset + 230, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                y_offset += 35 if i == 0 else 25

        # Confidence indicator
        y_offset += 20
        conf = self.current_prediction.get("confidence", 0) if self.current_prediction else 0
        cv2.putText(frame, f"Confidence: {conf*100:.0f}%", (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # FPS and latency
        y_offset = h - 80
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        avg_latency = np.mean(self.latency_history) if self.latency_history else 0

        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
        cv2.putText(frame, f"Latency: {avg_latency:.0f}ms", (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Recording indicator
        if self.is_recording:
            cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (55, 38),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Accumulated text (subtitle style at bottom)
        if self.accumulated_text:
            text = " ".join(self.accumulated_text[-10:])  # Last 10 signs
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]

            # Background
            cv2.rectangle(frame, (10, h - 60), (10 + text_size[0] + 20, h - 20),
                         (0, 0, 0), -1)

            cv2.putText(frame, text, (20, h - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Instructions
        cv2.putText(frame, "SPACE: Clear | R: Record | Q: Quit",
                   (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return frame

    def start_recording(self):
        """Start video recording."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_path = PROJECT_ROOT / "demo" / "recordings" / f"demo_{timestamp}.mp4"
        self.recording_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(self.recording_path), fourcc, self.target_fps, (1280, 720)
        )
        self.is_recording = True
        print(f"Recording started: {self.recording_path}")

    def stop_recording(self):
        """Stop video recording."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.is_recording = False
        print(f"Recording saved: {self.recording_path}")

    def run(self):
        """Main demo loop."""
        print("=" * 60)
        print("SonZo AI - Real-Time Sign Language Recognition Demo")
        print("=" * 60)
        print(f"Model: {len(self.label_map)} signs")
        print(f"Device: {self.device}")
        print("\nControls:")
        print("  SPACE - Clear accumulated text")
        print("  R     - Start/stop recording")
        print("  Q     - Quit")
        print()

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        print("Camera opened. Press Q to quit.\n")

        frame_interval = 1.0 / self.target_fps
        last_prediction_time = 0
        prediction_interval = 0.1  # Predict every 100ms

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = time.time()

                # Track FPS
                dt = current_time - self.last_frame_time
                if dt > 0:
                    self.fps_history.append(1.0 / dt)
                self.last_frame_time = current_time

                # Add frame to buffer
                self.frame_buffer.add(frame.copy())

                # Run prediction periodically
                if current_time - last_prediction_time >= prediction_interval:
                    frames = self.frame_buffer.get_sequence()

                    if len(frames) >= self.sequence_length // 2:
                        prediction = self.predict(frames)
                        self.smoother.add(prediction)

                        # Get smoothed prediction
                        smoothed = self.smoother.get_smoothed()
                        if smoothed:
                            self.current_prediction = {
                                "top_predictions": prediction["top_predictions"],
                                "confidence": smoothed["confidence"]
                            }

                            # Add to accumulated text if confident and new
                            top_sign = prediction["top_predictions"][0]["sign"]
                            if (smoothed["confidence"] >= self.confidence_threshold and
                                top_sign != self.last_added_sign):

                                self.accumulated_text.append(top_sign)
                                self.last_added_sign = top_sign

                    last_prediction_time = current_time

                # Draw UI
                display_frame = self.draw_ui(frame)

                # Record if enabled
                if self.is_recording and self.video_writer:
                    self.video_writer.write(display_frame)

                # Show frame
                cv2.imshow("SonZo AI - Sign Language Demo", display_frame)

                # Handle key press
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord(' '):
                    self.accumulated_text.clear()
                    self.last_added_sign = None
                    self.smoother.clear()
                    print("Text cleared")
                elif key == ord('r') or key == ord('R'):
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording()

        finally:
            if self.is_recording:
                self.stop_recording()
            cap.release()
            cv2.destroyAllWindows()

        # Print statistics
        print("\n" + "=" * 60)
        print("Session Statistics")
        print("=" * 60)
        print(f"Average FPS: {np.mean(self.fps_history):.1f}")
        print(f"Average Latency: {np.mean(self.latency_history):.1f}ms")
        print(f"Signs recognized: {len(self.accumulated_text)}")


def main():
    parser = argparse.ArgumentParser(description='Real-time ASL recognition demo')

    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS')
    parser.add_argument('--sequence-length', type=int, default=16,
                       help='Number of frames per prediction')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--record', action='store_true',
                       help='Start recording immediately')

    args = parser.parse_args()

    demo = RealTimeDemo(
        model_path=args.model,
        camera_id=args.camera,
        target_fps=args.fps,
        sequence_length=args.sequence_length,
        confidence_threshold=args.threshold
    )

    if args.record:
        demo.start_recording()

    demo.run()


if __name__ == "__main__":
    main()
