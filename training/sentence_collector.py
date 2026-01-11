#!/usr/bin/env python3
"""
Sentence Data Collection Tool for OAK-D Camera
===============================================
Record yourself signing scripted sentences with automatic annotation.

Workflow:
1. Tool shows you a sentence to sign
2. You sign it while OAK-D records
3. Recording is saved with the annotation (what you signed)
4. Repeat until you have enough data
5. Data exports in CTC training format

This creates perfectly annotated sentence-level data because
YOU know what you're signing - no manual annotation needed!

Usage:
    # With OAK-D camera
    python training/sentence_collector.py --camera oak

    # With webcam fallback
    python training/sentence_collector.py --camera webcam

    # Custom sentence file
    python training/sentence_collector.py --sentences my_sentences.txt

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
from typing import Dict, List, Optional, Any
import threading
import queue

import cv2
import numpy as np

# Try to import OAK-D dependencies
try:
    import depthai as dai
    OAK_AVAILABLE = True
except ImportError:
    OAK_AVAILABLE = False
    print("Note: DepthAI not installed. OAK-D camera unavailable.")
    print("      Install with: pip install depthai")


# ==============================================================================
# SENTENCE PROMPTS
# ==============================================================================

# Default sentences to collect - ASL grammar structure
DEFAULT_SENTENCES = [
    # Greetings (short)
    ["HELLO"],
    ["GOODBYE"],
    ["THANK_YOU"],

    # Greetings (longer)
    ["HELLO", "HOW", "YOU"],
    ["HELLO", "NICE", "MEET", "YOU"],
    ["GOODBYE", "SEE", "YOU"],

    # Questions
    ["WHAT", "NAME", "YOU"],
    ["WHERE", "YOU", "LIVE"],
    ["HOW", "YOU"],
    ["WHAT", "YOU", "WANT"],
    ["YOU", "UNDERSTAND"],
    ["YOU", "HUNGRY"],

    # Statements
    ["I", "UNDERSTAND"],
    ["I", "LOVE", "YOU"],
    ["I", "WANT", "LEARN"],
    ["I", "NEED", "HELP"],
    ["I", "HAPPY"],
    ["I", "SAD"],

    # Responses
    ["YES"],
    ["NO"],
    ["I", "KNOW"],
    ["I", "NOT", "KNOW"],
    ["I", "NOT", "UNDERSTAND"],

    # Instructions
    ["PLEASE", "HELP"],
    ["PLEASE", "WAIT"],
    ["AGAIN"],
    ["SLOW"],

    # Common phrases
    ["SORRY"],
    ["PLEASE"],
    ["FINISH"],
]


# ==============================================================================
# CAMERA HANDLERS
# ==============================================================================

class WebcamHandler:
    """Simple webcam handler as fallback."""

    def __init__(self, camera_id: int = 0, output_dir: str = "recordings"):
        self.camera_id = camera_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cap = None
        self.recording = False
        self.video_writer = None
        self.current_filename = None

    def initialize(self) -> bool:
        """Initialize webcam."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print("❌ Failed to open webcam")
            return False

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print("✅ Webcam initialized")
        return True

    def start_recording(self, filename: str):
        """Start recording to file."""
        self.current_filename = str(self.output_dir / filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.current_filename, fourcc, 30.0, (640, 480)
        )
        self.recording = True
        print(f"📹 Recording: {filename}")

    def stop_recording(self) -> Optional[str]:
        """Stop recording and return filename."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        return self.current_filename

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame."""
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        if self.recording and self.video_writer:
            self.video_writer.write(frame)

        return frame

    def close(self):
        """Release camera."""
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()


class OAKCameraHandler:
    """OAK-D camera handler for high-quality capture."""

    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = None
        self.device = None
        self.recording = False
        self.video_writer = None
        self.current_filename = None
        self.q_rgb = None

    def initialize(self) -> bool:
        """Initialize OAK-D camera."""
        if not OAK_AVAILABLE:
            print("❌ DepthAI not installed")
            return False

        try:
            # Create pipeline
            self.pipeline = dai.Pipeline()

            # RGB camera
            cam_rgb = self.pipeline.create(dai.node.ColorCamera)
            xout_rgb = self.pipeline.create(dai.node.XLinkOut)
            xout_rgb.setStreamName("rgb")

            # Configure camera
            cam_rgb.setPreviewSize(640, 480)
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam_rgb.setFps(30)

            # Auto settings
            cam_rgb.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
            cam_rgb.initialControl.setAutoExposureEnable()

            # Link
            cam_rgb.preview.link(xout_rgb.input)

            # Connect
            self.device = dai.Device(self.pipeline)
            self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            print("✅ OAK-D camera initialized")
            return True

        except Exception as e:
            print(f"❌ OAK-D initialization failed: {e}")
            return False

    def start_recording(self, filename: str):
        """Start recording to file."""
        self.current_filename = str(self.output_dir / filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.current_filename, fourcc, 30.0, (640, 480)
        )
        self.recording = True
        print(f"📹 Recording: {filename}")

    def stop_recording(self) -> Optional[str]:
        """Stop recording and return filename."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        return self.current_filename

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame."""
        if self.q_rgb is None:
            return None

        try:
            in_rgb = self.q_rgb.get()
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()

                if self.recording and self.video_writer:
                    self.video_writer.write(frame)

                return frame
        except Exception as e:
            print(f"⚠️ Frame capture error: {e}")

        return None

    def close(self):
        """Release camera."""
        if self.video_writer:
            self.video_writer.release()
        if self.device:
            self.device.close()


# ==============================================================================
# DATA COLLECTOR
# ==============================================================================

class SentenceCollector:
    """
    Interactive sentence data collection tool.

    Shows sentences to sign, records with camera, saves with annotation.
    """

    def __init__(
        self,
        camera_type: str = "oak",
        output_dir: str = "sentence_data",
        sentences: Optional[List[List[str]]] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)

        # Initialize camera
        if camera_type == "oak" and OAK_AVAILABLE:
            self.camera = OAKCameraHandler(str(self.output_dir / "videos"))
        else:
            print("Using webcam (OAK-D not available or not selected)")
            self.camera = WebcamHandler(output_dir=str(self.output_dir / "videos"))

        # Sentences to collect
        self.sentences = sentences or DEFAULT_SENTENCES
        self.current_idx = 0

        # Collection state
        self.collected = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Recording state
        self.is_recording = False
        self.record_start_time = None
        self.countdown = 0

        # UI state
        self.window_name = "SonZo - Sentence Collector"

    def load_progress(self):
        """Load previous collection progress."""
        progress_path = self.output_dir / "progress.json"
        if progress_path.exists():
            with open(progress_path, 'r') as f:
                data = json.load(f)
                self.collected = data.get("collected", [])
                self.current_idx = len(self.collected) % len(self.sentences)
                print(f"📂 Loaded {len(self.collected)} previous recordings")

    def save_progress(self):
        """Save collection progress."""
        progress_path = self.output_dir / "progress.json"
        with open(progress_path, 'w') as f:
            json.dump({
                "collected": self.collected,
                "session_id": self.session_id,
                "total_sentences": len(self.sentences)
            }, f, indent=2)

    def export_for_training(self):
        """Export collected data in CTC training format."""
        if not self.collected:
            print("No data to export")
            return

        # Build vocabulary
        all_signs = set()
        for item in self.collected:
            all_signs.update(item["signs"])

        sign_to_idx = {"<blank>": 0}
        for i, sign in enumerate(sorted(all_signs)):
            sign_to_idx[sign] = i + 1

        # Save vocabulary
        vocab_path = self.output_dir / "vocabulary.json"
        with open(vocab_path, 'w') as f:
            json.dump({
                "sign_to_idx": sign_to_idx,
                "idx_to_sign": {v: k for k, v in sign_to_idx.items()},
                "vocab_size": len(sign_to_idx)
            }, f, indent=2)

        # Create sentence metadata
        sentences = []
        for item in self.collected:
            sentences.append({
                "id": item["id"],
                "signs": item["signs"],
                "video_path": item["video_path"],
                "duration_sec": item.get("duration", 0),
                "split": "train"  # Can be split later
            })

        # Split 80/10/10
        random.shuffle(sentences)
        n = len(sentences)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        for i, s in enumerate(sentences):
            if i < n_train:
                s["split"] = "train"
            elif i < n_train + n_val:
                s["split"] = "val"
            else:
                s["split"] = "test"

        # Save splits
        train = [s for s in sentences if s["split"] == "train"]
        val = [s for s in sentences if s["split"] == "val"]
        test = [s for s in sentences if s["split"] == "test"]

        with open(self.output_dir / "train_sentences.json", 'w') as f:
            json.dump({"sentences": train}, f, indent=2)
        with open(self.output_dir / "val_sentences.json", 'w') as f:
            json.dump({"sentences": val}, f, indent=2)
        with open(self.output_dir / "test_sentences.json", 'w') as f:
            json.dump({"sentences": test}, f, indent=2)

        print(f"\n✅ Exported {len(sentences)} sentences for training")
        print(f"   Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        print(f"   Vocabulary: {len(sign_to_idx)} signs")
        print(f"   Output: {self.output_dir}")

    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI overlay on frame."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Semi-transparent overlay at top
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        # Current sentence to sign
        current_sentence = self.sentences[self.current_idx]
        sentence_text = " ".join(current_sentence)

        # Title
        cv2.putText(display, "Sign this sentence:", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Sentence (larger, yellow)
        cv2.putText(display, sentence_text, (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # Progress
        progress_text = f"Progress: {len(self.collected)}/{len(self.sentences) * 3}"
        cv2.putText(display, progress_text, (20, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Recording indicator
        if self.is_recording:
            # Blinking red dot
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(display, (w - 30, 30), 15, (0, 0, 255), -1)
            cv2.putText(display, "REC", (w - 70, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Recording time
            elapsed = time.time() - self.record_start_time
            time_text = f"{elapsed:.1f}s"
            cv2.putText(display, time_text, (w - 70, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        elif self.countdown > 0:
            # Countdown display
            cv2.putText(display, str(self.countdown), (w // 2 - 30, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)

        # Instructions at bottom
        instructions = [
            "SPACE: Start/Stop recording",
            "N: Next sentence",
            "P: Previous sentence",
            "E: Export for training",
            "Q: Quit"
        ]

        y_pos = h - 20 - (len(instructions) * 25)
        for instr in instructions:
            cv2.putText(display, instr, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y_pos += 25

        return display

    def run(self):
        """Main collection loop."""
        print("\n" + "=" * 60)
        print("SonZo AI - Sentence Data Collector")
        print("=" * 60)

        # Initialize camera
        if not self.camera.initialize():
            print("Failed to initialize camera")
            return

        # Load previous progress
        self.load_progress()

        print(f"\nSentences to collect: {len(self.sentences)}")
        print(f"Already collected: {len(self.collected)}")
        print("\nControls:")
        print("  SPACE - Start/Stop recording")
        print("  N     - Next sentence")
        print("  P     - Previous sentence")
        print("  E     - Export for training")
        print("  Q     - Quit")
        print("\n" + "=" * 60)

        cv2.namedWindow(self.window_name)

        try:
            while True:
                # Capture frame
                frame = self.camera.capture_frame()
                if frame is None:
                    continue

                # Draw UI
                display = self.draw_ui(frame)
                cv2.imshow(self.window_name, display)

                # Handle countdown
                if self.countdown > 0:
                    time.sleep(0.5)
                    self.countdown -= 1
                    if self.countdown == 0:
                        self._start_recording()
                    continue

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                elif key == ord(' '):  # Space - toggle recording
                    if not self.is_recording:
                        # Start countdown
                        self.countdown = 3
                        print("Starting in 3...")
                    else:
                        self._stop_recording()

                elif key == ord('n'):  # Next sentence
                    self.current_idx = (self.current_idx + 1) % len(self.sentences)

                elif key == ord('p'):  # Previous sentence
                    self.current_idx = (self.current_idx - 1) % len(self.sentences)

                elif key == ord('e'):  # Export
                    self.export_for_training()

        finally:
            self.camera.close()
            cv2.destroyAllWindows()
            self.save_progress()
            print(f"\n✅ Session complete. Collected {len(self.collected)} recordings.")

    def _start_recording(self):
        """Start recording current sentence."""
        sentence = self.sentences[self.current_idx]
        sentence_slug = "_".join(sentence).lower()
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"{self.session_id}_{sentence_slug}_{timestamp}.mp4"

        self.camera.start_recording(filename)
        self.is_recording = True
        self.record_start_time = time.time()

    def _stop_recording(self):
        """Stop recording and save annotation."""
        video_path = self.camera.stop_recording()
        duration = time.time() - self.record_start_time

        self.is_recording = False

        # Save annotation
        sentence = self.sentences[self.current_idx]
        record_id = f"rec_{len(self.collected):05d}"

        self.collected.append({
            "id": record_id,
            "signs": sentence,
            "video_path": video_path,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })

        self.save_progress()

        print(f"✅ Saved: {' '.join(sentence)} ({duration:.1f}s)")

        # Move to next sentence
        self.current_idx = (self.current_idx + 1) % len(self.sentences)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Collect sentence-level sign language data')

    parser.add_argument('--camera', type=str, default='oak',
                       choices=['oak', 'webcam'],
                       help='Camera type to use')
    parser.add_argument('--output', type=str, default='./sentence_data',
                       help='Output directory')
    parser.add_argument('--sentences', type=str,
                       help='Path to custom sentences file (one per line, signs space-separated)')

    args = parser.parse_args()

    # Load custom sentences if provided
    sentences = None
    if args.sentences:
        with open(args.sentences, 'r') as f:
            sentences = [line.strip().split() for line in f if line.strip()]
        print(f"Loaded {len(sentences)} custom sentences")

    # Create collector
    collector = SentenceCollector(
        camera_type=args.camera,
        output_dir=args.output,
        sentences=sentences
    )

    # Run collection
    collector.run()


if __name__ == "__main__":
    main()
