#!/usr/bin/env python3
"""
SonZo AI - Demo Recognition API
=================================
A simulated recognition API for investor demos.
Uses hand detection + gesture heuristics instead of full ML model.

This allows demos to work before the model is fully trained.

Usage:
    python demo_recognition_api.py --port 8082

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import argparse
import base64
import io
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Try to import MediaPipe for hand detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available - using simulated hand detection")


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Demo configuration."""
    # Simulated recognition settings
    DEMO_SIGNS = [
        "HELLO", "THANK_YOU", "YES", "NO", "PLEASE", "SORRY",
        "HELP", "LOVE", "FRIEND", "GOOD", "BAD", "WANT",
        "A", "B", "C", "D", "E", "1", "2", "3"
    ]

    # Recognition simulation
    RECOGNITION_DELAY = 0.3  # Simulate processing time
    CONFIDENCE_RANGE = (0.75, 0.98)  # Random confidence range

    # Hand detection
    MIN_HAND_SIZE = 0.1  # Minimum hand size relative to frame


# =============================================================================
# Models
# =============================================================================

class RecognitionRequest(BaseModel):
    """Recognition request."""
    image: str  # Base64 encoded image
    timestamp: Optional[int] = None
    single: bool = False


class HandLandmark(BaseModel):
    """Hand landmark position."""
    x: float
    y: float
    z: float = 0.0


class HandData(BaseModel):
    """Hand detection data."""
    landmarks: List[HandLandmark]
    handedness: str = "Right"
    confidence: float = 0.9


class RecognitionResult(BaseModel):
    """Recognition result."""
    sign: Optional[str] = None
    confidence: float = 0.0
    hands: Optional[List[HandData]] = None
    state: str = "ready"  # ready, processing, recognized, error
    top_predictions: Optional[List[Dict[str, Any]]] = None
    latency_ms: float = 0.0


# =============================================================================
# Hand Detector
# =============================================================================

class HandDetector:
    """Hand detection wrapper."""

    def __init__(self):
        self.hands = None

        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )

    def detect(self, image: np.ndarray) -> Optional[List[HandData]]:
        """Detect hands in image."""
        if self.hands is None:
            # Simulate hand detection
            return self._simulate_detection(image)

        # Use MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        if not results.multi_hand_landmarks:
            return None

        hands = []
        for i, landmarks in enumerate(results.multi_hand_landmarks):
            hand_landmarks = [
                HandLandmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in landmarks.landmark
            ]

            handedness = "Right"
            if results.multi_handedness:
                handedness = results.multi_handedness[i].classification[0].label

            hands.append(HandData(
                landmarks=hand_landmarks,
                handedness=handedness,
                confidence=0.9
            ))

        return hands

    def _simulate_detection(self, image: np.ndarray) -> Optional[List[HandData]]:
        """Simulate hand detection for demos."""
        # Simple skin color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Skin color range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest contour (likely hand)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        frame_area = image.shape[0] * image.shape[1]

        if area / frame_area < Config.MIN_HAND_SIZE:
            return None

        # Generate simulated landmarks
        hull = cv2.convexHull(largest)
        center = np.mean(hull, axis=0)[0]

        landmarks = self._generate_simulated_landmarks(center, image.shape)

        return [HandData(
            landmarks=landmarks,
            handedness="Right",
            confidence=0.85
        )]

    def _generate_simulated_landmarks(
        self,
        center: Tuple[float, float],
        shape: Tuple[int, int, int]
    ) -> List[HandLandmark]:
        """Generate simulated hand landmarks."""
        h, w = shape[:2]
        cx, cy = center[0] / w, center[1] / h

        # 21 landmarks for hand
        landmarks = []

        # Wrist
        landmarks.append(HandLandmark(x=cx, y=cy + 0.15))

        # Thumb
        for i in range(4):
            landmarks.append(HandLandmark(
                x=cx - 0.08 + i * 0.02,
                y=cy + 0.1 - i * 0.05
            ))

        # Fingers
        for finger in range(4):
            base_x = cx - 0.04 + finger * 0.03
            for joint in range(4):
                landmarks.append(HandLandmark(
                    x=base_x,
                    y=cy + 0.08 - joint * 0.05
                ))

        return landmarks[:21]


# =============================================================================
# Demo Recognizer
# =============================================================================

class DemoRecognizer:
    """Demo sign language recognizer."""

    def __init__(self):
        self.detector = HandDetector()
        self.last_recognition_time = 0
        self.recognition_cooldown = 1.5  # seconds
        self.buffer: List[str] = []

        # Demo sequence for scripted demos
        self.demo_sequence = ["HELLO", "THANK_YOU", "NICE", "MEET", "YOU"]
        self.demo_index = 0
        self.scripted_mode = False

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """Recognize sign in image."""
        start_time = time.time()

        # Detect hands
        hands = self.detector.detect(image)

        if not hands:
            return RecognitionResult(
                state="ready",
                latency_ms=(time.time() - start_time) * 1000
            )

        # Simulate processing delay
        time.sleep(Config.RECOGNITION_DELAY)

        # Check cooldown
        now = time.time()
        if now - self.last_recognition_time < self.recognition_cooldown:
            return RecognitionResult(
                hands=[h.dict() for h in hands],
                state="processing",
                latency_ms=(time.time() - start_time) * 1000
            )

        # Generate recognition
        sign, confidence = self._generate_recognition(hands)

        self.last_recognition_time = now

        # Top predictions
        top_predictions = self._generate_top_predictions(sign, confidence)

        return RecognitionResult(
            sign=sign,
            confidence=confidence,
            hands=[h.dict() for h in hands],
            state="recognized",
            top_predictions=top_predictions,
            latency_ms=(time.time() - start_time) * 1000
        )

    def _generate_recognition(
        self,
        hands: List[HandData]
    ) -> Tuple[str, float]:
        """Generate recognition result."""
        if self.scripted_mode:
            # Use demo sequence
            sign = self.demo_sequence[self.demo_index]
            self.demo_index = (self.demo_index + 1) % len(self.demo_sequence)
        else:
            # Analyze hand pose for heuristic recognition
            sign = self._heuristic_recognition(hands[0])

        confidence = random.uniform(*Config.CONFIDENCE_RANGE)

        return sign, confidence

    def _heuristic_recognition(self, hand: HandData) -> str:
        """Use simple heuristics to guess sign."""
        landmarks = hand.landmarks

        if len(landmarks) < 21:
            return random.choice(Config.DEMO_SIGNS)

        # Get finger positions
        wrist = landmarks[0]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        thumb_tip = landmarks[4]

        # Simple gesture heuristics
        fingers_up = sum([
            index_tip.y < landmarks[6].y,   # Index up
            middle_tip.y < landmarks[10].y,  # Middle up
            ring_tip.y < landmarks[14].y,    # Ring up
            pinky_tip.y < landmarks[18].y,   # Pinky up
        ])

        thumb_out = abs(thumb_tip.x - wrist.x) > 0.1

        # Map gestures to signs
        if fingers_up == 0 and thumb_out:
            return "GOOD"  # Thumbs up
        elif fingers_up == 1:
            return "ONE" if not thumb_out else "D"
        elif fingers_up == 2:
            if thumb_out:
                return "I_LOVE_YOU"
            return "PEACE"
        elif fingers_up == 4:
            return "HELLO"  # Open hand
        elif fingers_up == 0:
            return "A"  # Fist
        else:
            # Use common demo signs
            return random.choice(["THANK_YOU", "YES", "NO", "PLEASE"])

    def _generate_top_predictions(
        self,
        top_sign: str,
        top_confidence: float
    ) -> List[Dict[str, Any]]:
        """Generate top 3 predictions."""
        predictions = [{"sign": top_sign, "confidence": top_confidence}]

        # Add 2 more predictions with lower confidence
        remaining = [s for s in Config.DEMO_SIGNS if s != top_sign]

        for i in range(2):
            if remaining:
                sign = random.choice(remaining)
                remaining.remove(sign)
                confidence = top_confidence * (0.5 - i * 0.15)
                predictions.append({"sign": sign, "confidence": confidence})

        return predictions

    def set_scripted_mode(self, enabled: bool):
        """Enable scripted demo mode."""
        self.scripted_mode = enabled
        self.demo_index = 0


# =============================================================================
# API
# =============================================================================

app = FastAPI(
    title="SonZo AI Recognition API (Demo)",
    description="Demo recognition API with simulated responses",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recognizer
recognizer = DemoRecognizer()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mode": "demo",
        "mediapipe_available": MEDIAPIPE_AVAILABLE
    }


@app.post("/api/recognize", response_model=RecognitionResult)
async def recognize(request: RecognitionRequest):
    """Recognize sign in image."""
    try:
        # Decode image
        image_data = request.image
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # Recognize
        result = recognizer.recognize(image)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/demo/scripted")
async def set_scripted_mode(enabled: bool = True):
    """Enable scripted demo mode."""
    recognizer.set_scripted_mode(enabled)
    return {"scripted_mode": enabled}


@app.post("/api/demo/sequence")
async def set_demo_sequence(signs: List[str]):
    """Set custom demo sequence."""
    recognizer.demo_sequence = signs
    recognizer.demo_index = 0
    return {"sequence": signs}


@app.get("/api/signs")
async def get_available_signs():
    """Get available demo signs."""
    return {"signs": Config.DEMO_SIGNS}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SonZo Demo Recognition API")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--host", type=str, default="0.0.0.0")

    args = parser.parse_args()

    print("=" * 60)
    print("SonZo AI - Demo Recognition API")
    print("=" * 60)
    print(f"Mode: DEMO (simulated recognition)")
    print(f"MediaPipe: {'Available' if MEDIAPIPE_AVAILABLE else 'Not available'}")
    print(f"Server: http://{args.host}:{args.port}")
    print()

    uvicorn.run(app, host=args.host, port=args.port)
