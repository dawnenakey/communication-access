#!/usr/bin/env python3
"""
Landmark-Based Recognition API
===============================
Production-ready API for real-time sign language recognition
using the landmark-based model.

This replaces the demo/heuristic-based recognition with actual
trained model predictions.

Endpoints:
- POST /recognize - Recognize sign from image
- POST /recognize-video - Recognize from video clip
- POST /translate - Translate ASL gloss to English (AWS Bedrock)
- GET /health - Health check
- GET /glossary - Get supported signs

Usage:
    python demo/landmark_recognition_api.py --model ./models/best_landmark_model.pt --port 8082

For production (with gunicorn):
    gunicorn demo.landmark_recognition_api:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8082

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import threading

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# AWS Bedrock for ASL-to-English translation
try:
    import boto3
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False
    print("Warning: boto3 not available - translation disabled")

# Add parent directory for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

# Import PyTorch
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available - using mock predictions")

# Import MediaPipe
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        mp_holistic = mp.solutions.holistic
    else:
        from mediapipe import solutions as mp_solutions
        mp_holistic = mp_solutions.holistic
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available")


# =============================================================================
# ASL-to-English Translator (AWS Bedrock)
# =============================================================================

class ASLTranslator:
    """Translates ASL gloss sequences to natural English using AWS Bedrock."""

    def __init__(self, region: str = "us-east-1"):
        self.client = None
        self.region = region
        self.model_id = "anthropic.claude-3-haiku-20240307-v1:0"

        if BEDROCK_AVAILABLE:
            try:
                self.client = boto3.client(
                    'bedrock-runtime',
                    region_name=region
                )
                print(f"Bedrock translator initialized (region: {region})")
            except Exception as e:
                print(f"Failed to initialize Bedrock: {e}")
                self.client = None

    def translate(self, signs: List[str]) -> Optional[str]:
        """Convert ASL gloss sequence to natural English."""
        if not self.client or not signs:
            return None

        # Build prompt for ASL-to-English translation
        gloss_sequence = " ".join(signs)

        prompt = f"""You are an ASL (American Sign Language) to English translator.
Convert the following ASL gloss sequence into natural, grammatically correct English.

ASL Grammar Notes:
- ASL uses topic-comment structure (e.g., "STORE I GO" means "I'm going to the store")
- ASL often omits articles (a, an, the) and auxiliary verbs (is, are, was)
- Time signs often come first (e.g., "TOMORROW I WORK" means "I will work tomorrow")
- Questions use specific facial grammar, assume statements unless context suggests otherwise

ASL Gloss: {gloss_sequence}

Respond with ONLY the natural English translation, nothing else."""

        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 200,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response['body'].read())
            translation = response_body['content'][0]['text'].strip()
            return translation

        except Exception as e:
            print(f"Translation error: {e}")
            return None

    def is_available(self) -> bool:
        """Check if translator is available."""
        return self.client is not None


# Global translator instance
translator: Optional[ASLTranslator] = None


# =============================================================================
# Request/Response Models
# =============================================================================

class RecognitionRequest(BaseModel):
    """Recognition request with base64 image."""
    image: str  # Base64 encoded image
    include_landmarks: bool = False


class VideoRecognitionRequest(BaseModel):
    """Recognition request with multiple frames."""
    frames: List[str]  # List of base64 encoded frames
    fps: int = 15


class LandmarkPoint(BaseModel):
    """Single landmark point."""
    x: float
    y: float
    z: float = 0.0


class RecognitionResult(BaseModel):
    """Recognition result."""
    success: bool = True
    sign: Optional[str] = None
    confidence: float = 0.0
    top_predictions: List[Dict[str, Any]] = []
    landmarks: Optional[Dict[str, List[LandmarkPoint]]] = None
    latency_ms: float = 0.0
    message: str = ""
    english_translation: Optional[str] = None  # ASL-to-English via Bedrock


class GlossaryResponse(BaseModel):
    """Glossary of supported signs."""
    signs: List[str]
    count: int


class TranslationRequest(BaseModel):
    """Request for ASL-to-English translation."""
    signs: List[str]  # List of ASL gloss signs


class TranslationResult(BaseModel):
    """Translation result."""
    success: bool = True
    asl_gloss: str = ""
    english: Optional[str] = None
    latency_ms: float = 0.0
    message: str = ""


# =============================================================================
# Landmark Model Wrapper
# =============================================================================

class LandmarkModel:
    """Wrapper for the trained landmark model."""

    # Feature dimensions (must match training)
    HAND_LANDMARKS = 21
    POSE_LANDMARKS = 13  # Upper body subset

    # Confidence and motion thresholds
    MIN_CONFIDENCE = 0.65  # Minimum confidence to show prediction
    MOTION_THRESHOLD = 0.02  # Minimum motion to consider as signing
    MIN_HAND_LANDMARKS = 10  # Minimum hand landmarks detected

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.label_map = {}
        self.idx_to_label = {}
        self.feature_dim = 165  # 2 hands * 21 * 3 + 13 * 3

        # Frame buffer for temporal predictions
        self.frame_buffer = deque(maxlen=32)
        self.buffer_lock = threading.Lock()

        # Motion detection
        self.prev_landmarks = None
        self.motion_history = deque(maxlen=10)
        self.hand_detected_history = deque(maxlen=10)

        # Load model
        if os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print(f"Warning: Model not found at {model_path}")

        # Initialize MediaPipe
        if MEDIAPIPE_AVAILABLE:
            self.holistic = mp_holistic.Holistic(
                static_image_mode=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.holistic = None

    def _load_model(self, path: str):
        """Load model from checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Get config
            config = checkpoint.get("config", {})
            self.label_map = checkpoint.get("label_map", {})
            self.idx_to_label = {v: k for k, v in self.label_map.items()}

            # Recreate model architecture
            from train_landmarks import LandmarkLSTM, TrainingConfig

            model_config = TrainingConfig(
                input_dim=config.get("input_dim", 165),
                hidden_size=config.get("hidden_size", 256),
                num_layers=config.get("num_layers", 2),
                num_classes=len(self.label_map),
                dropout=config.get("dropout", 0.3),
                bidirectional=config.get("bidirectional", True)
            )

            self.model = LandmarkLSTM(model_config).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            print(f"Loaded model with {len(self.label_map)} classes")

        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def extract_landmarks(self, image: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """
        Extract landmarks from a single frame.

        Returns:
            features: Landmark feature vector
            hands_detected: Whether at least one hand was detected
            motion_amount: Amount of motion from previous frame
        """
        if not MEDIAPIPE_AVAILABLE or self.holistic is None:
            return np.zeros(self.feature_dim, dtype=np.float32), False, 0.0

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_image)

        # Initialize feature vector
        features = np.zeros(self.feature_dim, dtype=np.float32)
        idx = 0
        hand_landmarks_count = 0

        # Left hand (21 landmarks * 3 coords = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                features[idx:idx+3] = [lm.x * 2 - 1, lm.y * 2 - 1, lm.z * 2 - 1]
                idx += 3
            hand_landmarks_count += 21
        else:
            idx += 63

        # Right hand (21 landmarks * 3 coords = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                features[idx:idx+3] = [lm.x * 2 - 1, lm.y * 2 - 1, lm.z * 2 - 1]
                idx += 3
            hand_landmarks_count += 21
        else:
            idx += 63

        # Upper body pose (13 landmarks * 3 coords = 39)
        if results.pose_landmarks:
            upper_body_indices = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24]
            for i in upper_body_indices:
                lm = results.pose_landmarks.landmark[i]
                features[idx:idx+3] = [lm.x * 2 - 1, lm.y * 2 - 1, lm.z * 2 - 1]
                idx += 3

        # Calculate motion from previous frame
        motion_amount = 0.0
        if self.prev_landmarks is not None:
            # Only compare hand landmarks (first 126 features)
            hand_features = features[:126]
            prev_hand_features = self.prev_landmarks[:126]
            motion_amount = np.mean(np.abs(hand_features - prev_hand_features))

        self.prev_landmarks = features.copy()

        # Track hand detection and motion
        hands_detected = hand_landmarks_count >= self.MIN_HAND_LANDMARKS
        self.hand_detected_history.append(hands_detected)
        self.motion_history.append(motion_amount)

        return features, hands_detected, motion_amount

    def add_frame(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Add frame to buffer for temporal prediction.

        Returns:
            hands_detected: Whether hands were detected
            motion_amount: Amount of motion detected
        """
        landmarks, hands_detected, motion_amount = self.extract_landmarks(image)
        with self.buffer_lock:
            self.frame_buffer.append(landmarks)
        return hands_detected, motion_amount

    def has_sufficient_motion(self) -> bool:
        """Check if there's enough motion to indicate signing."""
        if len(self.motion_history) < 5:
            return False
        avg_motion = sum(self.motion_history) / len(self.motion_history)
        return avg_motion > self.MOTION_THRESHOLD

    def has_hands_detected(self) -> bool:
        """Check if hands have been consistently detected."""
        if len(self.hand_detected_history) < 3:
            return False
        # At least 60% of recent frames should have hands
        return sum(self.hand_detected_history) / len(self.hand_detected_history) >= 0.6

    def predict(self, min_frames: int = 8) -> Tuple[str, float, List[Dict], str]:
        """
        Predict sign from buffered frames.

        Returns:
            sign: Predicted sign (or None)
            confidence: Prediction confidence
            top_predictions: Top 5 predictions
            status: Status message ("signing", "no_hands", "no_motion", "collecting")
        """
        with self.buffer_lock:
            frames = list(self.frame_buffer)

        # Check prerequisites
        if len(frames) < min_frames:
            return None, 0.0, [], "collecting"

        if not self.has_hands_detected():
            return None, 0.0, [], "no_hands"

        if not self.has_sufficient_motion():
            return None, 0.0, [], "no_motion"

        if self.model is None:
            # Return mock prediction if no model
            return "HELLO", 0.85, [
                {"sign": "HELLO", "confidence": 0.85},
                {"sign": "HI", "confidence": 0.10},
                {"sign": "WAVE", "confidence": 0.05}
            ], "signing"

        # Prepare input
        sequence = np.array(frames, dtype=np.float32)

        # Pad or truncate to 64 frames
        max_len = 64
        if len(sequence) > max_len:
            # Sample uniformly
            indices = np.linspace(0, len(sequence) - 1, max_len, dtype=int)
            sequence = sequence[indices]
        elif len(sequence) < max_len:
            # Pad with zeros
            padding = np.zeros((max_len - len(sequence), self.feature_dim), dtype=np.float32)
            sequence = np.concatenate([sequence, padding], axis=0)

        # Convert to tensor
        x = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        lengths = torch.tensor([len(frames)], dtype=torch.long).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(x, lengths)
            probs = F.softmax(logits, dim=-1)

            # Get top predictions
            top_k = min(5, len(self.label_map))
            top_probs, top_indices = probs.topk(top_k, dim=-1)

            top_predictions = []
            for i in range(top_k):
                idx = top_indices[0, i].item()
                conf = top_probs[0, i].item()
                sign = self.idx_to_label.get(idx, f"SIGN_{idx}")
                top_predictions.append({"sign": sign, "confidence": round(conf, 3)})

            # Best prediction
            best_sign = top_predictions[0]["sign"]
            best_conf = top_predictions[0]["confidence"]

            # Apply minimum confidence threshold
            if best_conf < self.MIN_CONFIDENCE:
                return None, best_conf, top_predictions, "low_confidence"

        return best_sign, best_conf, top_predictions, "signing"

    def clear_buffer(self):
        """Clear frame buffer."""
        with self.buffer_lock:
            self.frame_buffer.clear()

    def get_glossary(self) -> List[str]:
        """Get list of supported signs."""
        return sorted(self.label_map.keys())


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="SonZo AI - Sign Language Recognition API",
    description="Real-time ASL recognition using landmark-based deep learning",
    version="2.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: Optional[LandmarkModel] = None


def decode_image(base64_str: str) -> np.ndarray:
    """Decode base64 image to numpy array."""
    # Remove data URL prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    img_bytes = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return image


@app.get("/", response_class=HTMLResponse)
async def serve_demo():
    """Serve the demo HTML page."""
    demo_path = SCRIPT_DIR / "recognition_demo.html"
    if demo_path.exists():
        return FileResponse(demo_path, media_type="text/html")
    return HTMLResponse("<h1>Demo not found</h1>", status_code=404)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None and model.model is not None,
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "bedrock_available": translator is not None and translator.is_available(),
        "device": str(model.device) if model else "N/A",
        "num_classes": len(model.label_map) if model else 0
    }


@app.get("/glossary", response_model=GlossaryResponse)
async def get_glossary():
    """Get list of supported signs."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    signs = model.get_glossary()
    return GlossaryResponse(signs=signs, count=len(signs))


@app.post("/recognize", response_model=RecognitionResult)
async def recognize_sign(request: RecognitionRequest):
    """Recognize sign from a single image frame."""
    start_time = time.time()

    if model is None:
        return RecognitionResult(
            success=False,
            message="Model not loaded",
            latency_ms=(time.time() - start_time) * 1000
        )

    try:
        # Decode image
        image = decode_image(request.image)
        if image is None:
            return RecognitionResult(
                success=False,
                message="Failed to decode image",
                latency_ms=(time.time() - start_time) * 1000
            )

        # Add frame to buffer (also returns detection info)
        hands_detected, motion_amount = model.add_frame(image)

        # Predict
        sign, confidence, top_predictions, status = model.predict(min_frames=8)

        # Generate user-friendly status messages
        status_messages = {
            "signing": "Recognition successful",
            "collecting": "Collecting frames...",
            "no_hands": "Position hands in view",
            "no_motion": "Waiting for sign...",
            "low_confidence": "Sign unclear - try again"
        }

        # Get landmarks if requested
        landmarks = None
        if request.include_landmarks and MEDIAPIPE_AVAILABLE:
            # Return raw landmarks for visualization
            pass  # TODO: implement if needed

        return RecognitionResult(
            success=True,
            sign=sign,
            confidence=confidence,
            top_predictions=top_predictions,
            landmarks=landmarks,
            latency_ms=(time.time() - start_time) * 1000,
            message=status_messages.get(status, "Processing...")
        )

    except Exception as e:
        return RecognitionResult(
            success=False,
            message=f"Error: {str(e)}",
            latency_ms=(time.time() - start_time) * 1000
        )


@app.post("/recognize-video", response_model=RecognitionResult)
async def recognize_video(request: VideoRecognitionRequest):
    """Recognize sign from multiple video frames."""
    start_time = time.time()

    if model is None:
        return RecognitionResult(
            success=False,
            message="Model not loaded",
            latency_ms=(time.time() - start_time) * 1000
        )

    try:
        # Clear buffer and add all frames
        model.clear_buffer()

        for frame_b64 in request.frames:
            image = decode_image(frame_b64)
            if image is not None:
                model.add_frame(image)

        # Predict
        sign, confidence, top_predictions = model.predict(min_frames=4)

        return RecognitionResult(
            success=True,
            sign=sign,
            confidence=confidence,
            top_predictions=top_predictions,
            latency_ms=(time.time() - start_time) * 1000,
            message="Recognition successful" if sign else "Not enough valid frames"
        )

    except Exception as e:
        return RecognitionResult(
            success=False,
            message=f"Error: {str(e)}",
            latency_ms=(time.time() - start_time) * 1000
        )


@app.post("/clear")
async def clear_buffer():
    """Clear the frame buffer."""
    if model:
        model.clear_buffer()
    return {"status": "cleared"}


@app.post("/translate", response_model=TranslationResult)
async def translate_asl(request: TranslationRequest):
    """Translate ASL gloss sequence to natural English using AWS Bedrock."""
    start_time = time.time()

    if translator is None or not translator.is_available():
        return TranslationResult(
            success=False,
            asl_gloss=" ".join(request.signs),
            message="Translation service not available",
            latency_ms=(time.time() - start_time) * 1000
        )

    try:
        english = translator.translate(request.signs)

        return TranslationResult(
            success=True,
            asl_gloss=" ".join(request.signs),
            english=english,
            latency_ms=(time.time() - start_time) * 1000,
            message="Translation successful" if english else "Translation failed"
        )

    except Exception as e:
        return TranslationResult(
            success=False,
            asl_gloss=" ".join(request.signs),
            message=f"Error: {str(e)}",
            latency_ms=(time.time() - start_time) * 1000
        )


# =============================================================================
# Main
# =============================================================================

def main():
    global model, translator

    parser = argparse.ArgumentParser(description='Landmark Recognition API')
    parser.add_argument('--model', type=str, default='./models/best_landmark_model.pt',
                       help='Path to trained model')
    parser.add_argument('--port', type=int, default=8082,
                       help='Port to run server on')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--ssl-cert', type=str, default=None,
                       help='Path to SSL certificate (fullchain.pem)')
    parser.add_argument('--ssl-key', type=str, default=None,
                       help='Path to SSL private key (privkey.pem)')
    parser.add_argument('--aws-region', type=str, default='us-east-1',
                       help='AWS region for Bedrock')
    parser.add_argument('--no-translate', action='store_true',
                       help='Disable ASL-to-English translation')

    args = parser.parse_args()

    # Initialize model
    print(f"Loading model from {args.model}...")
    model = LandmarkModel(args.model, device=args.device)

    # Initialize ASL-to-English translator (AWS Bedrock)
    if not args.no_translate and BEDROCK_AVAILABLE:
        print(f"Initializing Bedrock translator (region: {args.aws_region})...")
        translator = ASLTranslator(region=args.aws_region)
        if translator.is_available():
            print("✓ ASL-to-English translation enabled")
        else:
            print("✗ Bedrock not available - check AWS credentials")
    else:
        translator = None
        print("Translation disabled")

    # Run server
    ssl_config = {}
    if args.ssl_cert and args.ssl_key:
        ssl_config = {
            'ssl_certfile': args.ssl_cert,
            'ssl_keyfile': args.ssl_key
        }
        print(f"Starting HTTPS server on {args.host}:{args.port}")
    else:
        print(f"Starting HTTP server on {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, **ssl_config)


if __name__ == "__main__":
    main()
