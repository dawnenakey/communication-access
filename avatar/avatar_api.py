#!/usr/bin/env python3
"""
Avatar Generation API
======================
FastAPI backend for personalized signing avatar generation.

Endpoints:
- POST /api/avatar/create - Create avatar from face photo
- POST /api/avatar/{id}/sign - Generate signing video
- GET /api/phrases - List available phrases
- GET /api/avatar/{id}/videos - Get generated videos

Usage:
    uvicorn avatar_api:app --host 0.0.0.0 --port 8080

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import asyncio
import base64
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Module imports (lazy loaded)
face_capture = None
face_swap = None


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """API configuration."""
    DATA_DIR = Path(os.environ.get("AVATAR_DATA_DIR", "./avatar_data"))
    VIDEO_LIBRARY = Path(os.environ.get("VIDEO_LIBRARY", "./avatar/video_library"))
    MAX_AVATARS_PER_USER = 5
    VIDEO_EXPIRY_HOURS = 24
    ALLOWED_ORIGINS = ["*"]


# Create directories
Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
(Config.DATA_DIR / "avatars").mkdir(exist_ok=True)
(Config.DATA_DIR / "outputs").mkdir(exist_ok=True)


# =============================================================================
# Models
# =============================================================================

class AvatarCreateRequest(BaseModel):
    """Request to create avatar."""
    photo: str = Field(..., description="Base64 encoded face photo")
    name: Optional[str] = Field(None, description="Avatar name")


class AvatarCreateResponse(BaseModel):
    """Response from avatar creation."""
    avatar_id: str
    preview_url: str
    quality_score: float
    validation: Dict[str, Any]


class SignRequest(BaseModel):
    """Request to generate signing video."""
    phrase: str = Field(..., description="Phrase to sign (e.g., 'HELLO')")


class SignResponse(BaseModel):
    """Response from sign generation."""
    video_url: str
    phrase: str
    processing_time: float
    status: str


class PhraseInfo(BaseModel):
    """Information about an available phrase."""
    phrase: str
    category: str
    description: Optional[str] = None
    has_video: bool = True


class AvatarInfo(BaseModel):
    """Avatar information."""
    avatar_id: str
    name: Optional[str]
    created_at: str
    quality_score: float
    generated_videos: List[str]


# =============================================================================
# API Application
# =============================================================================

app = FastAPI(
    title="SonZo AI Avatar API",
    description="Personalized signing avatar generation",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=str(Config.DATA_DIR)), name="static")


# =============================================================================
# Helper Functions
# =============================================================================

def get_face_capture():
    """Lazy load face capture module."""
    global face_capture
    if face_capture is None:
        from face_capture import FaceCapture
        face_capture = FaceCapture()
    return face_capture


def get_face_swapper():
    """Lazy load face swapper module."""
    global face_swap
    if face_swap is None:
        from face_swap import FaceSwapper
        face_swap = FaceSwapper()
    return face_swap


def load_avatar_metadata(avatar_id: str) -> Optional[Dict]:
    """Load avatar metadata from disk."""
    metadata_path = Config.DATA_DIR / "avatars" / avatar_id / "metadata.json"
    if not metadata_path.exists():
        return None
    with open(metadata_path, 'r') as f:
        return json.load(f)


def save_avatar_metadata(avatar_id: str, metadata: Dict):
    """Save avatar metadata to disk."""
    avatar_dir = Config.DATA_DIR / "avatars" / avatar_id
    avatar_dir.mkdir(parents=True, exist_ok=True)
    with open(avatar_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def get_available_phrases() -> List[PhraseInfo]:
    """Get list of available phrases with video support."""
    phrases = []

    # Define phrase categories
    phrase_data = {
        "GREETINGS": ["HELLO", "GOODBYE", "NICE_TO_MEET_YOU"],
        "COMMON": ["THANK_YOU", "PLEASE", "SORRY", "YES", "NO", "HELP"],
        "QUESTIONS": ["WHAT", "WHERE", "WHO", "WHY", "HOW", "WHEN"],
        "FEELINGS": ["I_LOVE_YOU", "HAPPY", "SAD", "UNDERSTAND"],
        "ACTIONS": ["WANT", "NEED", "LIKE", "KNOW", "LEARN", "FINISH"]
    }

    for category, phrase_list in phrase_data.items():
        for phrase in phrase_list:
            video_path = Config.VIDEO_LIBRARY / f"{phrase}.mp4"
            phrases.append(PhraseInfo(
                phrase=phrase,
                category=category,
                has_video=video_path.exists()
            ))

    return phrases


async def generate_signing_video(
    avatar_id: str,
    phrase: str
) -> str:
    """Generate signing video with user's face (background task)."""
    import time
    start_time = time.time()

    # Load avatar
    metadata = load_avatar_metadata(avatar_id)
    if not metadata:
        raise ValueError(f"Avatar not found: {avatar_id}")

    # Check for pre-recorded video
    video_path = Config.VIDEO_LIBRARY / f"{phrase.upper()}.mp4"
    if not video_path.exists():
        raise ValueError(f"No video available for phrase: {phrase}")

    # Get face image
    face_path = Config.DATA_DIR / "avatars" / avatar_id / "face.png"
    if not face_path.exists():
        raise ValueError("Avatar face not found")

    # Output path
    output_dir = Config.DATA_DIR / "outputs" / avatar_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{phrase.upper()}.mp4"

    # Perform face swap
    swapper = get_face_swapper()
    swapper.swap_video(
        str(face_path),
        str(video_path),
        str(output_path)
    )

    # Update metadata
    if "generated_videos" not in metadata:
        metadata["generated_videos"] = []
    if phrase.upper() not in metadata["generated_videos"]:
        metadata["generated_videos"].append(phrase.upper())
    save_avatar_metadata(avatar_id, metadata)

    processing_time = time.time() - start_time
    print(f"Generated {phrase} video in {processing_time:.1f}s")

    return str(output_path)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """API root."""
    return {
        "name": "SonZo AI Avatar API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/phrases", response_model=List[PhraseInfo])
async def list_phrases():
    """Get list of available phrases."""
    return get_available_phrases()


@app.post("/api/avatar/create", response_model=AvatarCreateResponse)
async def create_avatar(request: AvatarCreateRequest):
    """
    Create a new avatar from face photo.

    Accepts base64 encoded image and returns avatar ID for future use.
    """
    try:
        # Generate avatar ID
        avatar_id = str(uuid.uuid4())[:8]

        # Decode image
        try:
            photo_data = request.photo
            if photo_data.startswith('data:image'):
                photo_data = photo_data.split(',')[1]
            image_bytes = base64.b64decode(photo_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

        # Extract face
        capture = get_face_capture()

        import numpy as np
        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        face_data = capture.extract_face(img)

        if face_data is None:
            raise HTTPException(status_code=400, detail="No face detected in image")

        # Validate face
        validation = capture.validate_face(face_data)

        if not validation["is_valid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Face quality too low",
                    "issues": validation["issues"],
                    "recommendations": validation["recommendations"]
                }
            )

        # Save avatar
        avatar_dir = Config.DATA_DIR / "avatars" / avatar_id
        avatar_dir.mkdir(parents=True, exist_ok=True)

        # Save face image
        face_path = avatar_dir / "face.png"
        capture.save_face(face_data, str(face_path))

        # Save embedding
        np.save(str(avatar_dir / "embedding.npy"), face_data.embedding)

        # Create metadata
        metadata = {
            "avatar_id": avatar_id,
            "name": request.name,
            "created_at": datetime.utcnow().isoformat(),
            "quality_score": face_data.quality_score,
            "age": face_data.age,
            "gender": face_data.gender,
            "generated_videos": []
        }
        save_avatar_metadata(avatar_id, metadata)

        return AvatarCreateResponse(
            avatar_id=avatar_id,
            preview_url=f"/static/avatars/{avatar_id}/face.png",
            quality_score=face_data.quality_score,
            validation=validation
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/avatar/{avatar_id}/sign", response_model=SignResponse)
async def generate_sign(
    avatar_id: str,
    request: SignRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate signing video for avatar.

    Returns immediately with status "processing" and video URL.
    Poll the video URL until it's available.
    """
    import time
    start_time = time.time()

    # Validate avatar exists
    metadata = load_avatar_metadata(avatar_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Avatar not found")

    # Validate phrase
    phrase = request.phrase.upper()
    video_path = Config.VIDEO_LIBRARY / f"{phrase}.mp4"

    if not video_path.exists():
        available = [p.phrase for p in get_available_phrases() if p.has_video]
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"No video available for phrase: {phrase}",
                "available_phrases": available[:10]
            }
        )

    # Check if already generated
    output_path = Config.DATA_DIR / "outputs" / avatar_id / f"{phrase}.mp4"
    if output_path.exists():
        return SignResponse(
            video_url=f"/static/outputs/{avatar_id}/{phrase}.mp4",
            phrase=phrase,
            processing_time=0,
            status="ready"
        )

    # Generate video (synchronously for MVP, could be async)
    try:
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: generate_signing_video(avatar_id, phrase)
        )
        processing_time = time.time() - start_time

        return SignResponse(
            video_url=f"/static/outputs/{avatar_id}/{phrase}.mp4",
            phrase=phrase,
            processing_time=processing_time,
            status="ready"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/avatar/{avatar_id}", response_model=AvatarInfo)
async def get_avatar(avatar_id: str):
    """Get avatar information."""
    metadata = load_avatar_metadata(avatar_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Avatar not found")

    return AvatarInfo(
        avatar_id=metadata["avatar_id"],
        name=metadata.get("name"),
        created_at=metadata["created_at"],
        quality_score=metadata["quality_score"],
        generated_videos=metadata.get("generated_videos", [])
    )


@app.get("/api/avatar/{avatar_id}/videos")
async def list_avatar_videos(avatar_id: str):
    """List all generated videos for avatar."""
    metadata = load_avatar_metadata(avatar_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Avatar not found")

    videos = []
    output_dir = Config.DATA_DIR / "outputs" / avatar_id

    if output_dir.exists():
        for video_file in output_dir.glob("*.mp4"):
            phrase = video_file.stem
            videos.append({
                "phrase": phrase,
                "url": f"/static/outputs/{avatar_id}/{phrase}.mp4",
                "filename": video_file.name
            })

    return {"avatar_id": avatar_id, "videos": videos}


@app.delete("/api/avatar/{avatar_id}")
async def delete_avatar(avatar_id: str):
    """Delete avatar and all generated videos."""
    avatar_dir = Config.DATA_DIR / "avatars" / avatar_id
    output_dir = Config.DATA_DIR / "outputs" / avatar_id

    if not avatar_dir.exists():
        raise HTTPException(status_code=404, detail="Avatar not found")

    # Delete directories
    shutil.rmtree(avatar_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)

    return {"status": "deleted", "avatar_id": avatar_id}


@app.get("/api/avatar/{avatar_id}/video/{phrase}")
async def download_video(avatar_id: str, phrase: str):
    """Download generated video."""
    video_path = Config.DATA_DIR / "outputs" / avatar_id / f"{phrase.upper()}.mp4"

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{phrase.upper()}_avatar.mp4"
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Avatar Generation API')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--reload', action='store_true')

    args = parser.parse_args()

    print("=" * 60)
    print("SonZo AI - Avatar Generation API")
    print("=" * 60)
    print(f"Data directory: {Config.DATA_DIR}")
    print(f"Video library: {Config.VIDEO_LIBRARY}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    print()

    uvicorn.run(
        "avatar_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
