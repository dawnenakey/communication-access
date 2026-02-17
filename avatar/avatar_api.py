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
- POST /api/generate-sign - Generate sign video on-the-fly (SMPL-X)
- GET /api/signs/available - List all available signs

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
import subprocess
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
realtime_generator = None


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


def get_realtime_generator():
    """Lazy load real-time avatar generator (SMPL-X)."""
    global realtime_generator
    if realtime_generator is None:
        try:
            from realtime_avatar_generator import RealtimeAvatarGenerator
            realtime_generator = RealtimeAvatarGenerator(
                output_dir=str(Config.DATA_DIR / "generated")
            )
            print("✅ Real-time avatar generator loaded")
        except Exception as e:
            print(f"⚠️ Real-time generator not available: {e}")
            realtime_generator = None
    return realtime_generator


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
# Real-Time Sign Generation (SMPL-X)
# =============================================================================

class GenerateSignRequest(BaseModel):
    """Request to generate a sign video on-the-fly."""
    sign: str = Field(..., description="Sign name (e.g., 'HELLO')")
    quality: str = Field("standard", description="Quality: preview, standard, high, production")


class GenerateSequenceRequest(BaseModel):
    """Request to generate a sequence of signs (sentence in ASL)."""
    signs: List[str] = Field(..., description="List of sign names (e.g., ['HELLO', 'HOW', 'YOU'])")
    quality: str = Field("standard", description="Quality: preview, standard, high, production")
    avatar_id: Optional[str] = Field(None, description="Avatar ID for face swap (optional)")


class GenerateSignResponse(BaseModel):
    """Response from sign generation."""
    sign: str
    video_url: str
    duration_seconds: float
    quality: str
    generated_at: str


@app.get("/api/signs/available")
async def list_available_signs():
    """
    Get list of all signs that can be generated dynamically.
    These don't require pre-recorded videos - generated via SMPL-X.
    """
    generator = get_realtime_generator()

    if generator is None:
        # Fallback to pre-recorded phrases
        return {
            "realtime_available": False,
            "signs": [p.phrase for p in get_available_phrases() if p.has_video],
            "count": len([p for p in get_available_phrases() if p.has_video]),
            "message": "Real-time generation not available. Using pre-recorded videos."
        }

    signs = generator.get_available_signs()
    sign_details = []

    for sign_name in signs:
        info = generator.get_sign_info(sign_name)
        if info:
            sign_details.append(info)

    return {
        "realtime_available": True,
        "signs": signs,
        "count": len(signs),
        "details": sign_details,
        "categories": {
            "greetings": ["HELLO", "GOODBYE", "NICE_TO_MEET_YOU"],
            "common": ["THANK_YOU", "PLEASE", "SORRY", "YES", "NO"],
            "questions": ["WHAT", "WHERE", "WHO", "WHY", "HOW", "WHEN"],
            "feelings": ["I_LOVE_YOU", "HAPPY", "SAD", "UNDERSTAND"],
            "actions": ["HELP", "WANT", "NEED", "LIKE", "KNOW", "LEARN", "FINISH"],
            "pronouns": ["ME", "YOU", "MY", "YOUR", "WE"],
            "descriptors": ["GOOD", "BAD", "MORE", "AGAIN"],
            "daily": ["EAT", "DRINK", "SLEEP", "WORK"],
            "movement": ["WAIT", "STOP", "GO", "COME", "NAME"]
        }
    }


@app.post("/api/generate-sign", response_model=GenerateSignResponse)
async def generate_sign_realtime(request: GenerateSignRequest):
    """
    Generate a sign video on-the-fly using SMPL-X avatar.

    No pre-recorded videos needed - this generates any supported sign dynamically.
    Optionally applies face swap if avatar_id is provided.
    """
    import time
    start_time = time.time()

    generator = get_realtime_generator()

    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Real-time avatar generation not available"
        )

    sign_name = request.sign.upper()

    # Validate sign exists
    if sign_name not in generator.get_available_signs():
        available = generator.get_available_signs()
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Sign '{sign_name}' not available",
                "available_signs": available[:20],
                "total_available": len(available)
            }
        )

    # Map quality string to enum
    from realtime_avatar_generator import RenderQuality
    quality_map = {
        "preview": RenderQuality.PREVIEW,
        "standard": RenderQuality.STANDARD,
        "high": RenderQuality.HIGH,
        "production": RenderQuality.PRODUCTION
    }
    quality = quality_map.get(request.quality, RenderQuality.STANDARD)

    # Generate video
    try:
        video_path = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: generator.generate_sign_video(sign_name, quality)
        )

        if not video_path:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate video for {sign_name}"
            )

        # Optionally apply face swap
        if request.avatar_id:
            metadata = load_avatar_metadata(request.avatar_id)
            if metadata:
                face_path = Config.DATA_DIR / "avatars" / request.avatar_id / "face.png"
                if face_path.exists():
                    try:
                        swapper = get_face_swapper()
                        output_path = Config.DATA_DIR / "generated" / f"{sign_name.lower()}_{request.avatar_id}.mp4"
                        swapper.swap_video(str(face_path), video_path, str(output_path))
                        video_path = str(output_path)
                    except Exception as e:
                        print(f"Face swap failed: {e}")
                        # Continue with original video

        # Get sign info for duration
        sign_info = generator.get_sign_info(sign_name)
        duration = sign_info["duration_seconds"] if sign_info else 1.0

        processing_time = time.time() - start_time
        print(f"Generated {sign_name} in {processing_time:.1f}s")

        # Create URL path
        relative_path = Path(video_path).relative_to(Config.DATA_DIR)
        video_url = f"/static/{relative_path}"

        return GenerateSignResponse(
            sign=sign_name,
            video_url=video_url,
            duration_seconds=duration,
            quality=request.quality,
            generated_at=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _concatenate_videos_ffmpeg(video_paths: List[Path], output_path: Path) -> Optional[str]:
    """Concatenate videos using ffmpeg. Returns output path or None."""
    if not video_paths:
        return None
    if len(video_paths) == 1:
        shutil.copy2(video_paths[0], output_path)
        return str(output_path)
    list_path = output_path.parent / "concat_list.txt"
    try:
        with open(list_path, 'w') as f:
            for p in video_paths:
                f.write(f"file '{p.absolute()}'\n")
        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(output_path)],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and output_path.exists():
            return str(output_path)
    except Exception as e:
        print(f"Video concat error: {e}")
    return None


@app.post("/api/generate-sequence")
async def generate_sign_sequence(request: GenerateSequenceRequest):
    """
    Generate a video for a sequence of signs (e.g., a sentence in ASL).

    Example: ["HELLO", "HOW", "YOU"] -> Single video saying "Hello, how are you?"

    Uses real-time SMPL-X generation when available, or pre-recorded videos from
    VIDEO_LIBRARY (e.g. avatar/video_library/HELLO.mp4) as fallback.
    """
    signs = [s.upper() for s in request.signs]
    quality = request.quality

    generator = get_realtime_generator()

    if generator is not None:
        # Real-time generation (Blender + SMPL-X)
        available = set(generator.get_available_signs())
        invalid_signs = [s for s in signs if s not in available]
        if invalid_signs:
            raise HTTPException(
                status_code=400,
                detail={"message": f"Some signs not available: {invalid_signs}", "available_signs": list(available)}
            )
        from realtime_avatar_generator import RenderQuality
        quality_enum = RenderQuality(quality.lower()) if quality else RenderQuality.STANDARD
        try:
            video_path = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: generator.generate_sign_sequence(signs, quality_enum)
            )
            if video_path:
                relative_path = Path(video_path).relative_to(Config.DATA_DIR)
                return {"signs": signs, "video_url": f"/static/{relative_path}", "quality": quality}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Fallback: pre-recorded videos from VIDEO_LIBRARY
    video_lib = Path(Config.VIDEO_LIBRARY)
    video_paths = []
    for sign in signs:
        p = video_lib / f"{sign}.mp4"
        if p.exists():
            video_paths.append(p)
        else:
            p_alt = video_lib / f"{sign.lower()}.mp4"
            if p_alt.exists():
                video_paths.append(p_alt)
    if not video_paths:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Real-time avatar generation not available. Install Blender + SMPL-X models, or add pre-recorded videos to avatar/video_library/ (e.g. HELLO.mp4, HOW.mp4, YOU.mp4).",
                "video_library_path": str(video_lib.absolute()),
                "requested_signs": signs
            }
        )
    output_dir = Config.DATA_DIR / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sequence_{'_'.join(signs)}.mp4"
    result_path = _concatenate_videos_ffmpeg(video_paths, output_path)
    if not result_path:
        raise HTTPException(status_code=500, detail="Failed to concatenate videos (ffmpeg required)")
    relative_path = Path(result_path).relative_to(Config.DATA_DIR)
    return {"signs": signs, "video_url": f"/static/{relative_path}", "quality": quality, "source": "video_library"}


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
