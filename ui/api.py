#!/usr/bin/env python3
"""
SonZo AI - UI Backend API
==========================
FastAPI backend for the personalized UI/UX experience.

Endpoints:
- User management (profile, preferences, progress)
- Gamification (achievements, streaks, XP)
- Conversations
- Recognition integration

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8081

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """API configuration."""
    DATA_DIR = Path(os.environ.get("SONZO_DATA_DIR", "./data"))
    USERS_DIR = DATA_DIR / "users"
    MAX_CONVERSATIONS = 100


# Create directories
Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
Config.USERS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Models
# =============================================================================

class UserProfile(BaseModel):
    """User profile data."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "User"
    avatar_id: Optional[str] = None
    avatar_url: Optional[str] = None
    goal: Optional[str] = None  # learn, communicate, both
    level: str = "beginner"  # beginner, intermediate, fluent
    signing_speed: float = 1.0
    left_handed: bool = False
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class UserSettings(BaseModel):
    """User settings."""
    dark_mode: bool = False
    high_contrast: bool = False
    reduce_motion: bool = False
    large_text: bool = False
    haptic: bool = True
    show_skeleton: bool = True
    practice_reminders: bool = True
    achievement_alerts: bool = True


class UserProgress(BaseModel):
    """User progress and gamification data."""
    signs_learned: List[str] = []
    accuracy: Dict[str, Dict[str, int]] = {}  # sign -> {correct, total}
    practice_time: int = 0  # seconds
    streak: int = 0
    last_practice: Optional[str] = None
    xp: int = 0
    level: int = 1
    achievements: List[str] = []
    conversations_completed: int = 0


class ConversationMessage(BaseModel):
    """A message in a conversation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: str  # user, avatar
    sign: str
    video_url: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class Conversation(BaseModel):
    """A conversation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    messages: List[ConversationMessage] = []
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class Achievement(BaseModel):
    """An achievement."""
    id: str
    name: str
    description: str
    icon: str
    xp: int
    unlocked: bool = False
    unlocked_at: Optional[str] = None


class DailyChallenge(BaseModel):
    """Daily challenge."""
    type: str
    title: str
    description: str
    target: int
    progress: int = 0
    reward: int
    completed: bool = False


# =============================================================================
# API Application
# =============================================================================

app = FastAPI(
    title="SonZo AI UI API",
    description="Backend for personalized UI/UX experience",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI files
ui_dir = Path(__file__).parent
if (ui_dir / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(ui_dir / "assets")), name="assets")
if (ui_dir / "index.html").exists():
    app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")


# =============================================================================
# Helper Functions
# =============================================================================

def get_user_dir(user_id: str) -> Path:
    """Get user data directory."""
    return Config.USERS_DIR / user_id


def load_user_data(user_id: str, filename: str, default: Any = None) -> Any:
    """Load user data file."""
    path = get_user_dir(user_id) / f"{filename}.json"
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return default


def save_user_data(user_id: str, filename: str, data: Any):
    """Save user data file."""
    user_dir = get_user_dir(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)

    path = user_dir / f"{filename}.json"
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


# Achievement definitions
ACHIEVEMENTS = {
    "first_sign": Achievement(
        id="first_sign", name="First Sign",
        description="Learn your first sign", icon="🎯", xp=10
    ),
    "alphabet_master": Achievement(
        id="alphabet_master", name="Alphabet Master",
        description="Learn all 26 letter signs", icon="🔤", xp=100
    ),
    "number_ninja": Achievement(
        id="number_ninja", name="Number Ninja",
        description="Learn all number signs (0-9)", icon="🔢", xp=50
    ),
    "streak_5": Achievement(
        id="streak_5", name="5 Day Streak",
        description="Practice for 5 days in a row", icon="🔥", xp=50
    ),
    "streak_10": Achievement(
        id="streak_10", name="10 Day Streak",
        description="Practice for 10 days in a row", icon="🔥", xp=100
    ),
    "streak_30": Achievement(
        id="streak_30", name="30 Day Streak",
        description="Practice for 30 days in a row", icon="⭐", xp=300
    ),
    "signs_10": Achievement(
        id="signs_10", name="Getting Started",
        description="Learn 10 signs", icon="🌱", xp=30
    ),
    "signs_50": Achievement(
        id="signs_50", name="Half Century",
        description="Learn 50 signs", icon="💯", xp=150
    ),
    "signs_100": Achievement(
        id="signs_100", name="Century Club",
        description="Learn 100 signs", icon="🏆", xp=300
    ),
}


# =============================================================================
# User Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Serve main UI."""
    index_path = ui_dir / "index.html"
    if index_path.exists():
        response = FileResponse(str(index_path))
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        return response
    return {"message": "SonZo AI UI API", "version": "1.0.0"}


@app.get("/api/health")
async def health_check():
    """Health check."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/users", response_model=UserProfile)
async def create_user(profile: Optional[UserProfile] = None):
    """Create new user."""
    if profile is None:
        profile = UserProfile()

    # Save profile
    save_user_data(profile.id, "profile", profile.dict())

    # Initialize settings and progress
    save_user_data(profile.id, "settings", UserSettings().dict())
    save_user_data(profile.id, "progress", UserProgress().dict())
    save_user_data(profile.id, "conversations", [])

    return profile


@app.get("/api/users/{user_id}", response_model=UserProfile)
async def get_user(user_id: str):
    """Get user profile."""
    data = load_user_data(user_id, "profile")
    if not data:
        raise HTTPException(status_code=404, detail="User not found")
    return UserProfile(**data)


@app.put("/api/users/{user_id}", response_model=UserProfile)
async def update_user(user_id: str, updates: Dict[str, Any]):
    """Update user profile."""
    data = load_user_data(user_id, "profile")
    if not data:
        raise HTTPException(status_code=404, detail="User not found")

    data.update(updates)
    save_user_data(user_id, "profile", data)

    return UserProfile(**data)


@app.delete("/api/users/{user_id}")
async def delete_user(user_id: str):
    """Delete user and all data."""
    user_dir = get_user_dir(user_id)
    if user_dir.exists():
        import shutil
        shutil.rmtree(user_dir)
    return {"status": "deleted"}


# =============================================================================
# Settings Endpoints
# =============================================================================

@app.get("/api/users/{user_id}/settings", response_model=UserSettings)
async def get_settings(user_id: str):
    """Get user settings."""
    data = load_user_data(user_id, "settings", UserSettings().dict())
    return UserSettings(**data)


@app.put("/api/users/{user_id}/settings", response_model=UserSettings)
async def update_settings(user_id: str, settings: UserSettings):
    """Update user settings."""
    save_user_data(user_id, "settings", settings.dict())
    return settings


@app.patch("/api/users/{user_id}/settings")
async def patch_settings(user_id: str, updates: Dict[str, Any]):
    """Partially update settings."""
    data = load_user_data(user_id, "settings", UserSettings().dict())
    data.update(updates)
    save_user_data(user_id, "settings", data)
    return data


# =============================================================================
# Progress Endpoints
# =============================================================================

@app.get("/api/users/{user_id}/progress", response_model=UserProgress)
async def get_progress(user_id: str):
    """Get user progress."""
    data = load_user_data(user_id, "progress", UserProgress().dict())
    return UserProgress(**data)


@app.post("/api/users/{user_id}/progress/sign-learned")
async def record_sign_learned(user_id: str, sign: str):
    """Record that a sign was learned."""
    progress = load_user_data(user_id, "progress", UserProgress().dict())

    if sign not in progress["signs_learned"]:
        progress["signs_learned"].append(sign)

        # Add XP
        progress["xp"] += 10

        # Check achievements
        sign_count = len(progress["signs_learned"])
        new_achievements = []

        if sign_count == 1:
            new_achievements.append("first_sign")
        if sign_count >= 10:
            new_achievements.append("signs_10")
        if sign_count >= 50:
            new_achievements.append("signs_50")
        if sign_count >= 100:
            new_achievements.append("signs_100")

        # Check alphabet
        alphabet = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if alphabet.issubset(set(progress["signs_learned"])):
            new_achievements.append("alphabet_master")

        # Check numbers
        numbers = set("0123456789")
        if numbers.issubset(set(progress["signs_learned"])):
            new_achievements.append("number_ninja")

        # Unlock new achievements
        for ach_id in new_achievements:
            if ach_id not in progress["achievements"]:
                progress["achievements"].append(ach_id)
                progress["xp"] += ACHIEVEMENTS[ach_id].xp

        save_user_data(user_id, "progress", progress)

        return {
            "signs_learned": len(progress["signs_learned"]),
            "new_achievements": new_achievements,
            "xp": progress["xp"]
        }

    return {"signs_learned": len(progress["signs_learned"])}


@app.post("/api/users/{user_id}/progress/accuracy")
async def record_accuracy(user_id: str, sign: str, correct: bool):
    """Record sign recognition accuracy."""
    progress = load_user_data(user_id, "progress", UserProgress().dict())

    if sign not in progress["accuracy"]:
        progress["accuracy"][sign] = {"correct": 0, "total": 0}

    progress["accuracy"][sign]["total"] += 1
    if correct:
        progress["accuracy"][sign]["correct"] += 1
        progress["xp"] += 5

    save_user_data(user_id, "progress", progress)

    return progress["accuracy"][sign]


@app.post("/api/users/{user_id}/progress/practice")
async def record_practice(user_id: str, duration_seconds: int = 0):
    """Record practice session."""
    progress = load_user_data(user_id, "progress", UserProgress().dict())

    # Add practice time
    progress["practice_time"] += duration_seconds

    # Update streak
    today = datetime.utcnow().date().isoformat()
    last_practice = progress.get("last_practice")

    if last_practice:
        last_date = datetime.fromisoformat(last_practice).date()
        days_diff = (datetime.utcnow().date() - last_date).days

        if days_diff == 0:
            pass  # Same day
        elif days_diff == 1:
            progress["streak"] += 1
        else:
            progress["streak"] = 1
    else:
        progress["streak"] = 1

    progress["last_practice"] = datetime.utcnow().isoformat()

    # Check streak achievements
    new_achievements = []
    if progress["streak"] >= 5 and "streak_5" not in progress["achievements"]:
        new_achievements.append("streak_5")
        progress["achievements"].append("streak_5")
        progress["xp"] += ACHIEVEMENTS["streak_5"].xp
    if progress["streak"] >= 10 and "streak_10" not in progress["achievements"]:
        new_achievements.append("streak_10")
        progress["achievements"].append("streak_10")
        progress["xp"] += ACHIEVEMENTS["streak_10"].xp
    if progress["streak"] >= 30 and "streak_30" not in progress["achievements"]:
        new_achievements.append("streak_30")
        progress["achievements"].append("streak_30")
        progress["xp"] += ACHIEVEMENTS["streak_30"].xp

    save_user_data(user_id, "progress", progress)

    return {
        "streak": progress["streak"],
        "practice_time": progress["practice_time"],
        "new_achievements": new_achievements
    }


# =============================================================================
# Achievements Endpoints
# =============================================================================

@app.get("/api/users/{user_id}/achievements")
async def get_achievements(user_id: str):
    """Get all achievements with unlock status."""
    progress = load_user_data(user_id, "progress", UserProgress().dict())
    unlocked = progress.get("achievements", [])

    achievements = []
    for ach_id, ach in ACHIEVEMENTS.items():
        ach_dict = ach.dict()
        ach_dict["unlocked"] = ach_id in unlocked
        achievements.append(ach_dict)

    return achievements


@app.post("/api/users/{user_id}/achievements/{achievement_id}")
async def unlock_achievement(user_id: str, achievement_id: str):
    """Manually unlock an achievement."""
    if achievement_id not in ACHIEVEMENTS:
        raise HTTPException(status_code=404, detail="Achievement not found")

    progress = load_user_data(user_id, "progress", UserProgress().dict())

    if achievement_id not in progress["achievements"]:
        progress["achievements"].append(achievement_id)
        progress["xp"] += ACHIEVEMENTS[achievement_id].xp
        save_user_data(user_id, "progress", progress)
        return {"unlocked": True, "xp_gained": ACHIEVEMENTS[achievement_id].xp}

    return {"unlocked": False, "message": "Already unlocked"}


# =============================================================================
# Conversation Endpoints
# =============================================================================

@app.get("/api/users/{user_id}/conversations")
async def get_conversations(user_id: str, limit: int = 20):
    """Get user conversations."""
    conversations = load_user_data(user_id, "conversations", [])
    return conversations[:limit]


@app.post("/api/users/{user_id}/conversations")
async def create_conversation(user_id: str):
    """Create new conversation."""
    conversations = load_user_data(user_id, "conversations", [])

    conversation = Conversation()
    conversations.insert(0, conversation.dict())

    # Limit stored conversations
    if len(conversations) > Config.MAX_CONVERSATIONS:
        conversations = conversations[:Config.MAX_CONVERSATIONS]

    save_user_data(user_id, "conversations", conversations)

    return conversation


@app.get("/api/users/{user_id}/conversations/{conversation_id}")
async def get_conversation(user_id: str, conversation_id: str):
    """Get specific conversation."""
    conversations = load_user_data(user_id, "conversations", [])

    for conv in conversations:
        if conv["id"] == conversation_id:
            return conv

    raise HTTPException(status_code=404, detail="Conversation not found")


@app.post("/api/users/{user_id}/conversations/{conversation_id}/messages")
async def add_message(user_id: str, conversation_id: str, message: ConversationMessage):
    """Add message to conversation."""
    conversations = load_user_data(user_id, "conversations", [])

    for i, conv in enumerate(conversations):
        if conv["id"] == conversation_id:
            conv["messages"].append(message.dict())
            conv["updated_at"] = datetime.utcnow().isoformat()
            conversations[i] = conv
            save_user_data(user_id, "conversations", conversations)
            return conv

    raise HTTPException(status_code=404, detail="Conversation not found")


@app.delete("/api/users/{user_id}/conversations/{conversation_id}")
async def delete_conversation(user_id: str, conversation_id: str):
    """Delete conversation."""
    conversations = load_user_data(user_id, "conversations", [])

    conversations = [c for c in conversations if c["id"] != conversation_id]
    save_user_data(user_id, "conversations", conversations)

    return {"status": "deleted"}


@app.delete("/api/users/{user_id}/conversations")
async def clear_conversations(user_id: str):
    """Clear all conversations."""
    save_user_data(user_id, "conversations", [])
    return {"status": "cleared"}


# =============================================================================
# Favorites Endpoints
# =============================================================================

@app.get("/api/users/{user_id}/favorites")
async def get_favorites(user_id: str):
    """Get user favorites."""
    return load_user_data(user_id, "favorites", ["HELLO", "THANK_YOU", "I_LOVE_YOU"])


@app.put("/api/users/{user_id}/favorites")
async def set_favorites(user_id: str, favorites: List[str]):
    """Set user favorites."""
    save_user_data(user_id, "favorites", favorites)
    return favorites


@app.post("/api/users/{user_id}/favorites/{phrase}")
async def add_favorite(user_id: str, phrase: str):
    """Add favorite phrase."""
    favorites = load_user_data(user_id, "favorites", [])

    if phrase not in favorites:
        favorites.append(phrase)
        save_user_data(user_id, "favorites", favorites)

    return favorites


@app.delete("/api/users/{user_id}/favorites/{phrase}")
async def remove_favorite(user_id: str, phrase: str):
    """Remove favorite phrase."""
    favorites = load_user_data(user_id, "favorites", [])

    if phrase in favorites:
        favorites.remove(phrase)
        save_user_data(user_id, "favorites", favorites)

    return favorites


# =============================================================================
# Daily Challenge
# =============================================================================

@app.get("/api/users/{user_id}/daily-challenge")
async def get_daily_challenge(user_id: str):
    """Get daily challenge."""
    # Generate challenge based on day
    day_of_year = datetime.utcnow().timetuple().tm_yday

    challenges = [
        DailyChallenge(
            type="learn_new",
            title="Learn 3 new question signs",
            description="Expand your vocabulary with question signs like WHAT, WHERE, WHO",
            target=3,
            reward=50
        ),
        DailyChallenge(
            type="practice",
            title="Practice 5 signs you know",
            description="Reinforce your knowledge by practicing familiar signs",
            target=5,
            reward=30
        ),
        DailyChallenge(
            type="conversation",
            title="Have a conversation",
            description="Use your avatar to have a signing conversation",
            target=1,
            reward=40
        ),
        DailyChallenge(
            type="streak",
            title="Maintain your streak",
            description="Practice today to keep your streak going",
            target=1,
            reward=25
        ),
    ]

    return challenges[day_of_year % len(challenges)]


# =============================================================================
# Export/Import
# =============================================================================

@app.get("/api/users/{user_id}/export")
async def export_user_data(user_id: str):
    """Export all user data."""
    return {
        "profile": load_user_data(user_id, "profile", {}),
        "settings": load_user_data(user_id, "settings", {}),
        "progress": load_user_data(user_id, "progress", {}),
        "favorites": load_user_data(user_id, "favorites", []),
        "conversations": load_user_data(user_id, "conversations", []),
        "exported_at": datetime.utcnow().isoformat()
    }


@app.post("/api/users/{user_id}/import")
async def import_user_data(user_id: str, data: Dict[str, Any]):
    """Import user data."""
    if "profile" in data:
        save_user_data(user_id, "profile", data["profile"])
    if "settings" in data:
        save_user_data(user_id, "settings", data["settings"])
    if "progress" in data:
        save_user_data(user_id, "progress", data["progress"])
    if "favorites" in data:
        save_user_data(user_id, "favorites", data["favorites"])
    if "conversations" in data:
        save_user_data(user_id, "conversations", data["conversations"])

    return {"status": "imported"}


@app.get("/{path:path}")
async def serve_spa(path: str):
    """Serve index.html for SPA routes (React Router handles /conversationaldemo, etc.)."""
    index_path = ui_dir / "index.html"
    if index_path.exists():
        response = FileResponse(str(index_path))
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        return response
    return {"message": "SonZo AI UI API", "version": "1.0.0"}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='SonZo AI UI API')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--reload', action='store_true')

    args = parser.parse_args()

    print("=" * 60)
    print("SonZo AI - UI Backend API")
    print("=" * 60)
    print(f"Data directory: {Config.DATA_DIR}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    print()

    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
