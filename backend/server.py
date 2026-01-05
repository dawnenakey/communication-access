from fastapi import FastAPI, APIRouter, HTTPException, Request, Response, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import base64

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============== MODELS ==============

class User(BaseModel):
    user_id: str
    email: str
    name: str
    picture: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserSession(BaseModel):
    user_id: str
    session_token: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SignWord(BaseModel):
    sign_id: str = Field(default_factory=lambda: f"sign_{uuid.uuid4().hex[:12]}")
    word: str
    description: Optional[str] = None
    image_data: str  # Base64 encoded image
    image_type: str  # e.g., 'image/png', 'image/jpeg'
    created_by: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SignWordCreate(BaseModel):
    word: str
    description: Optional[str] = None

class SignWordUpdate(BaseModel):
    word: Optional[str] = None
    description: Optional[str] = None

class TranslationHistory(BaseModel):
    history_id: str = Field(default_factory=lambda: f"hist_{uuid.uuid4().hex[:12]}")
    user_id: str
    input_type: str  # 'asl_to_text' or 'text_to_asl'
    input_content: str  # Either recognized text or input text
    output_content: str  # Either text result or sign_ids
    confidence: Optional[float] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TranslationHistoryCreate(BaseModel):
    input_type: str
    input_content: str
    output_content: str
    confidence: Optional[float] = None

# ============== AUTH HELPERS ==============

async def get_current_user(request: Request) -> User:
    """Get current user from session token in cookie or Authorization header"""
    session_token = request.cookies.get("session_token")
    
    if not session_token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            session_token = auth_header.split(" ")[1]
    
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Find session
    session_doc = await db.user_sessions.find_one(
        {"session_token": session_token},
        {"_id": 0}
    )
    
    if not session_doc:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    # Check expiry with timezone handling
    expires_at = session_doc["expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Session expired")
    
    # Find user
    user_doc = await db.users.find_one(
        {"user_id": session_doc["user_id"]},
        {"_id": 0}
    )
    
    if not user_doc:
        raise HTTPException(status_code=401, detail="User not found")
    
    return User(**user_doc)

async def get_optional_user(request: Request) -> Optional[User]:
    """Get current user if authenticated, otherwise return None"""
    try:
        return await get_current_user(request)
    except HTTPException:
        return None

# ============== AUTH ROUTES ==============

class LoginRequest(BaseModel):
    email: str
    name: str
    picture: Optional[str] = None

@api_router.post("/auth/login")
async def login(login_data: LoginRequest, response: Response):
    """Create or update user and create session"""
    # Check if user exists
    existing_user = await db.users.find_one(
        {"email": login_data.email},
        {"_id": 0}
    )

    if existing_user:
        user_id = existing_user["user_id"]
        # Update user info if needed
        await db.users.update_one(
            {"user_id": user_id},
            {"$set": {
                "name": login_data.name,
                "picture": login_data.picture
            }}
        )
    else:
        # Create new user
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        new_user = {
            "user_id": user_id,
            "email": login_data.email,
            "name": login_data.name,
            "picture": login_data.picture,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.users.insert_one(new_user)

    # Create session
    session_token = f"sess_{uuid.uuid4().hex}"
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)

    session_doc = {
        "user_id": user_id,
        "session_token": session_token,
        "expires_at": expires_at.isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    # Remove old sessions for this user
    await db.user_sessions.delete_many({"user_id": user_id})
    await db.user_sessions.insert_one(session_doc)

    # Set cookie
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        max_age=7 * 24 * 60 * 60  # 7 days
    )

    # Get user doc
    user_doc = await db.users.find_one({"user_id": user_id}, {"_id": 0})

    return {
        "user": user_doc,
        "session_token": session_token
    }

@api_router.post("/auth/session")
async def create_session(request: Request, response: Response):
    """Create session from email/name (simplified auth)"""
    data = await request.json()

    email = data.get("email")
    name = data.get("name", "User")
    picture = data.get("picture")

    if not email:
        raise HTTPException(status_code=400, detail="email required")

    # Check if user exists
    existing_user = await db.users.find_one(
        {"email": email},
        {"_id": 0}
    )

    if existing_user:
        user_id = existing_user["user_id"]
        await db.users.update_one(
            {"user_id": user_id},
            {"$set": {"name": name, "picture": picture}}
        )
    else:
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        new_user = {
            "user_id": user_id,
            "email": email,
            "name": name,
            "picture": picture,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.users.insert_one(new_user)

    session_token = f"sess_{uuid.uuid4().hex}"
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)

    session_doc = {
        "user_id": user_id,
        "session_token": session_token,
        "expires_at": expires_at.isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    await db.user_sessions.delete_many({"user_id": user_id})
    await db.user_sessions.insert_one(session_doc)

    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        max_age=7 * 24 * 60 * 60
    )

    user_doc = await db.users.find_one({"user_id": user_id}, {"_id": 0})

    return {
        "user": user_doc,
        "session_token": session_token
    }

@api_router.get("/auth/me")
async def get_me(user: User = Depends(get_current_user)):
    """Get current authenticated user"""
    return user.model_dump()

@api_router.post("/auth/logout")
async def logout(request: Request, response: Response):
    """Logout user and clear session"""
    session_token = request.cookies.get("session_token")
    
    if session_token:
        await db.user_sessions.delete_many({"session_token": session_token})
    
    response.delete_cookie(
        key="session_token",
        path="/",
        secure=True,
        samesite="none"
    )
    
    return {"message": "Logged out successfully"}

# ============== SIGN DICTIONARY ROUTES ==============

@api_router.get("/signs", response_model=List[dict])
async def get_signs(user: Optional[User] = Depends(get_optional_user)):
    """Get all signs in the dictionary"""
    signs = await db.signs.find({}, {"_id": 0}).to_list(1000)
    return signs

@api_router.get("/signs/{sign_id}")
async def get_sign(sign_id: str):
    """Get a specific sign by ID"""
    sign = await db.signs.find_one({"sign_id": sign_id}, {"_id": 0})
    if not sign:
        raise HTTPException(status_code=404, detail="Sign not found")
    return sign

@api_router.post("/signs")
async def create_sign(
    word: str = Form(...),
    description: str = Form(None),
    image: UploadFile = File(...),
    user: User = Depends(get_current_user)
):
    """Create a new sign entry with image upload"""
    # Read and encode image
    image_content = await image.read()
    image_base64 = base64.b64encode(image_content).decode('utf-8')
    
    sign_doc = {
        "sign_id": f"sign_{uuid.uuid4().hex[:12]}",
        "word": word.lower().strip(),
        "description": description,
        "image_data": image_base64,
        "image_type": image.content_type,
        "created_by": user.user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.signs.insert_one(sign_doc)
    
    # Return without _id
    sign_doc.pop("_id", None)
    return sign_doc

@api_router.put("/signs/{sign_id}")
async def update_sign(
    sign_id: str,
    word: str = Form(None),
    description: str = Form(None),
    image: UploadFile = File(None),
    user: User = Depends(get_current_user)
):
    """Update a sign entry"""
    existing = await db.signs.find_one({"sign_id": sign_id}, {"_id": 0})
    if not existing:
        raise HTTPException(status_code=404, detail="Sign not found")
    
    update_data = {"updated_at": datetime.now(timezone.utc).isoformat()}
    
    if word:
        update_data["word"] = word.lower().strip()
    if description is not None:
        update_data["description"] = description
    if image:
        image_content = await image.read()
        update_data["image_data"] = base64.b64encode(image_content).decode('utf-8')
        update_data["image_type"] = image.content_type
    
    await db.signs.update_one({"sign_id": sign_id}, {"$set": update_data})
    
    updated = await db.signs.find_one({"sign_id": sign_id}, {"_id": 0})
    return updated

@api_router.delete("/signs/{sign_id}")
async def delete_sign(sign_id: str, user: User = Depends(get_current_user)):
    """Delete a sign entry"""
    result = await db.signs.delete_one({"sign_id": sign_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Sign not found")
    return {"message": "Sign deleted successfully"}

@api_router.get("/signs/search/{word}")
async def search_signs(word: str):
    """Search signs by word"""
    signs = await db.signs.find(
        {"word": {"$regex": word.lower(), "$options": "i"}},
        {"_id": 0}
    ).to_list(100)
    return signs

# ============== TRANSLATION HISTORY ROUTES ==============

@api_router.get("/history", response_model=List[dict])
async def get_history(user: User = Depends(get_current_user)):
    """Get translation history for current user"""
    history = await db.translation_history.find(
        {"user_id": user.user_id},
        {"_id": 0}
    ).sort("created_at", -1).to_list(100)
    return history

@api_router.post("/history")
async def create_history(
    history_data: TranslationHistoryCreate,
    user: User = Depends(get_current_user)
):
    """Save a translation to history"""
    history_doc = {
        "history_id": f"hist_{uuid.uuid4().hex[:12]}",
        "user_id": user.user_id,
        "input_type": history_data.input_type,
        "input_content": history_data.input_content,
        "output_content": history_data.output_content,
        "confidence": history_data.confidence,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.translation_history.insert_one(history_doc)
    history_doc.pop("_id", None)
    return history_doc

@api_router.delete("/history/{history_id}")
async def delete_history(history_id: str, user: User = Depends(get_current_user)):
    """Delete a history entry"""
    result = await db.translation_history.delete_one({
        "history_id": history_id,
        "user_id": user.user_id
    })
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="History entry not found")
    return {"message": "History entry deleted"}

@api_router.delete("/history")
async def clear_history(user: User = Depends(get_current_user)):
    """Clear all history for current user"""
    await db.translation_history.delete_many({"user_id": user.user_id})
    return {"message": "History cleared"}

# ============== UTILITY ROUTES ==============

@api_router.get("/")
async def root():
    return {"message": "SignSync AI API", "version": "1.0.0"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
