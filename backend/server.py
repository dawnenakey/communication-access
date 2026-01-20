from fastapi import FastAPI, APIRouter, HTTPException, Request, Response, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import base64
import httpx
import json

# AWS Bedrock for translation
try:
    import boto3
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False
    print("Warning: boto3 not available - LLM translation disabled")

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# SonZo SLR API Configuration
SONZO_SLR_API_URL = os.environ.get('SONZO_SLR_API_URL', 'https://api.sonzo.io')
SONZO_SLR_API_KEY = os.environ.get('SONZO_SLR_API_KEY', '')

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

# ============== SLR MODELS ==============

class SLRRecognizeRequest(BaseModel):
    frames: List[str] = Field(..., description="Base64 encoded video frames")
    return_alternatives: bool = Field(default=False, description="Return alternative interpretations")
    confidence_threshold: float = Field(default=0.5, ge=0, le=1, description="Minimum confidence threshold")

class SLRRecognizeResponse(BaseModel):
    sign: str
    confidence: float
    alternatives: Optional[List[dict]] = None
    processing_time_ms: float
    request_id: str

# ============== CONVERSATION MODELS ==============

class EnglishToASLRequest(BaseModel):
    text: str = Field(..., description="English text to translate to ASL gloss")

class EnglishToASLResponse(BaseModel):
    english: str
    asl_gloss: List[str]
    processing_time_ms: float

class ConversationMessage(BaseModel):
    role: str  # 'user' (signing) or 'system' (response)
    content: str
    asl_gloss: Optional[List[str]] = None
    video_url: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ConversationRequest(BaseModel):
    frames: Optional[List[str]] = Field(None, description="Video frames for sign recognition (if user is signing)")
    text: Optional[str] = Field(None, description="Text input (if typing instead of signing)")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID to continue")
    avatar_id: Optional[str] = Field(None, description="Avatar ID for response video generation")

class ConversationResponse(BaseModel):
    conversation_id: str
    user_message: ConversationMessage
    system_response: ConversationMessage
    available_signs: List[str]
    processing_time_ms: float

# ============== BEDROCK TRANSLATOR ==============

class BedrockTranslator:
    """Bidirectional ASL/English translator using AWS Bedrock."""

    def __init__(self, region: str = "us-east-1"):
        self.client = None
        self.region = region
        self.model_id = "anthropic.claude-3-haiku-20240307-v1:0"

        if BEDROCK_AVAILABLE:
            try:
                self.client = boto3.client('bedrock-runtime', region_name=region)
                logger.info(f"Bedrock translator initialized (region: {region})")
            except Exception as e:
                logger.warning(f"Failed to initialize Bedrock: {e}")
                self.client = None

    def english_to_asl_gloss(self, text: str) -> List[str]:
        """Convert English text to ASL gloss sequence."""
        if not self.client:
            # Fallback: simple word extraction
            return self._simple_gloss(text)

        prompt = f"""You are an English to ASL (American Sign Language) gloss translator.
Convert the following English sentence into ASL gloss format.

ASL Gloss Rules:
- Use CAPITAL LETTERS for each sign
- Use underscores for multi-word concepts (e.g., ICE_CREAM)
- ASL uses topic-comment structure
- Remove articles (a, an, the) and helper verbs (is, are, was, were)
- Time concepts typically come first
- Questions have specific markers (WHO, WHAT, WHERE, WHY, HOW, WHEN)
- Use common ASL vocabulary when possible

Available signs in our system (prefer these when possible):
HELLO, GOODBYE, NICE_TO_MEET_YOU, THANK_YOU, PLEASE, SORRY, YES, NO, HELP,
WHAT, WHERE, WHO, WHY, HOW, WHEN, I_LOVE_YOU, HAPPY, SAD, UNDERSTAND,
WANT, NEED, LIKE, KNOW, LEARN, FINISH, NAME, MY, YOUR, YOU, ME, WE,
GOOD, BAD, MORE, AGAIN, WAIT, STOP, GO, COME, EAT, DRINK, SLEEP, WORK

English: {text}

Respond with ONLY the ASL gloss signs separated by spaces, nothing else.
Example input: "Hello, how are you?"
Example output: HELLO HOW YOU"""

        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}]
            })

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response['body'].read())
            gloss_text = response_body['content'][0]['text'].strip()
            return gloss_text.split()

        except Exception as e:
            logger.error(f"English to ASL translation error: {e}")
            return self._simple_gloss(text)

    def asl_gloss_to_english(self, signs: List[str]) -> str:
        """Convert ASL gloss sequence to natural English."""
        if not self.client:
            return " ".join(signs).replace("_", " ").title()

        gloss_sequence = " ".join(signs)

        prompt = f"""You are an ASL (American Sign Language) to English translator.
Convert the following ASL gloss sequence into natural, grammatically correct English.

ASL Grammar Notes:
- ASL uses topic-comment structure (e.g., "STORE I GO" means "I'm going to the store")
- ASL often omits articles (a, an, the) and auxiliary verbs (is, are, was)
- Time signs often come first (e.g., "TOMORROW I WORK" means "I will work tomorrow")

ASL Gloss: {gloss_sequence}

Respond with ONLY the natural English translation, nothing else."""

        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}]
            })

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text'].strip()

        except Exception as e:
            logger.error(f"ASL to English translation error: {e}")
            return " ".join(signs).replace("_", " ").title()

    def generate_response(self, user_message: str, context: List[Dict] = None) -> str:
        """Generate a conversational response to user's message."""
        if not self.client:
            return self._fallback_response(user_message)

        context_str = ""
        if context:
            context_str = "Previous conversation:\n"
            for msg in context[-5:]:  # Last 5 messages
                context_str += f"- {msg['role']}: {msg['content']}\n"

        prompt = f"""You are a friendly AI assistant communicating with a deaf user through sign language.
Keep your responses short and simple (1-2 sentences max) since they will be translated to ASL.
Use vocabulary that's common in ASL. Avoid complex idioms or figures of speech.

{context_str}

User said: {user_message}

Respond naturally and helpfully in simple English."""

        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": prompt}]
            })

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text'].strip()

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return self._fallback_response(user_message)

    def _simple_gloss(self, text: str) -> List[str]:
        """Simple fallback gloss conversion."""
        # Remove punctuation and convert to uppercase
        import re
        words = re.sub(r'[^\w\s]', '', text).upper().split()
        # Remove common articles
        skip_words = {'A', 'AN', 'THE', 'IS', 'ARE', 'WAS', 'WERE', 'AM', 'BE', 'BEEN'}
        return [w for w in words if w not in skip_words]

    def _fallback_response(self, user_message: str) -> str:
        """Simple fallback responses."""
        user_lower = user_message.lower()
        if any(g in user_lower for g in ['hello', 'hi', 'hey']):
            return "Hello! Nice to meet you."
        elif any(q in user_lower for q in ['how are you', 'how you']):
            return "I am good, thank you! How are you?"
        elif 'thank' in user_lower:
            return "You're welcome!"
        elif any(q in user_lower for q in ['bye', 'goodbye']):
            return "Goodbye! Have a great day!"
        elif '?' in user_message:
            return "That's a good question. Let me help you."
        else:
            return "I understand. How can I help you?"

    def is_available(self) -> bool:
        return self.client is not None


# Global translator instance
bedrock_translator: Optional[BedrockTranslator] = None

def get_translator() -> BedrockTranslator:
    global bedrock_translator
    if bedrock_translator is None:
        region = os.environ.get('AWS_REGION', 'us-east-1')
        bedrock_translator = BedrockTranslator(region=region)
    return bedrock_translator

# Available signs that have video support
AVAILABLE_SIGNS = [
    "HELLO", "GOODBYE", "NICE_TO_MEET_YOU", "THANK_YOU", "PLEASE", "SORRY",
    "YES", "NO", "HELP", "WHAT", "WHERE", "WHO", "WHY", "HOW", "WHEN",
    "I_LOVE_YOU", "HAPPY", "SAD", "UNDERSTAND", "WANT", "NEED", "LIKE",
    "KNOW", "LEARN", "FINISH", "NAME", "MY", "YOUR", "YOU", "ME", "WE",
    "GOOD", "BAD", "MORE", "AGAIN", "WAIT", "STOP", "GO", "COME",
    "EAT", "DRINK", "SLEEP", "WORK"
]

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

# ============== SLR (Sign Language Recognition) ROUTES ==============

@api_router.post("/slr/recognize", response_model=SLRRecognizeResponse)
async def recognize_sign(
    request: SLRRecognizeRequest,
    user: User = Depends(get_current_user)
):
    """
    Recognize ASL signs from video frames.
    Proxies request to SonZo SLR API (api.sonzo.io)
    """
    if not SONZO_SLR_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="SLR service not configured. Set SONZO_SLR_API_KEY environment variable."
        )

    # Validate frame count
    if len(request.frames) < 1 or len(request.frames) > 30:
        raise HTTPException(status_code=400, detail="1-30 frames required")

    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.post(
                f"{SONZO_SLR_API_URL}/api/slr/recognize",
                json={
                    "frames": request.frames,
                    "return_alternatives": request.return_alternatives,
                    "confidence_threshold": request.confidence_threshold
                },
                headers={
                    "X-API-Key": SONZO_SLR_API_KEY,
                    "Content-Type": "application/json"
                }
            )

            if response.status_code == 401:
                raise HTTPException(status_code=503, detail="SLR API authentication failed")
            elif response.status_code == 429:
                raise HTTPException(status_code=429, detail="SLR API rate limit exceeded")
            elif response.status_code != 200:
                logger.error(f"SLR API error: {response.status_code} - {response.text}")
                raise HTTPException(status_code=502, detail="SLR service error")

            result = response.json()

            # Auto-save to history if confidence is high enough
            if result.get("confidence", 0) >= request.confidence_threshold:
                history_doc = {
                    "history_id": f"hist_{uuid.uuid4().hex[:12]}",
                    "user_id": user.user_id,
                    "input_type": "asl_to_text",
                    "input_content": f"[{len(request.frames)} frames]",
                    "output_content": result.get("sign", "UNKNOWN"),
                    "confidence": result.get("confidence"),
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                await db.translation_history.insert_one(history_doc)

            return SLRRecognizeResponse(
                sign=result.get("sign", "UNKNOWN"),
                confidence=result.get("confidence", 0),
                alternatives=result.get("alternatives"),
                processing_time_ms=result.get("processing_time_ms", 0),
                request_id=result.get("request_id", str(uuid.uuid4()))
            )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="SLR service timeout")
    except httpx.RequestError as e:
        logger.error(f"SLR API connection error: {e}")
        raise HTTPException(status_code=503, detail="SLR service unavailable")

@api_router.get("/slr/signs")
async def get_supported_signs(user: User = Depends(get_current_user)):
    """Get list of all supported ASL signs from SonZo SLR API"""
    if not SONZO_SLR_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="SLR service not configured"
        )

    try:
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            response = await http_client.get(
                f"{SONZO_SLR_API_URL}/api/slr/signs",
                headers={"X-API-Key": SONZO_SLR_API_KEY}
            )

            if response.status_code != 200:
                raise HTTPException(status_code=502, detail="SLR service error")

            return response.json()

    except httpx.RequestError as e:
        logger.error(f"SLR API connection error: {e}")
        raise HTTPException(status_code=503, detail="SLR service unavailable")

@api_router.get("/slr/usage")
async def get_slr_usage(user: User = Depends(get_current_user)):
    """Get SLR API usage statistics"""
    if not SONZO_SLR_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="SLR service not configured"
        )

    try:
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            response = await http_client.get(
                f"{SONZO_SLR_API_URL}/api/slr/usage",
                headers={"X-API-Key": SONZO_SLR_API_KEY}
            )

            if response.status_code != 200:
                raise HTTPException(status_code=502, detail="SLR service error")

            return response.json()

    except httpx.RequestError as e:
        logger.error(f"SLR API connection error: {e}")
        raise HTTPException(status_code=503, detail="SLR service unavailable")

@api_router.get("/slr/health")
async def slr_health_check():
    """Check SonZo SLR API health"""
    if not SONZO_SLR_API_KEY:
        return {
            "status": "not_configured",
            "message": "SONZO_SLR_API_KEY not set"
        }

    try:
        async with httpx.AsyncClient(timeout=5.0) as http_client:
            response = await http_client.get(f"{SONZO_SLR_API_URL}/health")

            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "api_url": SONZO_SLR_API_URL,
                    "response": response.json()
                }
            else:
                return {
                    "status": "unhealthy",
                    "api_url": SONZO_SLR_API_URL,
                    "error": f"Status {response.status_code}"
                }

    except httpx.RequestError as e:
        return {
            "status": "unreachable",
            "api_url": SONZO_SLR_API_URL,
            "error": str(e)
        }

# ============== CONVERSATION ROUTES ==============

@api_router.post("/translate/english-to-asl", response_model=EnglishToASLResponse)
async def translate_english_to_asl(
    request: EnglishToASLRequest,
    user: User = Depends(get_current_user)
):
    """
    Translate English text to ASL gloss sequence.
    Uses AWS Bedrock Claude for intelligent translation.
    """
    import time
    start_time = time.time()

    translator = get_translator()
    asl_gloss = translator.english_to_asl_gloss(request.text)

    processing_time = (time.time() - start_time) * 1000

    return EnglishToASLResponse(
        english=request.text,
        asl_gloss=asl_gloss,
        processing_time_ms=processing_time
    )


@api_router.post("/conversation", response_model=ConversationResponse)
async def process_conversation(
    request: ConversationRequest,
    user: User = Depends(get_current_user)
):
    """
    Process a bidirectional sign language conversation.

    Flow:
    1. User signs (frames) OR types text
    2. System recognizes sign / processes text
    3. System generates response in English
    4. System converts response to ASL gloss
    5. Returns response with available signs for avatar

    This enables:
    - Deaf user signs → System responds in signs
    - Hearing user types → Response shown as signs
    """
    import time
    start_time = time.time()

    translator = get_translator()
    user_text = ""
    user_asl_gloss = []

    # Step 1: Get user's message (either from sign recognition or text)
    if request.frames and len(request.frames) > 0:
        # Recognize sign from video frames via SonZo SLR API
        if not SONZO_SLR_API_KEY:
            raise HTTPException(
                status_code=503,
                detail="SLR service not configured"
            )

        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                response = await http_client.post(
                    f"{SONZO_SLR_API_URL}/api/slr/recognize",
                    json={
                        "frames": request.frames,
                        "return_alternatives": False,
                        "confidence_threshold": 0.5
                    },
                    headers={
                        "X-API-Key": SONZO_SLR_API_KEY,
                        "Content-Type": "application/json"
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    recognized_sign = result.get("sign", "UNKNOWN")
                    if recognized_sign != "UNKNOWN":
                        user_asl_gloss = [recognized_sign]
                        # Convert ASL to English for context
                        user_text = translator.asl_gloss_to_english(user_asl_gloss)
                    else:
                        user_text = "Hello"  # Default if recognition fails
                        user_asl_gloss = ["HELLO"]
                else:
                    user_text = "Hello"
                    user_asl_gloss = ["HELLO"]

        except Exception as e:
            logger.error(f"SLR recognition error in conversation: {e}")
            user_text = "Hello"
            user_asl_gloss = ["HELLO"]

    elif request.text:
        # User typed text
        user_text = request.text
        user_asl_gloss = translator.english_to_asl_gloss(request.text)
    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'frames' or 'text' is required"
        )

    # Step 2: Generate or retrieve conversation context
    conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"

    # Get existing conversation context from DB
    conversation_doc = await db.conversations.find_one(
        {"conversation_id": conversation_id, "user_id": user.user_id}
    )

    context = []
    if conversation_doc:
        context = conversation_doc.get("messages", [])

    # Step 3: Generate system response
    response_english = translator.generate_response(user_text, context)

    # Step 4: Convert response to ASL gloss
    response_asl_gloss = translator.english_to_asl_gloss(response_english)

    # Step 5: Filter to available signs and find video URLs
    available_response_signs = [
        sign for sign in response_asl_gloss
        if sign in AVAILABLE_SIGNS
    ]

    # If no available signs, provide fallback
    if not available_response_signs:
        # Find closest available signs
        if any(g in response_english.lower() for g in ['hello', 'hi']):
            available_response_signs = ["HELLO"]
        elif 'thank' in response_english.lower():
            available_response_signs = ["THANK_YOU"]
        elif 'good' in response_english.lower():
            available_response_signs = ["GOOD"]
        else:
            available_response_signs = ["HELLO"]  # Default

    # Build video URL (if avatar API is available)
    video_url = None
    if request.avatar_id and available_response_signs:
        # Video would be generated by avatar API
        video_url = f"/api/avatar/{request.avatar_id}/video/{available_response_signs[0]}"

    # Create message objects
    user_message = ConversationMessage(
        role="user",
        content=user_text,
        asl_gloss=user_asl_gloss
    )

    system_response = ConversationMessage(
        role="system",
        content=response_english,
        asl_gloss=available_response_signs,
        video_url=video_url
    )

    # Save conversation to database
    new_messages = [
        {"role": "user", "content": user_text, "asl_gloss": user_asl_gloss, "timestamp": datetime.now(timezone.utc).isoformat()},
        {"role": "system", "content": response_english, "asl_gloss": available_response_signs, "timestamp": datetime.now(timezone.utc).isoformat()}
    ]

    if conversation_doc:
        # Update existing conversation
        await db.conversations.update_one(
            {"conversation_id": conversation_id},
            {"$push": {"messages": {"$each": new_messages}}}
        )
    else:
        # Create new conversation
        await db.conversations.insert_one({
            "conversation_id": conversation_id,
            "user_id": user.user_id,
            "messages": new_messages,
            "created_at": datetime.now(timezone.utc).isoformat()
        })

    processing_time = (time.time() - start_time) * 1000

    return ConversationResponse(
        conversation_id=conversation_id,
        user_message=user_message,
        system_response=system_response,
        available_signs=AVAILABLE_SIGNS,
        processing_time_ms=processing_time
    )


@api_router.get("/conversation/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user)
):
    """Get conversation history."""
    conversation = await db.conversations.find_one(
        {"conversation_id": conversation_id, "user_id": user.user_id},
        {"_id": 0}
    )

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return conversation


@api_router.delete("/conversation/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user)
):
    """Delete a conversation."""
    result = await db.conversations.delete_one({
        "conversation_id": conversation_id,
        "user_id": user.user_id
    })

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"message": "Conversation deleted"}


@api_router.get("/conversation/signs/available")
async def get_available_signs():
    """Get list of signs that have video support for avatar responses."""
    return {
        "signs": AVAILABLE_SIGNS,
        "count": len(AVAILABLE_SIGNS),
        "categories": {
            "greetings": ["HELLO", "GOODBYE", "NICE_TO_MEET_YOU"],
            "common": ["THANK_YOU", "PLEASE", "SORRY", "YES", "NO", "HELP"],
            "questions": ["WHAT", "WHERE", "WHO", "WHY", "HOW", "WHEN"],
            "feelings": ["I_LOVE_YOU", "HAPPY", "SAD", "UNDERSTAND"],
            "actions": ["WANT", "NEED", "LIKE", "KNOW", "LEARN", "FINISH", "WAIT", "STOP", "GO", "COME"],
            "pronouns": ["NAME", "MY", "YOUR", "YOU", "ME", "WE"],
            "descriptors": ["GOOD", "BAD", "MORE", "AGAIN"],
            "daily": ["EAT", "DRINK", "SLEEP", "WORK"]
        }
    }


# ============== UTILITY ROUTES ==============

@api_router.get("/")
async def root():
    return {"message": "SignSync AI API", "version": "1.0.0", "features": ["slr", "conversation", "avatar"]}

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
