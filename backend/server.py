from fastapi import FastAPI, APIRouter, HTTPException, Request, Response, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import base64
import httpx
import json
import sqlite3
import smtplib
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Password hashing
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Stripe
try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False
    print("Warning: stripe not available - payment processing disabled")

# AWS Bedrock for translation
try:
    import boto3
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False
    print("Warning: boto3 not available - LLM translation disabled")

# GenASL Service for realistic avatar videos
from genasl_service import GenASLService, get_genasl_service, GenASLStatus, GenASLVideoResult

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# SonZo SLR API Configuration
SONZO_SLR_API_URL = os.environ.get('SONZO_SLR_API_URL', 'https://api.sonzo.io')
SONZO_SLR_API_KEY = os.environ.get('SONZO_SLR_API_KEY', '')

# GenASL Configuration (AWS GenAI ASL Avatar)
GENASL_ENABLED = os.environ.get('GENASL_ENABLED', 'true').lower() == 'true'
GENASL_STATE_MACHINE_ARN = os.environ.get('GENASL_STATE_MACHINE_ARN', '')
GENASL_VIDEOS_BUCKET = os.environ.get('GENASL_VIDEOS_BUCKET', 'genasl-videos')
GENASL_SIGNS_TABLE = os.environ.get('GENASL_SIGNS_TABLE', 'genasl-signs')

# Stripe Configuration
STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', '')
STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY', '')
STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET', '')
STRIPE_PRO_PRICE_ID = os.environ.get('STRIPE_PRO_PRICE_ID', '')
STRIPE_PRO_ANNUAL_PRICE_ID = os.environ.get('STRIPE_PRO_ANNUAL_PRICE_ID', '')
STRIPE_ENTERPRISE_PRICE_ID = os.environ.get('STRIPE_ENTERPRISE_PRICE_ID', '')
STRIPE_ENTERPRISE_ANNUAL_PRICE_ID = os.environ.get('STRIPE_ENTERPRISE_ANNUAL_PRICE_ID', '')
FRONTEND_URL = os.environ.get('FRONTEND_URL', 'http://localhost:3000')

# Intake form configuration
INTAKE_DB_PATH = os.environ.get('INTAKE_DB_PATH', '/var/www/sonzo/data/intake.db')
INTAKE_EMAIL_TO = os.environ.get('INTAKE_EMAIL_TO', '')
SMTP_HOST = os.environ.get('SMTP_HOST', '')
SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))
SMTP_USE_SSL = os.environ.get('SMTP_USE_SSL', '').lower() == 'true'
SMTP_USERNAME = os.environ.get('SMTP_USERNAME', '')
SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD', '')
SMTP_FROM = os.environ.get('SMTP_FROM', '')
SES_REGION = os.environ.get('SES_REGION', '')
SES_FROM = os.environ.get('SES_FROM', '')
SES_TO = os.environ.get('SES_TO', '')

if STRIPE_AVAILABLE and STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

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
    password_hash: Optional[str] = None
    stripe_customer_id: Optional[str] = None
    subscription_plan: str = "free"  # free, pro, enterprise
    subscription_status: str = "active"  # active, inactive, canceled, past_due
    subscription_id: Optional[str] = None
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

class IntakeSubmission(BaseModel):
    firstName: str
    lastName: str
    email: str
    phone: Optional[str] = None
    organization: Optional[str] = None
    role: Optional[str] = None
    serviceType: Optional[str] = None
    useCase: Optional[str] = None
    timeline: Optional[str] = None
    additionalNotes: Optional[str] = None

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

# ============== GENASL MODELS ==============

class GenASLGenerateRequest(BaseModel):
    text: str = Field(..., description="English text to translate and generate ASL video")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio for speech input")
    wait_for_completion: bool = Field(True, description="Wait for video generation to complete")
    max_wait_seconds: int = Field(60, ge=5, le=300, description="Max seconds to wait for completion")

class GenASLGenerateResponse(BaseModel):
    execution_id: str
    status: str
    video_url: Optional[str] = None
    gloss_sequence: Optional[List[str]] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    processing_time_ms: float

class GenASLStatusResponse(BaseModel):
    execution_id: str
    status: str
    video_url: Optional[str] = None
    gloss_sequence: Optional[List[str]] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None

class GenASLTranslateRequest(BaseModel):
    text: str = Field(..., description="English text to translate to ASL gloss")

class GenASLTranslateResponse(BaseModel):
    english: str
    gloss_sequence: List[str]
    available_videos: List[Dict[str, Any]]
    fingerspelling_needed: List[str]
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

def init_intake_db() -> None:
    db_dir = os.path.dirname(INTAKE_DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    with sqlite3.connect(INTAKE_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS intake_submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT,
                organization TEXT,
                role TEXT,
                service_type TEXT,
                use_case TEXT,
                timeline TEXT,
                additional_notes TEXT,
                file_name TEXT,
                file_path TEXT,
                file_type TEXT,
                file_size INTEGER,
                ip_address TEXT,
                user_agent TEXT
            )
            """
        )
        # Add new columns if the table already exists without them
        for column, column_type in [
            ("file_name", "TEXT"),
            ("file_path", "TEXT"),
            ("file_type", "TEXT"),
            ("file_size", "INTEGER"),
        ]:
            try:
                conn.execute(f"ALTER TABLE intake_submissions ADD COLUMN {column} {column_type}")
            except sqlite3.OperationalError:
                pass

def save_intake_submission(payload: IntakeSubmission, metadata: Dict[str, str], file_info: Dict[str, Any]) -> int:
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(INTAKE_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO intake_submissions (
                created_at, first_name, last_name, email, phone,
                organization, role, service_type, use_case, timeline,
                additional_notes, file_name, file_path, file_type, file_size,
                ip_address, user_agent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                payload.firstName,
                payload.lastName,
                payload.email,
                payload.phone,
                payload.organization,
                payload.role,
                payload.serviceType,
                payload.useCase,
                payload.timeline,
                payload.additionalNotes,
                file_info.get("file_name"),
                file_info.get("file_path"),
                file_info.get("file_type"),
                file_info.get("file_size"),
                metadata.get("ip_address"),
                metadata.get("user_agent"),
            ),
        )
        return cursor.lastrowid

def send_intake_email(payload: IntakeSubmission, metadata: Dict[str, str], attachment: Dict[str, Any]) -> None:
    if not (SMTP_HOST and SMTP_USERNAME and SMTP_PASSWORD and SMTP_FROM and INTAKE_EMAIL_TO):
        raise RuntimeError("SMTP settings are not fully configured")

    subject = f"New SonZo intake: {payload.firstName} {payload.lastName}"
    if payload.serviceType:
        subject += f" ({payload.serviceType})"

    lines = [
        "New intake submission received:",
        "",
        f"Name: {payload.firstName} {payload.lastName}",
        f"Email: {payload.email}",
        f"Phone: {payload.phone or '-'}",
        f"Organization: {payload.organization or '-'}",
        f"Role: {payload.role or '-'}",
        f"Service Type: {payload.serviceType or '-'}",
        f"Timeline: {payload.timeline or '-'}",
        f"Use Case: {payload.useCase or '-'}",
        f"Additional Notes: {payload.additionalNotes or '-'}",
        "",
        f"IP Address: {metadata.get('ip_address', '-')}",
        f"User Agent: {metadata.get('user_agent', '-')}",
    ]

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = SMTP_FROM
    message["To"] = INTAKE_EMAIL_TO
    message.set_content("\n".join(lines))
    if attachment:
        message.add_attachment(
            attachment["content"],
            maintype=attachment["maintype"],
            subtype=attachment["subtype"],
            filename=attachment["filename"],
        )

    if SMTP_USE_SSL or SMTP_PORT == 465:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=15) as smtp:
            smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
            smtp.send_message(message)
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as smtp:
            smtp.starttls()
            smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
            smtp.send_message(message)

def send_intake_email_ses(payload: IntakeSubmission, metadata: Dict[str, str], attachment: Dict[str, Any]) -> None:
    if not (SES_REGION and SES_FROM and SES_TO):
        raise RuntimeError("SES settings are not fully configured")
    if not BEDROCK_AVAILABLE:
        raise RuntimeError("boto3 not available for SES")

    subject = f"New SonZo intake: {payload.firstName} {payload.lastName}"
    if payload.serviceType:
        subject += f" ({payload.serviceType})"

    lines = [
        "New intake submission received:",
        "",
        f"Name: {payload.firstName} {payload.lastName}",
        f"Email: {payload.email}",
        f"Phone: {payload.phone or '-'}",
        f"Organization: {payload.organization or '-'}",
        f"Role: {payload.role or '-'}",
        f"Service Type: {payload.serviceType or '-'}",
        f"Timeline: {payload.timeline or '-'}",
        f"Use Case: {payload.useCase or '-'}",
        f"Additional Notes: {payload.additionalNotes or '-'}",
        "",
        f"IP Address: {metadata.get('ip_address', '-')}",
        f"User Agent: {metadata.get('user_agent', '-')}",
    ]

    if attachment:
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = SES_FROM
        msg["To"] = SES_TO
        msg.attach(MIMEText("\n".join(lines), "plain"))

        part = MIMEBase(attachment["maintype"], attachment["subtype"])
        part.set_payload(attachment["content"])
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{attachment["filename"]}"')
        msg.attach(part)

        client = boto3.client("ses", region_name=SES_REGION)
        client.send_raw_email(
            Source=SES_FROM,
            Destinations=[SES_TO],
            RawMessage={"Data": msg.as_string()},
        )
    else:
        client = boto3.client("ses", region_name=SES_REGION)
        client.send_email(
            Source=SES_FROM,
            Destination={"ToAddresses": [SES_TO]},
            Message={
                "Subject": {"Data": subject},
                "Body": {"Text": {"Data": "\n".join(lines)}},
            },
        )

def send_intake_email_safe(payload: IntakeSubmission, metadata: Dict[str, str], attachment: Dict[str, Any]) -> None:
    try:
        if SMTP_HOST and SMTP_USERNAME and SMTP_PASSWORD and SMTP_FROM and INTAKE_EMAIL_TO:
            send_intake_email(payload, metadata, attachment)
        elif SES_REGION and SES_FROM and SES_TO:
            send_intake_email_ses(payload, metadata, attachment)
        else:
            raise RuntimeError("No email configuration available")
    except Exception as exc:
        logger.warning(f"Failed to send intake email: {exc}")

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

# ============== INTAKE ROUTES ==============

@api_router.post("/intake")
async def submit_intake(request: Request, background_tasks: BackgroundTasks):
    content_type = request.headers.get("content-type", "")
    file_info: Dict[str, Any] = {}
    attachment: Dict[str, Any] = {}

    if content_type.startswith("application/json"):
        data = await request.json()
        try:
            payload = IntakeSubmission(**data)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors())
    else:
        form = await request.form()
        data = {k: form.get(k) for k in form.keys() if k != "logoFile"}
        try:
            payload = IntakeSubmission(**data)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors())

        upload = form.get("logoFile")
        if upload:
            max_bytes = 10 * 1024 * 1024
            content = await upload.read()
            if len(content) > max_bytes:
                raise HTTPException(status_code=400, detail="File must be 10MB or smaller")

            uploads_dir = "/var/www/sonzo/data/uploads"
            os.makedirs(uploads_dir, exist_ok=True)
            safe_name = upload.filename or "upload"
            file_id = uuid.uuid4().hex
            stored_name = f"{file_id}_{safe_name}"
            file_path = os.path.join(uploads_dir, stored_name)
            with open(file_path, "wb") as handle:
                handle.write(content)

            file_info = {
                "file_name": safe_name,
                "file_path": file_path,
                "file_type": upload.content_type,
                "file_size": len(content),
            }
            maintype = "application"
            subtype = "octet-stream"
            if upload.content_type and "/" in upload.content_type:
                maintype, subtype = upload.content_type.split("/", 1)
            attachment = {
                "filename": safe_name,
                "content": content,
                "maintype": maintype,
                "subtype": subtype,
            }

    if not payload.firstName or not payload.lastName or not payload.email:
        raise HTTPException(status_code=400, detail="Missing required fields")
    if "@" not in payload.email:
        raise HTTPException(status_code=400, detail="Invalid email address")

    metadata = {
        "ip_address": request.client.host if request.client else "",
        "user_agent": request.headers.get("user-agent", ""),
    }

    await run_in_threadpool(init_intake_db)
    submission_id = await run_in_threadpool(save_intake_submission, payload, metadata, file_info)

    email_queued = False
    if (
        (SMTP_HOST and SMTP_USERNAME and SMTP_PASSWORD and SMTP_FROM and INTAKE_EMAIL_TO)
        or (SES_REGION and SES_FROM and SES_TO)
    ):
        background_tasks.add_task(send_intake_email_safe, payload, metadata, attachment)
        email_queued = True

    return {
        "status": "ok",
        "submission_id": submission_id,
        "email_queued": email_queued,
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
    data = user.model_dump()
    data.pop("password_hash", None)
    return data

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

# ============== PASSWORD AUTH ROUTES ==============

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str

class LoginPasswordRequest(BaseModel):
    email: str
    password: str

@api_router.post("/auth/register")
async def register(data: RegisterRequest, response: Response):
    """Register a new user with email and password"""
    if not data.email or "@" not in data.email:
        raise HTTPException(status_code=400, detail="Invalid email address")

    if len(data.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    if not data.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")

    existing_user = await db.users.find_one({"email": data.email.lower().strip()})
    if existing_user:
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    password_hash = pwd_context.hash(data.password)

    stripe_customer_id = None
    if STRIPE_AVAILABLE and STRIPE_SECRET_KEY:
        try:
            customer = stripe.Customer.create(
                email=data.email.lower().strip(),
                name=data.name.strip(),
                metadata={"source": "sonzo_signup"}
            )
            stripe_customer_id = customer.id
        except Exception as e:
            logger.warning(f"Failed to create Stripe customer: {e}")

    user_id = f"user_{uuid.uuid4().hex[:12]}"
    new_user = {
        "user_id": user_id,
        "email": data.email.lower().strip(),
        "name": data.name.strip(),
        "picture": None,
        "password_hash": password_hash,
        "stripe_customer_id": stripe_customer_id,
        "subscription_plan": "free",
        "subscription_status": "active",
        "subscription_id": None,
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

    user_doc = await db.users.find_one({"user_id": user_id}, {"_id": 0, "password_hash": 0})

    return {
        "user": user_doc,
        "session_token": session_token
    }

@api_router.post("/auth/login-password")
async def login_password(data: LoginPasswordRequest, response: Response):
    """Login with email and password"""
    if not data.email or not data.password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    user_doc = await db.users.find_one({"email": data.email.lower().strip()})

    if not user_doc:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user_doc.get("password_hash"):
        raise HTTPException(status_code=401, detail="This account uses a different login method")

    if not pwd_context.verify(data.password, user_doc["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user_id = user_doc["user_id"]

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

    safe_user = {k: v for k, v in user_doc.items() if k not in ("_id", "password_hash")}

    return {
        "user": safe_user,
        "session_token": session_token
    }

# ============== STRIPE ROUTES ==============

class CheckoutSessionRequest(BaseModel):
    plan_id: str  # "pro" or "enterprise"
    annual: bool = False

@api_router.post("/stripe/create-checkout-session")
async def create_checkout_session(
    data: CheckoutSessionRequest,
    user: User = Depends(get_current_user)
):
    """Create a Stripe Checkout session for subscription"""
    if not STRIPE_AVAILABLE or not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Payment processing is not configured")

    price_map = {
        ("pro", False): STRIPE_PRO_PRICE_ID,
        ("pro", True): STRIPE_PRO_ANNUAL_PRICE_ID,
        ("enterprise", False): STRIPE_ENTERPRISE_PRICE_ID,
        ("enterprise", True): STRIPE_ENTERPRISE_ANNUAL_PRICE_ID,
    }

    price_id = price_map.get((data.plan_id, data.annual))
    if not price_id:
        raise HTTPException(status_code=400, detail="Invalid plan selected")

    user_doc = await db.users.find_one({"user_id": user.user_id})
    stripe_customer_id = user_doc.get("stripe_customer_id")

    if not stripe_customer_id:
        try:
            customer = stripe.Customer.create(
                email=user.email,
                name=user.name,
                metadata={"user_id": user.user_id}
            )
            stripe_customer_id = customer.id
            await db.users.update_one(
                {"user_id": user.user_id},
                {"$set": {"stripe_customer_id": stripe_customer_id}}
            )
        except Exception as e:
            logger.error(f"Failed to create Stripe customer: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize payment")

    try:
        checkout_session = stripe.checkout.Session.create(
            customer=stripe_customer_id,
            payment_method_types=["card"],
            line_items=[{
                "price": price_id,
                "quantity": 1,
            }],
            mode="subscription",
            success_url=f"{FRONTEND_URL}/?session_id={{CHECKOUT_SESSION_ID}}&status=success",
            cancel_url=f"{FRONTEND_URL}/pricing?status=canceled",
            metadata={
                "user_id": user.user_id,
                "plan_id": data.plan_id,
            },
            subscription_data={
                "metadata": {
                    "user_id": user.user_id,
                    "plan_id": data.plan_id,
                }
            }
        )

        return {"checkout_url": checkout_session.url, "session_id": checkout_session.id}
    except Exception as e:
        logger.error(f"Failed to create checkout session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create checkout session")

@api_router.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events"""
    if not STRIPE_AVAILABLE or not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe not configured")

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        if STRIPE_WEBHOOK_SECRET:
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET
            )
        else:
            event = json.loads(payload)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_type = event.get("type", "")
    data_object = event.get("data", {}).get("object", {})

    logger.info(f"Stripe webhook received: {event_type}")

    if event_type == "checkout.session.completed":
        user_id = data_object.get("metadata", {}).get("user_id")
        subscription_id = data_object.get("subscription")
        plan_id = data_object.get("metadata", {}).get("plan_id", "pro")

        if user_id:
            await db.users.update_one(
                {"user_id": user_id},
                {"$set": {
                    "subscription_plan": plan_id,
                    "subscription_status": "active",
                    "subscription_id": subscription_id,
                }}
            )
            logger.info(f"User {user_id} subscribed to {plan_id}")

    elif event_type == "customer.subscription.updated":
        subscription_id = data_object.get("id")
        status = data_object.get("status")
        user_id = data_object.get("metadata", {}).get("user_id")
        plan_id = data_object.get("metadata", {}).get("plan_id")

        if user_id:
            update_fields = {"subscription_status": status}
            if plan_id:
                update_fields["subscription_plan"] = plan_id

            await db.users.update_one(
                {"user_id": user_id},
                {"$set": update_fields}
            )
            logger.info(f"User {user_id} subscription updated: {status}")

    elif event_type == "customer.subscription.deleted":
        user_id = data_object.get("metadata", {}).get("user_id")

        if user_id:
            await db.users.update_one(
                {"user_id": user_id},
                {"$set": {
                    "subscription_plan": "free",
                    "subscription_status": "canceled",
                    "subscription_id": None,
                }}
            )
            logger.info(f"User {user_id} subscription canceled")

    elif event_type == "invoice.payment_failed":
        customer_id = data_object.get("customer")
        if customer_id:
            user_doc = await db.users.find_one({"stripe_customer_id": customer_id})
            if user_doc:
                await db.users.update_one(
                    {"user_id": user_doc["user_id"]},
                    {"$set": {"subscription_status": "past_due"}}
                )
                logger.info(f"User {user_doc['user_id']} payment failed")

    return {"status": "ok"}

@api_router.post("/stripe/create-portal-session")
async def create_portal_session(user: User = Depends(get_current_user)):
    """Create a Stripe Customer Portal session for managing subscription"""
    if not STRIPE_AVAILABLE or not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Payment processing is not configured")

    user_doc = await db.users.find_one({"user_id": user.user_id})
    stripe_customer_id = user_doc.get("stripe_customer_id")

    if not stripe_customer_id:
        raise HTTPException(status_code=400, detail="No payment account found. Please subscribe first.")

    try:
        portal_session = stripe.billing_portal.Session.create(
            customer=stripe_customer_id,
            return_url=f"{FRONTEND_URL}/pricing",
        )
        return {"portal_url": portal_session.url}
    except Exception as e:
        logger.error(f"Failed to create portal session: {e}")
        raise HTTPException(status_code=500, detail="Failed to open billing portal")

@api_router.get("/stripe/subscription")
async def get_subscription(user: User = Depends(get_current_user)):
    """Get current user subscription status"""
    user_doc = await db.users.find_one(
        {"user_id": user.user_id},
        {"_id": 0, "password_hash": 0}
    )

    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")

    subscription_info = {
        "plan": user_doc.get("subscription_plan", "free"),
        "status": user_doc.get("subscription_status", "active"),
        "stripe_customer_id": user_doc.get("stripe_customer_id"),
        "subscription_id": user_doc.get("subscription_id"),
    }

    if (subscription_info["subscription_id"] and STRIPE_AVAILABLE and STRIPE_SECRET_KEY):
        try:
            sub = stripe.Subscription.retrieve(subscription_info["subscription_id"])
            subscription_info["current_period_end"] = sub.current_period_end
            subscription_info["cancel_at_period_end"] = sub.cancel_at_period_end
        except Exception as e:
            logger.warning(f"Failed to fetch subscription details: {e}")

    return subscription_info

@api_router.get("/stripe/config")
async def get_stripe_config():
    """Get Stripe publishable key for frontend"""
    return {
        "publishable_key": STRIPE_PUBLISHABLE_KEY,
        "enabled": bool(STRIPE_AVAILABLE and STRIPE_SECRET_KEY),
    }

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


# ============== GENASL ROUTES ==============

@api_router.post("/genasl/generate", response_model=GenASLGenerateResponse)
async def generate_asl_video(
    request: GenASLGenerateRequest,
    user: User = Depends(get_current_user)
):
    """
    Generate realistic ASL avatar video from English text or speech.

    Uses AWS GenASL pipeline with 3,300+ signs from ASLLVD dataset.
    Returns presigned S3 URL for the generated video.
    """
    import time
    start_time = time.time()

    if not GENASL_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="GenASL service is disabled. Set GENASL_ENABLED=true"
        )

    genasl = get_genasl_service()

    if not genasl.is_available():
        raise HTTPException(
            status_code=503,
            detail="GenASL service not configured. Set GENASL_STATE_MACHINE_ARN"
        )

    # Decode audio if provided
    audio_bytes = None
    if request.audio_base64:
        try:
            audio_bytes = base64.b64decode(request.audio_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio base64: {e}")

    # Generate video
    result = await genasl.generate_sentence_video(
        english_text=request.text,
        audio_input=audio_bytes,
        execution_wait=request.wait_for_completion,
        max_wait_seconds=request.max_wait_seconds
    )

    # Save to history
    if result.status == GenASLStatus.SUCCEEDED:
        history_doc = {
            "history_id": f"hist_{uuid.uuid4().hex[:12]}",
            "user_id": user.user_id,
            "input_type": "text_to_asl_video",
            "input_content": request.text,
            "output_content": json.dumps({
                "gloss": result.gloss_sequence,
                "video_url": result.video_url
            }),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.translation_history.insert_one(history_doc)

    return GenASLGenerateResponse(
        execution_id=result.execution_id,
        status=result.status.value,
        video_url=result.video_url,
        gloss_sequence=result.gloss_sequence,
        duration_seconds=result.duration_seconds,
        error_message=result.error_message,
        processing_time_ms=(time.time() - start_time) * 1000
    )


@api_router.get("/genasl/status/{execution_id}", response_model=GenASLStatusResponse)
async def get_genasl_status(
    execution_id: str,
    user: User = Depends(get_current_user)
):
    """
    Check the status of a GenASL video generation execution.

    Use this to poll for async execution results when wait_for_completion=false.
    """
    if not GENASL_ENABLED:
        raise HTTPException(status_code=503, detail="GenASL service is disabled")

    genasl = get_genasl_service()
    result = await genasl.get_execution_status(execution_id)

    return GenASLStatusResponse(
        execution_id=result.execution_id,
        status=result.status.value,
        video_url=result.video_url,
        gloss_sequence=result.gloss_sequence,
        duration_seconds=result.duration_seconds,
        error_message=result.error_message
    )


@api_router.post("/genasl/translate", response_model=GenASLTranslateResponse)
async def translate_to_asl_gloss(
    request: GenASLTranslateRequest,
    user: User = Depends(get_current_user)
):
    """
    Translate English text to ASL gloss sequence.

    Uses AWS Bedrock (Claude Sonnet) for intelligent translation.
    Returns gloss sequence with video availability information.
    """
    import time
    start_time = time.time()

    if not GENASL_ENABLED:
        raise HTTPException(status_code=503, detail="GenASL service is disabled")

    genasl = get_genasl_service()

    # Translate to gloss
    gloss_sequence = await genasl.translate_to_gloss(request.text)

    # Check video availability for each gloss
    available_videos = []
    fingerspelling_needed = []

    for gloss in gloss_sequence:
        if gloss.startswith('#'):
            # Fingerspelling
            fingerspelling_needed.append(gloss[1:])
            available_videos.append({
                "gloss": gloss,
                "type": "fingerspelling",
                "available": True
            })
        else:
            video_info = await genasl.lookup_sign_video(gloss)
            if video_info:
                available_videos.append({
                    "gloss": gloss,
                    "type": "sign",
                    "available": True,
                    "video_url": video_info.get('video_url'),
                    "duration": video_info.get('duration')
                })
            else:
                available_videos.append({
                    "gloss": gloss,
                    "type": "sign",
                    "available": False
                })

    return GenASLTranslateResponse(
        english=request.text,
        gloss_sequence=gloss_sequence,
        available_videos=available_videos,
        fingerspelling_needed=fingerspelling_needed,
        processing_time_ms=(time.time() - start_time) * 1000
    )


@api_router.get("/genasl/signs")
async def get_genasl_available_signs():
    """
    Get all available ASL signs from the GenASL ASLLVD dataset.

    Returns 3,300+ signs organized by category.
    """
    if not GENASL_ENABLED:
        raise HTTPException(status_code=503, detail="GenASL service is disabled")

    genasl = get_genasl_service()
    return await genasl.get_available_signs()


@api_router.get("/genasl/signs/{gloss}")
async def get_sign_video(
    gloss: str,
    user: User = Depends(get_current_user)
):
    """
    Get the video URL for a specific ASL sign.

    Returns presigned S3 URL for the sign video.
    """
    if not GENASL_ENABLED:
        raise HTTPException(status_code=503, detail="GenASL service is disabled")

    genasl = get_genasl_service()
    video_info = await genasl.lookup_sign_video(gloss.upper())

    if not video_info:
        raise HTTPException(status_code=404, detail=f"Sign '{gloss}' not found in database")

    return video_info


@api_router.get("/genasl/health")
async def genasl_health_check():
    """Check GenASL service health and configuration."""
    if not GENASL_ENABLED:
        return {
            "status": "disabled",
            "message": "GenASL is disabled. Set GENASL_ENABLED=true to enable."
        }

    genasl = get_genasl_service()
    health = await genasl.health_check()
    health["enabled"] = GENASL_ENABLED

    return health


# ============== UPDATED CONVERSATION WITH GENASL ==============

@api_router.post("/conversation/genasl", response_model=ConversationResponse)
async def process_conversation_with_genasl(
    request: ConversationRequest,
    user: User = Depends(get_current_user)
):
    """
    Process a bidirectional conversation with realistic GenASL avatar responses.

    Same as /conversation but uses GenASL for video generation:
    - 3,300+ available signs vs ~50 previously
    - Realistic human avatar from ASLLVD dataset
    - Full sentence support instead of single signs

    Flow:
    1. User signs (frames) OR types text
    2. System recognizes sign / processes text
    3. System generates response in English
    4. GenASL translates to ASL and generates realistic video
    5. Returns response with video URL
    """
    import time
    start_time = time.time()

    if not GENASL_ENABLED:
        # Fall back to regular conversation endpoint
        return await process_conversation(request, user)

    genasl = get_genasl_service()
    translator = get_translator()
    user_text = ""
    user_asl_gloss = []

    # Step 1: Get user's message (either from sign recognition or text)
    if request.frames and len(request.frames) > 0:
        # Recognize sign from video frames via SonZo SLR API
        if not SONZO_SLR_API_KEY:
            raise HTTPException(status_code=503, detail="SLR service not configured")

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
                        user_text = translator.asl_gloss_to_english(user_asl_gloss)
                    else:
                        user_text = "Hello"
                        user_asl_gloss = ["HELLO"]
                else:
                    user_text = "Hello"
                    user_asl_gloss = ["HELLO"]

        except Exception as e:
            logger.error(f"SLR recognition error: {e}")
            user_text = "Hello"
            user_asl_gloss = ["HELLO"]

    elif request.text:
        user_text = request.text
        user_asl_gloss = await genasl.translate_to_gloss(request.text)
    else:
        raise HTTPException(status_code=400, detail="Either 'frames' or 'text' is required")

    # Step 2: Generate or retrieve conversation context
    conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"

    conversation_doc = await db.conversations.find_one(
        {"conversation_id": conversation_id, "user_id": user.user_id}
    )

    context = []
    if conversation_doc:
        context = conversation_doc.get("messages", [])

    # Step 3: Generate system response
    response_english = translator.generate_response(user_text, context)

    # Step 4: Generate ASL video using GenASL
    video_result = await genasl.generate_sentence_video(
        english_text=response_english,
        execution_wait=True,
        max_wait_seconds=45
    )

    response_asl_gloss = video_result.gloss_sequence or await genasl.translate_to_gloss(response_english)
    video_url = video_result.video_url

    # Create message objects
    user_message = ConversationMessage(
        role="user",
        content=user_text,
        asl_gloss=user_asl_gloss
    )

    system_response = ConversationMessage(
        role="system",
        content=response_english,
        asl_gloss=response_asl_gloss,
        video_url=video_url
    )

    # Save conversation
    new_messages = [
        {
            "role": "user",
            "content": user_text,
            "asl_gloss": user_asl_gloss,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        {
            "role": "system",
            "content": response_english,
            "asl_gloss": response_asl_gloss,
            "video_url": video_url,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    ]

    if conversation_doc:
        await db.conversations.update_one(
            {"conversation_id": conversation_id},
            {"$push": {"messages": {"$each": new_messages}}}
        )
    else:
        await db.conversations.insert_one({
            "conversation_id": conversation_id,
            "user_id": user.user_id,
            "messages": new_messages,
            "created_at": datetime.now(timezone.utc).isoformat()
        })

    # Get available signs from GenASL
    available_signs_data = await genasl.get_available_signs()
    available_signs = available_signs_data.get('signs', AVAILABLE_SIGNS)[:100]  # Return subset

    processing_time = (time.time() - start_time) * 1000

    return ConversationResponse(
        conversation_id=conversation_id,
        user_message=user_message,
        system_response=system_response,
        available_signs=available_signs,
        processing_time_ms=processing_time
    )


# ============== UTILITY ROUTES ==============

@api_router.get("/")
async def root():
    return {
        "message": "SignSync AI API",
        "version": "2.0.0",
        "features": ["slr", "conversation", "avatar", "genasl"],
        "genasl_enabled": GENASL_ENABLED
    }

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
