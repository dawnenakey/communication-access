"""
Broadcast Service for Real-Time ASL Communication

This service provides:
- WebSocket support for real-time bidirectional communication
- SSE (Server-Sent Events) for one-way streaming to viewers
- Room/channel management for broadcasts
- Integration with SLR endpoints for real-time sign recognition
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
SLR_API_URL = os.getenv("SLR_API_URL", "http://localhost:8000/api/slr")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")


class BroadcastRole(str, Enum):
    BROADCASTER = "broadcaster"
    VIEWER = "viewer"
    INTERPRETER = "interpreter"


class MessageType(str, Enum):
    # Connection events
    JOIN = "join"
    LEAVE = "leave"
    ERROR = "error"

    # Broadcast events
    BROADCAST_START = "broadcast_start"
    BROADCAST_END = "broadcast_end"
    BROADCAST_INFO = "broadcast_info"

    # Recognition events
    FRAME_DATA = "frame_data"
    RECOGNITION_RESULT = "recognition_result"
    RECOGNITION_PARTIAL = "recognition_partial"

    # Chat/caption events
    CAPTION = "caption"
    CHAT = "chat"

    # Control events
    PING = "ping"
    PONG = "pong"


@dataclass
class Participant:
    """Represents a participant in a broadcast room"""
    id: str
    role: BroadcastRole
    name: str = "Anonymous"
    websocket: Optional[WebSocket] = None
    sse_queue: Optional[asyncio.Queue] = None
    joined_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "role": self.role.value,
            "name": self.name,
            "joined_at": self.joined_at.isoformat()
        }


@dataclass
class BroadcastRoom:
    """Represents a broadcast room/channel"""
    room_id: str
    title: str
    broadcaster_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    participants: Dict[str, Participant] = field(default_factory=dict)
    recognition_enabled: bool = True
    captions_buffer: List[dict] = field(default_factory=list)
    max_caption_buffer: int = 100

    def to_dict(self) -> dict:
        return {
            "room_id": self.room_id,
            "title": self.title,
            "broadcaster_id": self.broadcaster_id,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active,
            "participant_count": len(self.participants),
            "recognition_enabled": self.recognition_enabled
        }


class ConnectionManager:
    """Manages WebSocket connections and broadcast rooms"""

    def __init__(self):
        self.rooms: Dict[str, BroadcastRoom] = {}
        self.participant_to_room: Dict[str, str] = {}
        self._http_client: Optional[httpx.AsyncClient] = None
        self._recognition_tasks: Dict[str, asyncio.Task] = {}

    async def get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self):
        if self._http_client:
            await self._http_client.aclose()
        # Cancel all recognition tasks
        for task in self._recognition_tasks.values():
            task.cancel()

    def create_room(self, title: str, room_id: Optional[str] = None) -> BroadcastRoom:
        """Create a new broadcast room"""
        if room_id is None:
            room_id = str(uuid.uuid4())[:8]

        room = BroadcastRoom(room_id=room_id, title=title)
        self.rooms[room_id] = room
        logger.info(f"Created room: {room_id} - {title}")
        return room

    def get_room(self, room_id: str) -> Optional[BroadcastRoom]:
        return self.rooms.get(room_id)

    def list_rooms(self, active_only: bool = True) -> List[dict]:
        rooms = self.rooms.values()
        if active_only:
            rooms = [r for r in rooms if r.is_active]
        return [r.to_dict() for r in rooms]

    async def add_participant(
        self,
        room_id: str,
        websocket: Optional[WebSocket] = None,
        sse_queue: Optional[asyncio.Queue] = None,
        role: BroadcastRole = BroadcastRole.VIEWER,
        name: str = "Anonymous"
    ) -> Participant:
        """Add a participant to a room"""
        room = self.get_room(room_id)
        if not room:
            raise ValueError(f"Room {room_id} not found")

        participant_id = str(uuid.uuid4())[:8]
        participant = Participant(
            id=participant_id,
            role=role,
            name=name,
            websocket=websocket,
            sse_queue=sse_queue
        )

        room.participants[participant_id] = participant
        self.participant_to_room[participant_id] = room_id

        # Set broadcaster if role is broadcaster
        if role == BroadcastRole.BROADCASTER:
            room.broadcaster_id = participant_id

        # Notify others
        await self.broadcast_to_room(room_id, {
            "type": MessageType.JOIN.value,
            "participant": participant.to_dict(),
            "participant_count": len(room.participants)
        }, exclude={participant_id})

        logger.info(f"Participant {participant_id} ({role.value}) joined room {room_id}")
        return participant

    async def remove_participant(self, participant_id: str):
        """Remove a participant from their room"""
        room_id = self.participant_to_room.get(participant_id)
        if not room_id:
            return

        room = self.get_room(room_id)
        if not room:
            return

        participant = room.participants.pop(participant_id, None)
        del self.participant_to_room[participant_id]

        # If broadcaster left, end the broadcast
        if room.broadcaster_id == participant_id:
            room.is_active = False
            await self.broadcast_to_room(room_id, {
                "type": MessageType.BROADCAST_END.value,
                "reason": "Broadcaster left"
            })
        else:
            # Notify others
            await self.broadcast_to_room(room_id, {
                "type": MessageType.LEAVE.value,
                "participant_id": participant_id,
                "participant_count": len(room.participants)
            })

        logger.info(f"Participant {participant_id} left room {room_id}")

    async def broadcast_to_room(
        self,
        room_id: str,
        message: dict,
        exclude: Optional[Set[str]] = None
    ):
        """Broadcast a message to all participants in a room"""
        room = self.get_room(room_id)
        if not room:
            return

        exclude = exclude or set()
        message_json = json.dumps(message)

        for participant_id, participant in room.participants.items():
            if participant_id in exclude:
                continue

            try:
                # WebSocket participant
                if participant.websocket:
                    await participant.websocket.send_text(message_json)
                # SSE participant
                elif participant.sse_queue:
                    await participant.sse_queue.put(message)
            except Exception as e:
                logger.warning(f"Failed to send to {participant_id}: {e}")

    async def send_to_participant(self, participant_id: str, message: dict):
        """Send a message to a specific participant"""
        room_id = self.participant_to_room.get(participant_id)
        if not room_id:
            return

        room = self.get_room(room_id)
        if not room:
            return

        participant = room.participants.get(participant_id)
        if not participant:
            return

        try:
            message_json = json.dumps(message)
            if participant.websocket:
                await participant.websocket.send_text(message_json)
            elif participant.sse_queue:
                await participant.sse_queue.put(message)
        except Exception as e:
            logger.warning(f"Failed to send to {participant_id}: {e}")

    async def process_recognition(
        self,
        room_id: str,
        frames: List[str],
        participant_id: str
    ) -> Optional[dict]:
        """Send frames to SLR endpoint for recognition"""
        try:
            client = await self.get_http_client()
            response = await client.post(
                f"{SLR_API_URL}/recognize",
                json={"frames": frames},
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()

                # Broadcast recognition result to room
                room = self.get_room(room_id)
                if room:
                    caption_entry = {
                        "type": MessageType.RECOGNITION_RESULT.value,
                        "result": result,
                        "from_participant": participant_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }

                    # Add to captions buffer
                    room.captions_buffer.append(caption_entry)
                    if len(room.captions_buffer) > room.max_caption_buffer:
                        room.captions_buffer.pop(0)

                    await self.broadcast_to_room(room_id, caption_entry)

                return result
            else:
                logger.error(f"SLR recognition failed: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return None

    def add_caption(self, room_id: str, text: str, source: str = "manual"):
        """Add a caption to the room's buffer"""
        room = self.get_room(room_id)
        if not room:
            return

        caption_entry = {
            "type": MessageType.CAPTION.value,
            "text": text,
            "source": source,
            "timestamp": datetime.utcnow().isoformat()
        }

        room.captions_buffer.append(caption_entry)
        if len(room.captions_buffer) > room.max_caption_buffer:
            room.captions_buffer.pop(0)


# Create connection manager instance
manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("Broadcast service starting...")
    yield
    logger.info("Broadcast service shutting down...")
    await manager.close()


# Create FastAPI app
app = FastAPI(
    title="ASL Broadcast Service",
    description="Real-time ASL broadcast with WebSocket and SSE support",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class CreateRoomRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    room_id: Optional[str] = None


class CreateRoomResponse(BaseModel):
    room_id: str
    title: str
    created_at: str


class RoomInfo(BaseModel):
    room_id: str
    title: str
    broadcaster_id: Optional[str]
    is_active: bool
    participant_count: int


class RecognitionRequest(BaseModel):
    frames: List[str] = Field(..., min_items=1, max_items=32)


class CaptionRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = "manual"


# HTTP Endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "broadcast",
        "rooms_active": len([r for r in manager.rooms.values() if r.is_active])
    }


@app.post("/rooms", response_model=CreateRoomResponse)
async def create_room(request: CreateRoomRequest):
    """Create a new broadcast room"""
    room = manager.create_room(request.title, request.room_id)
    return CreateRoomResponse(
        room_id=room.room_id,
        title=room.title,
        created_at=room.created_at.isoformat()
    )


@app.get("/rooms")
async def list_rooms(active_only: bool = True):
    """List all broadcast rooms"""
    return {"rooms": manager.list_rooms(active_only)}


@app.get("/rooms/{room_id}")
async def get_room(room_id: str):
    """Get information about a specific room"""
    room = manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room.to_dict()


@app.delete("/rooms/{room_id}")
async def end_room(room_id: str):
    """End a broadcast room"""
    room = manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    room.is_active = False
    await manager.broadcast_to_room(room_id, {
        "type": MessageType.BROADCAST_END.value,
        "reason": "Room closed by host"
    })

    return {"status": "ended", "room_id": room_id}


@app.get("/rooms/{room_id}/captions")
async def get_captions(room_id: str, limit: int = 50):
    """Get recent captions for a room"""
    room = manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    return {"captions": room.captions_buffer[-limit:]}


@app.post("/rooms/{room_id}/captions")
async def add_caption(room_id: str, request: CaptionRequest):
    """Add a manual caption to a room"""
    room = manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    manager.add_caption(room_id, request.text, request.source)

    # Broadcast to room
    await manager.broadcast_to_room(room_id, {
        "type": MessageType.CAPTION.value,
        "text": request.text,
        "source": request.source,
        "timestamp": datetime.utcnow().isoformat()
    })

    return {"status": "added"}


# WebSocket endpoint
@app.websocket("/ws/{room_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    room_id: str,
    role: str = Query("viewer"),
    name: str = Query("Anonymous")
):
    """WebSocket endpoint for real-time broadcast communication"""
    await websocket.accept()

    # Validate room
    room = manager.get_room(room_id)
    if not room:
        await websocket.send_json({
            "type": MessageType.ERROR.value,
            "message": "Room not found"
        })
        await websocket.close(code=4004)
        return

    if not room.is_active:
        await websocket.send_json({
            "type": MessageType.ERROR.value,
            "message": "Broadcast has ended"
        })
        await websocket.close(code=4003)
        return

    # Parse role
    try:
        participant_role = BroadcastRole(role)
    except ValueError:
        participant_role = BroadcastRole.VIEWER

    # Check if broadcaster already exists
    if participant_role == BroadcastRole.BROADCASTER and room.broadcaster_id:
        await websocket.send_json({
            "type": MessageType.ERROR.value,
            "message": "Broadcaster already exists"
        })
        await websocket.close(code=4001)
        return

    # Add participant
    try:
        participant = await manager.add_participant(
            room_id=room_id,
            websocket=websocket,
            role=participant_role,
            name=name
        )
    except Exception as e:
        await websocket.send_json({
            "type": MessageType.ERROR.value,
            "message": str(e)
        })
        await websocket.close(code=4000)
        return

    # Send room info
    await websocket.send_json({
        "type": MessageType.BROADCAST_INFO.value,
        "room": room.to_dict(),
        "participant_id": participant.id,
        "recent_captions": room.captions_buffer[-10:]
    })

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == MessageType.PING.value:
                await websocket.send_json({"type": MessageType.PONG.value})

            elif message_type == MessageType.FRAME_DATA.value:
                # Only broadcasters can send frames
                if participant.role != BroadcastRole.BROADCASTER:
                    continue

                frames = data.get("frames", [])
                if frames and room.recognition_enabled:
                    # Process recognition asynchronously
                    asyncio.create_task(
                        manager.process_recognition(room_id, frames, participant.id)
                    )

            elif message_type == MessageType.CAPTION.value:
                # Only broadcasters and interpreters can add captions
                if participant.role not in [BroadcastRole.BROADCASTER, BroadcastRole.INTERPRETER]:
                    continue

                text = data.get("text", "")
                if text:
                    manager.add_caption(room_id, text, f"live:{participant.role.value}")
                    await manager.broadcast_to_room(room_id, {
                        "type": MessageType.CAPTION.value,
                        "text": text,
                        "from_participant": participant.id,
                        "source": f"live:{participant.role.value}",
                        "timestamp": datetime.utcnow().isoformat()
                    })

            elif message_type == MessageType.CHAT.value:
                # Anyone can chat
                text = data.get("text", "")
                if text:
                    await manager.broadcast_to_room(room_id, {
                        "type": MessageType.CHAT.value,
                        "text": text,
                        "from_participant": participant.id,
                        "from_name": participant.name,
                        "timestamp": datetime.utcnow().isoformat()
                    })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {participant.id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await manager.remove_participant(participant.id)


# SSE endpoint for viewers who prefer HTTP streaming
@app.get("/sse/{room_id}")
async def sse_endpoint(
    request: Request,
    room_id: str,
    name: str = Query("Anonymous")
):
    """Server-Sent Events endpoint for one-way streaming to viewers"""
    room = manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    if not room.is_active:
        raise HTTPException(status_code=410, detail="Broadcast has ended")

    # Create SSE queue for this participant
    sse_queue: asyncio.Queue = asyncio.Queue()

    # Add participant
    participant = await manager.add_participant(
        room_id=room_id,
        sse_queue=sse_queue,
        role=BroadcastRole.VIEWER,
        name=name
    )

    async def event_generator():
        try:
            # Send initial room info
            yield f"event: {MessageType.BROADCAST_INFO.value}\n"
            yield f"data: {json.dumps({'room': room.to_dict(), 'participant_id': participant.id})}\n\n"

            # Send recent captions
            for caption in room.captions_buffer[-10:]:
                yield f"event: caption\n"
                yield f"data: {json.dumps(caption)}\n\n"

            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                try:
                    # Wait for messages with timeout
                    message = await asyncio.wait_for(sse_queue.get(), timeout=30.0)
                    event_type = message.get("type", "message")
                    yield f"event: {event_type}\n"
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f"event: ping\n"
                    yield f"data: {json.dumps({'timestamp': datetime.utcnow().isoformat()})}\n\n"

        finally:
            await manager.remove_participant(participant.id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# Recognition proxy endpoint for HTTP-based recognition
@app.post("/rooms/{room_id}/recognize")
async def recognize_in_room(room_id: str, request: RecognitionRequest):
    """Process recognition and broadcast results to room"""
    room = manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    result = await manager.process_recognition(room_id, request.frames, "api")
    if result:
        return result
    else:
        raise HTTPException(status_code=500, detail="Recognition failed")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8083"))
    uvicorn.run(app, host="0.0.0.0", port=port)
