/**
 * useBroadcast Hook
 *
 * Provides real-time broadcast functionality with WebSocket and SSE support.
 * Used for live ASL streaming with real-time captions and recognition.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

// Message types matching backend
export enum MessageType {
  JOIN = 'join',
  LEAVE = 'leave',
  ERROR = 'error',
  BROADCAST_START = 'broadcast_start',
  BROADCAST_END = 'broadcast_end',
  BROADCAST_INFO = 'broadcast_info',
  FRAME_DATA = 'frame_data',
  RECOGNITION_RESULT = 'recognition_result',
  RECOGNITION_PARTIAL = 'recognition_partial',
  CAPTION = 'caption',
  CHAT = 'chat',
  PING = 'ping',
  PONG = 'pong',
}

export enum BroadcastRole {
  BROADCASTER = 'broadcaster',
  VIEWER = 'viewer',
  INTERPRETER = 'interpreter',
}

export interface Participant {
  id: string;
  role: BroadcastRole;
  name: string;
  joined_at: string;
}

export interface RoomInfo {
  room_id: string;
  title: string;
  broadcaster_id: string | null;
  is_active: boolean;
  participant_count: number;
  recognition_enabled: boolean;
}

export interface Caption {
  type: string;
  text?: string;
  result?: {
    recognized_sign: string;
    confidence: number;
    english_translation?: string;
  };
  source?: string;
  from_participant?: string;
  timestamp: string;
}

export interface ChatMessage {
  type: string;
  text: string;
  from_participant: string;
  from_name: string;
  timestamp: string;
}

export interface BroadcastState {
  isConnected: boolean;
  isConnecting: boolean;
  room: RoomInfo | null;
  participantId: string | null;
  captions: Caption[];
  chatMessages: ChatMessage[];
  participantCount: number;
  error: string | null;
}

interface UseBroadcastOptions {
  role?: BroadcastRole;
  name?: string;
  useSSE?: boolean; // Use SSE instead of WebSocket (for viewers)
  onCaption?: (caption: Caption) => void;
  onChat?: (message: ChatMessage) => void;
  onParticipantJoin?: (participant: Participant) => void;
  onParticipantLeave?: (participantId: string) => void;
  onBroadcastEnd?: (reason: string) => void;
  onRecognitionResult?: (result: Caption) => void;
}

const getBaseUrl = () => {
  if (typeof window === 'undefined') return '';
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  return { ws: `${protocol}//${host}`, http: window.location.origin };
};

export function useBroadcast(roomId: string | null, options: UseBroadcastOptions = {}) {
  const {
    role = BroadcastRole.VIEWER,
    name = 'Anonymous',
    useSSE = false,
    onCaption,
    onChat,
    onParticipantJoin,
    onParticipantLeave,
    onBroadcastEnd,
    onRecognitionResult,
  } = options;

  const [state, setState] = useState<BroadcastState>({
    isConnected: false,
    isConnecting: false,
    room: null,
    participantId: null,
    captions: [],
    chatMessages: [],
    participantCount: 0,
    error: null,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const sseRef = useRef<EventSource | null>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const maxCaptions = 100;

  const handleMessage = useCallback((data: Record<string, unknown>) => {
    const type = data.type as string;

    switch (type) {
      case MessageType.BROADCAST_INFO:
        setState(prev => ({
          ...prev,
          room: data.room as RoomInfo,
          participantId: data.participant_id as string,
          captions: (data.recent_captions as Caption[]) || [],
          participantCount: (data.room as RoomInfo)?.participant_count || 0,
        }));
        break;

      case MessageType.JOIN:
        setState(prev => ({
          ...prev,
          participantCount: (data.participant_count as number) || prev.participantCount,
        }));
        if (onParticipantJoin) {
          onParticipantJoin(data.participant as Participant);
        }
        break;

      case MessageType.LEAVE:
        setState(prev => ({
          ...prev,
          participantCount: (data.participant_count as number) || prev.participantCount,
        }));
        if (onParticipantLeave) {
          onParticipantLeave(data.participant_id as string);
        }
        break;

      case MessageType.CAPTION:
      case MessageType.RECOGNITION_RESULT:
      case MessageType.RECOGNITION_PARTIAL: {
        const caption = data as Caption;
        setState(prev => ({
          ...prev,
          captions: [...prev.captions.slice(-maxCaptions + 1), caption],
        }));
        if (onCaption) onCaption(caption);
        if (type === MessageType.RECOGNITION_RESULT && onRecognitionResult) {
          onRecognitionResult(caption);
        }
        break;
      }

      case MessageType.CHAT: {
        const chatMsg = data as ChatMessage;
        setState(prev => ({
          ...prev,
          chatMessages: [...prev.chatMessages.slice(-100), chatMsg],
        }));
        if (onChat) onChat(chatMsg);
        break;
      }

      case MessageType.BROADCAST_END:
        setState(prev => ({
          ...prev,
          room: prev.room ? { ...prev.room, is_active: false } : null,
        }));
        if (onBroadcastEnd) {
          onBroadcastEnd((data.reason as string) || 'Broadcast ended');
        }
        break;

      case MessageType.ERROR:
        setState(prev => ({
          ...prev,
          error: (data.message as string) || 'Unknown error',
        }));
        break;
    }
  }, [onCaption, onChat, onParticipantJoin, onParticipantLeave, onBroadcastEnd, onRecognitionResult]);

  // Connect to room
  const connect = useCallback(() => {
    if (!roomId) return;

    setState(prev => ({ ...prev, isConnecting: true, error: null }));

    const urls = getBaseUrl();

    if (useSSE) {
      // Use SSE for viewers
      const sseUrl = `${urls.http}/sse/broadcast/${roomId}?name=${encodeURIComponent(name)}`;
      const eventSource = new EventSource(sseUrl);
      sseRef.current = eventSource;

      eventSource.onopen = () => {
        setState(prev => ({ ...prev, isConnected: true, isConnecting: false }));
      };

      eventSource.onerror = () => {
        setState(prev => ({
          ...prev,
          isConnected: false,
          isConnecting: false,
          error: 'SSE connection error'
        }));
      };

      // Listen for different event types
      const eventTypes = [
        'broadcast_info', 'join', 'leave', 'caption',
        'recognition_result', 'recognition_partial',
        'chat', 'broadcast_end', 'error'
      ];

      eventTypes.forEach(eventType => {
        eventSource.addEventListener(eventType, (event: MessageEvent) => {
          try {
            const data = JSON.parse(event.data);
            handleMessage({ ...data, type: eventType });
          } catch (e) {
            console.error('Failed to parse SSE message:', e);
          }
        });
      });

    } else {
      // Use WebSocket
      const wsUrl = `${urls.ws}/ws/broadcast/${roomId}?role=${role}&name=${encodeURIComponent(name)}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setState(prev => ({ ...prev, isConnected: true, isConnecting: false }));

        // Start ping interval
        pingIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: MessageType.PING }));
          }
        }, 30000);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleMessage(data);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onerror = () => {
        setState(prev => ({
          ...prev,
          error: 'WebSocket connection error'
        }));
      };

      ws.onclose = (event) => {
        setState(prev => ({
          ...prev,
          isConnected: false,
          isConnecting: false,
        }));

        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
        }

        // Reconnect on abnormal close (not manual disconnect)
        if (event.code !== 1000 && event.code !== 4003) {
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, 3000);
        }
      };
    }
  }, [roomId, role, name, useSSE, handleMessage]);

  // Disconnect from room
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close(1000);
      wsRef.current = null;
    }

    if (sseRef.current) {
      sseRef.current.close();
      sseRef.current = null;
    }

    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
    }

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    setState({
      isConnected: false,
      isConnecting: false,
      room: null,
      participantId: null,
      captions: [],
      chatMessages: [],
      participantCount: 0,
      error: null,
    });
  }, []);

  // Send frames for recognition (broadcaster only)
  const sendFrames = useCallback((frames: string[]) => {
    if (wsRef.current?.readyState === WebSocket.OPEN && role === BroadcastRole.BROADCASTER) {
      wsRef.current.send(JSON.stringify({
        type: MessageType.FRAME_DATA,
        frames,
      }));
    }
  }, [role]);

  // Send caption (broadcaster/interpreter only)
  const sendCaption = useCallback((text: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: MessageType.CAPTION,
        text,
      }));
    }
  }, []);

  // Send chat message
  const sendChat = useCallback((text: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: MessageType.CHAT,
        text,
      }));
    }
  }, []);

  // Auto-connect when roomId changes
  useEffect(() => {
    if (roomId) {
      connect();
    } else {
      disconnect();
    }

    return () => {
      disconnect();
    };
  }, [roomId, connect, disconnect]);

  return {
    ...state,
    connect,
    disconnect,
    sendFrames,
    sendCaption,
    sendChat,
  };
}

// API functions for room management
export async function createRoom(title: string, roomId?: string): Promise<RoomInfo> {
  const response = await fetch('/api/broadcast/rooms', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, room_id: roomId }),
  });

  if (!response.ok) {
    throw new Error('Failed to create room');
  }

  return response.json();
}

export async function listRooms(activeOnly = true): Promise<RoomInfo[]> {
  const response = await fetch(`/api/broadcast/rooms?active_only=${activeOnly}`);

  if (!response.ok) {
    throw new Error('Failed to list rooms');
  }

  const data = await response.json();
  return data.rooms;
}

export async function getRoom(roomId: string): Promise<RoomInfo> {
  const response = await fetch(`/api/broadcast/rooms/${roomId}`);

  if (!response.ok) {
    throw new Error('Room not found');
  }

  return response.json();
}

export async function endRoom(roomId: string): Promise<void> {
  const response = await fetch(`/api/broadcast/rooms/${roomId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error('Failed to end room');
  }
}

export async function getCaptions(roomId: string, limit = 50): Promise<Caption[]> {
  const response = await fetch(`/api/broadcast/rooms/${roomId}/captions?limit=${limit}`);

  if (!response.ok) {
    throw new Error('Failed to get captions');
  }

  const data = await response.json();
  return data.captions;
}
