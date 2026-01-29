import { useState, useRef, useEffect, useCallback } from "react";
import axios from "axios";
import { toast } from "sonner";
import { API } from "../App";
import { Button } from "./ui/button";
import { ScrollArea } from "./ui/scroll-area";
import { Textarea } from "./ui/textarea";
import {
  Hand, Camera, CameraOff, Send, Loader2, Play, Square,
  MessageCircle, Video, Volume2, RefreshCw, User, Bot
} from "lucide-react";

/**
 * ConversationMode - Bidirectional Sign Language Communication
 *
 * This component enables:
 * 1. Deaf user signs -> System recognizes -> Responds in signs (via GenASL avatar video)
 * 2. Hearing user types -> Response shown as realistic ASL video
 *
 * The conversation loop:
 * User signs/types -> Recognition -> LLM Response -> GenASL Video Generation -> Realistic Avatar
 *
 * Uses GenASL (AWS GenAI ASL Avatar) for:
 * - 3,300+ signs from ASLLVD dataset
 * - Realistic human signing videos
 * - Full sentence support
 */
export default function ConversationMode() {
  // Camera states
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isRecording, setIsRecording] = useState(false);

  // Conversation states
  const [messages, setMessages] = useState([]);
  const [textInput, setTextInput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [availableSigns, setAvailableSigns] = useState([]);

  // GenASL configuration
  const [genaslEnabled, setGenaslEnabled] = useState(true);
  const [signCount, setSignCount] = useState(3300);

  // Video playback
  const [currentVideo, setCurrentVideo] = useState(null);
  const [isPlayingResponse, setIsPlayingResponse] = useState(false);

  // Frame capture
  const [frameBuffer, setFrameBuffer] = useState([]);
  const FRAME_BUFFER_SIZE = 16;
  const CAPTURE_INTERVAL_MS = 100;

  // Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const captureIntervalRef = useRef(null);
  const responseVideoRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Load available signs from GenASL on mount
  useEffect(() => {
    loadAvailableSigns();
    checkGenASLHealth();
  }, []);

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const loadAvailableSigns = async () => {
    try {
      // Try GenASL endpoint first (3,300+ signs)
      const response = await axios.get(`${API}/genasl/signs`);
      if (response.data?.signs) {
        setAvailableSigns(response.data.signs);
        setSignCount(response.data.count || response.data.signs.length);
      }
    } catch (error) {
      // Fallback to legacy endpoint
      try {
        const fallbackResponse = await axios.get(`${API}/conversation/signs/available`);
        setAvailableSigns(fallbackResponse.data.signs || []);
        setSignCount(fallbackResponse.data.signs?.length || 50);
      } catch (fallbackError) {
        console.error("Failed to load available signs:", fallbackError);
      }
    }
  };

  const checkGenASLHealth = async () => {
    try {
      const response = await axios.get(`${API}/genasl/health`);
      setGenaslEnabled(response.data?.enabled && response.data?.configured);
    } catch (error) {
      console.log("GenASL not available, using fallback");
      setGenaslEnabled(false);
    }
  };

  // Camera controls
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsCameraOn(true);
        toast.success("Camera ready for signing");
      }
    } catch (error) {
      console.error("Camera error:", error);
      toast.error("Failed to access camera");
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraOn(false);
    stopRecording();
  };

  // Capture frame as base64
  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return null;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
    return dataUrl.split(',')[1];
  }, []);

  // Start recording sign
  const startRecording = useCallback(() => {
    if (!isCameraOn) {
      toast.error("Please start the camera first");
      return;
    }

    setIsRecording(true);
    setFrameBuffer([]);
    toast.info("Recording your sign...");

    let frames = [];
    captureIntervalRef.current = setInterval(() => {
      const frame = captureFrame();
      if (frame) {
        frames.push(frame);
        setFrameBuffer([...frames]);

        // Auto-stop after collecting enough frames
        if (frames.length >= FRAME_BUFFER_SIZE) {
          stopRecording(frames);
        }
      }
    }, CAPTURE_INTERVAL_MS);
  }, [isCameraOn, captureFrame]);

  // Stop recording and process
  const stopRecording = useCallback((frames = null) => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }

    setIsRecording(false);

    // Process captured frames
    const framesToProcess = frames || frameBuffer;
    if (framesToProcess.length >= 8) {
      processConversation(framesToProcess, null);
    } else if (framesToProcess.length > 0) {
      toast.error("Please sign for a bit longer");
    }

    setFrameBuffer([]);
  }, [frameBuffer]);

  // Process conversation - uses GenASL for realistic avatar videos
  const processConversation = async (frames, text) => {
    setIsProcessing(true);

    try {
      const payload = {
        conversation_id: conversationId,
        avatar_id: null
      };

      if (frames && frames.length > 0) {
        payload.frames = frames;
      } else if (text) {
        payload.text = text;
      }

      // Use GenASL endpoint for realistic videos (3,300+ signs from ASLLVD)
      const endpoint = genaslEnabled ? `${API}/conversation/genasl` : `${API}/conversation`;
      const response = await axios.post(endpoint, payload, {
        timeout: 60000 // GenASL video generation may take longer
      });
      const data = response.data;

      // Update conversation ID
      if (!conversationId) {
        setConversationId(data.conversation_id);
      }

      // Add messages to conversation
      const newUserMessage = {
        id: Date.now(),
        role: "user",
        content: data.user_message.content,
        asl_gloss: data.user_message.asl_gloss,
        timestamp: new Date().toISOString()
      };

      const newSystemMessage = {
        id: Date.now() + 1,
        role: "system",
        content: data.system_response.content,
        asl_gloss: data.system_response.asl_gloss,
        video_url: data.system_response.video_url,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, newUserMessage, newSystemMessage]);

      // Auto-play response as speech (optional TTS)
      // speakResponse(data.system_response.content);

      toast.success(`Response: ${data.system_response.asl_gloss.join(" ")}`);

    } catch (error) {
      console.error("Conversation error:", error);
      if (error.response?.status === 503) {
        toast.error("Sign recognition service not available");
      } else {
        toast.error("Failed to process conversation");
      }
    } finally {
      setIsProcessing(false);
    }
  };

  // Send text message
  const sendTextMessage = async () => {
    if (!textInput.trim()) return;

    const text = textInput;
    setTextInput("");
    await processConversation(null, text);
  };

  // Handle enter key
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendTextMessage();
    }
  };

  // Speak text using TTS
  const speakText = (text) => {
    if (!text) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    window.speechSynthesis.speak(utterance);
  };

  // Start new conversation
  const startNewConversation = () => {
    setMessages([]);
    setConversationId(null);
    toast.info("Started new conversation");
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
      }
      stopCamera();
    };
  }, []);

  return (
    <div className="grid lg:grid-cols-2 gap-6 h-[calc(100vh-200px)]">
      {/* Left: Camera/Input */}
      <div className="space-y-4">
        {/* Camera View */}
        <div className="glass-card overflow-hidden">
          <div className="relative aspect-video bg-card">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
            />
            <canvas ref={canvasRef} className="hidden" />

            {!isCameraOn && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-card/80">
                <CameraOff className="w-12 h-12 text-muted-foreground mb-4" />
                <p className="text-muted-foreground mb-4">Camera is off</p>
                <Button onClick={startCamera} className="rounded-full">
                  <Camera className="w-4 h-4 mr-2" />
                  Start Camera to Sign
                </Button>
              </div>
            )}

            {/* Recording indicator */}
            {isRecording && (
              <div className="absolute top-4 left-4 glass px-4 py-2 rounded-full flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
                <span className="text-sm font-medium">
                  Recording... ({frameBuffer.length}/{FRAME_BUFFER_SIZE})
                </span>
              </div>
            )}

            {/* Processing indicator */}
            {isProcessing && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                <div className="glass px-6 py-4 rounded-xl flex items-center gap-3">
                  <Loader2 className="w-6 h-6 animate-spin text-cyan-400" />
                  <span>Processing...</span>
                </div>
              </div>
            )}
          </div>

          {/* Camera Controls */}
          <div className="p-4 flex flex-wrap gap-3">
            {isCameraOn ? (
              <>
                <Button
                  onClick={isRecording ? () => stopRecording() : startRecording}
                  disabled={isProcessing}
                  className={`rounded-full flex-1 ${isRecording ? 'bg-red-500 hover:bg-red-600' : 'bg-primary'}`}
                >
                  {isRecording ? (
                    <>
                      <Square className="w-4 h-4 mr-2" />
                      Stop Recording
                    </>
                  ) : (
                    <>
                      <Hand className="w-4 h-4 mr-2" />
                      Record Sign
                    </>
                  )}
                </Button>
                <Button
                  onClick={stopCamera}
                  variant="outline"
                  className="rounded-full"
                >
                  <CameraOff className="w-4 h-4 mr-2" />
                  Stop
                </Button>
              </>
            ) : (
              <Button onClick={startCamera} className="rounded-full flex-1">
                <Camera className="w-4 h-4 mr-2" />
                Start Camera
              </Button>
            )}
          </div>
        </div>

        {/* Text Input (alternative to signing) */}
        <div className="glass-card p-4 space-y-3">
          <p className="text-sm text-muted-foreground">
            Or type your message:
          </p>
          <div className="flex gap-2">
            <Textarea
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type a message..."
              className="min-h-[60px] bg-black/20 border-white/10 rounded-xl resize-none"
              disabled={isProcessing}
            />
            <Button
              onClick={sendTextMessage}
              disabled={!textInput.trim() || isProcessing}
              className="rounded-xl px-4"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Right: Conversation */}
      <div className="glass-card flex flex-col h-full">
        {/* Header */}
        <div className="p-4 border-b border-white/5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <MessageCircle className="w-5 h-5 text-cyan-400" />
            <h3 className="font-heading font-bold">Conversation</h3>
          </div>
          <Button
            onClick={startNewConversation}
            variant="ghost"
            size="sm"
            className="text-muted-foreground"
          >
            <RefreshCw className="w-4 h-4 mr-1" />
            New
          </Button>
        </div>

        {/* Messages */}
        <ScrollArea className="flex-1 p-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center py-12">
              <Hand className="w-16 h-16 text-muted-foreground/30 mb-4" />
              <h4 className="font-medium text-lg mb-2">Start a Conversation</h4>
              <p className="text-sm text-muted-foreground max-w-xs">
                Sign in front of the camera or type a message to start communicating
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {msg.role === 'system' && (
                    <div className="w-8 h-8 rounded-full bg-cyan-500/20 flex items-center justify-center flex-shrink-0">
                      <Bot className="w-4 h-4 text-cyan-400" />
                    </div>
                  )}

                  <div
                    className={`max-w-[80%] rounded-2xl p-4 ${
                      msg.role === 'user'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-card/80 border border-white/10'
                    }`}
                  >
                    <p className="text-sm">{msg.content}</p>

                    {/* ASL Gloss */}
                    {msg.asl_gloss && msg.asl_gloss.length > 0 && (
                      <div className="mt-2 pt-2 border-t border-white/10">
                        <p className="text-xs text-muted-foreground mb-1">ASL:</p>
                        <div className="flex flex-wrap gap-1">
                          {msg.asl_gloss.map((sign, i) => (
                            <span
                              key={i}
                              className={`text-xs px-2 py-0.5 rounded-full ${
                                availableSigns.includes(sign)
                                  ? 'bg-green-500/20 text-green-400'
                                  : 'bg-amber-500/20 text-amber-400'
                              }`}
                            >
                              {sign}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Action buttons for system messages */}
                    {msg.role === 'system' && (
                      <div className="mt-3 flex gap-2">
                        <Button
                          onClick={() => speakText(msg.content)}
                          variant="ghost"
                          size="sm"
                          className="h-7 text-xs"
                        >
                          <Volume2 className="w-3 h-3 mr-1" />
                          Speak
                        </Button>
                        {msg.video_url && (
                          <Button
                            onClick={() => setCurrentVideo(msg.video_url)}
                            variant="ghost"
                            size="sm"
                            className="h-7 text-xs"
                          >
                            <Video className="w-3 h-3 mr-1" />
                            Watch Sign
                          </Button>
                        )}
                      </div>
                    )}
                  </div>

                  {msg.role === 'user' && (
                    <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0">
                      <User className="w-4 h-4 text-primary" />
                    </div>
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </ScrollArea>

        {/* Available Signs Footer */}
        <div className="p-3 border-t border-white/5">
          <p className="text-xs text-muted-foreground">
            <span className="text-green-400">{signCount.toLocaleString()}+</span> signs available
            {genaslEnabled && (
              <span className="ml-2 text-cyan-400">GenASL Realistic Avatar</span>
            )}
          </p>
        </div>
      </div>

      {/* Video Modal (for watching avatar responses) */}
      {currentVideo && (
        <div
          className="fixed inset-0 bg-black/80 flex items-center justify-center z-50"
          onClick={() => setCurrentVideo(null)}
        >
          <div className="glass-card p-4 max-w-lg w-full mx-4" onClick={e => e.stopPropagation()}>
            <div className="flex justify-between items-center mb-4">
              <h4 className="font-heading font-bold">Avatar Response</h4>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setCurrentVideo(null)}
              >
                Close
              </Button>
            </div>
            <video
              ref={responseVideoRef}
              src={currentVideo}
              controls
              autoPlay
              className="w-full rounded-xl"
            />
          </div>
        </div>
      )}
    </div>
  );
}
