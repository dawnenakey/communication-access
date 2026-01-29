import { useState, useRef, useEffect, useCallback } from "react";
import axios from "axios";
import { toast, Toaster } from "sonner";
import { Button } from "../components/ui/button";
import { ScrollArea } from "../components/ui/scroll-area";
import { Textarea } from "../components/ui/textarea";
import {
  Hand, Camera, CameraOff, Send, Loader2, Play, Square,
  MessageCircle, Video, Volume2, RefreshCw, User, Bot, Mic
} from "lucide-react";

// API endpoint - can be configured for production
const API = import.meta.env?.VITE_API_URL || process.env.REACT_APP_API_URL || "https://api.sonzo.io";

/**
 * ConversationalDemo - Standalone Bidirectional Sign Language Demo
 *
 * Access at: sonzo.io/conversationaldemo
 *
 * Features:
 * - Sign via webcam → System recognizes → Responds in ASL
 * - Type text → System responds in ASL video (GenASL)
 * - No login required for demo
 * - Uses GenASL for realistic avatar videos with 3,300+ signs
 */
export default function ConversationalDemo() {
  // Camera states
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isRecording, setIsRecording] = useState(false);

  // Conversation states
  const [messages, setMessages] = useState([
    {
      id: 0,
      role: "system",
      content: "Welcome! I'm your ASL conversation assistant. Sign to me or type a message to start chatting.",
      asl_gloss: ["HELLO", "WELCOME"],
      timestamp: new Date().toISOString()
    }
  ]);
  const [textInput, setTextInput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [conversationId, setConversationId] = useState(null);

  // GenASL configuration
  const [useGenASL, setUseGenASL] = useState(true);
  const [genaslEnabled, setGenaslEnabled] = useState(true);
  const [currentVideo, setCurrentVideo] = useState(null);

  // Available signs - GenASL has 3,300+ signs from ASLLVD dataset
  const [availableSigns, setAvailableSigns] = useState([]);
  const [signCount, setSignCount] = useState(3300); // GenASL default

  // Frame capture for sign recognition
  const [frameBuffer, setFrameBuffer] = useState([]);
  const FRAME_BUFFER_SIZE = 16;
  const CAPTURE_INTERVAL_MS = 100;

  // Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const captureIntervalRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Load available signs from GenASL on mount
  useEffect(() => {
    loadAvailableSigns();
    checkGenASLHealth();
  }, []);

  const loadAvailableSigns = async () => {
    try {
      const response = await axios.get(`${API}/api/genasl/signs`);
      if (response.data?.signs) {
        setAvailableSigns(response.data.signs);
        setSignCount(response.data.count || response.data.signs.length);
      }
    } catch (error) {
      console.log("Using fallback signs list");
      // Fallback to local list if GenASL not available
    }
  };

  const checkGenASLHealth = async () => {
    try {
      const response = await axios.get(`${API}/api/genasl/health`);
      setGenaslEnabled(response.data?.enabled && response.data?.configured);
    } catch (error) {
      console.log("GenASL health check failed, using fallback mode");
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
        toast.success("Camera ready - sign away!");
      }
    } catch (error) {
      console.error("Camera error:", error);
      toast.error("Could not access camera. Please check permissions.");
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

    const framesToProcess = frames || frameBuffer;
    if (framesToProcess.length >= 8) {
      processConversation(framesToProcess, null);
    } else if (framesToProcess.length > 0) {
      toast.error("Please hold the sign a bit longer");
    }

    setFrameBuffer([]);
  }, [frameBuffer]);

  // Process conversation - uses GenASL for realistic avatar videos
  const processConversation = async (frames, text) => {
    setIsProcessing(true);

    try {
      const payload = {
        conversation_id: conversationId,
      };

      if (frames && frames.length > 0) {
        payload.frames = frames;
      } else if (text) {
        payload.text = text;
      }

      // Use GenASL endpoint for realistic avatar videos (3,300+ signs)
      let response;
      const endpoint = useGenASL && genaslEnabled
        ? `${API}/api/conversation/genasl`
        : `${API}/api/conversation`;

      try {
        response = await axios.post(endpoint, payload, {
          withCredentials: true,
          timeout: 60000 // GenASL may take longer for video generation
        });
      } catch (authError) {
        // If auth fails, use demo/fallback response
        console.log("Using demo mode");
        response = generateDemoResponse(text || "Hello");
      }

      const data = response.data || response;

      if (!conversationId && data.conversation_id) {
        setConversationId(data.conversation_id);
      }

      // Add messages with video URL from GenASL
      const newUserMessage = {
        id: Date.now(),
        role: "user",
        content: data.user_message?.content || text || "Sign",
        asl_gloss: data.user_message?.asl_gloss || [],
        timestamp: new Date().toISOString()
      };

      const newSystemMessage = {
        id: Date.now() + 1,
        role: "system",
        content: data.system_response?.content || "I understand!",
        asl_gloss: data.system_response?.asl_gloss || ["UNDERSTAND"],
        video_url: data.system_response?.video_url || null,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, newUserMessage, newSystemMessage]);

      // Show notification with video indicator if available
      const videoIndicator = newSystemMessage.video_url ? " (video ready)" : "";
      toast.success(`Response: ${newSystemMessage.asl_gloss.slice(0, 5).join(" ")}${newSystemMessage.asl_gloss.length > 5 ? '...' : ''}${videoIndicator}`);

      // Auto-play video if available
      if (newSystemMessage.video_url) {
        setCurrentVideo(newSystemMessage.video_url);
      }

    } catch (error) {
      console.error("Conversation error:", error);
      // Still show a response in demo mode
      const fallback = generateDemoResponse(text || "Hello");
      setMessages(prev => [...prev,
        { id: Date.now(), role: "user", content: text || "Sign", asl_gloss: [], timestamp: new Date().toISOString() },
        { id: Date.now() + 1, role: "system", content: fallback.system_response.content, asl_gloss: fallback.system_response.asl_gloss, timestamp: new Date().toISOString() }
      ]);
    } finally {
      setIsProcessing(false);
    }
  };

  // Demo fallback responses
  const generateDemoResponse = (userText) => {
    const lower = userText.toLowerCase();
    let response = { content: "I understand. How can I help?", asl_gloss: ["UNDERSTAND", "HOW", "HELP"] };

    if (lower.includes("hello") || lower.includes("hi")) {
      response = { content: "Hello! Nice to meet you!", asl_gloss: ["HELLO", "NICE_TO_MEET_YOU"] };
    } else if (lower.includes("how are you")) {
      response = { content: "I am good, thank you! How are you?", asl_gloss: ["GOOD", "THANK_YOU", "HOW", "YOU"] };
    } else if (lower.includes("thank")) {
      response = { content: "You're welcome!", asl_gloss: ["WELCOME"] };
    } else if (lower.includes("bye") || lower.includes("goodbye")) {
      response = { content: "Goodbye! Have a great day!", asl_gloss: ["GOODBYE", "GOOD"] };
    } else if (lower.includes("name")) {
      response = { content: "My name is SonZo AI. Nice to meet you!", asl_gloss: ["MY", "NAME", "NICE_TO_MEET_YOU"] };
    } else if (lower.includes("help")) {
      response = { content: "I'm here to help! What do you need?", asl_gloss: ["HELP", "WHAT", "NEED"] };
    } else if (lower.includes("love")) {
      response = { content: "I love you too!", asl_gloss: ["I_LOVE_YOU"] };
    }

    return {
      user_message: { content: userText, asl_gloss: userText.toUpperCase().split(/\s+/).filter(w => !['A', 'AN', 'THE', 'IS', 'ARE'].includes(w)) },
      system_response: response
    };
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

  // Text to speech
  const speakText = (text) => {
    if (!text) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    window.speechSynthesis.speak(utterance);
  };

  // Start new conversation
  const startNewConversation = () => {
    setMessages([{
      id: 0,
      role: "system",
      content: "New conversation started! Sign or type to chat.",
      asl_gloss: ["HELLO", "READY"],
      timestamp: new Date().toISOString()
    }]);
    setConversationId(null);
    toast.info("Started new conversation");
  };

  // Cleanup
  useEffect(() => {
    return () => {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
      }
      stopCamera();
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      <Toaster position="top-center" richColors />

      {/* Header */}
      <header className="border-b border-white/10 bg-black/20 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-500 flex items-center justify-center">
              <Hand className="w-5 h-5 text-black" />
            </div>
            <div>
              <h1 className="font-bold text-xl">SonZo AI</h1>
              <p className="text-xs text-cyan-400">Conversational Sign Language</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <span className="px-2 py-1 rounded-full bg-green-500/20 text-green-400 text-xs">
              LIVE DEMO
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto p-4 lg:p-6">
        <div className="grid lg:grid-cols-2 gap-6 h-[calc(100vh-140px)]">

          {/* Left: Camera/Input */}
          <div className="space-y-4">
            {/* Camera View */}
            <div className="rounded-2xl overflow-hidden border border-white/10 bg-black/30 backdrop-blur">
              <div className="relative aspect-video bg-slate-900">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                />
                <canvas ref={canvasRef} className="hidden" />

                {!isCameraOn && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <CameraOff className="w-16 h-16 text-slate-600 mb-4" />
                    <p className="text-slate-400 mb-4">Camera is off</p>
                    <Button
                      onClick={startCamera}
                      className="rounded-full bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400"
                    >
                      <Camera className="w-4 h-4 mr-2" />
                      Start Camera to Sign
                    </Button>
                  </div>
                )}

                {/* Recording indicator */}
                {isRecording && (
                  <div className="absolute top-4 left-4 bg-black/60 backdrop-blur px-4 py-2 rounded-full flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
                    <span className="text-sm font-medium">
                      Recording... ({frameBuffer.length}/{FRAME_BUFFER_SIZE})
                    </span>
                  </div>
                )}

                {/* Processing overlay */}
                {isProcessing && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm">
                    <div className="bg-black/80 px-6 py-4 rounded-xl flex items-center gap-3">
                      <Loader2 className="w-6 h-6 animate-spin text-cyan-400" />
                      <span>Processing...</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Camera Controls */}
              <div className="p-4 flex flex-wrap gap-3 border-t border-white/10">
                {isCameraOn ? (
                  <>
                    <Button
                      onClick={isRecording ? () => stopRecording() : startRecording}
                      disabled={isProcessing}
                      className={`rounded-full flex-1 ${
                        isRecording
                          ? 'bg-red-500 hover:bg-red-600'
                          : 'bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400'
                      }`}
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
                      className="rounded-full border-white/20 hover:bg-white/10"
                    >
                      <CameraOff className="w-4 h-4" />
                    </Button>
                  </>
                ) : (
                  <Button
                    onClick={startCamera}
                    className="rounded-full flex-1 bg-gradient-to-r from-cyan-500 to-blue-500"
                  >
                    <Camera className="w-4 h-4 mr-2" />
                    Start Camera
                  </Button>
                )}
              </div>
            </div>

            {/* Text Input */}
            <div className="rounded-2xl border border-white/10 bg-black/30 backdrop-blur p-4 space-y-3">
              <p className="text-sm text-slate-400 flex items-center gap-2">
                <Mic className="w-4 h-4" />
                Or type your message:
              </p>
              <div className="flex gap-2">
                <Textarea
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type a message... (e.g., 'Hello, how are you?')"
                  className="min-h-[60px] bg-white/5 border-white/10 rounded-xl resize-none text-white placeholder:text-slate-500"
                  disabled={isProcessing}
                />
                <Button
                  onClick={sendTextMessage}
                  disabled={!textInput.trim() || isProcessing}
                  className="rounded-xl px-4 bg-gradient-to-r from-cyan-500 to-blue-500"
                >
                  <Send className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </div>

          {/* Right: Conversation */}
          <div className="rounded-2xl border border-white/10 bg-black/30 backdrop-blur flex flex-col h-full">
            {/* Header */}
            <div className="p-4 border-b border-white/10 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <MessageCircle className="w-5 h-5 text-cyan-400" />
                <h3 className="font-bold">Conversation</h3>
              </div>
              <Button
                onClick={startNewConversation}
                variant="ghost"
                size="sm"
                className="text-slate-400 hover:text-white hover:bg-white/10"
              >
                <RefreshCw className="w-4 h-4 mr-1" />
                New
              </Button>
            </div>

            {/* Messages */}
            <ScrollArea className="flex-1 p-4">
              <div className="space-y-4">
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    {msg.role === 'system' && (
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center flex-shrink-0">
                        <Bot className="w-4 h-4 text-white" />
                      </div>
                    )}

                    <div
                      className={`max-w-[80%] rounded-2xl p-4 ${
                        msg.role === 'user'
                          ? 'bg-gradient-to-r from-cyan-500/20 to-blue-500/20 border border-cyan-500/30'
                          : 'bg-white/5 border border-white/10'
                      }`}
                    >
                      <p className="text-sm">{msg.content}</p>

                      {/* ASL Gloss */}
                      {msg.asl_gloss && msg.asl_gloss.length > 0 && (
                        <div className="mt-2 pt-2 border-t border-white/10">
                          <p className="text-xs text-slate-500 mb-1">ASL Gloss:</p>
                          <div className="flex flex-wrap gap-1">
                            {msg.asl_gloss.map((sign, i) => (
                              <span
                                key={i}
                                className={`text-xs px-2 py-0.5 rounded-full ${
                                  availableSigns.includes(sign)
                                    ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                                    : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                                }`}
                              >
                                {sign}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Actions for system messages */}
                      {msg.role === 'system' && msg.id !== 0 && (
                        <div className="mt-3 flex gap-2">
                          <Button
                            onClick={() => speakText(msg.content)}
                            variant="ghost"
                            size="sm"
                            className="h-7 text-xs hover:bg-white/10"
                          >
                            <Volume2 className="w-3 h-3 mr-1" />
                            Speak
                          </Button>
                          {msg.video_url && (
                            <Button
                              onClick={() => setCurrentVideo(msg.video_url)}
                              variant="ghost"
                              size="sm"
                              className="h-7 text-xs hover:bg-white/10 text-cyan-400"
                            >
                              <Video className="w-3 h-3 mr-1" />
                              Watch ASL
                            </Button>
                          )}
                        </div>
                      )}
                    </div>

                    {msg.role === 'user' && (
                      <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center flex-shrink-0">
                        <User className="w-4 h-4 text-slate-300" />
                      </div>
                    )}
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            </ScrollArea>

            {/* Footer */}
            <div className="p-3 border-t border-white/10 text-center">
              <p className="text-xs text-slate-500">
                <span className="text-green-400">{signCount.toLocaleString()}+</span> ASL signs supported
                <span className="mx-2">•</span>
                {genaslEnabled ? (
                  <span className="text-cyan-400">GenASL Realistic Avatar</span>
                ) : (
                  <span>Powered by AWS Bedrock</span>
                )}
                <span className="mx-2">•</span>
                ASLLVD Dataset
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* Video Modal for GenASL avatar responses */}
      {currentVideo && (
        <div
          className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50"
          onClick={() => setCurrentVideo(null)}
        >
          <div
            className="bg-slate-900 border border-white/10 rounded-2xl p-4 max-w-2xl w-full mx-4 shadow-2xl"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex justify-between items-center mb-4">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center">
                  <Hand className="w-4 h-4 text-white" />
                </div>
                <div>
                  <h4 className="font-bold">ASL Avatar Response</h4>
                  <p className="text-xs text-slate-400">Realistic signing from GenASL</p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setCurrentVideo(null)}
                className="hover:bg-white/10"
              >
                Close
              </Button>
            </div>
            <video
              src={currentVideo}
              controls
              autoPlay
              className="w-full rounded-xl bg-black"
              onEnded={() => setCurrentVideo(null)}
            >
              Your browser does not support video playback.
            </video>
            <p className="text-xs text-slate-500 mt-3 text-center">
              Video generated using AWS GenASL with ASLLVD dataset (3,300+ signs)
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
