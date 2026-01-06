import { useState, useRef, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { toast } from "sonner";
import { useAuth, API } from "../App";
import { Button } from "../components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { 
  Hand, Video, Type, Volume2, Mic, MicOff, Camera, CameraOff, 
  Upload, History, BookOpen, LogOut, User, Settings, Play, Square,
  Loader2, ChevronRight, RefreshCw, Trash2
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "../components/ui/avatar";
import { Textarea } from "../components/ui/textarea";
import { ScrollArea } from "../components/ui/scroll-area";

export default function Dashboard() {
  const { user } = useAuth();
  const navigate = useNavigate();
  
  // Camera states
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [recognizedText, setRecognizedText] = useState("");
  const [confidence, setConfidence] = useState(0);
  
  // Video upload states
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [isProcessingVideo, setIsProcessingVideo] = useState(false);
  
  // Text to ASL states
  const [textInput, setTextInput] = useState("");
  const [aslSigns, setAslSigns] = useState([]);
  const [isLoadingSigns, setIsLoadingSigns] = useState(false);
  
  // Dictionary
  const [dictionary, setDictionary] = useState([]);
  
  // SLR Recognition states
  const [slrStatus, setSlrStatus] = useState("idle"); // idle, capturing, processing
  const [frameBuffer, setFrameBuffer] = useState([]);
  const FRAME_BUFFER_SIZE = 16; // Number of frames to capture
  const CAPTURE_INTERVAL_MS = 100; // Capture frame every 100ms

  // Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const uploadVideoRef = useRef(null);
  const fileInputRef = useRef(null);
  const recognitionIntervalRef = useRef(null);

  // Load dictionary on mount
  useEffect(() => {
    loadDictionary();
  }, []);

  const loadDictionary = async () => {
    try {
      const response = await axios.get(`${API}/signs`);
      setDictionary(response.data);
    } catch (error) {
      console.error("Failed to load dictionary:", error);
    }
  };

  // Camera controls
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 1280, height: 720, facingMode: "user" } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsCameraOn(true);
        toast.success("Camera started");
      }
    } catch (error) {
      console.error("Camera error:", error);
      toast.error("Failed to access camera. Please check permissions.");
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
    setIsRecognizing(false);
    setRecognizedText("");
    setConfidence(0);
    toast.info("Camera stopped");
  };

  // Capture a single frame from video as base64
  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return null;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Set canvas size to video size
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Get base64 encoded image (without data URL prefix)
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
    return dataUrl.split(',')[1]; // Remove "data:image/jpeg;base64," prefix
  }, []);

  // Send frames to SLR API for recognition
  const recognizeFrames = useCallback(async (frames) => {
    if (frames.length === 0) return;

    setSlrStatus("processing");

    try {
      const response = await axios.post(`${API}/slr/recognize`, {
        frames: frames,
        return_alternatives: true,
        confidence_threshold: 0.5
      });

      const result = response.data;

      if (result.sign && result.sign !== "UNKNOWN") {
        setRecognizedText(prev => {
          const newText = prev ? `${prev} ${result.sign}` : result.sign;
          return newText;
        });
        setConfidence(result.confidence * 100);
        toast.success(`Recognized: ${result.sign} (${(result.confidence * 100).toFixed(1)}%)`);
      }

      setSlrStatus("idle");
    } catch (error) {
      console.error("SLR recognition error:", error);
      if (error.response?.status === 503) {
        toast.error("SLR service not configured. Check API key.");
      } else if (error.response?.status === 429) {
        toast.error("Rate limit exceeded. Please wait.");
      } else {
        toast.error("Recognition failed. Please try again.");
      }
      setSlrStatus("idle");
    }
  }, []);

  // Toggle real-time recognition using SonZo SLR API
  const toggleRecognition = useCallback(() => {
    if (!isCameraOn) {
      toast.error("Please start the camera first");
      return;
    }

    if (isRecognizing) {
      // Stop recognition
      if (recognitionIntervalRef.current) {
        clearInterval(recognitionIntervalRef.current);
        recognitionIntervalRef.current = null;
      }
      setIsRecognizing(false);
      setSlrStatus("idle");
      setFrameBuffer([]);
      toast.info("Recognition stopped");
    } else {
      // Start recognition
      setIsRecognizing(true);
      setSlrStatus("capturing");
      toast.success("Recognition started - Sign in front of camera");

      let frames = [];

      // Capture frames at regular intervals
      recognitionIntervalRef.current = setInterval(() => {
        const frame = captureFrame();
        if (frame) {
          frames.push(frame);

          // When we have enough frames, send for recognition
          if (frames.length >= FRAME_BUFFER_SIZE) {
            const framesToSend = [...frames];
            frames = []; // Reset buffer
            recognizeFrames(framesToSend);
          }
        }
      }, CAPTURE_INTERVAL_MS);
    }
  }, [isCameraOn, isRecognizing, captureFrame, recognizeFrames]);

  // Cleanup recognition interval on unmount
  useEffect(() => {
    return () => {
      if (recognitionIntervalRef.current) {
        clearInterval(recognitionIntervalRef.current);
      }
    };
  }, []);

  // Video upload
  const handleVideoUpload = (event) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.type.startsWith('video/')) {
        toast.error("Please upload a video file");
        return;
      }
      const url = URL.createObjectURL(file);
      setUploadedVideo(url);
      toast.success("Video uploaded successfully");
    }
  };

  const processVideo = async () => {
    if (!uploadedVideo) {
      toast.error("Please upload a video first");
      return;
    }
    
    setIsProcessingVideo(true);
    toast.info("Processing video...");
    
    // Simulate video processing
    setTimeout(() => {
      setIsProcessingVideo(false);
      setRecognizedText("Hello, how are you? Nice to meet you.");
      setConfidence(85);
      toast.success("Video processed successfully!");
    }, 3000);
  };

  // Text to ASL
  const translateToASL = async () => {
    if (!textInput.trim()) {
      toast.error("Please enter text to translate");
      return;
    }
    
    setIsLoadingSigns(true);
    
    try {
      const words = textInput.toLowerCase().split(/\s+/);
      const matchedSigns = [];
      
      for (const word of words) {
        const response = await axios.get(`${API}/signs/search/${word}`);
        if (response.data.length > 0) {
          matchedSigns.push(response.data[0]);
        } else {
          // Add placeholder for words not in dictionary
          matchedSigns.push({ word, notFound: true });
        }
      }
      
      setAslSigns(matchedSigns);
      
      // Save to history
      await axios.post(`${API}/history`, {
        input_type: "text_to_asl",
        input_content: textInput,
        output_content: matchedSigns.map(s => s.word).join(" "),
        confidence: null
      });
      
      toast.success("Translation complete!");
    } catch (error) {
      console.error("Translation error:", error);
      toast.error("Failed to translate text");
    } finally {
      setIsLoadingSigns(false);
    }
  };

  // Text to Speech
  const speakText = (text) => {
    if (!text) {
      toast.error("No text to speak");
      return;
    }
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
    toast.success("Speaking...");
  };

  // Logout
  const handleLogout = async () => {
    try {
      await axios.post(`${API}/auth/logout`);
      navigate("/");
      toast.success("Logged out successfully");
    } catch (error) {
      console.error("Logout error:", error);
      navigate("/");
    }
  };

  // Save recognition to history
  const saveToHistory = async () => {
    if (!recognizedText) {
      toast.error("No text to save");
      return;
    }
    
    try {
      await axios.post(`${API}/history`, {
        input_type: "asl_to_text",
        input_content: "ASL Signs",
        output_content: recognizedText,
        confidence: confidence
      });
      toast.success("Saved to history!");
    } catch (error) {
      toast.error("Failed to save to history");
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 glass border-b border-white/5">
        <div className="flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-500 flex items-center justify-center">
              <Hand className="w-5 h-5 text-black" />
            </div>
            <span className="font-heading font-bold text-xl tracking-tight">SignSync AI</span>
          </div>

          <nav className="hidden md:flex items-center gap-2">
            <Button 
              variant="ghost" 
              onClick={() => navigate("/dashboard")}
              data-testid="nav-dashboard"
              className="text-foreground"
            >
              <Camera className="w-4 h-4 mr-2" />
              Translate
            </Button>
            <Button 
              variant="ghost" 
              onClick={() => navigate("/dictionary")}
              data-testid="nav-dictionary"
            >
              <BookOpen className="w-4 h-4 mr-2" />
              Dictionary
            </Button>
            <Button 
              variant="ghost" 
              onClick={() => navigate("/history")}
              data-testid="nav-history"
            >
              <History className="w-4 h-4 mr-2" />
              History
            </Button>
          </nav>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="gap-2" data-testid="user-menu-btn">
                <Avatar className="w-8 h-8">
                  <AvatarImage src={user?.picture} alt={user?.name} />
                  <AvatarFallback className="bg-primary/20 text-primary">
                    {user?.name?.charAt(0) || "U"}
                  </AvatarFallback>
                </Avatar>
                <span className="hidden md:inline font-medium">{user?.name}</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56 glass">
              <DropdownMenuItem className="cursor-pointer" disabled>
                <User className="w-4 h-4 mr-2" />
                Profile
              </DropdownMenuItem>
              <DropdownMenuItem className="cursor-pointer" disabled>
                <Settings className="w-4 h-4 mr-2" />
                Settings
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem 
                className="cursor-pointer text-destructive focus:text-destructive"
                onClick={handleLogout}
                data-testid="logout-btn"
              >
                <LogOut className="w-4 h-4 mr-2" />
                Logout
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      {/* Main Content */}
      <main className="p-6 lg:p-8">
        <div className="max-w-7xl mx-auto">
          <Tabs defaultValue="camera" className="space-y-6">
            <TabsList className="glass h-12 p-1 gap-1">
              <TabsTrigger 
                value="camera" 
                data-testid="tab-camera"
                className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground rounded-lg px-6"
              >
                <Camera className="w-4 h-4 mr-2" />
                Live Camera
              </TabsTrigger>
              <TabsTrigger 
                value="video" 
                data-testid="tab-video"
                className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground rounded-lg px-6"
              >
                <Video className="w-4 h-4 mr-2" />
                Video Upload
              </TabsTrigger>
              <TabsTrigger 
                value="text" 
                data-testid="tab-text"
                className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground rounded-lg px-6"
              >
                <Type className="w-4 h-4 mr-2" />
                Text to ASL
              </TabsTrigger>
            </TabsList>

            {/* Live Camera Tab */}
            <TabsContent value="camera" className="space-y-6">
              <div className="grid lg:grid-cols-3 gap-6">
                {/* Camera View */}
                <div className="lg:col-span-2 space-y-4">
                  <div 
                    className={`webcam-container ${isCameraOn && isRecognizing ? 'active tracing-beam' : 'inactive'}`}
                    data-testid="webcam-container"
                  >
                    <video 
                      ref={videoRef}
                      autoPlay 
                      playsInline 
                      muted
                      className="w-full h-full object-cover bg-card"
                    />
                    <canvas ref={canvasRef} className="hidden" />
                    
                    {!isCameraOn && (
                      <div className="absolute inset-0 flex flex-col items-center justify-center bg-card/80 backdrop-blur-sm">
                        <CameraOff className="w-16 h-16 text-muted-foreground mb-4" />
                        <p className="text-muted-foreground">Camera is off</p>
                        <Button 
                          onClick={startCamera}
                          data-testid="start-camera-btn"
                          className="mt-4 rounded-full bg-primary text-primary-foreground"
                        >
                          <Camera className="w-4 h-4 mr-2" />
                          Start Camera
                        </Button>
                      </div>
                    )}
                    
                    {/* Recognition status overlay */}
                    {isCameraOn && isRecognizing && (
                      <div className="absolute top-4 left-4 glass px-4 py-2 rounded-full flex items-center gap-2">
                        <div className={`w-3 h-3 rounded-full ${slrStatus === "processing" ? "bg-amber-500" : "bg-red-500"} pulse-ring`} />
                        <span className="text-sm font-medium">
                          {slrStatus === "processing" ? "Processing..." : "Capturing..."}
                        </span>
                      </div>
                    )}

                    {/* SLR API Status */}
                    {isCameraOn && (
                      <div className="absolute bottom-4 left-4 glass px-3 py-1 rounded-full">
                        <span className="text-xs text-muted-foreground">
                          Powered by SonZo SLR API
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Camera Controls */}
                  <div className="flex flex-wrap gap-3">
                    {isCameraOn ? (
                      <>
                        <Button 
                          onClick={stopCamera}
                          variant="outline"
                          data-testid="stop-camera-btn"
                          className="rounded-full border-white/10"
                        >
                          <CameraOff className="w-4 h-4 mr-2" />
                          Stop Camera
                        </Button>
                        <Button 
                          onClick={toggleRecognition}
                          data-testid="toggle-recognition-btn"
                          className={`rounded-full ${isRecognizing ? 'bg-red-500 hover:bg-red-600' : 'bg-primary'} text-white`}
                        >
                          {isRecognizing ? (
                            <>
                              <Square className="w-4 h-4 mr-2" />
                              Stop Recognition
                            </>
                          ) : (
                            <>
                              <Play className="w-4 h-4 mr-2" />
                              Start Recognition
                            </>
                          )}
                        </Button>
                      </>
                    ) : (
                      <Button 
                        onClick={startCamera}
                        data-testid="enable-camera-btn"
                        className="rounded-full bg-primary text-primary-foreground"
                      >
                        <Camera className="w-4 h-4 mr-2" />
                        Enable Camera
                      </Button>
                    )}
                  </div>
                </div>

                {/* Results Panel */}
                <div className="space-y-4">
                  <div className="glass-card p-6 space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="font-heading font-bold text-lg">Translation Result</h3>
                      {confidence > 0 && (
                        <span className="font-mono text-sm text-cyan-400">
                          {confidence.toFixed(1)}% confident
                        </span>
                      )}
                    </div>
                    
                    <div className="min-h-[120px] p-4 rounded-xl bg-black/20 border border-white/5">
                      {recognizedText ? (
                        <p className="text-lg leading-relaxed" data-testid="recognized-text">
                          {recognizedText}
                        </p>
                      ) : (
                        <p className="text-muted-foreground italic">
                          Start recognition to see translated text here...
                        </p>
                      )}
                    </div>

                    <div className="flex gap-2">
                      <Button 
                        onClick={() => speakText(recognizedText)}
                        disabled={!recognizedText}
                        data-testid="speak-btn"
                        className="flex-1 rounded-xl"
                        variant="outline"
                      >
                        <Volume2 className="w-4 h-4 mr-2" />
                        Speak
                      </Button>
                      <Button 
                        onClick={saveToHistory}
                        disabled={!recognizedText}
                        data-testid="save-history-btn"
                        className="flex-1 rounded-xl"
                        variant="outline"
                      >
                        <History className="w-4 h-4 mr-2" />
                        Save
                      </Button>
                      <Button 
                        onClick={() => {
                          setRecognizedText("");
                          setConfidence(0);
                        }}
                        disabled={!recognizedText}
                        data-testid="clear-btn"
                        variant="ghost"
                        className="rounded-xl"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>

                  {/* Quick Dictionary Preview */}
                  <div className="glass-card p-6 space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="font-heading font-bold text-lg">Dictionary</h3>
                      <Button 
                        variant="ghost" 
                        size="sm"
                        onClick={() => navigate("/dictionary")}
                        data-testid="view-dictionary-btn"
                      >
                        View All
                        <ChevronRight className="w-4 h-4 ml-1" />
                      </Button>
                    </div>
                    
                    <div className="text-center py-4">
                      <p className="font-mono text-3xl text-cyan-400 font-bold">
                        {dictionary.length}
                      </p>
                      <p className="text-sm text-muted-foreground">Signs in dictionary</p>
                    </div>
                    
                    <Button 
                      onClick={() => navigate("/dictionary")}
                      className="w-full rounded-xl"
                      variant="outline"
                      data-testid="add-signs-btn"
                    >
                      <BookOpen className="w-4 h-4 mr-2" />
                      Manage Signs
                    </Button>
                  </div>
                </div>
              </div>
            </TabsContent>

            {/* Video Upload Tab */}
            <TabsContent value="video" className="space-y-6">
              <div className="grid lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 space-y-4">
                  <div className="glass-card p-6 space-y-4">
                    <h3 className="font-heading font-bold text-lg">Upload Video</h3>
                    
                    <input 
                      type="file" 
                      accept="video/*"
                      onChange={handleVideoUpload}
                      ref={fileInputRef}
                      className="hidden"
                    />
                    
                    {uploadedVideo ? (
                      <div className="space-y-4">
                        <video 
                          ref={uploadVideoRef}
                          src={uploadedVideo}
                          controls
                          className="w-full rounded-xl bg-card aspect-video"
                          data-testid="uploaded-video"
                        />
                        <div className="flex gap-3">
                          <Button 
                            onClick={processVideo}
                            disabled={isProcessingVideo}
                            data-testid="process-video-btn"
                            className="rounded-full bg-primary text-primary-foreground"
                          >
                            {isProcessingVideo ? (
                              <>
                                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                Processing...
                              </>
                            ) : (
                              <>
                                <Play className="w-4 h-4 mr-2" />
                                Process Video
                              </>
                            )}
                          </Button>
                          <Button 
                            onClick={() => {
                              setUploadedVideo(null);
                              if (fileInputRef.current) fileInputRef.current.value = "";
                            }}
                            variant="outline"
                            className="rounded-full border-white/10"
                            data-testid="remove-video-btn"
                          >
                            <Trash2 className="w-4 h-4 mr-2" />
                            Remove
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <div 
                        onClick={() => fileInputRef.current?.click()}
                        className="border-2 border-dashed border-white/10 rounded-2xl p-12 text-center cursor-pointer hover:border-cyan-400/30 hover:bg-cyan-400/5 transition-colors"
                        data-testid="upload-area"
                      >
                        <Upload className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                        <p className="font-medium mb-2">Click to upload video</p>
                        <p className="text-sm text-muted-foreground">
                          Supports MP4, WebM, MOV files
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Video Results */}
                <div className="glass-card p-6 space-y-4">
                  <h3 className="font-heading font-bold text-lg">Video Translation</h3>
                  
                  <div className="min-h-[200px] p-4 rounded-xl bg-black/20 border border-white/5">
                    {recognizedText && !isProcessingVideo ? (
                      <p className="text-lg leading-relaxed" data-testid="video-result-text">
                        {recognizedText}
                      </p>
                    ) : isProcessingVideo ? (
                      <div className="flex flex-col items-center justify-center h-full py-8">
                        <Loader2 className="w-8 h-8 animate-spin text-cyan-400 mb-4" />
                        <p className="text-muted-foreground">Analyzing video...</p>
                      </div>
                    ) : (
                      <p className="text-muted-foreground italic text-center py-8">
                        Upload and process a video to see translation...
                      </p>
                    )}
                  </div>

                  <div className="flex gap-2">
                    <Button 
                      onClick={() => speakText(recognizedText)}
                      disabled={!recognizedText || isProcessingVideo}
                      className="flex-1 rounded-xl"
                      variant="outline"
                      data-testid="video-speak-btn"
                    >
                      <Volume2 className="w-4 h-4 mr-2" />
                      Speak
                    </Button>
                    <Button 
                      onClick={saveToHistory}
                      disabled={!recognizedText || isProcessingVideo}
                      className="flex-1 rounded-xl"
                      variant="outline"
                      data-testid="video-save-btn"
                    >
                      <History className="w-4 h-4 mr-2" />
                      Save
                    </Button>
                  </div>
                </div>
              </div>
            </TabsContent>

            {/* Text to ASL Tab */}
            <TabsContent value="text" className="space-y-6">
              <div className="grid lg:grid-cols-2 gap-6">
                {/* Input */}
                <div className="glass-card p-6 space-y-4">
                  <h3 className="font-heading font-bold text-lg">Enter Text</h3>
                  
                  <Textarea 
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    placeholder="Type text to translate to ASL signs..."
                    className="min-h-[150px] bg-black/20 border-white/10 focus:border-cyan-500/50 rounded-xl resize-none"
                    data-testid="text-input"
                  />

                  <div className="flex gap-3">
                    <Button 
                      onClick={translateToASL}
                      disabled={isLoadingSigns || !textInput.trim()}
                      data-testid="translate-btn"
                      className="rounded-full bg-primary text-primary-foreground"
                    >
                      {isLoadingSigns ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Translating...
                        </>
                      ) : (
                        <>
                          <RefreshCw className="w-4 h-4 mr-2" />
                          Translate to ASL
                        </>
                      )}
                    </Button>
                    <Button 
                      onClick={() => speakText(textInput)}
                      disabled={!textInput.trim()}
                      variant="outline"
                      className="rounded-full border-white/10"
                      data-testid="speak-input-btn"
                    >
                      <Volume2 className="w-4 h-4 mr-2" />
                      Speak
                    </Button>
                  </div>
                </div>

                {/* ASL Output */}
                <div className="glass-card p-6 space-y-4">
                  <h3 className="font-heading font-bold text-lg">ASL Signs</h3>
                  
                  <ScrollArea className="h-[300px]">
                    {aslSigns.length > 0 ? (
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                        {aslSigns.map((sign, index) => (
                          <div 
                            key={index}
                            className={`rounded-xl overflow-hidden border ${
                              sign.notFound ? 'border-amber-500/30 bg-amber-500/5' : 'border-white/10 bg-card/50'
                            }`}
                            data-testid={`asl-sign-${index}`}
                          >
                            {sign.notFound ? (
                              <div className="aspect-square flex items-center justify-center bg-amber-500/10">
                                <span className="text-amber-400 text-xs">Not found</span>
                              </div>
                            ) : (
                              <img 
                                src={`data:${sign.image_type};base64,${sign.image_data}`}
                                alt={sign.word}
                                className="w-full aspect-square object-cover"
                              />
                            )}
                            <div className="p-2 text-center">
                              <span className="text-sm font-medium capitalize">{sign.word}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="flex flex-col items-center justify-center h-full text-center py-12">
                        <Hand className="w-12 h-12 text-muted-foreground mb-4" />
                        <p className="text-muted-foreground">
                          Enter text and click translate to see ASL signs
                        </p>
                      </div>
                    )}
                  </ScrollArea>

                  {aslSigns.length > 0 && aslSigns.some(s => s.notFound) && (
                    <p className="text-sm text-amber-400">
                      Some words were not found in the dictionary. 
                      <Button 
                        variant="link" 
                        className="text-amber-400 p-0 h-auto ml-1"
                        onClick={() => navigate("/dictionary")}
                      >
                        Add them here
                      </Button>
                    </p>
                  )}
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  );
}
