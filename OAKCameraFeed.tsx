import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Camera, CameraOff, Maximize2, Minimize2, 
  RotateCcw, Settings2, Layers, Eye, EyeOff,
  Cpu, Activity, Wifi
} from 'lucide-react';

interface OAKCameraFeedProps {
  isActive: boolean;
  onToggle: () => void;
  onSentenceRecognized: (sentence: string, confidence: number) => void;
  language: string;
}

interface HandLandmark {
  x: number;
  y: number;
  z: number;
  visibility: number;
}

interface RecognitionFrame {
  landmarks: HandLandmark[];
  timestamp: number;
}

const OAKCameraFeed: React.FC<OAKCameraFeedProps> = ({
  isActive,
  onToggle,
  onSentenceRecognized,
  language
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const depthCanvasRef = useRef<HTMLCanvasElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showDepth, setShowDepth] = useState(true);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [fps, setFps] = useState(30);
  const [processingTime, setProcessingTime] = useState(0);
  const [frameBuffer, setFrameBuffer] = useState<RecognitionFrame[]>([]);
  const [currentSentence, setCurrentSentence] = useState<string[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const animationRef = useRef<number>();

  // Simulated sentences for different languages
  const sentencePatterns: Record<string, string[]> = {
    ASL: [
      "Hello, how are you today?",
      "My name is SonZo AI",
      "Nice to meet you",
      "What is your name?",
      "Thank you very much",
      "I understand sign language",
      "Can you help me please?",
      "Where is the bathroom?",
      "I am learning to sign",
      "Have a great day"
    ],
    BSL: [
      "Hello, how are you?",
      "My name is SonZo",
      "Pleased to meet you",
      "What's your name?",
      "Thank you so much",
      "I can understand signing",
      "Could you help me?",
      "Where's the toilet?",
      "I'm learning to sign",
      "Have a lovely day"
    ],
    ISL: [
      "Namaste, how are you?",
      "My name is SonZo AI",
      "Nice meeting you",
      "What is your name?",
      "Thank you very much",
      "I understand sign language",
      "Please help me",
      "Where is the restroom?",
      "I am learning signing",
      "Have a good day"
    ]
  };

  // Generate simulated hand landmarks
  const generateLandmarks = useCallback((): HandLandmark[] => {
    const basePositions = [
      { x: 0.5, y: 0.8, z: 0 },   // Wrist
      { x: 0.5, y: 0.7, z: 0.02 }, // Palm base
      { x: 0.5, y: 0.6, z: 0.03 }, // Palm center
      { x: 0.5, y: 0.5, z: 0.04 }, // Palm top
      // Thumb
      { x: 0.35, y: 0.65, z: 0.05 },
      { x: 0.3, y: 0.55, z: 0.06 },
      { x: 0.28, y: 0.45, z: 0.07 },
      { x: 0.25, y: 0.35, z: 0.08 },
      // Index
      { x: 0.42, y: 0.45, z: 0.05 },
      { x: 0.42, y: 0.35, z: 0.06 },
      { x: 0.42, y: 0.25, z: 0.07 },
      { x: 0.42, y: 0.18, z: 0.08 },
      // Middle
      { x: 0.5, y: 0.43, z: 0.05 },
      { x: 0.5, y: 0.32, z: 0.06 },
      { x: 0.5, y: 0.22, z: 0.07 },
      { x: 0.5, y: 0.15, z: 0.08 },
      // Ring
      { x: 0.58, y: 0.45, z: 0.05 },
      { x: 0.58, y: 0.35, z: 0.06 },
      { x: 0.58, y: 0.27, z: 0.07 },
      { x: 0.58, y: 0.2, z: 0.08 },
      // Pinky
      { x: 0.65, y: 0.5, z: 0.05 },
      { x: 0.68, y: 0.42, z: 0.06 },
      { x: 0.7, y: 0.35, z: 0.07 },
      { x: 0.72, y: 0.3, z: 0.08 },
    ];

    const time = Date.now() / 1000;
    return basePositions.map((pos, i) => ({
      x: pos.x + Math.sin(time * 2 + i * 0.5) * 0.02,
      y: pos.y + Math.cos(time * 1.5 + i * 0.3) * 0.015,
      z: pos.z + Math.sin(time * 3 + i * 0.7) * 0.01,
      visibility: 0.9 + Math.random() * 0.1
    }));
  }, []);

  // Draw camera feed simulation
  const drawFrame = useCallback(() => {
    const canvas = canvasRef.current;
    const depthCanvas = depthCanvasRef.current;
    if (!canvas || !depthCanvas) return;

    const ctx = canvas.getContext('2d');
    const depthCtx = depthCanvas.getContext('2d');
    if (!ctx || !depthCtx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear canvases
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, width, height);

    // Draw simulated camera feed (gradient background)
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, '#16213e');
    gradient.addColorStop(1, '#0f0f23');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    // Add noise effect
    const imageData = ctx.getImageData(0, 0, width, height);
    for (let i = 0; i < imageData.data.length; i += 4) {
      const noise = (Math.random() - 0.5) * 10;
      imageData.data[i] += noise;
      imageData.data[i + 1] += noise;
      imageData.data[i + 2] += noise;
    }
    ctx.putImageData(imageData, 0, 0);

    // Generate and draw landmarks
    const landmarks = generateLandmarks();

    if (showLandmarks) {
      // Draw connections
      const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8], // Index
        [0, 9], [9, 10], [10, 11], [11, 12], // Middle
        [0, 13], [13, 14], [14, 15], [15, 16], // Ring
        [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
        [5, 9], [9, 13], [13, 17] // Palm connections
      ];

      ctx.strokeStyle = 'rgba(139, 92, 246, 0.6)';
      ctx.lineWidth = 2;
      connections.forEach(([start, end]) => {
        if (landmarks[start] && landmarks[end]) {
          ctx.beginPath();
          ctx.moveTo(landmarks[start].x * width, landmarks[start].y * height);
          ctx.lineTo(landmarks[end].x * width, landmarks[end].y * height);
          ctx.stroke();
        }
      });

      // Draw landmark points
      landmarks.forEach((landmark, i) => {
        const x = landmark.x * width;
        const y = landmark.y * height;
        const depth = landmark.z;

        // Color based on depth
        const hue = 260 + depth * 500;
        ctx.fillStyle = `hsla(${hue}, 80%, 60%, ${landmark.visibility})`;
        
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();

        // Outer ring
        ctx.strokeStyle = `hsla(${hue}, 80%, 60%, 0.3)`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.stroke();
      });
    }

    // Draw depth visualization
    if (showDepth) {
      depthCtx.fillStyle = '#000';
      depthCtx.fillRect(0, 0, depthCanvas.width, depthCanvas.height);

      landmarks.forEach((landmark) => {
        const x = landmark.x * depthCanvas.width;
        const y = landmark.y * depthCanvas.height;
        const depth = Math.abs(landmark.z);
        
        // Depth color: closer = warmer, farther = cooler
        const intensity = Math.min(255, depth * 2000);
        depthCtx.fillStyle = `rgb(${intensity}, ${100 - intensity * 0.3}, ${255 - intensity})`;
        
        depthCtx.beginPath();
        depthCtx.arc(x, y, 8 - depth * 30, 0, Math.PI * 2);
        depthCtx.fill();
      });
    }

    // Update frame buffer for sentence recognition
    if (isRecording) {
      setFrameBuffer(prev => {
        const newBuffer = [...prev, { landmarks, timestamp: Date.now() }];
        // Keep last 90 frames (3 seconds at 30fps)
        return newBuffer.slice(-90);
      });
    }

    // Simulate FPS and processing time
    setFps(28 + Math.floor(Math.random() * 4));
    setProcessingTime(Math.floor(20 + Math.random() * 15));

    if (isActive) {
      animationRef.current = requestAnimationFrame(drawFrame);
    }
  }, [isActive, showLandmarks, showDepth, generateLandmarks, isRecording]);

  // Simulate sentence recognition
  useEffect(() => {
    if (!isRecording || frameBuffer.length < 60) return;

    const recognitionInterval = setInterval(() => {
      const sentences = sentencePatterns[language] || sentencePatterns.ASL;
      const randomSentence = sentences[Math.floor(Math.random() * sentences.length)];
      const confidence = 0.85 + Math.random() * 0.12;
      
      onSentenceRecognized(randomSentence, confidence);
      setCurrentSentence(randomSentence.split(' '));
      setFrameBuffer([]);
    }, 4000);

    return () => clearInterval(recognitionInterval);
  }, [isRecording, frameBuffer.length, language, onSentenceRecognized]);

  useEffect(() => {
    if (isActive) {
      drawFrame();
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isActive, drawFrame]);

  return (
    <div className={`relative bg-card rounded-2xl border border-border overflow-hidden ${isFullscreen ? 'fixed inset-4 z-50' : ''}`}>
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between p-4 bg-gradient-to-b from-black/60 to-transparent">
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${
            isActive ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
          }`}>
            <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-green-500 status-online' : 'bg-red-500'}`} />
            {isActive ? 'OAK-D Pro Active' : 'Camera Off'}
          </div>
          {isActive && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/10 text-white text-xs">
              <Activity className="w-3 h-3" />
              {fps} FPS
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowLandmarks(!showLandmarks)}
            className={`p-2 rounded-lg transition-colors ${showLandmarks ? 'bg-primary/20 text-primary' : 'bg-white/10 text-white/70'}`}
            title="Toggle Landmarks"
          >
            {showLandmarks ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          </button>
          <button
            onClick={() => setShowDepth(!showDepth)}
            className={`p-2 rounded-lg transition-colors ${showDepth ? 'bg-primary/20 text-primary' : 'bg-white/10 text-white/70'}`}
            title="Toggle Depth View"
          >
            <Layers className="w-4 h-4" />
          </button>
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-2 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 transition-colors"
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Main Camera View */}
      <div className="relative aspect-video bg-black">
        {isActive ? (
          <>
            <canvas
              ref={canvasRef}
              width={640}
              height={480}
              className="w-full h-full object-cover"
            />
            
            {/* Depth Mini View */}
            {showDepth && (
              <div className="absolute bottom-4 right-4 w-32 h-24 rounded-lg overflow-hidden border border-white/20 shadow-lg">
                <div className="absolute top-1 left-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-black/60 text-cyan-400 z-10">
                  DEPTH
                </div>
                <canvas
                  ref={depthCanvasRef}
                  width={160}
                  height={120}
                  className="w-full h-full"
                />
              </div>
            )}

            {/* Recognition Status */}
            <div className="absolute bottom-4 left-4 right-40 flex items-center gap-3">
              <button
                onClick={() => setIsRecording(!isRecording)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                  isRecording 
                    ? 'bg-red-500 text-white animate-pulse' 
                    : 'bg-white/10 text-white hover:bg-white/20'
                }`}
              >
                <div className={`w-3 h-3 rounded-full ${isRecording ? 'bg-white' : 'bg-red-500'}`} />
                {isRecording ? 'Recording...' : 'Start Recording'}
              </button>

              {currentSentence.length > 0 && (
                <div className="flex-1 px-4 py-2 rounded-lg bg-black/60 backdrop-blur">
                  <p className="text-sm text-white/70">Recognized:</p>
                  <p className="text-white font-medium sentence-animate">
                    {currentSentence.join(' ')}
                  </p>
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-muted/50">
            <CameraOff className="w-16 h-16 text-muted-foreground mb-4" />
            <p className="text-lg font-medium text-muted-foreground mb-2">Camera Inactive</p>
            <p className="text-sm text-muted-foreground mb-4">Connect your OAK-D camera to begin</p>
            <button
              onClick={onToggle}
              className="flex items-center gap-2 px-6 py-3 bg-primary text-primary-foreground rounded-xl font-medium hover:bg-primary/90 transition-colors"
            >
              <Camera className="w-5 h-5" />
              Activate Camera
            </button>
          </div>
        )}
      </div>

      {/* Bottom Stats Bar */}
      {isActive && (
        <div className="flex items-center justify-between px-4 py-3 bg-muted/50 border-t border-border">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-primary" />
              <span className="text-sm">
                <span className="text-muted-foreground">Processing:</span>{' '}
                <span className="font-medium">{processingTime}ms</span>
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Wifi className="w-4 h-4 text-green-500" />
              <span className="text-sm">
                <span className="text-muted-foreground">Latency:</span>{' '}
                <span className="font-medium">&lt;50ms</span>
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-500" />
              <span className="text-sm">
                <span className="text-muted-foreground">Depth:</span>{' '}
                <span className="font-medium">Stereo 3D</span>
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setFrameBuffer([])}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm bg-muted hover:bg-muted/80 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset
            </button>
            <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm bg-muted hover:bg-muted/80 transition-colors">
              <Settings2 className="w-3.5 h-3.5" />
              Configure
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default OAKCameraFeed;
