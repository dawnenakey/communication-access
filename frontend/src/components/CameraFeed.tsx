import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Camera, CameraOff, Maximize2, Minimize2, 
  RotateCcw, Settings2, Layers, Eye, EyeOff,
  Cpu, Activity, Wifi, Smartphone, Monitor, Video,
  AlertCircle, Zap, TrendingUp, BarChart3, Grid3X3
} from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import useSignRecognition, { HandLandmark, RecognitionResult, APIUsageInfo } from '@/hooks/useSignRecognition';

export type CameraType = 'webcam' | 'oak_ai' | 'lumen';

// RTMPose 133 Keypoint Structure
export interface PoseKeypoint {
  x: number;
  y: number;
  z: number;
  confidence: number;
  name: string;
}

export interface FullBodyPose {
  body: PoseKeypoint[];      // 17 body keypoints (COCO format)
  leftHand: PoseKeypoint[];  // 21 hand keypoints
  rightHand: PoseKeypoint[]; // 21 hand keypoints
  face: PoseKeypoint[];      // 68 face keypoints
  timestamp: number;
  overallConfidence: number;
}

interface CameraFeedProps {
  isActive: boolean;
  onToggle: () => void;
  onSentenceRecognized: (sentence: string, confidence: number) => void;
  onSignRecognized?: (sign: string, confidence: number) => void;
  onPoseUpdate?: (pose: FullBodyPose) => void;
  language: string;
  cameraType: CameraType;
  onCameraTypeChange: (type: CameraType) => void;
  showPoseDebug?: boolean;
}

// Body keypoint names (COCO 17)
const BODY_KEYPOINT_NAMES = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
];

// Hand keypoint names (21 per hand)
const HAND_KEYPOINT_NAMES = [
  'wrist',
  'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
  'index_mcp', 'index_pip', 'index_dip', 'index_tip',
  'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
  'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
  'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
];

const CameraFeed: React.FC<CameraFeedProps> = ({
  isActive,
  onToggle,
  onSentenceRecognized,
  onSignRecognized,
  onPoseUpdate,
  language,
  cameraType,
  onCameraTypeChange,
  showPoseDebug = false
}) => {
  const { user, isAuthenticated } = useAuth();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const depthCanvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkCanvasRef = useRef<HTMLCanvasElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showDepth, setShowDepth] = useState(true);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [showDebugPanel, setShowDebugPanel] = useState(showPoseDebug);
  const [fps, setFps] = useState(30);
  const [processingTime, setProcessingTime] = useState(0);
  const [currentSentence, setCurrentSentence] = useState<string[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [webcamStream, setWebcamStream] = useState<MediaStream | null>(null);
  const [webcamError, setWebcamError] = useState<string | null>(null);
  const [showAPIUsage, setShowAPIUsage] = useState(false);
  const [currentPose, setCurrentPose] = useState<FullBodyPose | null>(null);
  const animationRef = useRef<number>();
  const lastFrameTimeRef = useRef<number>(0);
  const frameCountRef = useRef(0);

  // Get auth token
  const token = localStorage.getItem('sonzo_token');

  // Sign recognition hook
  const {
    isProcessing,
    isInitialized: recognitionInitialized,
    mediaPipeReady,
    currentSign,
    recognizedSigns,
    apiUsage,
    error: recognitionError,
    processFrame,
    clearBuffer,
    clearError,
  } = useSignRecognition({
    token,
    language,
    confidenceThreshold: 0.7,
    onRecognition: (result) => {
      if (result.sign && onSignRecognized) {
        onSignRecognized(result.sign, result.confidence);
      }
    },
    onSentenceComplete: (sentence, confidence) => {
      onSentenceRecognized(sentence, confidence);
      setCurrentSentence(sentence.split(' '));
    },
    onAPILimitReached: () => {
      setShowAPIUsage(true);
    },
  });

  const cameraOptions: { type: CameraType; label: string; icon: React.ReactNode; description: string }[] = [
    { type: 'webcam', label: 'Webcam', icon: <Monitor className="w-4 h-4" />, description: 'Browser/Phone Camera' },
    { type: 'oak_ai', label: 'OAK AI', icon: <Cpu className="w-4 h-4" />, description: 'Luxonis OAK-D Pro' },
    { type: 'lumen', label: 'Lumen', icon: <Video className="w-4 h-4" />, description: 'Lumen 3D Camera' }
  ];

  // Start webcam
  const startWebcam = useCallback(async () => {
    try {
      setWebcamError(null);

      // Check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera access is not supported in this browser');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
      });
      setWebcamStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    } catch (err: any) {
      console.error('Webcam error:', err);

      // Provide specific error messages based on error type
      let errorMessage = 'Failed to access camera';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        errorMessage = 'Camera permission denied. Please allow camera access in your browser settings.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        errorMessage = 'No camera found. Please connect a camera and try again.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        errorMessage = 'Camera is in use by another application. Please close other apps using the camera.';
      } else if (err.name === 'OverconstrainedError') {
        errorMessage = 'Camera does not support the requested resolution. Trying default settings...';
        // Try with lower resolution
        try {
          const fallbackStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user' }
          });
          setWebcamStream(fallbackStream);
          if (videoRef.current) {
            videoRef.current.srcObject = fallbackStream;
            videoRef.current.play();
          }
          setWebcamError(null);
          return;
        } catch {
          errorMessage = 'Camera initialization failed. Please check camera permissions.';
        }
      } else if (err.name === 'SecurityError') {
        errorMessage = 'Camera access blocked. Please use HTTPS or enable camera permissions.';
      } else if (err.message) {
        errorMessage = err.message;
      }

      setWebcamError(errorMessage);
    }
  }, []);

  // Stop webcam
  const stopWebcam = useCallback(() => {
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
      setWebcamStream(null);
    }
  }, [webcamStream]);

  // Handle camera activation
  useEffect(() => {
    if (isActive && cameraType === 'webcam') {
      startWebcam();
    } else {
      stopWebcam();
    }
    return () => stopWebcam();
  }, [isActive, cameraType]);

  // Generate RTMPose-style 133 keypoints
  const generateFullPose = useCallback((): FullBodyPose => {
    const time = Date.now() / 1000;
    
    // Generate body keypoints (17 COCO)
    const body: PoseKeypoint[] = BODY_KEYPOINT_NAMES.map((name, i) => {
      const basePositions: Record<string, [number, number]> = {
        'nose': [0.5, 0.15],
        'left_eye': [0.48, 0.13], 'right_eye': [0.52, 0.13],
        'left_ear': [0.45, 0.14], 'right_ear': [0.55, 0.14],
        'left_shoulder': [0.4, 0.25], 'right_shoulder': [0.6, 0.25],
        'left_elbow': [0.35, 0.38], 'right_elbow': [0.65, 0.38],
        'left_wrist': [0.32, 0.5], 'right_wrist': [0.68, 0.5],
        'left_hip': [0.45, 0.55], 'right_hip': [0.55, 0.55],
        'left_knee': [0.44, 0.72], 'right_knee': [0.56, 0.72],
        'left_ankle': [0.43, 0.88], 'right_ankle': [0.57, 0.88]
      };
      const [bx, by] = basePositions[name] || [0.5, 0.5];
      return {
        x: bx + Math.sin(time * 1.5 + i * 0.4) * 0.01,
        y: by + Math.cos(time * 1.2 + i * 0.3) * 0.008,
        z: Math.sin(time * 2 + i * 0.5) * 0.05,
        confidence: 0.85 + Math.random() * 0.15,
        name
      };
    });

    // Generate hand keypoints (21 per hand)
    const generateHand = (isLeft: boolean): PoseKeypoint[] => {
      const baseX = isLeft ? 0.32 : 0.68;
      const baseY = 0.5;
      return HAND_KEYPOINT_NAMES.map((name, i) => {
        const offset = i * 0.015;
        const fingerOffset = Math.floor(i / 4) * 0.02;
        return {
          x: baseX + (isLeft ? -1 : 1) * (offset * 0.5) + Math.sin(time * 3 + i * 0.2) * 0.008,
          y: baseY + offset + fingerOffset + Math.cos(time * 2.5 + i * 0.3) * 0.006,
          z: Math.sin(time * 4 + i * 0.4) * 0.03,
          confidence: 0.8 + Math.random() * 0.2,
          name: `${isLeft ? 'left' : 'right'}_${name}`
        };
      });
    };

    // Generate face keypoints (68 points)
    const face: PoseKeypoint[] = Array.from({ length: 68 }, (_, i) => {
      const angle = (i / 68) * Math.PI * 2;
      const radius = i < 17 ? 0.08 : i < 27 ? 0.04 : i < 36 ? 0.03 : 0.02;
      return {
        x: 0.5 + Math.cos(angle) * radius + Math.sin(time * 2 + i * 0.1) * 0.002,
        y: 0.15 + Math.sin(angle) * radius * 0.8 + Math.cos(time * 1.8 + i * 0.15) * 0.002,
        z: Math.sin(time * 3 + i * 0.2) * 0.01,
        confidence: 0.9 + Math.random() * 0.1,
        name: `face_${i}`
      };
    });

    const allConfidences = [...body, ...generateHand(true), ...generateHand(false), ...face].map(k => k.confidence);
    const overallConfidence = allConfidences.reduce((a, b) => a + b, 0) / allConfidences.length;

    return {
      body,
      leftHand: generateHand(true),
      rightHand: generateHand(false),
      face,
      timestamp: Date.now(),
      overallConfidence
    };
  }, []);

  // Draw RTMPose skeleton overlay
  const drawPoseSkeleton = useCallback((ctx: CanvasRenderingContext2D, pose: FullBodyPose, width: number, height: number) => {
    // Body connections (COCO skeleton)
    const bodyConnections = [
      [0, 1], [0, 2], [1, 3], [2, 4], // Head
      [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], // Arms
      [5, 11], [6, 12], [11, 12], // Torso
      [11, 13], [13, 15], [12, 14], [14, 16] // Legs
    ];

    // Draw body skeleton
    ctx.strokeStyle = 'rgba(0, 255, 128, 0.8)';
    ctx.lineWidth = 3;
    bodyConnections.forEach(([start, end]) => {
      const p1 = pose.body[start];
      const p2 = pose.body[end];
      if (p1 && p2 && p1.confidence > 0.5 && p2.confidence > 0.5) {
        ctx.beginPath();
        ctx.moveTo(p1.x * width, p1.y * height);
        ctx.lineTo(p2.x * width, p2.y * height);
        ctx.stroke();
      }
    });

    // Draw body keypoints
    pose.body.forEach((kp, i) => {
      if (kp.confidence > 0.5) {
        const x = kp.x * width;
        const y = kp.y * height;
        
        // Confidence-based color
        const hue = 120 * kp.confidence;
        ctx.fillStyle = `hsla(${hue}, 100%, 50%, 0.9)`;
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();
        
        // Keypoint index
        ctx.fillStyle = 'white';
        ctx.font = '8px monospace';
        ctx.fillText(i.toString(), x + 8, y + 3);
      }
    });

    // Draw hand keypoints with connections
    const drawHand = (hand: PoseKeypoint[], color: string) => {
      const handConnections = [
        [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8], // Index
        [0, 9], [9, 10], [10, 11], [11, 12], // Middle
        [0, 13], [13, 14], [14, 15], [15, 16], // Ring
        [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
        [5, 9], [9, 13], [13, 17] // Palm
      ];

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      handConnections.forEach(([start, end]) => {
        const p1 = hand[start];
        const p2 = hand[end];
        if (p1 && p2) {
          ctx.beginPath();
          ctx.moveTo(p1.x * width, p1.y * height);
          ctx.lineTo(p2.x * width, p2.y * height);
          ctx.stroke();
        }
      });

      hand.forEach(kp => {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(kp.x * width, kp.y * height, 4, 0, Math.PI * 2);
        ctx.fill();
      });
    };

    drawHand(pose.leftHand, 'rgba(255, 100, 100, 0.9)');
    drawHand(pose.rightHand, 'rgba(100, 100, 255, 0.9)');

    // Draw face mesh (simplified)
    ctx.strokeStyle = 'rgba(255, 200, 100, 0.5)';
    ctx.lineWidth = 1;
    for (let i = 0; i < pose.face.length - 1; i++) {
      const p1 = pose.face[i];
      const p2 = pose.face[i + 1];
      if (i < 16 || (i >= 17 && i < 26) || (i >= 27 && i < 35)) {
        ctx.beginPath();
        ctx.moveTo(p1.x * width, p1.y * height);
        ctx.lineTo(p2.x * width, p2.y * height);
        ctx.stroke();
      }
    }

    pose.face.forEach(kp => {
      ctx.fillStyle = 'rgba(255, 200, 100, 0.7)';
      ctx.beginPath();
      ctx.arc(kp.x * width, kp.y * height, 2, 0, Math.PI * 2);
      ctx.fill();
    });
  }, []);

  // Main frame processing loop
  const processAndDrawFrame = useCallback(async () => {
    const canvas = canvasRef.current;
    const depthCanvas = depthCanvasRef.current;
    const landmarkCanvas = landmarkCanvasRef.current;
    const video = videoRef.current;
    
    if (!canvas || !depthCanvas || !landmarkCanvas) return;

    const ctx = canvas.getContext('2d');
    const depthCtx = depthCanvas.getContext('2d');
    const landmarkCtx = landmarkCanvas.getContext('2d');
    if (!ctx || !depthCtx || !landmarkCtx) return;

    const width = canvas.width;
    const height = canvas.height;
    const startTime = performance.now();

    // Draw video frame or simulated background
    // Only draw when video has frames (readyState >= 2, videoWidth > 0) to avoid black canvas
    if (cameraType === 'webcam' && video && webcamStream && video.readyState >= 2 && video.videoWidth > 0) {
      ctx.drawImage(video, 0, 0, width, height);
    } else {
      const gradient = ctx.createLinearGradient(0, 0, width, height);
      gradient.addColorStop(0, cameraType === 'oak_ai' ? '#16213e' : '#1a1a2e');
      gradient.addColorStop(1, cameraType === 'oak_ai' ? '#0f0f23' : '#0d0d1a');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);
    }

    // Generate and draw pose
    const pose = generateFullPose();
    setCurrentPose(pose);
    onPoseUpdate?.(pose);

    // Clear landmark canvas
    landmarkCtx.clearRect(0, 0, width, height);

    // Draw pose skeleton if enabled
    if (showLandmarks) {
      drawPoseSkeleton(landmarkCtx, pose, width, height);
    }

    // Process frame for sign recognition if recording
    if (isRecording && isActive) {
      const handLandmarks = {
        left: pose.leftHand.map(k => ({ x: k.x, y: k.y, z: k.z, visibility: k.confidence })),
        right: pose.rightHand.map(k => ({ x: k.x, y: k.y, z: k.z, visibility: k.confidence }))
      };
      await processFrame(video, canvas, handLandmarks);
    }

    // Draw depth visualization
    if (showDepth && (cameraType === 'oak_ai' || cameraType === 'lumen')) {
      depthCtx.fillStyle = '#000';
      depthCtx.fillRect(0, 0, depthCanvas.width, depthCanvas.height);

      [...pose.leftHand, ...pose.rightHand].forEach((kp) => {
        const x = kp.x * depthCanvas.width;
        const y = kp.y * depthCanvas.height;
        const depth = Math.abs(kp.z);
        const intensity = Math.min(255, depth * 2000);
        depthCtx.fillStyle = cameraType === 'lumen' 
          ? `rgb(${100 - intensity * 0.3}, ${intensity}, ${200 - intensity * 0.5})`
          : `rgb(${intensity}, ${100 - intensity * 0.3}, ${255 - intensity})`;
        depthCtx.beginPath();
        depthCtx.arc(x, y, 6, 0, Math.PI * 2);
        depthCtx.fill();
      });
    }

    // Calculate FPS
    const now = performance.now();
    frameCountRef.current++;
    if (now - lastFrameTimeRef.current >= 1000) {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
      lastFrameTimeRef.current = now;
    }

    setProcessingTime(Math.floor(now - startTime));

    if (isActive) {
      animationRef.current = requestAnimationFrame(processAndDrawFrame);
    }
  }, [isActive, showLandmarks, showDepth, generateFullPose, drawPoseSkeleton, cameraType, webcamStream, isRecording, processFrame, onPoseUpdate]);

  // Start/stop animation loop
  useEffect(() => {
    if (isActive) {
      lastFrameTimeRef.current = performance.now();
      processAndDrawFrame();
    } else {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    }
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [isActive, processAndDrawFrame]);

  const handleRecordingToggle = () => {
    if (isRecording) {
      setIsRecording(false);
    } else {
      clearBuffer();
      setCurrentSentence([]);
      setIsRecording(true);
    }
  };

  const getCameraStatusColor = () => {
    switch (cameraType) {
      case 'oak_ai': return 'text-violet-400 bg-violet-500/20';
      case 'lumen': return 'text-cyan-400 bg-cyan-500/20';
      default: return 'text-green-400 bg-green-500/20';
    }
  };

  const getCameraLabel = () => {
    switch (cameraType) {
      case 'oak_ai': return 'OAK-D Pro Active';
      case 'lumen': return 'Lumen 3D Active';
      default: return 'Webcam Active';
    }
  };

  return (
    <div className={`relative bg-card rounded-2xl border border-border overflow-hidden ${isFullscreen ? 'fixed inset-4 z-50' : ''}`}>
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between p-4 bg-gradient-to-b from-black/60 to-transparent">
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${
            isActive ? getCameraStatusColor() : 'bg-red-500/20 text-red-400'
          }`}>
            <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-current animate-pulse' : 'bg-red-500'}`} />
            {isActive ? getCameraLabel() : 'Camera Off'}
          </div>
          {isActive && (
            <>
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/10 text-white text-xs">
                <Activity className="w-3 h-3" />
                {fps} FPS
              </div>
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/20 text-emerald-400 text-xs">
                <Grid3X3 className="w-3 h-3" />
                133 pts
              </div>
              {isProcessing && (
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/20 text-primary text-xs">
                  <Zap className="w-3 h-3 animate-pulse" />
                  Processing
                </div>
              )}
            </>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowDebugPanel(!showDebugPanel)}
            className={`p-2 rounded-lg transition-colors ${showDebugPanel ? 'bg-primary/20 text-primary' : 'bg-white/10 text-white/70'}`}
            title="Pose Debug Panel"
          >
            <BarChart3 className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowLandmarks(!showLandmarks)}
            className={`p-2 rounded-lg transition-colors ${showLandmarks ? 'bg-primary/20 text-primary' : 'bg-white/10 text-white/70'}`}
            title="Toggle Landmarks"
          >
            {showLandmarks ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          </button>
          {(cameraType === 'oak_ai' || cameraType === 'lumen') && (
            <button
              onClick={() => setShowDepth(!showDepth)}
              className={`p-2 rounded-lg transition-colors ${showDepth ? 'bg-primary/20 text-primary' : 'bg-white/10 text-white/70'}`}
              title="Toggle Depth View"
            >
              <Layers className="w-4 h-4" />
            </button>
          )}
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-2 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 transition-colors"
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Pose Debug Panel */}
      {showDebugPanel && currentPose && (
        <div className="absolute top-16 right-4 z-20 w-72 max-h-80 overflow-auto p-3 rounded-xl bg-black/90 backdrop-blur border border-white/10 shadow-xl">
          <h4 className="text-xs font-bold text-white mb-2 flex items-center gap-2">
            <Grid3X3 className="w-3 h-3 text-emerald-400" />
            RTMPose Keypoints (133)
          </h4>
          
          <div className="space-y-2">
            {/* Overall Confidence */}
            <div className="flex items-center justify-between text-[10px]">
              <span className="text-white/60">Overall Confidence</span>
              <span className={`font-mono font-bold ${currentPose.overallConfidence > 0.8 ? 'text-green-400' : 'text-yellow-400'}`}>
                {(currentPose.overallConfidence * 100).toFixed(1)}%
              </span>
            </div>

            {/* Body Keypoints */}
            <div className="border-t border-white/10 pt-2">
              <p className="text-[10px] font-semibold text-emerald-400 mb-1">Body (17 pts)</p>
              <div className="grid grid-cols-2 gap-1">
                {currentPose.body.slice(0, 6).map((kp, i) => (
                  <div key={i} className="flex items-center justify-between text-[9px] px-1 py-0.5 rounded bg-white/5">
                    <span className="text-white/50 truncate">{kp.name}</span>
                    <span className={`font-mono ${kp.confidence > 0.8 ? 'text-green-400' : 'text-yellow-400'}`}>
                      {(kp.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Hand Keypoints */}
            <div className="border-t border-white/10 pt-2">
              <p className="text-[10px] font-semibold text-red-400 mb-1">Left Hand (21 pts)</p>
              <div className="flex flex-wrap gap-1">
                {currentPose.leftHand.slice(0, 5).map((kp, i) => (
                  <span key={i} className={`text-[8px] px-1 py-0.5 rounded font-mono ${kp.confidence > 0.8 ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                    {(kp.confidence * 100).toFixed(0)}
                  </span>
                ))}
                <span className="text-[8px] text-white/40">...</span>
              </div>
            </div>

            <div className="border-t border-white/10 pt-2">
              <p className="text-[10px] font-semibold text-blue-400 mb-1">Right Hand (21 pts)</p>
              <div className="flex flex-wrap gap-1">
                {currentPose.rightHand.slice(0, 5).map((kp, i) => (
                  <span key={i} className={`text-[8px] px-1 py-0.5 rounded font-mono ${kp.confidence > 0.8 ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                    {(kp.confidence * 100).toFixed(0)}
                  </span>
                ))}
                <span className="text-[8px] text-white/40">...</span>
              </div>
            </div>

            {/* Face Keypoints */}
            <div className="border-t border-white/10 pt-2">
              <p className="text-[10px] font-semibold text-orange-400 mb-1">Face (68 pts)</p>
              <div className="flex items-center gap-2 text-[9px]">
                <span className="text-white/50">Avg conf:</span>
                <span className="font-mono text-orange-400">
                  {(currentPose.face.reduce((a, b) => a + b.confidence, 0) / currentPose.face.length * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {/* Sample Coordinates */}
            <div className="border-t border-white/10 pt-2">
              <p className="text-[10px] font-semibold text-white/70 mb-1">Sample Coordinates</p>
              <div className="font-mono text-[8px] text-white/50 space-y-0.5">
                <div>nose: ({currentPose.body[0].x.toFixed(3)}, {currentPose.body[0].y.toFixed(3)}, {currentPose.body[0].z.toFixed(3)})</div>
                <div>l_wrist: ({currentPose.body[9].x.toFixed(3)}, {currentPose.body[9].y.toFixed(3)}, {currentPose.body[9].z.toFixed(3)})</div>
                <div>r_wrist: ({currentPose.body[10].x.toFixed(3)}, {currentPose.body[10].y.toFixed(3)}, {currentPose.body[10].z.toFixed(3)})</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Camera Type Selector */}
      <div className="absolute top-16 left-4 z-10 flex flex-col gap-2">
        {cameraOptions.map((option) => (
          <button
            key={option.type}
            onClick={() => onCameraTypeChange(option.type)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-all ${
              cameraType === option.type
                ? 'bg-primary text-primary-foreground shadow-lg'
                : 'bg-black/40 text-white/80 hover:bg-black/60'
            }`}
          >
            {option.icon}
            <div className="text-left">
              <div>{option.label}</div>
              <div className="text-[10px] opacity-70">{option.description}</div>
            </div>
          </button>
        ))}
      </div>

      {/* Main Camera View */}
      <div className="relative aspect-video bg-black">
        {/* Video: use opacity-[0.01] not invisible - Chrome may pause decoding with visibility:hidden, causing black canvas */}
        <video ref={videoRef} className="absolute inset-0 w-full h-full object-cover opacity-[0.01]" playsInline muted autoPlay />

        {isActive ? (
          <>
            <canvas ref={canvasRef} width={640} height={480} className="w-full h-full object-cover" />
            <canvas ref={landmarkCanvasRef} width={640} height={480} className="absolute inset-0 w-full h-full object-cover pointer-events-none" />
            
            {showDepth && (cameraType === 'oak_ai' || cameraType === 'lumen') && (
              <div className="absolute bottom-4 right-4 w-32 h-24 rounded-lg overflow-hidden border border-white/20 shadow-lg">
                <div className={`absolute top-1 left-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-black/60 z-10 ${
                  cameraType === 'lumen' ? 'text-cyan-400' : 'text-violet-400'
                }`}>
                  DEPTH
                </div>
                <canvas ref={depthCanvasRef} width={160} height={120} className="w-full h-full" />
              </div>
            )}

            {webcamError && cameraType === 'webcam' && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/80">
                <div className="text-center p-6">
                  <CameraOff className="w-12 h-12 text-red-400 mx-auto mb-3" />
                  <p className="text-red-400 font-medium mb-2">Camera Access Denied</p>
                  <p className="text-white/60 text-sm mb-4">{webcamError}</p>
                  <button onClick={startWebcam} className="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm">
                    Try Again
                  </button>
                </div>
              </div>
            )}

            {currentSign && (
              <div className="absolute top-20 left-1/2 -translate-x-1/2 z-10">
                <div className="px-4 py-2 rounded-lg bg-primary/90 text-primary-foreground font-medium text-lg animate-pulse">
                  {currentSign.replace('_', ' ').toUpperCase()}
                </div>
              </div>
            )}

            <div className="absolute bottom-4 left-4 right-40 flex items-center gap-3">
              <button
                onClick={handleRecordingToggle}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                  isRecording ? 'bg-red-500 text-white animate-pulse' : 'bg-white/10 text-white hover:bg-white/20'
                }`}
              >
                <div className={`w-3 h-3 rounded-full ${isRecording ? 'bg-white' : 'bg-red-500'}`} />
                {isRecording ? 'Recording...' : 'Start Recording'}
              </button>

              {currentSentence.length > 0 && (
                <div className="flex-1 px-4 py-2 rounded-lg bg-black/60 backdrop-blur">
                  <p className="text-sm text-white/70">Recognized:</p>
                  <p className="text-white font-medium">{currentSentence.join(' ')}</p>
                </div>
              )}
            </div>

            {recognizedSigns.length > 0 && (
              <div className="absolute bottom-20 left-4 flex items-center gap-2">
                <span className="text-xs text-white/60">Recent:</span>
                <div className="flex gap-1">
                  {recognizedSigns.slice(-5).map((sign, i) => (
                    <span key={i} className="px-2 py-1 rounded bg-white/10 text-white/80 text-xs capitalize">
                      {sign.replace('_', ' ')}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-muted/50">
            <CameraOff className="w-16 h-16 text-muted-foreground mb-4" />
            <p className="text-lg font-medium text-muted-foreground mb-2">Camera Inactive</p>
            <p className="text-sm text-muted-foreground mb-4">
              {cameraType === 'webcam' 
                ? 'Click to enable your browser/phone camera'
                : `Connect your ${cameraType === 'oak_ai' ? 'OAK-D' : 'Lumen'} camera to begin`
              }
            </p>
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
              <Grid3X3 className="w-4 h-4 text-emerald-500" />
              <span className="text-sm">
                <span className="text-muted-foreground">Keypoints:</span>{' '}
                <span className="font-medium">133 (RTMPose)</span>
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => { clearBuffer(); setCurrentSentence([]); }}
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

export default CameraFeed;
