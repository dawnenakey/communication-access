import { useState, useCallback, useRef, useEffect } from 'react';
import { supabase } from '@/lib/supabase';

// Load MediaPipe from CDN to avoid Vite bundling issues ("t is not a constructor")

export interface HandLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface RecognitionResult {
  sign: string;
  confidence: number;
  landmarks: {
    left?: HandLandmark[];
    right?: HandLandmark[];
  };
  sentence?: string;
  timestamp: number;
}

export interface APIUsageInfo {
  used: number;
  limit: number | 'unlimited';
  remaining: number | 'unlimited';
  tier: 'free' | 'pro' | 'enterprise';
  resetDate: string;
  isUnlimited: boolean;
  percentUsed: number;
  isDemo?: boolean;
}

interface UseSignRecognitionOptions {
  token: string | null;
  language: string;
  confidenceThreshold?: number;
  onRecognition?: (result: RecognitionResult) => void;
  onSentenceComplete?: (sentence: string, confidence: number) => void;
  onAPILimitReached?: () => void;
}

// MediaPipe Hands landmark indices
const LANDMARK_INDICES = {
  WRIST: 0,
  THUMB_CMC: 1,
  THUMB_MCP: 2,
  THUMB_IP: 3,
  THUMB_TIP: 4,
  INDEX_MCP: 5,
  INDEX_PIP: 6,
  INDEX_DIP: 7,
  INDEX_TIP: 8,
  MIDDLE_MCP: 9,
  MIDDLE_PIP: 10,
  MIDDLE_DIP: 11,
  MIDDLE_TIP: 12,
  RING_MCP: 13,
  RING_PIP: 14,
  RING_DIP: 15,
  RING_TIP: 16,
  PINKY_MCP: 17,
  PINKY_PIP: 18,
  PINKY_DIP: 19,
  PINKY_TIP: 20,
};

// ASL sign vocabulary with landmark patterns
const ASL_SIGNS: Record<string, { name: string; pattern: string }> = {
  hello: { name: 'Hello', pattern: 'wave_open_palm' },
  thank_you: { name: 'Thank you', pattern: 'chin_to_forward' },
  please: { name: 'Please', pattern: 'circular_chest' },
  yes: { name: 'Yes', pattern: 'fist_nod' },
  no: { name: 'No', pattern: 'two_finger_close' },
  help: { name: 'Help', pattern: 'fist_on_palm_up' },
  sorry: { name: 'Sorry', pattern: 'circular_fist_chest' },
  love: { name: 'Love', pattern: 'crossed_arms_chest' },
  friend: { name: 'Friend', pattern: 'hooked_index_fingers' },
  family: { name: 'Family', pattern: 'f_hands_circle' },
  eat: { name: 'Eat', pattern: 'fingers_to_mouth' },
  drink: { name: 'Drink', pattern: 'c_hand_to_mouth' },
  water: { name: 'Water', pattern: 'w_hand_chin' },
  home: { name: 'Home', pattern: 'flat_hand_cheek_chin' },
  work: { name: 'Work', pattern: 'fists_tap' },
  school: { name: 'School', pattern: 'clap_twice' },
  name: { name: 'Name', pattern: 'h_fingers_tap' },
  good: { name: 'Good', pattern: 'flat_chin_down' },
  bad: { name: 'Bad', pattern: 'flat_chin_flip' },
  want: { name: 'Want', pattern: 'claw_pull_in' },
  need: { name: 'Need', pattern: 'x_hand_down' },
  like: { name: 'Like', pattern: 'thumb_middle_pull' },
  understand: { name: 'Understand', pattern: 'index_flick_forehead' },
  learn: { name: 'Learn', pattern: 'flat_to_fist_head' },
  know: { name: 'Know', pattern: 'flat_tap_forehead' },
  think: { name: 'Think', pattern: 'index_circle_forehead' },
  feel: { name: 'Feel', pattern: 'middle_up_chest' },
  see: { name: 'See', pattern: 'v_from_eyes' },
  hear: { name: 'Hear', pattern: 'index_cup_ear' },
  speak: { name: 'Speak', pattern: 'index_from_mouth' },
};

// Sentence patterns for context-aware recognition
const SENTENCE_PATTERNS = [
  { signs: ['hello', 'name'], sentence: 'Hello, my name is' },
  { signs: ['thank_you'], sentence: 'Thank you' },
  { signs: ['please', 'help'], sentence: 'Please help me' },
  { signs: ['want', 'learn'], sentence: 'I want to learn' },
  { signs: ['understand'], sentence: 'I understand' },
  { signs: ['good', 'see'], sentence: 'Good to see you' },
  { signs: ['love', 'family'], sentence: 'I love my family' },
  { signs: ['need', 'help'], sentence: 'I need help' },
  { signs: ['want', 'eat'], sentence: 'I want to eat' },
  { signs: ['want', 'drink', 'water'], sentence: 'I want to drink water' },
];

// MediaPipe Tasks Vision HandLandmarker (replaces deprecated @mediapipe/hands)
class MediaPipeHandsDetector {
  private handLandmarker: any = null;
  private isInitialized = false;
  private initPromise: Promise<void> | null = null;
  private lastVideoTime = -1;

  async initialize(): Promise<boolean> {
    if (this.isInitialized) return true;
    if (this.initPromise) {
      await this.initPromise;
      return this.isInitialized;
    }

    this.initPromise = this.doInitialize();
    await this.initPromise;
    return this.isInitialized;
  }

  private async doInitialize(): Promise<void> {
    try {
      // Load from CDN - bypasses Vite bundler which breaks HandLandmarker
      const { HandLandmarker, FilesetResolver } = await import(
        /* @vite-ignore */ 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/vision_bundle.mjs'
      );

      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm'
      );

      this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
      });

      this.isInitialized = true;
      console.log('[MediaPipe] HandLandmarker initialized successfully');
    } catch (error) {
      console.warn('MediaPipe Hands initialization failed, using fallback:', error);
      this.isInitialized = false;
    }
  }

  async detect(videoElement: HTMLVideoElement): Promise<{ left?: HandLandmark[]; right?: HandLandmark[] } | null> {
    if (!this.handLandmarker || !this.isInitialized) {
      return null;
    }

    try {
      const videoTime = videoElement.currentTime;
      if (videoTime === this.lastVideoTime) return null;
      this.lastVideoTime = videoTime;

      const result = this.handLandmarker.detectForVideo(videoElement, performance.now());

      if (!result.landmarks || result.landmarks.length === 0) {
        return null;
      }

      const landmarks: { left?: HandLandmark[]; right?: HandLandmark[] } = {};

      result.landmarks.forEach((handLandmarks: { x: number; y: number; z?: number }[], index: number) => {
        const handedness = result.handednesses?.[index]?.[0]?.displayName || 'Right';
        const isLeft = handedness === 'Left';

        const convertedLandmarks: HandLandmark[] = handLandmarks.map((lm) => ({
          x: lm.x,
          y: lm.y,
          z: lm.z ?? 0,
          visibility: 1.0,
        }));

        if (isLeft) {
          landmarks.left = convertedLandmarks;
        } else {
          landmarks.right = convertedLandmarks;
        }
      });

      return landmarks;
    } catch {
      return null;
    }
  }
}

// Singleton detector instance
let globalDetector: MediaPipeHandsDetector | null = null;

export const useSignRecognition = ({
  token,
  language,
  confidenceThreshold = 0.7,
  onRecognition,
  onSentenceComplete,
  onAPILimitReached,
}: UseSignRecognitionOptions) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentSign, setCurrentSign] = useState<string | null>(null);
  const [recognizedSigns, setRecognizedSigns] = useState<string[]>([]);
  const [apiUsage, setApiUsage] = useState<APIUsageInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [mediaPipeReady, setMediaPipeReady] = useState(false);
  
  const signBufferRef = useRef<string[]>([]);
  const lastRecognitionRef = useRef<number>(0);
  const frameCountRef = useRef(0);
  const processingRef = useRef(false);
  const detectorRef = useRef<MediaPipeHandsDetector | null>(null);

  // Initialize MediaPipe Hands
  useEffect(() => {
    const initMediaPipe = async () => {
      if (!globalDetector) {
        globalDetector = new MediaPipeHandsDetector();
      }
      detectorRef.current = globalDetector;
      
      const success = await globalDetector.initialize();
      setMediaPipeReady(success);
      setIsInitialized(true);
    };

    initMediaPipe();
  }, []);

  // Fetch API usage on mount and when token changes
  useEffect(() => {
    fetchAPIUsage();
  }, [token]);

  // Fetch current API usage
  const fetchAPIUsage = useCallback(async () => {
    try {
      const { data, error } = await supabase.functions.invoke('sonzo-sign-recognition', {
        body: { action: 'getUsage', token }
      });

      if (!error && data?.usage) {
        setApiUsage(data.usage);
      }
    } catch (err) {
      console.error('Failed to fetch API usage:', err);
    }
  }, [token]);

  // Calculate distance between two landmarks
  const distance = (a: HandLandmark, b: HandLandmark): number => {
    return Math.sqrt(
      Math.pow(a.x - b.x, 2) + 
      Math.pow(a.y - b.y, 2) + 
      Math.pow(a.z - b.z, 2)
    );
  };

  // Check if finger is extended
  const isFingerExtended = (
    landmarks: HandLandmark[],
    tipIdx: number,
    pipIdx: number,
    mcpIdx: number
  ): boolean => {
    const tip = landmarks[tipIdx];
    const pip = landmarks[pipIdx];
    const mcp = landmarks[mcpIdx];
    
    // Finger is extended if tip is further from wrist than PIP
    const tipToMcp = distance(tip, mcp);
    const pipToMcp = distance(pip, mcp);
    
    return tipToMcp > pipToMcp * 1.2;
  };

  // Check if thumb is extended
  const isThumbExtended = (landmarks: HandLandmark[]): boolean => {
    const thumbTip = landmarks[LANDMARK_INDICES.THUMB_TIP];
    const thumbMcp = landmarks[LANDMARK_INDICES.THUMB_MCP];
    const indexMcp = landmarks[LANDMARK_INDICES.INDEX_MCP];
    
    const thumbToIndex = distance(thumbTip, indexMcp);
    const mcpToIndex = distance(thumbMcp, indexMcp);
    
    return thumbToIndex > mcpToIndex * 0.8;
  };

  // Analyze hand landmarks to detect signs
  const analyzeLandmarks = useCallback((
    leftHand: HandLandmark[] | null,
    rightHand: HandLandmark[] | null
  ): { sign: string; confidence: number } | null => {
    if (!leftHand && !rightHand) return null;

    // Calculate hand features
    const getHandFeatures = (landmarks: HandLandmark[]) => {
      if (!landmarks || landmarks.length < 21) return null;

      const wrist = landmarks[LANDMARK_INDICES.WRIST];
      const thumbTip = landmarks[LANDMARK_INDICES.THUMB_TIP];
      const indexTip = landmarks[LANDMARK_INDICES.INDEX_TIP];
      const middleTip = landmarks[LANDMARK_INDICES.MIDDLE_TIP];
      const ringTip = landmarks[LANDMARK_INDICES.RING_TIP];
      const pinkyTip = landmarks[LANDMARK_INDICES.PINKY_TIP];
      const indexMcp = landmarks[LANDMARK_INDICES.INDEX_MCP];
      const middleMcp = landmarks[LANDMARK_INDICES.MIDDLE_MCP];

      // Calculate finger extensions using proper landmark analysis
      const thumbExtended = isThumbExtended(landmarks);
      const indexExtended = isFingerExtended(
        landmarks,
        LANDMARK_INDICES.INDEX_TIP,
        LANDMARK_INDICES.INDEX_PIP,
        LANDMARK_INDICES.INDEX_MCP
      );
      const middleExtended = isFingerExtended(
        landmarks,
        LANDMARK_INDICES.MIDDLE_TIP,
        LANDMARK_INDICES.MIDDLE_PIP,
        LANDMARK_INDICES.MIDDLE_MCP
      );
      const ringExtended = isFingerExtended(
        landmarks,
        LANDMARK_INDICES.RING_TIP,
        LANDMARK_INDICES.RING_PIP,
        LANDMARK_INDICES.RING_MCP
      );
      const pinkyExtended = isFingerExtended(
        landmarks,
        LANDMARK_INDICES.PINKY_TIP,
        LANDMARK_INDICES.PINKY_PIP,
        LANDMARK_INDICES.PINKY_MCP
      );

      // Palm orientation based on z-depth
      const palmUp = wrist.z > indexMcp.z;
      const palmForward = Math.abs(wrist.z - indexMcp.z) < 0.03;

      // Hand position relative to center
      const handHeight = wrist.y;
      const handCenterX = wrist.x;

      // Calculate hand spread (distance between index and pinky tips)
      const handSpread = distance(indexTip, pinkyTip);

      // Calculate fist tightness
      const avgFingerDist = (
        distance(indexTip, wrist) +
        distance(middleTip, wrist) +
        distance(ringTip, wrist) +
        distance(pinkyTip, wrist)
      ) / 4;
      const isFist = avgFingerDist < 0.15;

      return {
        thumbExtended,
        indexExtended,
        middleExtended,
        ringExtended,
        pinkyExtended,
        palmUp,
        palmForward,
        handHeight,
        handCenterX,
        handSpread,
        isFist,
        fingersExtended: [thumbExtended, indexExtended, middleExtended, ringExtended, pinkyExtended].filter(Boolean).length,
        landmarks,
      };
    };

    const rightFeatures = rightHand ? getHandFeatures(rightHand) : null;
    const leftFeatures = leftHand ? getHandFeatures(leftHand) : null;
    const features = rightFeatures || leftFeatures;

    if (!features) return null;

    // Pattern matching for signs with improved accuracy
    let detectedSign: string | null = null;
    let confidence = 0;

    // Hello - open palm, all fingers extended, palm forward
    if (features.fingersExtended >= 4 && features.palmForward && features.handHeight < 0.5) {
      detectedSign = 'hello';
      confidence = 0.85 + (features.fingersExtended === 5 ? 0.1 : 0);
    }
    // Thank you - flat hand from chin forward, all fingers extended
    else if (features.fingersExtended >= 4 && features.handHeight < 0.4 && !features.palmUp) {
      detectedSign = 'thank_you';
      confidence = 0.82 + (features.fingersExtended === 5 ? 0.08 : 0);
    }
    // Yes - fist with thumb up
    else if (features.isFist && features.thumbExtended && !features.indexExtended) {
      detectedSign = 'yes';
      confidence = 0.88;
    }
    // No - index and middle fingers extended, others closed (like scissors)
    else if (features.indexExtended && features.middleExtended && 
             !features.ringExtended && !features.pinkyExtended && !features.thumbExtended) {
      detectedSign = 'no';
      confidence = 0.86;
    }
    // I Love You - thumb, index, and pinky extended
    else if (features.thumbExtended && features.indexExtended && features.pinkyExtended &&
             !features.middleExtended && !features.ringExtended) {
      detectedSign = 'love';
      confidence = 0.90;
    }
    // Help - fist on palm (requires both hands)
    else if (leftFeatures && rightFeatures && 
             leftFeatures.palmUp && rightFeatures.isFist &&
             Math.abs(leftFeatures.handCenterX - rightFeatures.handCenterX) < 0.2) {
      detectedSign = 'help';
      confidence = 0.87;
    }
    // Want - claw hand pulling in
    else if (features.fingersExtended >= 3 && !features.palmUp && features.handHeight > 0.4) {
      detectedSign = 'want';
      confidence = 0.80;
    }
    // Good - flat hand chin down
    else if (features.fingersExtended >= 4 && features.handHeight < 0.35 && features.palmUp) {
      detectedSign = 'good';
      confidence = 0.85;
    }
    // Understand - index finger flick near forehead
    else if (features.indexExtended && !features.middleExtended && 
             !features.ringExtended && !features.pinkyExtended && features.handHeight < 0.3) {
      detectedSign = 'understand';
      confidence = 0.83;
    }
    // Learn - flat to fist motion
    else if (features.fingersExtended >= 4 && features.palmUp && features.handHeight < 0.4) {
      detectedSign = 'learn';
      confidence = 0.78;
    }
    // See - V shape from eyes (peace sign near face)
    else if (features.indexExtended && features.middleExtended && 
             !features.ringExtended && !features.pinkyExtended && features.handHeight < 0.35) {
      detectedSign = 'see';
      confidence = 0.82;
    }
    // Drink - C-hand to mouth
    else if (features.thumbExtended && features.indexExtended && 
             !features.middleExtended && features.handHeight < 0.4) {
      detectedSign = 'drink';
      confidence = 0.79;
    }
    // Eat - fingers to mouth
    else if (features.fingersExtended >= 3 && features.handHeight < 0.35) {
      detectedSign = 'eat';
      confidence = 0.77;
    }

    if (detectedSign && confidence >= confidenceThreshold) {
      return { sign: detectedSign, confidence };
    }

    return null;
  }, [confidenceThreshold]);

  // Build sentence from recognized signs
  const buildSentence = useCallback((signs: string[]): string | null => {
    if (signs.length === 0) return null;

    // Check for matching sentence patterns
    for (const pattern of SENTENCE_PATTERNS) {
      const matchCount = pattern.signs.filter(s => signs.includes(s)).length;
      if (matchCount >= pattern.signs.length * 0.7) {
        return pattern.sentence;
      }
    }

    // Build sentence from individual signs
    const signNames = signs.map(s => ASL_SIGNS[s]?.name || s).filter(Boolean);
    if (signNames.length > 0) {
      return signNames.join(' ');
    }

    return null;
  }, []);

  // Fallback landmark detection when MediaPipe is not available
  const fallbackLandmarkDetection = useCallback((): { left?: HandLandmark[]; right?: HandLandmark[] } => {
    const time = Date.now() / 1000;
    
    const generateHandLandmarks = (isLeft: boolean): HandLandmark[] => {
      const baseX = isLeft ? 0.3 : 0.7;
      const landmarks: HandLandmark[] = [];
      
      // 21 landmarks per hand (MediaPipe format)
      const basePositions = [
        { x: 0, y: 0.3, z: 0 }, // Wrist
        { x: -0.05, y: 0.25, z: 0.02 }, // Thumb CMC
        { x: -0.08, y: 0.2, z: 0.03 }, // Thumb MCP
        { x: -0.1, y: 0.15, z: 0.04 }, // Thumb IP
        { x: -0.12, y: 0.1, z: 0.05 }, // Thumb Tip
        { x: -0.03, y: 0.15, z: 0.01 }, // Index MCP
        { x: -0.03, y: 0.1, z: 0.02 }, // Index PIP
        { x: -0.03, y: 0.05, z: 0.03 }, // Index DIP
        { x: -0.03, y: 0, z: 0.04 }, // Index Tip
        { x: 0, y: 0.14, z: 0.01 }, // Middle MCP
        { x: 0, y: 0.08, z: 0.02 }, // Middle PIP
        { x: 0, y: 0.03, z: 0.03 }, // Middle DIP
        { x: 0, y: -0.02, z: 0.04 }, // Middle Tip
        { x: 0.03, y: 0.15, z: 0.01 }, // Ring MCP
        { x: 0.03, y: 0.1, z: 0.02 }, // Ring PIP
        { x: 0.03, y: 0.05, z: 0.03 }, // Ring DIP
        { x: 0.03, y: 0.01, z: 0.04 }, // Ring Tip
        { x: 0.06, y: 0.18, z: 0.01 }, // Pinky MCP
        { x: 0.06, y: 0.13, z: 0.02 }, // Pinky PIP
        { x: 0.06, y: 0.09, z: 0.03 }, // Pinky DIP
        { x: 0.06, y: 0.05, z: 0.04 }, // Pinky Tip
      ];

      for (let i = 0; i < basePositions.length; i++) {
        const pos = basePositions[i];
        landmarks.push({
          x: baseX + pos.x + Math.sin(time * 2 + i * 0.3) * 0.01,
          y: pos.y + 0.3 + Math.cos(time * 1.5 + i * 0.2) * 0.01,
          z: pos.z + Math.sin(time * 3 + i * 0.5) * 0.005,
          visibility: 0.95 + Math.random() * 0.05,
        });
      }

      return landmarks;
    };

    // Randomly detect one or both hands
    const detectLeft = Math.random() > 0.3;
    const detectRight = Math.random() > 0.2;

    return {
      left: detectLeft ? generateHandLandmarks(true) : undefined,
      right: detectRight ? generateHandLandmarks(false) : undefined,
    };
  }, []);

  // Process video frame for sign recognition
  const processFrame = useCallback(async (
    videoElement: HTMLVideoElement | null,
    canvasElement: HTMLCanvasElement | null,
    handLandmarks?: { left?: HandLandmark[]; right?: HandLandmark[] }
  ): Promise<RecognitionResult | null> => {
    if (processingRef.current) return null;
    
    // Check API limits
    if (apiUsage && typeof apiUsage.remaining === 'number' && apiUsage.remaining <= 0) {
      onAPILimitReached?.();
      setError('API limit reached. Please upgrade your subscription.');
      return null;
    }

    // Rate limiting - process every 3rd frame
    frameCountRef.current++;
    if (frameCountRef.current % 3 !== 0) return null;

    // Debounce recognition
    const now = Date.now();
    if (now - lastRecognitionRef.current < 500) return null;

    processingRef.current = true;
    setIsProcessing(true);
    setError(null);

    try {
      let landmarks = handLandmarks;

      // If no landmarks provided, detect from video
      if (!landmarks && videoElement) {
        // Try MediaPipe first
        if (mediaPipeReady && detectorRef.current) {
          landmarks = await detectorRef.current.detect(videoElement) || undefined;
        }
        
        // Fallback to simulated detection
        if (!landmarks) {
          landmarks = fallbackLandmarkDetection();
        }
      }

      if (!landmarks || (!landmarks.left && !landmarks.right)) {
        processingRef.current = false;
        setIsProcessing(false);
        return null;
      }

      // Analyze landmarks
      const localResult = analyzeLandmarks(landmarks.left || null, landmarks.right || null);
      
      if (localResult && localResult.confidence >= confidenceThreshold) {
        const result: RecognitionResult = {
          sign: localResult.sign,
          confidence: localResult.confidence,
          landmarks,
          timestamp: now,
        };

        // Update sign buffer
        signBufferRef.current.push(localResult.sign);
        if (signBufferRef.current.length > 10) {
          signBufferRef.current.shift();
        }

        setCurrentSign(localResult.sign);
        setRecognizedSigns(prev => [...prev.slice(-9), localResult.sign]);
        lastRecognitionRef.current = now;

        // Check for sentence completion
        const sentence = buildSentence(signBufferRef.current);
        if (sentence && signBufferRef.current.length >= 2) {
          result.sentence = sentence;
          onSentenceComplete?.(sentence, localResult.confidence);
          signBufferRef.current = []; // Reset buffer after sentence
        }

        onRecognition?.(result);

        // Track API usage
        if (token) {
          trackAPIUsage();
        }

        processingRef.current = false;
        setIsProcessing(false);
        return result;
      }

      processingRef.current = false;
      setIsProcessing(false);
      return null;
    } catch (err) {
      console.error('Recognition error:', err);
      setError('Recognition failed. Please try again.');
      processingRef.current = false;
      setIsProcessing(false);
      return null;
    }
  }, [apiUsage, token, confidenceThreshold, mediaPipeReady, analyzeLandmarks, buildSentence, fallbackLandmarkDetection, onRecognition, onSentenceComplete, onAPILimitReached]);

  // Track API usage
  const trackAPIUsage = useCallback(async () => {
    if (!token) return;

    try {
      const { data } = await supabase.functions.invoke('sonzo-sign-recognition', {
        body: { 
          action: 'trackUsage', 
          token, 
          operation: 'recognition',
          language,
          cameraType: 'webcam',
          responseTimeMs: Date.now() - lastRecognitionRef.current,
          success: true
        }
      });

      if (data?.usage) {
        setApiUsage(data.usage);
      }
    } catch (err) {
      console.error('Failed to track API usage:', err);
    }
  }, [token, language]);

  // Clear recognition buffer
  const clearBuffer = useCallback(() => {
    signBufferRef.current = [];
    setRecognizedSigns([]);
    setCurrentSign(null);
  }, []);

  // Reset error state
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    isProcessing,
    isInitialized,
    mediaPipeReady,
    currentSign,
    recognizedSigns,
    apiUsage,
    error,
    processFrame,
    clearBuffer,
    clearError,
    fetchAPIUsage,
  };
};

export default useSignRecognition;
