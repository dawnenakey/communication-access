import React, { useRef, useEffect, useState, Suspense, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { 
  OrbitControls, 
  Environment, 
  ContactShadows,
  Html,
  PerspectiveCamera,
  useGLTF,
  useAnimations
} from '@react-three/drei';
import * as THREE from 'three';
import { 
  Volume2, VolumeX, Maximize2, Minimize2, 
  Palette, Sparkles, User, Link2, Loader2,
  Play, Pause, SkipForward, Settings2, Mic,
  ShoppingBag, BookOpen, MessageSquare
} from 'lucide-react';
import { supabase } from '@/lib/supabase';
import { 
  SIGN_DATABASE, 
  SignData, 
  SignKeyframe,
  getSign,
  getSignNames,
  getCategories,
  getSignsByCategory
} from '@/data/SignDictionary';
import AvatarMarketplace from './AvatarMarketplace';
import SignLearningMode from '@/pages/SignLearningMode';

// ============================================
// TYPES
// ============================================

interface VoiceSettings {
  voice: 'female_warm' | 'female_young' | 'male_deep' | 'male_warm' | 'male_young' | 'female_professional';
  speed: number;
  pitch: number;
  enabled: boolean;
}

interface AvatarCustomization {
  skinTone: string;
  hairColor: string;
  shirtColor: string;
  eyeColor: string;
  hairStyle: 'short' | 'medium' | 'long';
  gender: 'female' | 'male';
}

interface ReadyPlayerMeAvatarProps {
  currentSentence: string;
  isResponding: boolean;
  language: string;
  onResponseComplete?: () => void;
  recognizedSign?: string | null;
  avatarUrl?: string;
  onAvatarUrlChange?: (url: string) => void;
}

// ============================================
// REALISTIC HUMAN AVATAR (Kara-style)
// ============================================

const RealisticHumanAvatar: React.FC<{
  currentKeyframe: SignKeyframe | null;
  customization: AvatarCustomization;
  mouthOpenness: number;
  isSpeaking: boolean;
}> = ({ currentKeyframe, customization, mouthOpenness, isSpeaking }) => {
  const groupRef = useRef<THREE.Group>(null);
  const [blinkProgress, setBlinkProgress] = useState(0);
  const [breatheOffset, setBreatheOffset] = useState(0);
  
  // Smooth animation values
  const smoothLeftArm = useRef({ x: -0.18, y: 0.35, z: 0 });
  const smoothRightArm = useRef({ x: 0.18, y: 0.35, z: 0 });
  const smoothHeadTilt = useRef([0, 0, 0]);

  useFrame((state, delta) => {
    const time = state.clock.elapsedTime;
    
    // Natural blinking (every 3-5 seconds)
    const blinkCycle = (time % 4) / 4;
    if (blinkCycle > 0.97 || blinkCycle < 0.03) {
      setBlinkProgress(prev => Math.min(1, prev + delta * 12));
    } else {
      setBlinkProgress(prev => Math.max(0, prev - delta * 8));
    }
    
    // Subtle breathing animation
    setBreatheOffset(Math.sin(time * 1.5) * 0.003);

    // Smooth arm movements
    if (currentKeyframe) {
      const targetLeft = currentKeyframe.leftArm.position;
      const targetRight = currentKeyframe.rightArm.position;
      
      smoothLeftArm.current.x = THREE.MathUtils.lerp(smoothLeftArm.current.x, targetLeft[0], delta * 4);
      smoothLeftArm.current.y = THREE.MathUtils.lerp(smoothLeftArm.current.y, targetLeft[1], delta * 4);
      smoothLeftArm.current.z = THREE.MathUtils.lerp(smoothLeftArm.current.z, targetLeft[2], delta * 4);
      
      smoothRightArm.current.x = THREE.MathUtils.lerp(smoothRightArm.current.x, targetRight[0], delta * 4);
      smoothRightArm.current.y = THREE.MathUtils.lerp(smoothRightArm.current.y, targetRight[1], delta * 4);
      smoothRightArm.current.z = THREE.MathUtils.lerp(smoothRightArm.current.z, targetRight[2], delta * 4);
      
      smoothHeadTilt.current[0] = THREE.MathUtils.lerp(smoothHeadTilt.current[0], currentKeyframe.body.headTilt[0], delta * 5);
      smoothHeadTilt.current[1] = THREE.MathUtils.lerp(smoothHeadTilt.current[1], currentKeyframe.body.headTilt[1], delta * 5);
      smoothHeadTilt.current[2] = THREE.MathUtils.lerp(smoothHeadTilt.current[2], currentKeyframe.body.headTilt[2], delta * 5);
    }
  });

  const headTilt = currentKeyframe?.body.headTilt || [0, 0, 0];
  const eyebrowRaise = currentKeyframe?.face.eyebrowRaise || 0;
  const mouthSmile = currentKeyframe?.face.mouthSmile || 0;
  const torsoTwist = currentKeyframe?.body.torsoTwist || 0;
  const shoulderShrug = currentKeyframe?.body.shoulderShrug || 0;

  // Skin material
  const skinMaterial = (
    <meshStandardMaterial 
      color={customization.skinTone} 
      roughness={0.6} 
      metalness={0.1}
    />
  );

  // Hair material
  const hairMaterial = (
    <meshStandardMaterial 
      color={customization.hairColor} 
      roughness={0.8}
    />
  );

  // Shirt material - blue like Kara
  const shirtMaterial = (
    <meshStandardMaterial 
      color={customization.shirtColor} 
      roughness={0.7}
    />
  );

  // Calculate arm IK positions
  const calculateArmRotation = (targetPos: { x: number; y: number; z: number }, isLeft: boolean) => {
    const shoulderY = 0.52;
    const upperArmLength = 0.12;
    const forearmLength = 0.11;
    
    const dx = targetPos.x - (isLeft ? -0.14 : 0.14);
    const dy = targetPos.y - shoulderY;
    const dz = targetPos.z;
    
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const maxReach = upperArmLength + forearmLength;
    const clampedDistance = Math.min(distance, maxReach * 0.95);
    
    // Simple IK approximation
    const shoulderAngle = Math.atan2(dz, Math.sqrt(dx * dx + dy * dy));
    const shoulderRotY = Math.atan2(dx, -dy);
    const elbowAngle = Math.acos(Math.min(1, clampedDistance / maxReach)) * 1.5;
    
    return { shoulderAngle, shoulderRotY, elbowAngle };
  };

  const leftArmIK = calculateArmRotation(smoothLeftArm.current, true);
  const rightArmIK = calculateArmRotation(smoothRightArm.current, false);

  // Hand component with fingers
  const Hand: React.FC<{ isLeft: boolean; curl: number }> = ({ isLeft, curl }) => {
    const fingerCurl = currentKeyframe 
      ? (isLeft ? currentKeyframe.leftArm.hand : currentKeyframe.rightArm.hand)
      : { thumb: 0.2, index: 0.2, middle: 0.2, ring: 0.2, pinky: 0.2, spread: 0.1 };
    
    return (
      <group rotation={[0, 0, isLeft ? 0.1 : -0.1]}>
        {/* Palm */}
        <mesh scale={[0.035, 0.045, 0.02]}>
          <boxGeometry args={[1, 1, 1]} />
          {skinMaterial}
        </mesh>
        
        {/* Fingers */}
        {[
          { x: -0.012, name: 'index', curl: fingerCurl.index },
          { x: -0.004, name: 'middle', curl: fingerCurl.middle },
          { x: 0.004, name: 'ring', curl: fingerCurl.ring },
          { x: 0.012, name: 'pinky', curl: fingerCurl.pinky }
        ].map((finger, i) => (
          <group key={finger.name} position={[finger.x, -0.035, 0]} rotation={[finger.curl * 1.2, 0, 0]}>
            {/* Proximal */}
            <mesh position={[0, -0.012, 0]}>
              <capsuleGeometry args={[0.005, 0.02, 4, 8]} />
              {skinMaterial}
            </mesh>
            {/* Middle */}
            <group position={[0, -0.028, 0]} rotation={[finger.curl * 0.8, 0, 0]}>
              <mesh position={[0, -0.008, 0]}>
                <capsuleGeometry args={[0.004, 0.014, 4, 8]} />
                {skinMaterial}
              </mesh>
              {/* Distal */}
              <group position={[0, -0.02, 0]} rotation={[finger.curl * 0.6, 0, 0]}>
                <mesh position={[0, -0.006, 0]}>
                  <capsuleGeometry args={[0.0035, 0.01, 4, 8]} />
                  {skinMaterial}
                </mesh>
              </group>
            </group>
          </group>
        ))}
        
        {/* Thumb */}
        <group position={[isLeft ? 0.022 : -0.022, -0.01, 0.008]} rotation={[0, isLeft ? -0.5 : 0.5, fingerCurl.thumb * 0.8]}>
          <mesh position={[0, -0.01, 0]}>
            <capsuleGeometry args={[0.006, 0.016, 4, 8]} />
            {skinMaterial}
          </mesh>
          <group position={[0, -0.022, 0]} rotation={[fingerCurl.thumb * 0.6, 0, 0]}>
            <mesh position={[0, -0.008, 0]}>
              <capsuleGeometry args={[0.005, 0.012, 4, 8]} />
              {skinMaterial}
            </mesh>
          </group>
        </group>
      </group>
    );
  };

  // Arm component with IK
  const Arm: React.FC<{ isLeft: boolean }> = ({ isLeft }) => {
    const ik = isLeft ? leftArmIK : rightArmIK;
    const xOffset = isLeft ? -0.14 : 0.14;
    
    return (
      <group position={[xOffset, 0.52 + shoulderShrug * 0.02, 0]}>
        {/* Shoulder joint */}
        <group rotation={[ik.shoulderAngle, ik.shoulderRotY * (isLeft ? 1 : -1), isLeft ? 0.2 : -0.2]}>
          {/* Upper arm */}
          <mesh position={[0, -0.06, 0]}>
            <capsuleGeometry args={[0.032, 0.1, 8, 16]} />
            {shirtMaterial}
          </mesh>
          
          {/* Elbow joint */}
          <group position={[0, -0.12, 0]} rotation={[ik.elbowAngle, 0, 0]}>
            {/* Forearm */}
            <mesh position={[0, -0.055, 0]}>
              <capsuleGeometry args={[0.026, 0.09, 8, 16]} />
              {skinMaterial}
            </mesh>
            
            {/* Wrist and hand */}
            <group position={[0, -0.12, 0]}>
              <Hand isLeft={isLeft} curl={0.2} />
            </group>
          </group>
        </group>
      </group>
    );
  };

  return (
    <group ref={groupRef} position={[0, -0.35, 0]}>
      {/* Body/Torso */}
      <group position={[0, 0.35, 0]} rotation={[0, torsoTwist, 0]}>
        {/* Main torso - blue shirt */}
        <mesh position={[0, 0, 0]}>
          <capsuleGeometry args={[0.11, 0.2, 12, 24]} />
          {shirtMaterial}
        </mesh>
        
        {/* Collar/neckline */}
        <mesh position={[0, 0.14, 0.02]} rotation={[0.3, 0, 0]}>
          <torusGeometry args={[0.045, 0.012, 8, 16, Math.PI]} />
          {shirtMaterial}
        </mesh>
        
        {/* Shoulders */}
        <mesh position={[-0.12, 0.08, 0]} rotation={[0, 0, 0.3]}>
          <capsuleGeometry args={[0.04, 0.06, 8, 16]} />
          {shirtMaterial}
        </mesh>
        <mesh position={[0.12, 0.08, 0]} rotation={[0, 0, -0.3]}>
          <capsuleGeometry args={[0.04, 0.06, 8, 16]} />
          {shirtMaterial}
        </mesh>

        {/* Arms */}
        <Arm isLeft={true} />
        <Arm isLeft={false} />
      </group>

      {/* Neck */}
      <mesh position={[0, 0.58 + breatheOffset, 0]}>
        <cylinderGeometry args={[0.032, 0.038, 0.08, 16]} />
        {skinMaterial}
      </mesh>

      {/* Head */}
      <group 
        position={[0, 0.72 + breatheOffset, 0]} 
        rotation={[
          smoothHeadTilt.current[0], 
          smoothHeadTilt.current[1], 
          smoothHeadTilt.current[2]
        ]}
      >
        {/* Head shape - more realistic oval */}
        <mesh scale={[0.9, 1.05, 0.95]}>
          <sphereGeometry args={[0.1, 32, 32]} />
          {skinMaterial}
        </mesh>
        
        {/* Jaw/chin definition */}
        <mesh position={[0, -0.06, 0.03]} scale={[0.7, 0.5, 0.6]}>
          <sphereGeometry args={[0.08, 16, 16]} />
          {skinMaterial}
        </mesh>

        {/* Hair - styled based on gender */}
        {customization.gender === 'female' ? (
          <>
            {/* Female hair - medium length like Kara */}
            <mesh position={[0, 0.03, -0.01]} scale={[1.05, 1, 1]}>
              <sphereGeometry args={[0.1, 32, 32, 0, Math.PI * 2, 0, Math.PI * 0.6]} />
              {hairMaterial}
            </mesh>
            {/* Side hair */}
            <mesh position={[-0.08, -0.02, 0]} scale={[0.4, 0.8, 0.5]}>
              <sphereGeometry args={[0.1, 16, 16]} />
              {hairMaterial}
            </mesh>
            <mesh position={[0.08, -0.02, 0]} scale={[0.4, 0.8, 0.5]}>
              <sphereGeometry args={[0.1, 16, 16]} />
              {hairMaterial}
            </mesh>
            {/* Back hair */}
            <mesh position={[0, -0.04, -0.06]} scale={[0.8, 1, 0.4]}>
              <sphereGeometry args={[0.1, 16, 16]} />
              {hairMaterial}
            </mesh>
          </>
        ) : (
          <>
            {/* Male hair - short */}
            <mesh position={[0, 0.04, 0]}>
              <sphereGeometry args={[0.098, 32, 32, 0, Math.PI * 2, 0, Math.PI * 0.5]} />
              {hairMaterial}
            </mesh>
          </>
        )}

        {/* Ears */}
        {[-0.095, 0.095].map((x, i) => (
          <mesh key={i} position={[x, 0, -0.01]} rotation={[0, x > 0 ? -0.2 : 0.2, 0]}>
            <sphereGeometry args={[0.018, 8, 8]} />
            {skinMaterial}
          </mesh>
        ))}

        {/* Eyes */}
        {[-0.032, 0.032].map((x, i) => (
          <group key={i} position={[x, 0.015, 0.08]}>
            {/* Eye socket shadow */}
            <mesh position={[0, 0, -0.005]} scale={[1.3, 1.1, 0.5]}>
              <sphereGeometry args={[0.016, 12, 12]} />
              <meshStandardMaterial color="#d4a574" roughness={0.8} />
            </mesh>
            
            {/* Eyeball */}
            <mesh scale={[1, 0.75 * (1 - blinkProgress * 0.9), 1]}>
              <sphereGeometry args={[0.016, 16, 16]} />
              <meshStandardMaterial color="#fefefe" roughness={0.3} />
            </mesh>
            
            {/* Iris */}
            <mesh position={[0, 0, 0.01]} scale={[1, 1 - blinkProgress * 0.9, 1]}>
              <sphereGeometry args={[0.008, 12, 12]} />
              <meshStandardMaterial color={customization.eyeColor} roughness={0.4} />
            </mesh>
            
            {/* Pupil */}
            <mesh position={[0, 0, 0.014]} scale={[1, 1 - blinkProgress * 0.9, 1]}>
              <sphereGeometry args={[0.004, 8, 8]} />
              <meshStandardMaterial color="#1a1a1a" roughness={0.2} />
            </mesh>
            
            {/* Eye highlight */}
            <mesh position={[0.003, 0.003, 0.015]} scale={[1, 1 - blinkProgress * 0.9, 1]}>
              <sphereGeometry args={[0.002, 6, 6]} />
              <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.5} />
            </mesh>
            
            {/* Eyelid */}
            <mesh 
              position={[0, 0.008 - blinkProgress * 0.012, 0.005]} 
              scale={[1.1, 0.3 + blinkProgress * 0.5, 0.8]}
            >
              <sphereGeometry args={[0.016, 12, 12, 0, Math.PI * 2, 0, Math.PI * 0.5]} />
              {skinMaterial}
            </mesh>
          </group>
        ))}

        {/* Eyebrows */}
        {[-0.035, 0.035].map((x, i) => (
          <mesh 
            key={`brow-${i}`}
            position={[x, 0.045 + eyebrowRaise * 0.008, 0.085]} 
            rotation={[0.1, 0, (i === 0 ? 0.12 : -0.12) - eyebrowRaise * 0.08]}
            scale={[1, 0.4, 0.6]}
          >
            <capsuleGeometry args={[0.008, 0.018, 4, 8]} />
            {hairMaterial}
          </mesh>
        ))}

        {/* Nose */}
        <group position={[0, -0.015, 0.09]}>
          {/* Bridge */}
          <mesh position={[0, 0.015, 0]} rotation={[-0.2, 0, 0]} scale={[0.6, 1, 0.8]}>
            <capsuleGeometry args={[0.008, 0.025, 4, 8]} />
            {skinMaterial}
          </mesh>
          {/* Tip */}
          <mesh position={[0, -0.01, 0.008]} scale={[1, 0.7, 0.8]}>
            <sphereGeometry args={[0.012, 12, 12]} />
            {skinMaterial}
          </mesh>
          {/* Nostrils */}
          {[-0.008, 0.008].map((x, i) => (
            <mesh key={i} position={[x, -0.015, 0]} scale={[0.6, 0.4, 0.5]}>
              <sphereGeometry args={[0.008, 8, 8]} />
              <meshStandardMaterial color="#c9908a" roughness={0.8} />
            </mesh>
          ))}
        </group>

        {/* Mouth */}
        <group position={[0, -0.055, 0.075]}>
          {/* Upper lip */}
          <mesh position={[0, 0.004 + mouthSmile * 0.003, 0]} scale={[1, 0.5, 0.6]}>
            <capsuleGeometry args={[0.018, 0.008, 4, 12]} />
            <meshStandardMaterial color="#c9706a" roughness={0.6} />
          </mesh>
          
          {/* Lower lip */}
          <mesh 
            position={[0, -0.008 - mouthOpenness * 0.015, 0]} 
            scale={[0.9, 0.55 + mouthOpenness * 0.2, 0.65]}
          >
            <capsuleGeometry args={[0.016, 0.01, 4, 12]} />
            <meshStandardMaterial color="#d4807a" roughness={0.6} />
          </mesh>
          
          {/* Mouth interior (visible when open) */}
          {mouthOpenness > 0.1 && (
            <mesh position={[0, -0.004, -0.008]} scale={[0.7, mouthOpenness * 0.8, 0.5]}>
              <sphereGeometry args={[0.015, 12, 12]} />
              <meshStandardMaterial color="#4a2020" roughness={0.9} />
            </mesh>
          )}
          
          {/* Teeth hint */}
          {mouthOpenness > 0.2 && (
            <mesh position={[0, 0, -0.005]} scale={[0.6, 0.15, 0.3]}>
              <boxGeometry args={[0.025, 0.01, 0.01]} />
              <meshStandardMaterial color="#f5f5f0" roughness={0.3} />
            </mesh>
          )}
        </group>
      </group>
    </group>
  );
};

// ============================================
// AVATAR MODEL LOADER (for GLTF/GLB)
// ============================================

const AvatarModel: React.FC<{
  url: string;
  currentKeyframe: SignKeyframe | null;
  mouthOpenness: number;
}> = ({ url, currentKeyframe, mouthOpenness }) => {
  const group = useRef<THREE.Group>(null);
  const { scene } = useGLTF(url);
  const clonedScene = React.useMemo(() => scene.clone(), [scene]);

  useEffect(() => {
    clonedScene.traverse((child) => {
      if (child instanceof THREE.SkinnedMesh && child.morphTargetInfluences && child.morphTargetDictionary) {
        const mouthOpenIndex = child.morphTargetDictionary['mouthOpen'] ?? 
                              child.morphTargetDictionary['jawOpen'] ??
                              child.morphTargetDictionary['viseme_aa'];
        if (mouthOpenIndex !== undefined) {
          child.morphTargetInfluences[mouthOpenIndex] = mouthOpenness;
        }
      }
    });
  }, [clonedScene, mouthOpenness]);

  return (
    <group ref={group} position={[0, -0.8, 0]} scale={[1, 1, 1]}>
      <primitive object={clonedScene} />
    </group>
  );
};

// ============================================
// MAIN AVATAR SCENE
// ============================================

const AvatarScene: React.FC<{
  avatarUrl: string | null;
  currentKeyframe: SignKeyframe | null;
  customization: AvatarCustomization;
  mouthOpenness: number;
  isSpeaking: boolean;
}> = ({ avatarUrl, currentKeyframe, customization, mouthOpenness, isSpeaking }) => {
  return (
    <>
      <PerspectiveCamera makeDefault position={[0, 0.5, 1.2]} fov={45} />
      <OrbitControls 
        enablePan={false} 
        enableZoom={true}
        minDistance={0.6}
        maxDistance={2}
        minPolarAngle={Math.PI / 4}
        maxPolarAngle={Math.PI / 1.8}
        target={[0, 0.35, 0]}
      />
      
      <ambientLight intensity={0.5} />
      <directionalLight position={[2, 4, 3]} intensity={0.8} castShadow />
      <directionalLight position={[-2, 2, 3]} intensity={0.4} />
      <directionalLight position={[0, 1, -2]} intensity={0.2} />
      
      <Environment preset="studio" />
      
      {avatarUrl ? (
        <Suspense fallback={
          <Html center>
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="w-8 h-8 animate-spin text-primary" />
              <p className="text-sm text-muted-foreground">Loading avatar...</p>
            </div>
          </Html>
        }>
          <AvatarModel 
            url={avatarUrl} 
            currentKeyframe={currentKeyframe}
            mouthOpenness={mouthOpenness}
          />
        </Suspense>
      ) : (
        <RealisticHumanAvatar 
          currentKeyframe={currentKeyframe}
          customization={customization}
          mouthOpenness={mouthOpenness}
          isSpeaking={isSpeaking}
        />
      )}
      
      <ContactShadows 
        position={[0, -0.35, 0]} 
        opacity={0.4} 
        scale={1.2} 
        blur={2} 
      />
    </>
  );
};

// ============================================
// MAIN COMPONENT
// ============================================

const ReadyPlayerMeAvatar: React.FC<ReadyPlayerMeAvatarProps> = ({
  currentSentence,
  isResponding,
  language,
  onResponseComplete,
  recognizedSign,
  avatarUrl: externalAvatarUrl,
  onAvatarUrlChange,
}) => {
  // State
  const [avatarUrl, setAvatarUrl] = useState<string | null>(externalAvatarUrl || null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showCustomization, setShowCustomization] = useState(false);
  const [showMarketplace, setShowMarketplace] = useState(false);
  const [showLearningMode, setShowLearningMode] = useState(false);
  const [customization, setCustomization] = useState<AvatarCustomization>({
    skinTone: '#e0ac69',
    hairColor: '#2c1810',
    shirtColor: '#2563eb', // Blue like Kara
    eyeColor: '#4a3728',
    hairStyle: 'medium',
    gender: 'female'
  });
  const [voiceSettings, setVoiceSettings] = useState<VoiceSettings>({
    voice: 'female_warm',
    speed: 1,
    pitch: 0,
    enabled: true,
  });
  const [isLoadingVoice, setIsLoadingVoice] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [currentKeyframe, setCurrentKeyframe] = useState<SignKeyframe | null>(null);
  const [currentSignName, setCurrentSignName] = useState<string | null>(null);
  const [mouthOpenness, setMouthOpenness] = useState(0);
  const [wordIndex, setWordIndex] = useState(0);
  const [signingQueue, setSigningQueue] = useState<SignData[]>([]);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const animationRef = useRef<number | null>(null);

  // Color options
  const skinTones = ['#ffdfc4', '#f0c8a0', '#e0ac69', '#c68642', '#8d5524', '#5c3d2e'];
  const hairColors = ['#2c1810', '#4a3728', '#8b4513', '#d4a574', '#1a1a1a', '#c0c0c0'];
  const shirtColors = ['#2563eb', '#7c3aed', '#10b981', '#f59e0b', '#ef4444', '#1f2937'];
  const eyeColors = ['#4a3728', '#2d5a27', '#1e3a5f', '#4a4a4a', '#8b4513'];

  // Handle avatar selection from marketplace
  const handleSelectAvatar = useCallback((url: string) => {
    setAvatarUrl(url);
    onAvatarUrlChange?.(url);
  }, [onAvatarUrlChange]);

  // Generate speech with TTS
  const generateSpeech = useCallback(async (text: string) => {
    if (!voiceSettings.enabled || !text) return;

    setIsLoadingVoice(true);
    try {
      const response = await fetch(`${supabase.supabaseUrl}/functions/v1/avatar-tts`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${supabase.supabaseKey}`,
        },
        body: JSON.stringify({
          text,
          voice: voiceSettings.voice,
          speed: voiceSettings.speed,
          pitch: voiceSettings.pitch,
        }),
      });

      if (!response.ok) throw new Error('TTS failed');

      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      
      if (audioRef.current) {
        audioRef.current.pause();
        URL.revokeObjectURL(audioRef.current.src);
      }

      const audio = new Audio(audioUrl);
      audioRef.current = audio;
      
      audio.addEventListener('play', () => {
        setIsPlayingAudio(true);
        animateLipSync();
      });
      
      audio.addEventListener('ended', () => {
        setIsPlayingAudio(false);
        setMouthOpenness(0);
        if (animationRef.current) cancelAnimationFrame(animationRef.current);
      });

      audio.play();
    } catch (error) {
      console.error('TTS error:', error);
    } finally {
      setIsLoadingVoice(false);
    }
  }, [voiceSettings]);

  // Lip sync animation
  const animateLipSync = useCallback(() => {
    const animate = () => {
      if (!audioRef.current || audioRef.current.paused) return;
      const time = performance.now() / 1000;
      const openness = 0.3 + Math.sin(time * 8) * 0.2 + Math.sin(time * 12) * 0.1;
      setMouthOpenness(Math.max(0, Math.min(1, openness)));
      animationRef.current = requestAnimationFrame(animate);
    };
    animate();
  }, []);

  // Play sign animation
  const playSignAnimation = useCallback((sign: SignData): Promise<void> => {
    return new Promise((resolve) => {
      setCurrentSignName(sign.name);
      
      const keyframes = sign.keyframes;
      const duration = sign.duration;
      const startTime = performance.now();

      const animate = () => {
        const elapsed = performance.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);

        let currentIdx = 0;
        for (let i = 0; i < keyframes.length - 1; i++) {
          if (progress >= keyframes[i].time && progress < keyframes[i + 1].time) {
            currentIdx = i;
            break;
          }
        }
        if (progress >= keyframes[keyframes.length - 1].time) {
          currentIdx = keyframes.length - 1;
        }

        setCurrentKeyframe(keyframes[currentIdx]);
        setMouthOpenness(keyframes[currentIdx].face.mouthOpen);

        if (progress < 1) {
          requestAnimationFrame(animate);
        } else {
          setTimeout(() => {
            setCurrentKeyframe(null);
            setCurrentSignName(null);
            setMouthOpenness(0);
            resolve();
          }, 100);
        }
      };

      requestAnimationFrame(animate);
    });
  }, []);

  // Handle sign from learning mode
  const handlePlaySignFromLearning = useCallback((sign: SignData) => {
    playSignAnimation(sign);
    if (voiceSettings.enabled) {
      generateSpeech(sign.name);
    }
  }, [playSignAnimation, voiceSettings.enabled, generateSpeech]);

  // Parse sentence into signs and play sequentially
  const playSentence = useCallback(async (sentence: string) => {
    const words = sentence.toLowerCase().split(/\s+/).filter(w => w.length > 0);
    const signs: SignData[] = [];
    
    // Map words to signs
    for (const word of words) {
      const cleanWord = word.replace(/[^a-z]/g, '');
      const sign = getSign(cleanWord);
      if (sign) {
        signs.push(sign);
      }
    }

    // Generate speech for the whole sentence
    if (voiceSettings.enabled) {
      generateSpeech(sentence);
    }

    // Play signs sequentially
    for (let i = 0; i < signs.length; i++) {
      setWordIndex(i);
      await playSignAnimation(signs[i]);
      await new Promise(resolve => setTimeout(resolve, 150)); // Brief pause between signs
    }

    onResponseComplete?.();
  }, [voiceSettings.enabled, generateSpeech, playSignAnimation, onResponseComplete]);

  // Handle sentence signing
  useEffect(() => {
    if (isResponding && currentSentence) {
      playSentence(currentSentence);
    }
  }, [isResponding, currentSentence, playSentence]);

  // Handle recognized sign
  useEffect(() => {
    if (recognizedSign) {
      const sign = getSign(recognizedSign);
      if (sign) {
        playSignAnimation(sign);
      }
    }
  }, [recognizedSign, playSignAnimation]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        URL.revokeObjectURL(audioRef.current.src);
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  const words = currentSentence?.split(' ').filter(w => w.length > 0) || [];

  return (
    <>
      <div className={`relative bg-card rounded-2xl border border-border overflow-hidden transition-all duration-300 ${
        isFullscreen ? 'fixed inset-4 z-50' : ''
      }`}>
        {/* Header */}
        <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between p-3 bg-gradient-to-b from-black/60 to-transparent">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <div>
              <p className="text-xs font-semibold text-white">SonZo 3D Avatar</p>
              <p className="text-[10px] text-white/60">{language} • {Object.keys(SIGN_DATABASE).length} signs</p>
            </div>
          </div>

          <div className="flex items-center gap-1">
            <button
              onClick={() => setShowLearningMode(true)}
              className="p-2 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 transition-all"
              title="Learn Signs"
            >
              <BookOpen className="w-4 h-4" />
            </button>
            <button
              onClick={() => setShowMarketplace(true)}
              className="p-2 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 transition-all"
              title="Avatar Marketplace"
            >
              <ShoppingBag className="w-4 h-4" />
            </button>
            <button
              onClick={() => setShowCustomization(!showCustomization)}
              className={`p-2 rounded-lg transition-all ${
                showCustomization 
                  ? 'bg-primary/30 text-primary' 
                  : 'bg-white/10 text-white/70 hover:bg-white/20'
              }`}
              title="Customize"
            >
              <Palette className="w-4 h-4" />
            </button>
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-2 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 transition-all"
            >
              {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Customization Panel */}
        {showCustomization && !avatarUrl && (
          <div className="absolute top-14 right-3 z-20 w-56 p-3 rounded-xl bg-card/95 backdrop-blur-xl border border-border shadow-2xl">
            <h4 className="text-xs font-bold mb-3 flex items-center gap-2">
              <Palette className="w-3 h-3 text-primary" />
              Customize Avatar
            </h4>
            
            {/* Gender */}
            <div className="mb-3">
              <p className="text-[10px] font-medium text-muted-foreground mb-1.5">Gender</p>
              <div className="flex gap-2">
                {(['female', 'male'] as const).map(g => (
                  <button
                    key={g}
                    onClick={() => setCustomization(prev => ({ ...prev, gender: g }))}
                    className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      customization.gender === g 
                        ? 'bg-primary text-primary-foreground' 
                        : 'bg-muted hover:bg-muted/80'
                    }`}
                  >
                    {g.charAt(0).toUpperCase() + g.slice(1)}
                  </button>
                ))}
              </div>
            </div>
            
            {/* Skin Tone */}
            <div className="mb-3">
              <p className="text-[10px] font-medium text-muted-foreground mb-1.5">Skin Tone</p>
              <div className="flex flex-wrap gap-1.5">
                {skinTones.map(tone => (
                  <button
                    key={tone}
                    onClick={() => setCustomization(prev => ({ ...prev, skinTone: tone }))}
                    className={`w-6 h-6 rounded-full border-2 transition-all hover:scale-110 ${
                      customization.skinTone === tone ? 'border-primary scale-110' : 'border-transparent'
                    }`}
                    style={{ backgroundColor: tone }}
                  />
                ))}
              </div>
            </div>

            {/* Hair Color */}
            <div className="mb-3">
              <p className="text-[10px] font-medium text-muted-foreground mb-1.5">Hair Color</p>
              <div className="flex flex-wrap gap-1.5">
                {hairColors.map(color => (
                  <button
                    key={color}
                    onClick={() => setCustomization(prev => ({ ...prev, hairColor: color }))}
                    className={`w-6 h-6 rounded-full border-2 transition-all hover:scale-110 ${
                      customization.hairColor === color ? 'border-primary scale-110' : 'border-transparent'
                    }`}
                    style={{ backgroundColor: color }}
                  />
                ))}
              </div>
            </div>

            {/* Shirt Color */}
            <div className="mb-3">
              <p className="text-[10px] font-medium text-muted-foreground mb-1.5">Shirt Color</p>
              <div className="flex flex-wrap gap-1.5">
                {shirtColors.map(color => (
                  <button
                    key={color}
                    onClick={() => setCustomization(prev => ({ ...prev, shirtColor: color }))}
                    className={`w-6 h-6 rounded-full border-2 transition-all hover:scale-110 ${
                      customization.shirtColor === color ? 'border-primary scale-110' : 'border-white/20'
                    }`}
                    style={{ backgroundColor: color }}
                  />
                ))}
              </div>
            </div>

            {/* Eye Color */}
            <div>
              <p className="text-[10px] font-medium text-muted-foreground mb-1.5">Eye Color</p>
              <div className="flex flex-wrap gap-1.5">
                {eyeColors.map(color => (
                  <button
                    key={color}
                    onClick={() => setCustomization(prev => ({ ...prev, eyeColor: color }))}
                    className={`w-6 h-6 rounded-full border-2 transition-all hover:scale-110 ${
                      customization.eyeColor === color ? 'border-primary scale-110' : 'border-transparent'
                    }`}
                    style={{ backgroundColor: color }}
                  />
                ))}
              </div>
            </div>
          </div>
        )}

        {/* 3D Canvas */}
        <div className={`bg-gradient-to-b from-slate-100 via-slate-200 to-slate-300 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 ${
          isFullscreen ? 'h-full' : 'aspect-square'
        }`}>
          <Canvas shadows>
            <Suspense fallback={
              <Html center>
                <div className="flex flex-col items-center gap-2">
                  <Loader2 className="w-8 h-8 animate-spin text-primary" />
                  <p className="text-sm text-muted-foreground">Loading...</p>
                </div>
              </Html>
            }>
              <AvatarScene 
                avatarUrl={avatarUrl}
                currentKeyframe={currentKeyframe}
                customization={customization}
                mouthOpenness={mouthOpenness}
                isSpeaking={isPlayingAudio}
              />
            </Suspense>
          </Canvas>
        </div>

        {/* Status & Current Sign Display */}
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/70 via-black/40 to-transparent">
          {isResponding && currentSentence ? (
            <div className="text-center">
              <p className="text-[10px] text-white/50 mb-1 uppercase tracking-wider font-medium">Signing Sentence</p>
              <p className="text-sm font-medium text-white leading-relaxed">
                {words.map((word, i) => (
                  <span
                    key={i}
                    className={`inline-block mx-0.5 transition-all duration-300 ${
                      i === wordIndex 
                        ? 'text-primary scale-105 font-bold' 
                        : i < wordIndex 
                          ? 'text-white/30' 
                          : 'text-white/60'
                    }`}
                  >
                    {word}
                  </span>
                ))}
              </p>
            </div>
          ) : currentSignName ? (
            <div className="text-center">
              <p className="text-[10px] text-white/50 mb-1 uppercase tracking-wider font-medium">Playing Sign</p>
              <p className="text-lg font-bold text-primary capitalize">{currentSignName}</p>
            </div>
          ) : (
            <div className="text-center">
              <p className="text-xs text-white/50">Ready to respond in {language}</p>
            </div>
          )}
        </div>

        {/* Status Indicator */}
        <div className="absolute top-14 left-3 flex flex-col gap-1.5">
          <div className={`flex items-center gap-1.5 px-2 py-1 rounded-full text-[10px] font-semibold backdrop-blur-sm ${
            isResponding 
              ? 'bg-green-500/20 text-green-400 border border-green-500/30' 
              : currentSignName
                ? 'bg-violet-500/20 text-violet-400 border border-violet-500/30'
                : 'bg-white/10 text-white/50 border border-white/10'
          }`}>
            <div className={`w-1.5 h-1.5 rounded-full ${
              isResponding 
                ? 'bg-green-500 animate-pulse' 
                : currentSignName
                  ? 'bg-violet-500 animate-pulse'
                  : 'bg-white/30'
            }`} />
            {isResponding ? 'Signing' : currentSignName ? 'Playing' : 'Idle'}
          </div>

          {isPlayingAudio && (
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-full text-[10px] font-semibold backdrop-blur-sm bg-blue-500/20 text-blue-400 border border-blue-500/30">
              <Volume2 className="w-3 h-3 animate-pulse" />
              Speaking
            </div>
          )}
        </div>
      </div>

      {/* Avatar Marketplace Modal */}
      <AvatarMarketplace
        isOpen={showMarketplace}
        onClose={() => setShowMarketplace(false)}
        onSelectAvatar={handleSelectAvatar}
        currentAvatarUrl={avatarUrl || undefined}
      />

      {/* Sign Learning Mode Modal */}
      <SignLearningMode
        isOpen={showLearningMode}
        onClose={() => setShowLearningMode(false)}
        onPlaySign={handlePlaySignFromLearning}
        recognizedSign={recognizedSign}
        language={language}
      />
    </>
  );
};

export default ReadyPlayerMeAvatar;
