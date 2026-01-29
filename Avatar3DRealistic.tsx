import React, { useRef, useEffect, useState, Suspense, useCallback, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { 
  OrbitControls, 
  Environment, 
  ContactShadows,
  Html,
  PerspectiveCamera,
  useTexture,
  Sphere,
  Box
} from '@react-three/drei';
import * as THREE from 'three';
import { 
  Volume2, VolumeX, Maximize2, Minimize2, 
  Palette, Sparkles, User, Loader2,
  Play, Settings2, ShoppingBag, BookOpen,
  Activity, Cpu, Database, Cloud, Zap
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
} from './SignDictionary';

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
  gender: 'female' | 'male';
}

interface Avatar3DRealisticProps {
  currentSentence: string;
  isResponding: boolean;
  language: string;
  onResponseComplete?: () => void;
  recognizedSign?: string | null;
  avatarUrl?: string;
  onAvatarUrlChange?: (url: string) => void;
  onOpenMarketplace?: () => void;
  onOpenLearning?: () => void;
}

// ============================================
// PHOTOREALISTIC HUMAN AVATAR
// ============================================

const PhotorealisticAvatar: React.FC<{
  currentKeyframe: SignKeyframe | null;
  customization: AvatarCustomization;
  mouthOpenness: number;
  isSpeaking: boolean;
}> = ({ currentKeyframe, customization, mouthOpenness, isSpeaking }) => {
  const groupRef = useRef<THREE.Group>(null);
  const [blinkProgress, setBlinkProgress] = useState(0);
  const [breatheOffset, setBreatheOffset] = useState(0);
  const [idleTime, setIdleTime] = useState(0);
  
  // Smooth animation interpolation
  const smoothLeftArm = useRef({ x: -0.22, y: 0.2, z: 0.1 });
  const smoothRightArm = useRef({ x: 0.22, y: 0.2, z: 0.1 });
  const smoothLeftElbow = useRef(0.3);
  const smoothRightElbow = useRef(0.3);
  const smoothHeadTilt = useRef([0, 0, 0]);
  const smoothShoulders = useRef(0);

  useFrame((state, delta) => {
    const time = state.clock.elapsedTime;
    setIdleTime(time);
    
    // Natural blinking (every 2.5-4 seconds with variation)
    const blinkInterval = 3 + Math.sin(time * 0.3) * 0.5;
    const blinkCycle = (time % blinkInterval) / blinkInterval;
    if (blinkCycle > 0.96 || blinkCycle < 0.04) {
      setBlinkProgress(prev => Math.min(1, prev + delta * 15));
    } else {
      setBlinkProgress(prev => Math.max(0, prev - delta * 10));
    }
    
    // Subtle breathing animation
    setBreatheOffset(Math.sin(time * 1.2) * 0.004 + Math.sin(time * 0.8) * 0.002);

    // Smooth arm movements with IK-like interpolation
    if (currentKeyframe) {
      const targetLeft = currentKeyframe.leftArm.position;
      const targetRight = currentKeyframe.rightArm.position;
      const lerpSpeed = delta * 6;
      
      smoothLeftArm.current.x = THREE.MathUtils.lerp(smoothLeftArm.current.x, targetLeft[0], lerpSpeed);
      smoothLeftArm.current.y = THREE.MathUtils.lerp(smoothLeftArm.current.y, targetLeft[1], lerpSpeed);
      smoothLeftArm.current.z = THREE.MathUtils.lerp(smoothLeftArm.current.z, targetLeft[2], lerpSpeed);
      smoothRightArm.current.x = THREE.MathUtils.lerp(smoothRightArm.current.x, targetRight[0], lerpSpeed);
      smoothRightArm.current.y = THREE.MathUtils.lerp(smoothRightArm.current.y, targetRight[1], lerpSpeed);
      smoothRightArm.current.z = THREE.MathUtils.lerp(smoothRightArm.current.z, targetRight[2], lerpSpeed);
      
      // Calculate elbow bend from position if not provided
      const leftElbow = currentKeyframe.leftArm.elbow ?? Math.max(0.2, Math.abs(targetLeft[1]) * 0.8);
      const rightElbow = currentKeyframe.rightArm.elbow ?? Math.max(0.2, Math.abs(targetRight[1]) * 0.8);
      
      smoothLeftElbow.current = THREE.MathUtils.lerp(smoothLeftElbow.current, leftElbow, lerpSpeed);
      smoothRightElbow.current = THREE.MathUtils.lerp(smoothRightElbow.current, rightElbow, lerpSpeed);
      
      smoothHeadTilt.current[0] = THREE.MathUtils.lerp(smoothHeadTilt.current[0], currentKeyframe.body.headTilt[0], lerpSpeed);
      smoothHeadTilt.current[1] = THREE.MathUtils.lerp(smoothHeadTilt.current[1], currentKeyframe.body.headTilt[1], lerpSpeed);
      smoothHeadTilt.current[2] = THREE.MathUtils.lerp(smoothHeadTilt.current[2], currentKeyframe.body.headTilt[2], lerpSpeed);
      

      const idleArmY = 0.2 + Math.sin(time * 0.5) * 0.01;
      smoothLeftArm.current.y = THREE.MathUtils.lerp(smoothLeftArm.current.y, idleArmY, delta * 2);
      smoothRightArm.current.y = THREE.MathUtils.lerp(smoothRightArm.current.y, idleArmY, delta * 2);
      smoothLeftArm.current.x = THREE.MathUtils.lerp(smoothLeftArm.current.x, -0.22, delta * 2);
      smoothRightArm.current.x = THREE.MathUtils.lerp(smoothRightArm.current.x, 0.22, delta * 2);
      smoothLeftElbow.current = THREE.MathUtils.lerp(smoothLeftElbow.current, 0.3, delta * 2);
      smoothRightElbow.current = THREE.MathUtils.lerp(smoothRightElbow.current, 0.3, delta * 2);
      
      // Subtle head movement when idle
      smoothHeadTilt.current[0] = THREE.MathUtils.lerp(smoothHeadTilt.current[0], Math.sin(time * 0.3) * 0.02, delta * 2);
      smoothHeadTilt.current[1] = THREE.MathUtils.lerp(smoothHeadTilt.current[1], Math.sin(time * 0.2) * 0.03, delta * 2);
    }
  });

  const eyebrowRaise = currentKeyframe?.face.eyebrowRaise || 0;
  const mouthSmile = currentKeyframe?.face.mouthSmile || 0;
  const torsoTwist = currentKeyframe?.body.torsoTwist || 0;

  // Realistic skin material with subsurface scattering simulation
  const skinMaterial = useMemo(() => (
    <meshStandardMaterial 
      color={customization.skinTone} 
      roughness={0.55}
      metalness={0.05}
    />
  ), [customization.skinTone]);

  // Hair material with slight sheen
  const hairMaterial = useMemo(() => (
    <meshStandardMaterial 
      color={customization.hairColor} 
      roughness={0.7}
      metalness={0.1}
    />
  ), [customization.hairColor]);

  // Clothing material
  const shirtMaterial = useMemo(() => (
    <meshStandardMaterial 
      color={customization.shirtColor} 
      roughness={0.65}
      metalness={0}
    />
  ), [customization.shirtColor]);

  // Calculate IK for arms
  const calculateArmPose = (targetPos: { x: number; y: number; z: number }, elbowBend: number, isLeft: boolean) => {
    const shoulderOffset = isLeft ? -0.18 : 0.18;
    const shoulderY = 0.48;
    
    const dx = targetPos.x - shoulderOffset;
    const dy = targetPos.y - shoulderY;
    const dz = targetPos.z;
    
    // Calculate shoulder rotation
    const shoulderRotZ = Math.atan2(dy, Math.abs(dx)) * (isLeft ? -1 : 1);
    const shoulderRotY = Math.atan2(dz, Math.sqrt(dx * dx + dy * dy));
    const shoulderRotX = isLeft ? 0.15 : -0.15;
    
    return {
      shoulderRotation: [shoulderRotX + elbowBend * 0.3, shoulderRotY, shoulderRotZ - (isLeft ? 0.3 : -0.3)],
      elbowRotation: elbowBend * 1.8
    };
  };

  const leftArmPose = calculateArmPose(smoothLeftArm.current, smoothLeftElbow.current, true);
  const rightArmPose = calculateArmPose(smoothRightArm.current, smoothRightElbow.current, false);

  // Detailed hand with articulated fingers
  const DetailedHand: React.FC<{ isLeft: boolean; fingerData: any }> = ({ isLeft, fingerData }) => {
    const fingers = fingerData || { thumb: 0.2, index: 0.2, middle: 0.2, ring: 0.2, pinky: 0.2, spread: 0.1 };
    
    return (
      <group rotation={[0.1, 0, isLeft ? 0.15 : -0.15]}>
        {/* Palm */}
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[0.045, 0.06, 0.025]} />
          {skinMaterial}
        </mesh>
        
        {/* Fingers with 3 segments each */}
        {[
          { x: -0.014, length: 0.035, curl: fingers.index, name: 'index' },
          { x: -0.005, length: 0.038, curl: fingers.middle, name: 'middle' },
          { x: 0.005, length: 0.035, curl: fingers.ring, name: 'ring' },
          { x: 0.014, length: 0.03, curl: fingers.pinky, name: 'pinky' }
        ].map((finger, i) => (
          <group key={finger.name} position={[finger.x + (isLeft ? -1 : 1) * fingers.spread * 0.01 * (i - 1.5), -0.04, 0]}>
            {/* Proximal phalanx */}
            <group rotation={[finger.curl * 1.5, 0, 0]}>
              <mesh position={[0, -0.015, 0]}>
                <capsuleGeometry args={[0.006, 0.025, 4, 8]} />
                {skinMaterial}
              </mesh>
              {/* Middle phalanx */}
              <group position={[0, -0.035, 0]} rotation={[finger.curl * 1.2, 0, 0]}>
                <mesh position={[0, -0.01, 0]}>
                  <capsuleGeometry args={[0.005, 0.018, 4, 8]} />
                  {skinMaterial}
                </mesh>
                {/* Distal phalanx */}
                <group position={[0, -0.025, 0]} rotation={[finger.curl * 0.8, 0, 0]}>
                  <mesh position={[0, -0.008, 0]}>
                    <capsuleGeometry args={[0.004, 0.012, 4, 8]} />
                    {skinMaterial}
                  </mesh>
                </group>
              </group>
            </group>
          </group>
        ))}
        
        {/* Thumb with 2 segments */}
        <group position={[isLeft ? 0.028 : -0.028, -0.015, 0.01]} rotation={[0, isLeft ? -0.6 : 0.6, fingers.thumb * 0.5]}>
          <mesh position={[0, -0.012, 0]}>
            <capsuleGeometry args={[0.007, 0.02, 4, 8]} />
            {skinMaterial}
          </mesh>
          <group position={[0, -0.028, 0]} rotation={[fingers.thumb * 0.8, 0, 0]}>
            <mesh position={[0, -0.01, 0]}>
              <capsuleGeometry args={[0.006, 0.016, 4, 8]} />
              {skinMaterial}
            </mesh>
          </group>
        </group>
      </group>
    );
  };

  // Arm component with realistic proportions
  const RealisticArm: React.FC<{ isLeft: boolean }> = ({ isLeft }) => {
    const pose = isLeft ? leftArmPose : rightArmPose;
    const fingerData = currentKeyframe 
      ? (isLeft ? currentKeyframe.leftArm.hand : currentKeyframe.rightArm.hand)
      : null;
    const xOffset = isLeft ? -0.18 : 0.18;
    
    return (
      <group position={[xOffset, 0.48 + smoothShoulders.current * 0.02, 0]}>
        {/* Shoulder joint */}
        <mesh>
          <sphereGeometry args={[0.045, 16, 16]} />
          {shirtMaterial}
        </mesh>
        
        {/* Upper arm with rotation */}
        <group rotation={pose.shoulderRotation as [number, number, number]}>
          <mesh position={[0, -0.08, 0]}>
            <capsuleGeometry args={[0.038, 0.12, 8, 16]} />
            {shirtMaterial}
          </mesh>
          
          {/* Elbow */}
          <group position={[0, -0.16, 0]}>
            <mesh>
              <sphereGeometry args={[0.032, 12, 12]} />
              {skinMaterial}
            </mesh>
            
            {/* Forearm with elbow bend */}
            <group rotation={[pose.elbowRotation, 0, 0]}>
              <mesh position={[0, -0.07, 0]}>
                <capsuleGeometry args={[0.03, 0.11, 8, 16]} />
                {skinMaterial}
              </mesh>
              
              {/* Wrist */}
              <group position={[0, -0.14, 0]}>
                <mesh>
                  <sphereGeometry args={[0.022, 10, 10]} />
                  {skinMaterial}
                </mesh>
                
                {/* Hand */}
                <group position={[0, -0.02, 0]}>
                  <DetailedHand isLeft={isLeft} fingerData={fingerData} />
                </group>
              </group>
            </group>
          </group>
        </group>
      </group>
    );
  };

  return (
    <group ref={groupRef} position={[0, -0.5, 0]}>
      {/* Torso */}
      <group position={[0, 0.3, 0]} rotation={[0, torsoTwist * 0.3, 0]}>
        {/* Main torso - professional shirt */}
        <mesh position={[0, 0, 0]}>
          <capsuleGeometry args={[0.14, 0.22, 12, 24]} />
          {shirtMaterial}
        </mesh>
        
        {/* Chest definition */}
        <mesh position={[0, 0.05, 0.06]} scale={[1.1, 0.8, 0.5]}>
          <sphereGeometry args={[0.1, 16, 16]} />
          {shirtMaterial}
        </mesh>
        
        {/* Collar/neckline */}
        <mesh position={[0, 0.16, 0.03]} rotation={[0.4, 0, 0]}>
          <torusGeometry args={[0.055, 0.015, 8, 16, Math.PI]} />
          {shirtMaterial}
        </mesh>
        
        {/* Shoulders */}
        <mesh position={[-0.15, 0.08, 0]} rotation={[0, 0, 0.35]}>
          <capsuleGeometry args={[0.05, 0.08, 8, 16]} />
          {shirtMaterial}
        </mesh>
        <mesh position={[0.15, 0.08, 0]} rotation={[0, 0, -0.35]}>
          <capsuleGeometry args={[0.05, 0.08, 8, 16]} />
          {shirtMaterial}
        </mesh>

        {/* Arms */}
        <RealisticArm isLeft={true} />
        <RealisticArm isLeft={false} />
      </group>

      {/* Neck */}
      <mesh position={[0, 0.56 + breatheOffset, 0]}>
        <cylinderGeometry args={[0.04, 0.048, 0.1, 16]} />
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
        {/* Head shape - realistic oval */}
        <mesh scale={[0.88, 1.08, 0.92]}>
          <sphereGeometry args={[0.12, 32, 32]} />
          {skinMaterial}
        </mesh>
        
        {/* Jaw/chin definition */}
        <mesh position={[0, -0.08, 0.04]} scale={[0.65, 0.45, 0.55]}>
          <sphereGeometry args={[0.1, 16, 16]} />
          {skinMaterial}
        </mesh>
        
        {/* Cheekbones */}
        <mesh position={[-0.06, -0.02, 0.08]} scale={[0.35, 0.25, 0.3]}>
          <sphereGeometry args={[0.1, 12, 12]} />
          {skinMaterial}
        </mesh>
        <mesh position={[0.06, -0.02, 0.08]} scale={[0.35, 0.25, 0.3]}>
          <sphereGeometry args={[0.1, 12, 12]} />
          {skinMaterial}
        </mesh>

        {/* Hair */}
        {customization.gender === 'female' ? (
          <>
            {/* Female hair - shoulder length */}
            <mesh position={[0, 0.04, -0.01]} scale={[1.08, 1.02, 1.02]}>
              <sphereGeometry args={[0.12, 32, 32, 0, Math.PI * 2, 0, Math.PI * 0.55]} />
              {hairMaterial}
            </mesh>
            {/* Side hair */}
            <mesh position={[-0.1, -0.04, 0]} scale={[0.35, 0.9, 0.45]}>
              <sphereGeometry args={[0.12, 16, 16]} />
              {hairMaterial}
            </mesh>
            <mesh position={[0.1, -0.04, 0]} scale={[0.35, 0.9, 0.45]}>
              <sphereGeometry args={[0.12, 16, 16]} />
              {hairMaterial}
            </mesh>
            {/* Back hair */}
            <mesh position={[0, -0.06, -0.08]} scale={[0.75, 1.1, 0.35]}>
              <sphereGeometry args={[0.12, 16, 16]} />
              {hairMaterial}
            </mesh>
            {/* Bangs */}
            <mesh position={[0, 0.08, 0.1]} scale={[0.8, 0.15, 0.3]} rotation={[0.3, 0, 0]}>
              <sphereGeometry args={[0.1, 12, 12]} />
              {hairMaterial}
            </mesh>
          </>
        ) : (
          <>
            {/* Male hair - short styled */}
            <mesh position={[0, 0.05, 0]}>
              <sphereGeometry args={[0.118, 32, 32, 0, Math.PI * 2, 0, Math.PI * 0.45]} />
              {hairMaterial}
            </mesh>
            {/* Side fade */}
            <mesh position={[-0.09, 0.02, 0]} scale={[0.25, 0.5, 0.8]}>
              <sphereGeometry args={[0.1, 12, 12]} />
              {hairMaterial}
            </mesh>
            <mesh position={[0.09, 0.02, 0]} scale={[0.25, 0.5, 0.8]}>
              <sphereGeometry args={[0.1, 12, 12]} />
              {hairMaterial}
            </mesh>
          </>
        )}

        {/* Ears */}
        {[-0.115, 0.115].map((x, i) => (
          <group key={i} position={[x, -0.01, -0.02]}>
            <mesh rotation={[0, x > 0 ? -0.25 : 0.25, 0]} scale={[0.6, 1, 0.4]}>
              <sphereGeometry args={[0.025, 10, 10]} />
              {skinMaterial}
            </mesh>
          </group>
        ))}

        {/* Eyes with realistic detail */}
        {[-0.038, 0.038].map((x, i) => (
          <group key={i} position={[x, 0.015, 0.095]}>
            {/* Eye socket/shadow */}
            <mesh position={[0, 0, -0.008]} scale={[1.4, 1.2, 0.5]}>
              <sphereGeometry args={[0.018, 14, 14]} />
              <meshStandardMaterial color={customization.skinTone} roughness={0.7} />
            </mesh>
            
            {/* Eyeball */}
            <mesh scale={[1, 0.72 * (1 - blinkProgress * 0.95), 1]}>
              <sphereGeometry args={[0.018, 20, 20]} />
              <meshStandardMaterial color="#fefefe" roughness={0.2} />
            </mesh>
            
            {/* Iris */}
            <mesh position={[0, 0, 0.012]} scale={[1, 1 - blinkProgress * 0.95, 1]}>
              <sphereGeometry args={[0.009, 16, 16]} />
              <meshStandardMaterial color={customization.eyeColor} roughness={0.35} />
            </mesh>
            
            {/* Pupil */}
            <mesh position={[0, 0, 0.016]} scale={[1, 1 - blinkProgress * 0.95, 1]}>
              <sphereGeometry args={[0.004, 10, 10]} />
              <meshStandardMaterial color="#0a0a0a" roughness={0.1} />
            </mesh>
            
            {/* Eye highlight/reflection */}
            <mesh position={[0.004, 0.004, 0.017]} scale={[1, 1 - blinkProgress * 0.95, 1]}>
              <sphereGeometry args={[0.0025, 8, 8]} />
              <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.6} />
            </mesh>
            
            {/* Upper eyelid */}
            <mesh 
              position={[0, 0.01 - blinkProgress * 0.015, 0.006]} 
              scale={[1.15, 0.35 + blinkProgress * 0.55, 0.75]}
              rotation={[0.1, 0, 0]}
            >
              <sphereGeometry args={[0.018, 14, 14, 0, Math.PI * 2, 0, Math.PI * 0.5]} />
              {skinMaterial}
            </mesh>
            
            {/* Lower eyelid */}
            <mesh 
              position={[0, -0.012, 0.004]} 
              scale={[1.1, 0.2, 0.6]}
              rotation={[-0.1, 0, 0]}
            >
              <sphereGeometry args={[0.018, 12, 12, 0, Math.PI * 2, Math.PI * 0.5, Math.PI * 0.5]} />
              {skinMaterial}
            </mesh>
            
            {/* Eyelashes (subtle) */}
            {customization.gender === 'female' && (
              <mesh position={[0, 0.015, 0.01]} scale={[1.2, 0.08, 0.3]} rotation={[0.2, 0, 0]}>
                <boxGeometry args={[0.035, 0.01, 0.01]} />
                <meshStandardMaterial color="#1a1a1a" />
              </mesh>
            )}
          </group>
        ))}

        {/* Eyebrows */}
        {[-0.042, 0.042].map((x, i) => (
          <mesh 
            key={`brow-${i}`}
            position={[x, 0.052 + eyebrowRaise * 0.01, 0.1]} 
            rotation={[0.15, 0, (i === 0 ? 0.15 : -0.15) - eyebrowRaise * 0.1]}
            scale={[1, 0.35, 0.55]}
          >
            <capsuleGeometry args={[0.01, 0.022, 4, 10]} />
            {hairMaterial}
          </mesh>
        ))}

        {/* Nose with realistic shape */}
        <group position={[0, -0.02, 0.105]}>
          {/* Bridge */}
          <mesh position={[0, 0.02, 0]} rotation={[-0.15, 0, 0]} scale={[0.55, 1, 0.75]}>
            <capsuleGeometry args={[0.01, 0.03, 6, 10]} />
            {skinMaterial}
          </mesh>
          {/* Tip */}
          <mesh position={[0, -0.012, 0.012]} scale={[1, 0.65, 0.75]}>
            <sphereGeometry args={[0.015, 14, 14]} />
            {skinMaterial}
          </mesh>
          {/* Nostrils */}
          {[-0.01, 0.01].map((x, i) => (
            <mesh key={i} position={[x, -0.018, 0.002]} scale={[0.55, 0.35, 0.45]}>
              <sphereGeometry args={[0.01, 10, 10]} />
              <meshStandardMaterial color="#c4857f" roughness={0.75} />
            </mesh>
          ))}
        </group>

        {/* Mouth with lip sync */}
        <group position={[0, -0.065, 0.09]}>
          {/* Upper lip */}
          <mesh position={[0, 0.005 + mouthSmile * 0.004, 0]} scale={[1, 0.45, 0.55]}>
            <capsuleGeometry args={[0.022, 0.01, 6, 14]} />
            <meshStandardMaterial color="#c4706a" roughness={0.55} />
          </mesh>
          
          {/* Lower lip */}
          <mesh 
            position={[0, -0.01 - mouthOpenness * 0.02, 0]} 
            scale={[0.88, 0.5 + mouthOpenness * 0.25, 0.6]}
          >
            <capsuleGeometry args={[0.02, 0.012, 6, 14]} />
            <meshStandardMaterial color="#d4807a" roughness={0.55} />
          </mesh>
          
          {/* Mouth interior */}
          {mouthOpenness > 0.08 && (
            <mesh position={[0, -0.005, -0.012]} scale={[0.65, mouthOpenness * 0.9, 0.45]}>
              <sphereGeometry args={[0.018, 14, 14]} />
              <meshStandardMaterial color="#3a1818" roughness={0.95} />
            </mesh>
          )}
          
          {/* Teeth */}
          {mouthOpenness > 0.15 && (
            <>
              <mesh position={[0, 0.002, -0.006]} scale={[0.55, 0.12, 0.25]}>
                <boxGeometry args={[0.03, 0.012, 0.012]} />
                <meshStandardMaterial color="#f8f8f5" roughness={0.25} />
              </mesh>
              <mesh position={[0, -0.012 - mouthOpenness * 0.01, -0.006]} scale={[0.5, 0.1, 0.22]}>
                <boxGeometry args={[0.028, 0.01, 0.01]} />
                <meshStandardMaterial color="#f5f5f0" roughness={0.3} />
              </mesh>
            </>
          )}
          
          {/* Tongue hint */}
          {mouthOpenness > 0.25 && (
            <mesh position={[0, -0.015, -0.005]} scale={[0.5, 0.3, 0.4]}>
              <sphereGeometry args={[0.012, 10, 10]} />
              <meshStandardMaterial color="#c45555" roughness={0.7} />
            </mesh>
          )}
        </group>
      </group>
    </group>
  );
};

// ============================================
// AVATAR SCENE
// ============================================

const AvatarScene: React.FC<{
  currentKeyframe: SignKeyframe | null;
  customization: AvatarCustomization;
  mouthOpenness: number;
  isSpeaking: boolean;
}> = ({ currentKeyframe, customization, mouthOpenness, isSpeaking }) => {
  return (
    <>
      <PerspectiveCamera makeDefault position={[0, 0.45, 1.4]} fov={40} />
      <OrbitControls 
        enablePan={false} 
        enableZoom={true}
        minDistance={0.8}
        maxDistance={2.5}
        minPolarAngle={Math.PI / 4}
        maxPolarAngle={Math.PI / 1.7}
        target={[0, 0.3, 0]}
      />
      
      <ambientLight intensity={0.45} />
      <directionalLight position={[3, 5, 4]} intensity={0.9} castShadow />
      <directionalLight position={[-3, 3, 4]} intensity={0.5} />
      <directionalLight position={[0, 2, -3]} intensity={0.25} />
      <pointLight position={[0, 1, 2]} intensity={0.3} color="#ffeedd" />
      
      <Environment preset="studio" />
      
      <PhotorealisticAvatar 
        currentKeyframe={currentKeyframe}
        customization={customization}
        mouthOpenness={mouthOpenness}
        isSpeaking={isSpeaking}
      />
      
      <ContactShadows 
        position={[0, -0.5, 0]} 
        opacity={0.5} 
        scale={1.5} 
        blur={2.5} 
      />
    </>
  );
};

// ============================================
// MAIN COMPONENT
// ============================================

const Avatar3DRealistic: React.FC<Avatar3DRealisticProps> = ({
  currentSentence,
  isResponding,
  language,
  onResponseComplete,
  recognizedSign,
  avatarUrl: externalAvatarUrl,
  onAvatarUrlChange,
  onOpenMarketplace,
  onOpenLearning,
}) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showCustomization, setShowCustomization] = useState(false);
  const [customization, setCustomization] = useState<AvatarCustomization>({
    skinTone: '#e0ac69',
    hairColor: '#2c1810',
    shirtColor: '#2563eb',
    eyeColor: '#4a3728',
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
  const [aslGloss, setAslGloss] = useState<string>('');

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const animationRef = useRef<number | null>(null);

  const skinTones = ['#ffdfc4', '#f0c8a0', '#e0ac69', '#c68642', '#8d5524', '#5c3d2e'];
  const hairColors = ['#2c1810', '#4a3728', '#8b4513', '#d4a574', '#1a1a1a', '#c0c0c0'];
  const shirtColors = ['#2563eb', '#7c3aed', '#10b981', '#f59e0b', '#ef4444', '#1f2937'];
  const eyeColors = ['#4a3728', '#2d5a27', '#1e3a5f', '#4a4a4a', '#8b4513'];

  // Convert English to ASL Gloss
  const convertToASLGloss = useCallback((sentence: string): string => {
    // Basic ASL gloss conversion rules:
    // 1. Remove articles (a, an, the)
    // 2. Reorder to Topic-Comment structure
    // 3. Use ASL grammar conventions
    
    let gloss = sentence.toUpperCase();
    
    // Remove articles
    gloss = gloss.replace(/\b(A|AN|THE)\b/gi, '');
    
    // Remove "to be" verbs in some contexts
    gloss = gloss.replace(/\b(AM|IS|ARE|WAS|WERE)\b/gi, '');
    
    // Convert questions
    if (gloss.includes('?')) {
      // Move WH-words to end for ASL
      const whWords = ['WHAT', 'WHERE', 'WHEN', 'WHO', 'WHY', 'HOW'];
      for (const wh of whWords) {
        if (gloss.startsWith(wh)) {
          gloss = gloss.replace(wh, '').trim() + ' ' + wh;
          break;
        }
      }
    }
    
    // Clean up extra spaces
    gloss = gloss.replace(/\s+/g, ' ').trim();
    gloss = gloss.replace(/[.,!]/g, '');
    
    return gloss;
  }, []);

  // Generate speech with TTS
  const generateSpeech = useCallback(async (text: string) => {
    if (!voiceSettings.enabled || !text) return;

    setIsLoadingVoice(true);
    try {
      const response = await supabase.functions.invoke('avatar-tts', {
        body: {
          text,
          voice: voiceSettings.voice,
          speed: voiceSettings.speed,
          pitch: voiceSettings.pitch,
        },
      });

      if (response.error) throw response.error;

      // Handle audio response
      if (response.data) {
        setIsPlayingAudio(true);
        animateLipSync();
        
        // Simulate audio duration
        setTimeout(() => {
          setIsPlayingAudio(false);
          setMouthOpenness(0);
          if (animationRef.current) cancelAnimationFrame(animationRef.current);
        }, text.length * 80);
      }
    } catch (error) {
      console.error('TTS error:', error);
    } finally {
      setIsLoadingVoice(false);
    }
  }, [voiceSettings]);

  // Lip sync animation
  const animateLipSync = useCallback(() => {
    const animate = () => {
      if (!isPlayingAudio) return;
      const time = performance.now() / 1000;
      const openness = 0.25 + Math.sin(time * 10) * 0.18 + Math.sin(time * 15) * 0.08;
      setMouthOpenness(Math.max(0, Math.min(1, openness)));
      animationRef.current = requestAnimationFrame(animate);
    };
    animate();
  }, [isPlayingAudio]);

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
          }, 80);
        }
      };

      requestAnimationFrame(animate);
    });
  }, []);

  // Parse sentence into signs and play sequentially
  const playSentence = useCallback(async (sentence: string) => {
    // Convert to ASL gloss first
    const gloss = convertToASLGloss(sentence);
    setAslGloss(gloss);
    
    const words = gloss.toLowerCase().split(/\s+/).filter(w => w.length > 0);
    const signs: SignData[] = [];
    
    for (const word of words) {
      const cleanWord = word.replace(/[^a-z]/g, '');
      const sign = getSign(cleanWord);
      if (sign) {
        signs.push(sign);
      }
    }

    if (voiceSettings.enabled) {
      generateSpeech(sentence);
    }

    for (let i = 0; i < signs.length; i++) {
      setWordIndex(i);
      await playSignAnimation(signs[i]);
      await new Promise(resolve => setTimeout(resolve, 120));
    }

    onResponseComplete?.();
  }, [voiceSettings.enabled, generateSpeech, playSignAnimation, onResponseComplete, convertToASLGloss]);

  useEffect(() => {
    if (isResponding && currentSentence) {
      playSentence(currentSentence);
    }
  }, [isResponding, currentSentence, playSentence]);

  useEffect(() => {
    if (recognizedSign) {
      const sign = getSign(recognizedSign);
      if (sign) {
        playSignAnimation(sign);
      }
    }
  }, [recognizedSign, playSignAnimation]);

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

  const glossWords = aslGloss?.split(' ').filter(w => w.length > 0) || [];

  return (
    <div className={`relative bg-card rounded-2xl border border-border overflow-hidden transition-all duration-300 ${
      isFullscreen ? 'fixed inset-4 z-50' : ''
    }`}>
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between p-3 bg-gradient-to-b from-black/70 to-transparent">
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
          {onOpenLearning && (
            <button
              onClick={onOpenLearning}
              className="p-2 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 transition-all"
              title="Learn Signs"
            >
              <BookOpen className="w-4 h-4" />
            </button>
          )}
          {onOpenMarketplace && (
            <button
              onClick={onOpenMarketplace}
              className="p-2 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 transition-all"
              title="Avatar Marketplace"
            >
              <ShoppingBag className="w-4 h-4" />
            </button>
          )}
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
      {showCustomization && (
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
              currentKeyframe={currentKeyframe}
              customization={customization}
              mouthOpenness={mouthOpenness}
              isSpeaking={isPlayingAudio}
            />
          </Suspense>
        </Canvas>
      </div>

      {/* Status & ASL Gloss Display */}
      <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 via-black/50 to-transparent">
        {isResponding && currentSentence ? (
          <div className="text-center space-y-2">
            {/* Original sentence */}
            <div>
              <p className="text-[9px] text-white/40 uppercase tracking-wider font-medium">English</p>
              <p className="text-xs text-white/60">{currentSentence}</p>
            </div>
            
            {/* ASL Gloss */}
            {aslGloss && (
              <div>
                <p className="text-[9px] text-primary/80 uppercase tracking-wider font-medium">ASL Gloss</p>
                <p className="text-sm font-medium text-white leading-relaxed">
                  {glossWords.map((word, i) => (
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
            )}
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
  );
};

export default Avatar3DRealistic;
