import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Volume2, VolumeX, Maximize2, Minimize2, 
  RefreshCw, Settings, User, Palette
} from 'lucide-react';

interface Avatar3DProps {
  currentSentence: string;
  isResponding: boolean;
  language: string;
  onResponseComplete?: () => void;
}

type AvatarPose = 'idle' | 'signing' | 'thinking' | 'greeting' | 'nodding';

interface AvatarCustomization {
  skinTone: string;
  hairColor: string;
  shirtColor: string;
}

const Avatar3D: React.FC<Avatar3DProps> = ({
  currentSentence,
  isResponding,
  language,
  onResponseComplete
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [pose, setPose] = useState<AvatarPose>('idle');
  const [currentWord, setCurrentWord] = useState('');
  const [wordIndex, setWordIndex] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [showCustomization, setShowCustomization] = useState(false);
  const [customization, setCustomization] = useState<AvatarCustomization>({
    skinTone: '#e0ac69',
    hairColor: '#2c1810',
    shirtColor: '#7c3aed'
  });
  const animationRef = useRef<number>();
  const frameRef = useRef(0);

  // Skin tone options
  const skinTones = ['#ffdfc4', '#f0c8a0', '#e0ac69', '#c68642', '#8d5524', '#5c3d2e'];
  const hairColors = ['#2c1810', '#4a3728', '#8b4513', '#d4a574', '#1a1a1a', '#c0c0c0'];
  const shirtColors = ['#7c3aed', '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#ec4899'];

  // Draw avatar on canvas
  const drawAvatar = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const frame = frameRef.current;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Background gradient
    const bgGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, height);
    bgGradient.addColorStop(0, 'rgba(139, 92, 246, 0.1)');
    bgGradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = bgGradient;
    ctx.fillRect(0, 0, width, height);

    // Animation offsets based on pose
    let headBob = 0;
    let shoulderOffset = 0;
    let armAngle = 0;
    let handOffset = { left: { x: 0, y: 0 }, right: { x: 0, y: 0 } };

    switch (pose) {
      case 'signing':
        headBob = Math.sin(frame * 0.1) * 3;
        shoulderOffset = Math.sin(frame * 0.15) * 2;
        armAngle = Math.sin(frame * 0.2) * 15;
        handOffset = {
          left: { x: Math.sin(frame * 0.25) * 20, y: Math.cos(frame * 0.2) * 15 - 30 },
          right: { x: Math.cos(frame * 0.25) * 25, y: Math.sin(frame * 0.2) * 20 - 40 }
        };
        break;
      case 'thinking':
        headBob = Math.sin(frame * 0.05) * 2;
        handOffset = {
          left: { x: 0, y: 0 },
          right: { x: 30, y: -60 }
        };
        break;
      case 'greeting':
        headBob = Math.sin(frame * 0.1) * 5;
        handOffset = {
          left: { x: 0, y: 0 },
          right: { x: 40 + Math.sin(frame * 0.3) * 10, y: -80 }
        };
        break;
      case 'nodding':
        headBob = Math.sin(frame * 0.2) * 8;
        break;
      default:
        headBob = Math.sin(frame * 0.03) * 2;
    }

    // Draw body/shirt
    ctx.fillStyle = customization.shirtColor;
    ctx.beginPath();
    ctx.ellipse(centerX, centerY + 120, 80, 60, 0, 0, Math.PI * 2);
    ctx.fill();

    // Shoulders
    ctx.beginPath();
    ctx.ellipse(centerX, centerY + 70 + shoulderOffset, 90, 30, 0, Math.PI, Math.PI * 2);
    ctx.fill();

    // Neck
    ctx.fillStyle = customization.skinTone;
    ctx.beginPath();
    ctx.ellipse(centerX, centerY + 40, 20, 25, 0, 0, Math.PI * 2);
    ctx.fill();

    // Head
    ctx.beginPath();
    ctx.ellipse(centerX, centerY - 20 + headBob, 55, 65, 0, 0, Math.PI * 2);
    ctx.fill();

    // Hair
    ctx.fillStyle = customization.hairColor;
    ctx.beginPath();
    ctx.ellipse(centerX, centerY - 55 + headBob, 50, 35, 0, Math.PI, Math.PI * 2);
    ctx.fill();
    
    // Side hair
    ctx.beginPath();
    ctx.ellipse(centerX - 45, centerY - 30 + headBob, 15, 40, -0.3, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(centerX + 45, centerY - 30 + headBob, 15, 40, 0.3, 0, Math.PI * 2);
    ctx.fill();

    // Eyes
    const eyeY = centerY - 25 + headBob;
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.ellipse(centerX - 18, eyeY, 12, 10, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(centerX + 18, eyeY, 12, 10, 0, 0, Math.PI * 2);
    ctx.fill();

    // Pupils (follow hand movement when signing)
    const pupilOffset = pose === 'signing' ? { x: handOffset.right.x * 0.05, y: handOffset.right.y * 0.03 } : { x: 0, y: 0 };
    ctx.fillStyle = '#2c1810';
    ctx.beginPath();
    ctx.arc(centerX - 18 + pupilOffset.x, eyeY + pupilOffset.y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(centerX + 18 + pupilOffset.x, eyeY + pupilOffset.y, 5, 0, Math.PI * 2);
    ctx.fill();

    // Eyebrows
    ctx.strokeStyle = customization.hairColor;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    const browRaise = pose === 'signing' ? 3 : 0;
    ctx.beginPath();
    ctx.moveTo(centerX - 28, eyeY - 15 - browRaise + headBob);
    ctx.quadraticCurveTo(centerX - 18, eyeY - 20 - browRaise + headBob, centerX - 8, eyeY - 15 - browRaise + headBob);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(centerX + 28, eyeY - 15 - browRaise + headBob);
    ctx.quadraticCurveTo(centerX + 18, eyeY - 20 - browRaise + headBob, centerX + 8, eyeY - 15 - browRaise + headBob);
    ctx.stroke();

    // Nose
    ctx.strokeStyle = `${customization.skinTone}cc`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY - 15 + headBob);
    ctx.lineTo(centerX - 5, centerY + headBob);
    ctx.stroke();

    // Mouth (changes with signing)
    ctx.strokeStyle = '#c9a9a9';
    ctx.lineWidth = 2;
    const mouthOpen = pose === 'signing' ? Math.abs(Math.sin(frame * 0.15)) * 8 : 0;
    if (mouthOpen > 2) {
      ctx.fillStyle = '#8b4040';
      ctx.beginPath();
      ctx.ellipse(centerX, centerY + 15 + headBob, 12, mouthOpen, 0, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.beginPath();
      ctx.moveTo(centerX - 15, centerY + 15 + headBob);
      ctx.quadraticCurveTo(centerX, centerY + 20 + headBob, centerX + 15, centerY + 15 + headBob);
      ctx.stroke();
    }

    // Arms
    ctx.fillStyle = customization.shirtColor;
    
    // Left arm
    ctx.save();
    ctx.translate(centerX - 70, centerY + 70);
    ctx.rotate((-20 + (pose === 'signing' ? -armAngle : 0)) * Math.PI / 180);
    ctx.beginPath();
    ctx.ellipse(0, 40, 18, 50, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Right arm
    ctx.save();
    ctx.translate(centerX + 70, centerY + 70);
    ctx.rotate((20 + (pose === 'signing' ? armAngle : 0)) * Math.PI / 180);
    ctx.beginPath();
    ctx.ellipse(0, 40, 18, 50, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Hands
    ctx.fillStyle = customization.skinTone;
    
    // Left hand
    const leftHandX = centerX - 90 + handOffset.left.x;
    const leftHandY = centerY + 130 + handOffset.left.y;
    ctx.beginPath();
    ctx.ellipse(leftHandX, leftHandY, 15, 18, 0, 0, Math.PI * 2);
    ctx.fill();
    
    // Left fingers
    for (let i = 0; i < 4; i++) {
      const angle = (-30 + i * 20) * Math.PI / 180;
      ctx.beginPath();
      ctx.ellipse(
        leftHandX + Math.cos(angle) * 18,
        leftHandY + Math.sin(angle) * 18 - 10,
        4, 12, angle, 0, Math.PI * 2
      );
      ctx.fill();
    }

    // Right hand
    const rightHandX = centerX + 90 + handOffset.right.x;
    const rightHandY = centerY + 130 + handOffset.right.y;
    ctx.beginPath();
    ctx.ellipse(rightHandX, rightHandY, 15, 18, 0, 0, Math.PI * 2);
    ctx.fill();
    
    // Right fingers
    for (let i = 0; i < 4; i++) {
      const angle = (30 - i * 20) * Math.PI / 180;
      const fingerExtend = pose === 'signing' ? Math.sin(frame * 0.2 + i) * 3 : 0;
      ctx.beginPath();
      ctx.ellipse(
        rightHandX + Math.cos(angle) * 18,
        rightHandY + Math.sin(angle) * 18 - 10 + fingerExtend,
        4, 12 + fingerExtend, angle, 0, Math.PI * 2
      );
      ctx.fill();
    }

    // Thumb (right hand)
    ctx.beginPath();
    ctx.ellipse(rightHandX - 15, rightHandY + 5, 5, 10, -0.5, 0, Math.PI * 2);
    ctx.fill();

    frameRef.current++;
    animationRef.current = requestAnimationFrame(drawAvatar);
  }, [pose, customization]);

  // Handle sentence signing animation
  useEffect(() => {
    if (isResponding && currentSentence) {
      const words = currentSentence.split(' ');
      setPose('signing');
      setWordIndex(0);

      const wordInterval = setInterval(() => {
        setWordIndex(prev => {
          if (prev >= words.length - 1) {
            clearInterval(wordInterval);
            setPose('idle');
            onResponseComplete?.();
            return prev;
          }
          return prev + 1;
        });
      }, 600);

      return () => clearInterval(wordInterval);
    }
  }, [isResponding, currentSentence, onResponseComplete]);

  useEffect(() => {
    if (currentSentence) {
      const words = currentSentence.split(' ');
      setCurrentWord(words[wordIndex] || '');
    }
  }, [wordIndex, currentSentence]);

  // Start animation loop
  useEffect(() => {
    drawAvatar();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [drawAvatar]);

  return (
    <div className={`relative bg-card rounded-2xl border border-border overflow-hidden ${isFullscreen ? 'fixed inset-4 z-50' : ''}`}>
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between p-4 bg-gradient-to-b from-black/40 to-transparent">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
            <User className="w-4 h-4 text-white" />
          </div>
          <div>
            <p className="text-sm font-medium text-white">SonZo Avatar</p>
            <p className="text-xs text-white/60">{language} Interpreter</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowCustomization(!showCustomization)}
            className={`p-2 rounded-lg transition-colors ${showCustomization ? 'bg-primary/20 text-primary' : 'bg-white/10 text-white/70 hover:bg-white/20'}`}
          >
            <Palette className="w-4 h-4" />
          </button>
          <button
            onClick={() => setIsMuted(!isMuted)}
            className="p-2 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 transition-colors"
          >
            {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
          </button>
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-2 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 transition-colors"
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Customization Panel */}
      {showCustomization && (
        <div className="absolute top-16 right-4 z-20 w-56 p-4 rounded-xl bg-card/95 backdrop-blur border border-border shadow-xl">
          <h4 className="text-sm font-semibold mb-3">Customize Avatar</h4>
          
          <div className="space-y-4">
            <div>
              <p className="text-xs text-muted-foreground mb-2">Skin Tone</p>
              <div className="flex gap-1">
                {skinTones.map(tone => (
                  <button
                    key={tone}
                    onClick={() => setCustomization(prev => ({ ...prev, skinTone: tone }))}
                    className={`w-6 h-6 rounded-full border-2 transition-transform hover:scale-110 ${
                      customization.skinTone === tone ? 'border-primary scale-110' : 'border-transparent'
                    }`}
                    style={{ backgroundColor: tone }}
                  />
                ))}
              </div>
            </div>

            <div>
              <p className="text-xs text-muted-foreground mb-2">Hair Color</p>
              <div className="flex gap-1">
                {hairColors.map(color => (
                  <button
                    key={color}
                    onClick={() => setCustomization(prev => ({ ...prev, hairColor: color }))}
                    className={`w-6 h-6 rounded-full border-2 transition-transform hover:scale-110 ${
                      customization.hairColor === color ? 'border-primary scale-110' : 'border-transparent'
                    }`}
                    style={{ backgroundColor: color }}
                  />
                ))}
              </div>
            </div>

            <div>
              <p className="text-xs text-muted-foreground mb-2">Shirt Color</p>
              <div className="flex gap-1">
                {shirtColors.map(color => (
                  <button
                    key={color}
                    onClick={() => setCustomization(prev => ({ ...prev, shirtColor: color }))}
                    className={`w-6 h-6 rounded-full border-2 transition-transform hover:scale-110 ${
                      customization.shirtColor === color ? 'border-primary scale-110' : 'border-transparent'
                    }`}
                    style={{ backgroundColor: color }}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Avatar Canvas */}
      <div className="aspect-square bg-gradient-to-b from-slate-900 to-slate-800">
        <canvas
          ref={canvasRef}
          width={400}
          height={400}
          className="w-full h-full"
        />
      </div>

      {/* Current Word/Sentence Display */}
      <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent">
        {isResponding && currentSentence ? (
          <div className="text-center">
            <p className="text-xs text-white/60 mb-1">Signing:</p>
            <p className="text-lg font-medium text-white">
              {currentSentence.split(' ').map((word, i) => (
                <span
                  key={i}
                  className={`inline-block mx-1 transition-all ${
                    i === wordIndex 
                      ? 'text-primary scale-110 font-bold' 
                      : i < wordIndex 
                        ? 'text-white/40' 
                        : 'text-white/70'
                  }`}
                >
                  {word}
                </span>
              ))}
            </p>
          </div>
        ) : (
          <div className="text-center">
            <p className="text-sm text-white/60">Ready to respond in {language}</p>
          </div>
        )}
      </div>

      {/* Status Indicator */}
      <div className={`absolute top-4 left-4 flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${
        isResponding 
          ? 'bg-green-500/20 text-green-400' 
          : 'bg-white/10 text-white/60'
      }`}>
        <div className={`w-2 h-2 rounded-full ${isResponding ? 'bg-green-500 status-processing' : 'bg-white/40'}`} />
        {isResponding ? 'Signing' : 'Idle'}
      </div>
    </div>
  );
};

export default Avatar3D;
