import React, { useEffect, useRef } from 'react';
import { Play, ArrowRight, Sparkles, Zap, Globe } from 'lucide-react';

interface HeroSectionProps {
  onStartRecognition: () => void;
}

const HeroSection: React.FC<HeroSectionProps> = ({ onStartRecognition }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resizeCanvas = () => {
      canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      canvas.height = canvas.offsetHeight * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Particle system
    const particles: Array<{
      x: number;
      y: number;
      vx: number;
      vy: number;
      size: number;
      opacity: number;
    }> = [];

    for (let i = 0; i < 50; i++) {
      particles.push({
        x: Math.random() * canvas.offsetWidth,
        y: Math.random() * canvas.offsetHeight,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        size: Math.random() * 3 + 1,
        opacity: Math.random() * 0.5 + 0.2,
      });
    }

    const animate = () => {
      ctx.clearRect(0, 0, canvas.offsetWidth, canvas.offsetHeight);

      particles.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0 || p.x > canvas.offsetWidth) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.offsetHeight) p.vy *= -1;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(139, 92, 246, ${p.opacity})`;
        ctx.fill();
      });

      // Draw connections
      particles.forEach((p1, i) => {
        particles.slice(i + 1).forEach((p2) => {
          const dx = p1.x - p2.x;
          const dy = p1.y - p2.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 100) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = `rgba(139, 92, 246, ${0.1 * (1 - dist / 100)})`;
            ctx.stroke();
          }
        });
      });

      requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
    };
  }, []);

  return (
    <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 animated-gradient opacity-10" />
      
      {/* Particle Canvas */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />

      {/* Gradient Orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-violet-500/20 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-purple-500/20 rounded-full blur-3xl animate-pulse delay-1000" />
      <div className="absolute top-1/2 right-1/3 w-64 h-64 bg-fuchsia-500/20 rounded-full blur-3xl animate-pulse delay-500" />

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Content */}
          <div className="text-center lg:text-left">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 mb-6">
              <Sparkles className="w-4 h-4 text-primary" />
              <span className="text-sm font-medium text-primary">Webcam, OAK AI & Lumen Cameras Supported</span>
            </div>

            {/* Headline */}
            <h1 className="text-responsive-xl font-bold leading-tight mb-6">
              <span className="gradient-text">Sentence-Level</span>
              <br />
              Sign Language
              <br />
              Recognition
            </h1>

            {/* Subheadline */}
            <p className="text-lg sm:text-xl text-muted-foreground mb-8 max-w-xl mx-auto lg:mx-0">
              Experience real-time sign language translation with our advanced 3D CNN + LSTM model. 
              Communicate in full sentences across 12+ global sign languages.
            </p>

            {/* Stats */}
            <div className="flex flex-wrap justify-center lg:justify-start gap-6 mb-8">
              <div className="text-center">
                <p className="text-3xl font-bold gradient-text">94.7%</p>
                <p className="text-sm text-muted-foreground">Accuracy</p>
              </div>
              <div className="text-center">
                <p className="text-3xl font-bold gradient-text">12+</p>
                <p className="text-sm text-muted-foreground">Languages</p>
              </div>
              <div className="text-center">
                <p className="text-3xl font-bold gradient-text">&lt;50ms</p>
                <p className="text-sm text-muted-foreground">Latency</p>
              </div>
            </div>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
              <button
                onClick={onStartRecognition}
                className="btn-shine inline-flex items-center justify-center gap-2 px-8 py-4 bg-gradient-to-r from-violet-600 to-purple-600 text-white font-semibold rounded-xl shadow-lg shadow-violet-500/30 hover:shadow-violet-500/50 transition-all hover:-translate-y-0.5"
              >
                <Play className="w-5 h-5" />
                Start Recognition
              </button>
              <a
                href="#learn"
                className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-muted hover:bg-muted/80 font-semibold rounded-xl transition-colors"
              >
                Learn More
                <ArrowRight className="w-5 h-5" />
              </a>
            </div>
          </div>

          {/* Right Content - Hand Landmark Visualization */}
          <div className="relative">
            <div className="relative w-full aspect-square max-w-lg mx-auto">
              {/* Outer Ring */}
              <div className="absolute inset-0 rounded-full border-2 border-dashed border-primary/30 animate-spin" style={{ animationDuration: '20s' }} />
              
              {/* Middle Ring */}
              <div className="absolute inset-8 rounded-full border border-primary/20" />
              
              {/* Inner Glow */}
              <div className="absolute inset-16 rounded-full bg-gradient-to-br from-violet-500/20 to-purple-500/20 blur-xl" />

              {/* Hand SVG with Landmarks */}
              <svg
                viewBox="0 0 200 200"
                className="absolute inset-0 w-full h-full p-12"
              >
                {/* Hand outline */}
                <path
                  d="M100 180 L100 120 L85 90 L85 50 L95 50 L95 85 L100 85 L100 40 L110 40 L110 85 L115 85 L115 45 L125 45 L125 85 L130 85 L130 55 L140 55 L140 100 L145 95 L155 100 L140 120 L140 140 L120 180 Z"
                  fill="none"
                  stroke="url(#handGradient)"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                
                {/* Gradient Definition */}
                <defs>
                  <linearGradient id="handGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#8b5cf6" />
                    <stop offset="100%" stopColor="#a855f7" />
                  </linearGradient>
                </defs>

                {/* Landmark Points - 21 points */}
                {[
                  { x: 100, y: 180, delay: 0 },    // Wrist
                  { x: 100, y: 150, delay: 0.1 },  // Palm
                  { x: 100, y: 120, delay: 0.2 },  // Palm 2
                  { x: 85, y: 90, delay: 0.3 },    // Thumb base
                  { x: 85, y: 70, delay: 0.4 },    // Thumb mid
                  { x: 85, y: 50, delay: 0.5 },    // Thumb tip
                  { x: 100, y: 85, delay: 0.6 },   // Index base
                  { x: 100, y: 60, delay: 0.7 },   // Index mid
                  { x: 100, y: 40, delay: 0.8 },   // Index tip
                  { x: 115, y: 85, delay: 0.9 },   // Middle base
                  { x: 115, y: 60, delay: 1.0 },   // Middle mid
                  { x: 115, y: 45, delay: 1.1 },   // Middle tip
                  { x: 130, y: 85, delay: 1.2 },   // Ring base
                  { x: 130, y: 65, delay: 1.3 },   // Ring mid
                  { x: 130, y: 55, delay: 1.4 },   // Ring tip
                  { x: 145, y: 95, delay: 1.5 },   // Pinky base
                  { x: 150, y: 97, delay: 1.6 },   // Pinky mid
                  { x: 155, y: 100, delay: 1.7 },  // Pinky tip
                ].map((point, i) => (
                  <g key={i}>
                    <circle
                      cx={point.x}
                      cy={point.y}
                      r="4"
                      fill="#8b5cf6"
                      className="landmark-point"
                      style={{ animationDelay: `${point.delay}s` }}
                    />
                    <circle
                      cx={point.x}
                      cy={point.y}
                      r="8"
                      fill="none"
                      stroke="#8b5cf6"
                      strokeWidth="1"
                      opacity="0.3"
                      className="landmark-point"
                      style={{ animationDelay: `${point.delay}s` }}
                    />
                  </g>
                ))}

                {/* Connection Lines */}
                <g stroke="#8b5cf6" strokeWidth="1" opacity="0.5">
                  <line x1="100" y1="180" x2="100" y2="120" />
                  <line x1="100" y1="120" x2="85" y2="90" />
                  <line x1="85" y1="90" x2="85" y2="50" />
                  <line x1="100" y1="120" x2="100" y2="40" />
                  <line x1="100" y1="120" x2="115" y2="45" />
                  <line x1="100" y1="120" x2="130" y2="55" />
                  <line x1="140" y1="100" x2="155" y2="100" />
                </g>
              </svg>

              {/* Floating Labels */}
              <div className="absolute top-4 right-4 px-3 py-1.5 rounded-lg bg-card/80 backdrop-blur border border-border text-xs font-medium">
                <span className="text-primary">21</span> Landmarks
              </div>
              <div className="absolute bottom-4 left-4 px-3 py-1.5 rounded-lg bg-card/80 backdrop-blur border border-border text-xs font-medium">
                <span className="text-primary">3D</span> Depth Data
              </div>
            </div>

            {/* Feature Pills */}
            <div className="absolute -left-4 top-1/4 px-4 py-2 rounded-xl bg-card/80 backdrop-blur border border-border shadow-lg float">
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-yellow-500" />
                <span className="text-sm font-medium">Real-time</span>
              </div>
            </div>
            <div className="absolute -right-4 bottom-1/4 px-4 py-2 rounded-xl bg-card/80 backdrop-blur border border-border shadow-lg float" style={{ animationDelay: '1s' }}>
              <div className="flex items-center gap-2">
                <Globe className="w-4 h-4 text-blue-500" />
                <span className="text-sm font-medium">12+ Languages</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Scroll Indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-muted-foreground">
        <span className="text-xs">Scroll to explore</span>
        <div className="w-6 h-10 rounded-full border-2 border-muted-foreground/30 flex items-start justify-center p-2">
          <div className="w-1.5 h-3 bg-primary rounded-full animate-bounce" />
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
