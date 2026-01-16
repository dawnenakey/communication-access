import { useNavigate } from "react-router-dom";
import { useEffect, useRef, useState } from "react";
import { Button } from "../components/ui/button";
import {
  Hand, Video, BookOpen, Volume2, ArrowRight, Zap, Users, Shield,
  Brain, Sparkles, Globe, MessageCircle, Mic, Eye, Accessibility
} from "lucide-react";

// Check for reduced motion preference
const usePrefersReducedMotion = () => {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReducedMotion(mediaQuery.matches);

    const handler = (e) => setPrefersReducedMotion(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  return prefersReducedMotion;
};

// Video background with AI/ML visuals
const VideoBackground = ({ reducedMotion }) => {
  const videoRef = useRef(null);

  // AI/ML themed video URLs (royalty-free)
  const videoSources = [
    "https://assets.mixkit.co/videos/preview/mixkit-digital-animation-of-futuristic-devices-99786-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-abstract-technology-network-connections-27866-large.mp4"
  ];

  if (reducedMotion) {
    // Static gradient fallback for reduced motion
    return (
      <div
        className="absolute inset-0 bg-gradient-to-br from-gray-900 via-red-950/20 to-black"
        aria-hidden="true"
      />
    );
  }

  return (
    <div className="absolute inset-0 overflow-hidden" aria-hidden="true">
      <video
        ref={videoRef}
        autoPlay
        loop
        muted
        playsInline
        className="absolute w-full h-full object-cover opacity-20"
        poster="https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=1920&q=80"
      >
        <source src={videoSources[0]} type="video/mp4" />
      </video>
      <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-black/80 to-black" />
    </div>
  );
};

// Animated particle background component (respects reduced motion)
const ParticleBackground = ({ reducedMotion }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (reducedMotion) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let animationFrameId;
    let particles = [];

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    resize();
    window.addEventListener('resize', resize);

    // Create particles
    for (let i = 0; i < 50; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        size: Math.random() * 2 + 1,
        opacity: Math.random() * 0.5 + 0.2
      });
    }

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.forEach((p, i) => {
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

        // Draw particle
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(220, 38, 38, ${p.opacity})`;
        ctx.fill();

        // Draw connections
        particles.forEach((p2, j) => {
          if (i === j) return;
          const dx = p.x - p2.x;
          const dy = p.y - p2.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 150) {
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = `rgba(220, 38, 38, ${0.1 * (1 - dist / 150)})`;
            ctx.stroke();
          }
        });
      });

      animationFrameId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animationFrameId);
    };
  }, [reducedMotion]);

  if (reducedMotion) return null;

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none opacity-60"
      style={{ zIndex: 1 }}
      aria-hidden="true"
    />
  );
};

// Floating orbs component (respects reduced motion)
const FloatingOrbs = ({ reducedMotion }) => (
  <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden="true">
    <div className={`absolute top-20 left-10 w-72 h-72 bg-red-500/20 rounded-full blur-[100px] ${!reducedMotion ? 'animate-pulse' : ''}`} />
    <div className={`absolute top-40 right-20 w-96 h-96 bg-orange-500/15 rounded-full blur-[120px] ${!reducedMotion ? 'animate-pulse' : ''}`} style={{ animationDelay: '1s' }} />
    <div className={`absolute bottom-20 left-1/3 w-80 h-80 bg-red-600/10 rounded-full blur-[100px] ${!reducedMotion ? 'animate-pulse' : ''}`} style={{ animationDelay: '2s' }} />
  </div>
);

export default function Landing() {
  const navigate = useNavigate();
  const reducedMotion = usePrefersReducedMotion();

  const handleLogin = () => {
    navigate("/dashboard");
  };

  const features = [
    {
      icon: <Brain className="w-6 h-6" />,
      title: "AI-Powered Recognition",
      description: "Advanced neural networks trained on thousands of ASL signs for real-time recognition",
      color: "text-red-400",
      bg: "bg-red-400/10",
      border: "border-red-400/20",
    },
    {
      icon: <MessageCircle className="w-6 h-6" />,
      title: "ASL ↔ English Translation",
      description: "Bidirectional translation powered by AWS Bedrock for natural conversations",
      color: "text-orange-400",
      bg: "bg-orange-400/10",
      border: "border-orange-400/20",
    },
    {
      icon: <Video className="w-6 h-6" />,
      title: "Real-time Video Processing",
      description: "Landmark extraction at 30fps using MediaPipe for fluid recognition",
      color: "text-amber-400",
      bg: "bg-amber-400/10",
      border: "border-amber-400/20",
    },
    {
      icon: <Globe className="w-6 h-6" />,
      title: "Universal Access",
      description: "Breaking communication barriers between Deaf and hearing communities",
      color: "text-red-300",
      bg: "bg-red-300/10",
      border: "border-red-300/20",
    },
  ];

  const stats = [
    { value: "100+", label: "Signs Recognized", icon: <Hand className="w-4 h-4" /> },
    { value: "53ms", label: "Latency", icon: <Zap className="w-4 h-4" /> },
    { value: "22%", label: "Model Accuracy", icon: <Eye className="w-4 h-4" /> },
  ];

  const capabilities = [
    { icon: <Hand />, label: "ASL Recognition" },
    { icon: <Mic />, label: "Voice Output" },
    { icon: <MessageCircle />, label: "Text Translation" },
    { icon: <Video />, label: "Video Analysis" },
  ];

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white overflow-hidden" role="main">
      {/* Skip to main content link for accessibility */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-red-500 focus:text-white focus:rounded-lg"
      >
        Skip to main content
      </a>

      {/* Animated Background - respects reduced motion */}
      <VideoBackground reducedMotion={reducedMotion} />
      <ParticleBackground reducedMotion={reducedMotion} />
      <FloatingOrbs reducedMotion={reducedMotion} />

      {/* Navigation */}
      <nav className="relative z-20 flex items-center justify-between p-6 lg:px-12" role="navigation" aria-label="Main navigation">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-red-500 to-orange-500 flex items-center justify-center shadow-lg shadow-red-500/25">
              <span className="font-bold text-xl text-white">S</span>
            </div>
            <div className="absolute -inset-1 rounded-2xl bg-gradient-to-br from-red-500 to-orange-500 blur opacity-30" />
          </div>
          <div>
            <span className="font-heading font-bold text-2xl tracking-tight">
              Son<span className="text-red-500">Zo</span>
            </span>
            <span className="block text-xs text-gray-500 -mt-1">AI Communication</span>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <a href="https://demo.sonzo.io" className="text-sm text-gray-400 hover:text-white transition-colors">
            Live Demo
          </a>
          <Button
            onClick={handleLogin}
            className="h-11 px-6 rounded-full bg-gradient-to-r from-red-500 to-orange-500 hover:from-red-600 hover:to-orange-600 text-white font-semibold shadow-lg shadow-red-500/25 transition-all hover:shadow-red-500/40"
          >
            Get Started
          </Button>
        </div>
      </nav>

      {/* Hero Section */}
      <section id="main-content" className="relative z-10 px-6 lg:px-12 py-16 lg:py-24" aria-labelledby="hero-heading">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            {/* Left Column - Text */}
            <div className="space-y-8">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-red-500/10 border border-red-500/20">
                <Sparkles className="w-4 h-4 text-red-400" />
                <span className="text-sm text-red-400 font-medium">Patent Pending Technology</span>
              </div>

              <h1 id="hero-heading" className="font-heading font-extrabold text-5xl md:text-7xl tracking-tight leading-[1.1]">
                <span className="text-white">AI-Powered</span>
                <br />
                <span className="bg-gradient-to-r from-red-400 via-orange-400 to-amber-400 bg-clip-text text-transparent">
                  Sign Language
                </span>
                <br />
                <span className="text-white">Communication</span>
              </h1>

              <p className="text-lg md:text-xl text-gray-400 leading-relaxed max-w-lg">
                Breaking barriers between Deaf and hearing communities with
                real-time ASL recognition, translation, and bidirectional communication.
              </p>

              {/* Capability Pills */}
              <div className="flex flex-wrap gap-3">
                {capabilities.map((cap, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 text-sm text-gray-300"
                  >
                    <span className="text-red-400">{cap.icon}</span>
                    {cap.label}
                  </div>
                ))}
              </div>

              <div className="flex flex-col sm:flex-row gap-4 pt-4">
                <Button
                  onClick={handleLogin}
                  className="h-14 px-8 rounded-full bg-gradient-to-r from-red-500 to-orange-500 hover:from-red-600 hover:to-orange-600 text-white font-bold text-lg shadow-lg shadow-red-500/25 transition-all hover:shadow-red-500/40 hover:scale-[1.02]"
                >
                  Start Translating
                  <ArrowRight className="w-5 h-5 ml-2" />
                </Button>

                <a href="https://demo.sonzo.io" target="_blank" rel="noopener noreferrer">
                  <Button
                    variant="outline"
                    className="h-14 px-8 rounded-full border-white/20 hover:bg-white/5 font-semibold text-lg w-full"
                  >
                    <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse mr-2" />
                    Try Live Demo
                  </Button>
                </a>
              </div>

              {/* Stats */}
              <div className="flex gap-8 pt-6 border-t border-white/10">
                {stats.map((stat, index) => (
                  <div key={index} className="text-center">
                    <div className="flex items-center justify-center gap-2 mb-1">
                      <span className="text-red-400">{stat.icon}</span>
                      <span className="font-heading font-bold text-2xl text-white">{stat.value}</span>
                    </div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider">{stat.label}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Right Column - Visual */}
            <div className="relative">
              {/* Main visual card */}
              <div className="relative rounded-3xl overflow-hidden bg-gradient-to-br from-gray-900 to-black border border-white/10 p-1">
                <div className="relative rounded-[22px] overflow-hidden bg-black">
                  {/* Demo preview image */}
                  <img
                    src="https://images.unsplash.com/photo-1531746790731-6c087fecd65a?w=800&q=80"
                    alt="AI Sign Language Recognition"
                    className="w-full aspect-[4/3] object-cover opacity-80"
                  />

                  {/* Overlay with recognition UI mockup */}
                  <div className="absolute inset-0 bg-gradient-to-t from-black via-black/50 to-transparent">
                    <div className="absolute bottom-0 left-0 right-0 p-6">
                      {/* Recognition result mockup */}
                      <div className="bg-black/80 backdrop-blur-xl rounded-2xl border border-white/10 p-4">
                        <div className="flex items-center justify-between mb-3">
                          <span className="text-xs text-gray-500 uppercase tracking-wider">Recognized Sign</span>
                          <span className="text-xs text-green-400 flex items-center gap-1">
                            <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                            Live
                          </span>
                        </div>
                        <div className="flex items-end justify-between">
                          <div>
                            <span className="text-3xl font-bold text-white">HELLO</span>
                            <span className="block text-sm text-gray-400 mt-1">→ "Hello, nice to meet you"</span>
                          </div>
                          <div className="text-right">
                            <span className="text-2xl font-bold text-red-400">97.5%</span>
                            <span className="block text-xs text-gray-500">confidence</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Corner decorations */}
                  <div className="absolute top-4 left-4 w-8 h-8 border-l-2 border-t-2 border-red-400/50 rounded-tl-lg" />
                  <div className="absolute top-4 right-4 w-8 h-8 border-r-2 border-t-2 border-red-400/50 rounded-tr-lg" />
                  <div className="absolute bottom-4 left-4 w-8 h-8 border-l-2 border-b-2 border-red-400/50 rounded-bl-lg" />
                  <div className="absolute bottom-4 right-4 w-8 h-8 border-r-2 border-b-2 border-red-400/50 rounded-br-lg" />
                </div>
              </div>

              {/* Floating tech badges */}
              <div className="absolute -top-4 -right-4 glass px-4 py-2 rounded-full flex items-center gap-2 shadow-xl">
                <Brain className="w-4 h-4 text-red-400" />
                <span className="text-sm font-medium">LSTM Neural Network</span>
              </div>

              <div className="absolute -bottom-4 -left-4 glass px-4 py-2 rounded-full flex items-center gap-2 shadow-xl">
                <Zap className="w-4 h-4 text-amber-400" />
                <span className="text-sm font-medium">AWS Bedrock</span>
              </div>

              {/* Glow effects */}
              <div className="absolute -top-8 -right-8 w-40 h-40 bg-red-500/30 rounded-full blur-3xl" />
              <div className="absolute -bottom-8 -left-8 w-40 h-40 bg-orange-500/20 rounded-full blur-3xl" />
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <section className="relative z-10 px-6 lg:px-12 py-24">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 mb-6">
              <Brain className="w-4 h-4 text-red-400" />
              <span className="text-sm text-gray-400">Powered by Machine Learning</span>
            </div>
            <h2 className="font-heading font-bold text-4xl md:text-5xl tracking-tight mb-4">
              Complete Communication
              <span className="block text-red-400">Suite</span>
            </h2>
            <p className="text-lg text-gray-400 max-w-2xl mx-auto">
              Everything you need for seamless communication between ASL and English
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {features.map((feature, index) => (
              <div
                key={index}
                className={`group relative rounded-3xl bg-gradient-to-br from-gray-900/80 to-black/80 backdrop-blur-xl border ${feature.border} p-8 transition-all duration-300 hover:scale-[1.02] hover:shadow-xl`}
              >
                <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-red-500/5 to-orange-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />

                <div className={`w-14 h-14 rounded-2xl ${feature.bg} flex items-center justify-center mb-6 relative`}>
                  <span className={feature.color}>{feature.icon}</span>
                </div>

                <h3 className="font-heading font-bold text-xl mb-3 text-white">{feature.title}</h3>
                <p className="text-gray-400 leading-relaxed">{feature.description}</p>

                {/* Corner accent */}
                <div className={`absolute top-0 right-0 w-20 h-20 ${feature.bg} blur-3xl opacity-50`} />
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="relative z-10 px-6 lg:px-12 py-24">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="font-heading font-bold text-4xl md:text-5xl tracking-tight mb-4">
              How It Works
            </h2>
            <p className="text-lg text-gray-400">Three steps to accessible communication</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                step: "01",
                title: "Capture",
                desc: "Enable your camera to capture ASL signs in real-time",
                icon: <Video className="w-6 h-6" />
              },
              {
                step: "02",
                title: "Process",
                desc: "Our AI extracts hand landmarks and recognizes signs instantly",
                icon: <Brain className="w-6 h-6" />
              },
              {
                step: "03",
                title: "Translate",
                desc: "Get natural English sentences from ASL gloss using AWS Bedrock",
                icon: <MessageCircle className="w-6 h-6" />
              },
            ].map((item, index) => (
              <div key={index} className="relative text-center group">
                <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-red-500/20 to-orange-500/20 border border-red-500/30 flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform">
                  <span className="text-red-400">{item.icon}</span>
                </div>

                <span className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-2 font-mono text-xs text-red-400 bg-red-400/10 px-2 py-1 rounded-full">
                  {item.step}
                </span>

                <h3 className="font-heading font-bold text-xl mb-2 text-white">{item.title}</h3>
                <p className="text-gray-400">{item.desc}</p>

                {index < 2 && (
                  <div className="hidden md:block absolute top-12 left-[60%] w-[80%] h-px bg-gradient-to-r from-red-500/50 to-transparent" />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 px-6 lg:px-12 py-24">
        <div className="max-w-4xl mx-auto">
          <div className="relative rounded-3xl bg-gradient-to-br from-red-500/10 to-orange-500/10 backdrop-blur-xl border border-red-500/20 p-12 text-center overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-red-500/5 via-transparent to-orange-500/5" />

            <div className="relative z-10">
              <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-red-500 to-orange-500 flex items-center justify-center mx-auto mb-6 shadow-lg shadow-red-500/25">
                <Users className="w-10 h-10 text-white" />
              </div>

              <h2 className="font-heading font-bold text-3xl md:text-4xl tracking-tight mb-4 text-white">
                Ready to Bridge the Gap?
              </h2>
              <p className="text-lg text-gray-400 mb-8 max-w-xl mx-auto">
                Join us in making communication accessible for everyone
              </p>

              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button
                  onClick={handleLogin}
                  className="h-14 px-10 rounded-full bg-gradient-to-r from-red-500 to-orange-500 hover:from-red-600 hover:to-orange-600 text-white font-bold text-lg shadow-lg shadow-red-500/25"
                >
                  Get Started Free
                  <ArrowRight className="w-5 h-5 ml-2" />
                </Button>

                <a href="https://demo.sonzo.io" target="_blank" rel="noopener noreferrer">
                  <Button
                    variant="outline"
                    className="h-14 px-10 rounded-full border-white/20 hover:bg-white/5 font-semibold text-lg"
                  >
                    View Live Demo
                  </Button>
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 px-6 lg:px-12 py-12 border-t border-white/5">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-red-500 to-orange-500 flex items-center justify-center">
              <span className="font-bold text-white">S</span>
            </div>
            <div>
              <span className="font-heading font-bold text-lg">Son<span className="text-red-500">Zo</span></span>
              <span className="block text-xs text-gray-500">AI Communication</span>
            </div>
          </div>

          <div className="flex items-center gap-2 text-sm text-gray-500">
            <Shield className="w-4 h-4" />
            <span>Your privacy is protected. Camera data stays on your device.</span>
          </div>

          <p className="text-sm text-gray-500">
            © 2025 SonZo AI. Patent Pending. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}
