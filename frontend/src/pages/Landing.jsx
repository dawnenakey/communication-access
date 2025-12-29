import { useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";
import { Hand, Video, BookOpen, History, Volume2, ArrowRight, Zap, Users, Shield } from "lucide-react";

export default function Landing() {
  const navigate = useNavigate();

  // REMINDER: DO NOT HARDCODE THE URL, OR ADD ANY FALLBACKS OR REDIRECT URLS, THIS BREAKS THE AUTH
  const handleLogin = () => {
    const redirectUrl = window.location.origin + "/dashboard";
    window.location.href = `https://auth.emergentagent.com/?redirect=${encodeURIComponent(redirectUrl)}`;
  };

  const features = [
    {
      icon: <Hand className="w-6 h-6" />,
      title: "Real-time Recognition",
      description: "Use your webcam to instantly recognize ASL signs and convert them to text",
      color: "text-cyan-400",
      bg: "bg-cyan-400/10",
    },
    {
      icon: <Video className="w-6 h-6" />,
      title: "Video Upload",
      description: "Upload videos of ASL conversations for batch translation",
      color: "text-pink-400",
      bg: "bg-pink-400/10",
    },
    {
      icon: <Volume2 className="w-6 h-6" />,
      title: "Text-to-Speech",
      description: "Hear the translated text spoken aloud for better understanding",
      color: "text-amber-400",
      bg: "bg-amber-400/10",
    },
    {
      icon: <BookOpen className="w-6 h-6" />,
      title: "Custom Dictionary",
      description: "Build your own sign dictionary with images and meanings",
      color: "text-emerald-400",
      bg: "bg-emerald-400/10",
    },
  ];

  const stats = [
    { value: "500+", label: "Signs in Dictionary" },
    { value: "Real-time", label: "Recognition Speed" },
    { value: "99%", label: "Accuracy Rate" },
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        {/* Background gradient */}
        <div className="absolute inset-0 hero-gradient" />
        
        {/* Navigation */}
        <nav className="relative z-10 flex items-center justify-between p-6 lg:px-12">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-500 flex items-center justify-center">
              <Hand className="w-5 h-5 text-black" />
            </div>
            <span className="font-heading font-bold text-xl tracking-tight">SignSync AI</span>
          </div>
          
          <Button 
            onClick={handleLogin}
            data-testid="login-btn"
            className="h-11 px-6 rounded-full bg-white/10 hover:bg-white/20 border border-white/10 backdrop-blur-sm transition-colors"
          >
            Sign In
          </Button>
        </nav>

        {/* Hero Content */}
        <div className="relative z-10 px-6 lg:px-12 py-20 lg:py-32">
          <div className="max-w-6xl mx-auto">
            <div className="grid lg:grid-cols-2 gap-12 items-center">
              {/* Left Column - Text */}
              <div className="space-y-8 animate-fade-in-up">
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cyan-400/10 border border-cyan-400/20">
                  <Zap className="w-4 h-4 text-cyan-400" />
                  <span className="text-sm text-cyan-400 font-medium">Powered by MediaPipe AI</span>
                </div>
                
                <h1 className="font-heading font-extrabold text-5xl md:text-7xl tracking-tight leading-none">
                  Bridge the
                  <span className="block bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
                    Communication Gap
                  </span>
                </h1>
                
                <p className="text-lg md:text-xl text-muted-foreground leading-relaxed max-w-lg">
                  Real-time American Sign Language recognition and translation. 
                  Convert signs to text, speech, and learn ASL with our intelligent platform.
                </p>

                <div className="flex flex-col sm:flex-row gap-4">
                  <Button 
                    onClick={handleLogin}
                    data-testid="get-started-btn"
                    className="h-14 px-8 rounded-full bg-primary text-primary-foreground font-bold text-lg btn-scale hover:shadow-[0_0_30px_rgba(34,211,238,0.4)]"
                  >
                    Get Started Free
                    <ArrowRight className="w-5 h-5 ml-2" />
                  </Button>
                  
                  <Button 
                    variant="outline"
                    data-testid="learn-more-btn"
                    className="h-14 px-8 rounded-full border-white/10 hover:bg-white/5 font-semibold text-lg"
                    onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
                  >
                    Learn More
                  </Button>
                </div>

                {/* Stats */}
                <div className="flex gap-8 pt-4">
                  {stats.map((stat, index) => (
                    <div key={index} className="text-center">
                      <div className="font-heading font-bold text-2xl text-cyan-400">{stat.value}</div>
                      <div className="text-sm text-muted-foreground">{stat.label}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Right Column - Hero Image */}
              <div className="relative animate-fade-in-up stagger-2">
                <div className="relative rounded-3xl overflow-hidden glass-card p-2">
                  <div className="absolute inset-0 bg-gradient-to-tr from-cyan-500/20 via-transparent to-purple-500/20 rounded-3xl" />
                  <img 
                    src="https://images.unsplash.com/photo-1733370446176-cf060c668a28?crop=entropy&cs=srgb&fm=jpg&q=85&w=800"
                    alt="ASL Communication"
                    className="rounded-2xl w-full aspect-[4/3] object-cover"
                  />
                  
                  {/* Floating badge */}
                  <div className="absolute bottom-6 left-6 glass px-4 py-2 rounded-full flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                    <span className="text-sm font-medium">Live Recognition Active</span>
                  </div>
                </div>
                
                {/* Decorative elements */}
                <div className="absolute -top-4 -right-4 w-24 h-24 bg-cyan-400/20 rounded-full blur-2xl" />
                <div className="absolute -bottom-8 -left-8 w-32 h-32 bg-purple-400/20 rounded-full blur-3xl" />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <section id="features" className="px-6 lg:px-12 py-24 bg-gradient-to-b from-background to-card/30">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16 animate-fade-in-up">
            <h2 className="font-heading font-bold text-3xl md:text-5xl tracking-tight mb-4">
              Everything You Need
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Comprehensive tools for ASL translation, learning, and communication
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {features.map((feature, index) => (
              <div 
                key={index}
                className={`glass-card p-8 card-lift animate-fade-in-up stagger-${index + 1}`}
                data-testid={`feature-card-${index}`}
              >
                <div className={`w-14 h-14 rounded-2xl ${feature.bg} flex items-center justify-center mb-6`}>
                  <span className={feature.color}>{feature.icon}</span>
                </div>
                <h3 className="font-heading font-bold text-xl mb-3">{feature.title}</h3>
                <p className="text-muted-foreground leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="px-6 lg:px-12 py-24">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="font-heading font-bold text-3xl md:text-5xl tracking-tight mb-4">
              How It Works
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Start translating ASL in three simple steps
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              { step: "01", title: "Enable Camera", desc: "Grant camera access to start real-time recognition" },
              { step: "02", title: "Sign Away", desc: "Perform ASL signs in front of your camera" },
              { step: "03", title: "Get Translation", desc: "See instant text translation and hear it spoken" },
            ].map((item, index) => (
              <div key={index} className="relative text-center">
                <div className="w-20 h-20 rounded-full bg-gradient-to-br from-cyan-400/20 to-blue-500/20 border border-cyan-400/30 flex items-center justify-center mx-auto mb-6">
                  <span className="font-mono font-bold text-2xl text-cyan-400">{item.step}</span>
                </div>
                <h3 className="font-heading font-bold text-xl mb-2">{item.title}</h3>
                <p className="text-muted-foreground">{item.desc}</p>
                
                {index < 2 && (
                  <div className="hidden md:block absolute top-10 left-[60%] w-[80%] h-px bg-gradient-to-r from-cyan-400/50 to-transparent" />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="px-6 lg:px-12 py-24">
        <div className="max-w-4xl mx-auto">
          <div className="glass rounded-3xl p-12 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 via-transparent to-purple-500/10" />
            
            <div className="relative z-10">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-cyan-400 to-blue-500 flex items-center justify-center mx-auto mb-6">
                <Users className="w-8 h-8 text-black" />
              </div>
              
              <h2 className="font-heading font-bold text-3xl md:text-4xl tracking-tight mb-4">
                Ready to Start Communicating?
              </h2>
              <p className="text-lg text-muted-foreground mb-8 max-w-xl mx-auto">
                Join thousands of users who are breaking communication barriers with SignSync AI
              </p>
              
              <Button 
                onClick={handleLogin}
                data-testid="cta-get-started-btn"
                className="h-14 px-10 rounded-full bg-primary text-primary-foreground font-bold text-lg btn-scale"
              >
                Get Started Now
                <ArrowRight className="w-5 h-5 ml-2" />
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-6 lg:px-12 py-12 border-t border-white/5">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-500 flex items-center justify-center">
              <Hand className="w-4 h-4 text-black" />
            </div>
            <span className="font-heading font-bold">SignSync AI</span>
          </div>
          
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Shield className="w-4 h-4" />
            <span>Your privacy is protected. Camera data stays on your device.</span>
          </div>
          
          <p className="text-sm text-muted-foreground">
            © 2025 SignSync AI. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}
