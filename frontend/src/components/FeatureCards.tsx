import React from 'react';
import { 
  Camera, Brain, Globe, MessageSquare, 
  Layers, Zap, Shield, Users, 
  Smartphone, Cloud, BarChart3, Sparkles
} from 'lucide-react';

const features = [
  {
    icon: Camera,
    title: 'OAK AI Camera',
    description: 'Advanced depth-sensing camera with on-device AI for precise 3D hand tracking and gesture recognition.',
    color: '#06b6d4',
    bgColor: 'bg-cyan-500/10'
  },
  {
    icon: Brain,
    title: '3D CNN + LSTM Model',
    description: 'Deep learning architecture combining spatial and temporal analysis for accurate sentence-level recognition.',
    color: '#8b5cf6',
    bgColor: 'bg-violet-500/10'
  },
  {
    icon: MessageSquare,
    title: 'Sentence Recognition',
    description: 'Understand complete sentences with context, grammar, and meaning - not just individual words or letters.',
    color: '#ec4899',
    bgColor: 'bg-pink-500/10'
  },
  {
    icon: Globe,
    title: '12+ Sign Languages',
    description: 'Support for ASL, BSL, ISL, LSF, DGS, JSL, Auslan, and more. Continuously expanding global coverage.',
    color: '#10b981',
    bgColor: 'bg-emerald-500/10'
  },
  {
    icon: Users,
    title: '3D Avatar Response',
    description: 'Customizable avatar that responds in sign language, making two-way communication seamless and natural.',
    color: '#f97316',
    bgColor: 'bg-orange-500/10'
  },
  {
    icon: Layers,
    title: 'Depth Visualization',
    description: 'Real-time 3D depth mapping shows hand positions in space for enhanced recognition accuracy.',
    color: '#6366f1',
    bgColor: 'bg-indigo-500/10'
  },
  {
    icon: Zap,
    title: 'Real-time Processing',
    description: 'Sub-50ms latency ensures natural conversation flow with instant sign language translation.',
    color: '#eab308',
    bgColor: 'bg-yellow-500/10'
  },
  {
    icon: Cloud,
    title: 'AWS Infrastructure',
    description: 'Powered by SageMaker, Lambda, and S3 for scalable, reliable cloud processing and model deployment.',
    color: '#0ea5e9',
    bgColor: 'bg-sky-500/10'
  },
  {
    icon: Shield,
    title: 'Privacy First',
    description: 'Local processing when possible, encrypted data transmission, and full GDPR compliance.',
    color: '#22c55e',
    bgColor: 'bg-green-500/10'
  },
  {
    icon: BarChart3,
    title: '94.7% Accuracy',
    description: 'Industry-leading recognition accuracy with continuous improvement through machine learning.',
    color: '#a855f7',
    bgColor: 'bg-purple-500/10'
  },
  {
    icon: Smartphone,
    title: 'Cross-Platform',
    description: 'Works on desktop, laptop, and mobile devices with any modern web browser.',
    color: '#f43f5e',
    bgColor: 'bg-rose-500/10'
  },
  {
    icon: Sparkles,
    title: 'AI-Powered Learning',
    description: 'Interactive tutorials and practice modes help users learn sign language with real-time feedback.',
    color: '#d946ef',
    bgColor: 'bg-fuchsia-500/10'
  }
];

const FeatureCards: React.FC = () => {
  return (
    <section id="features" className="py-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 mb-4">
            <Sparkles className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium text-primary">Features</span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            Powered by <span className="gradient-text">Advanced Technology</span>
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            SonZo AI combines cutting-edge hardware and AI to deliver the most accurate 
            and comprehensive sign language recognition platform available.
          </p>
        </div>

        {/* Feature Grid */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {features.map((feature, index) => (
            <div
              key={index}
              className="group p-6 bg-card rounded-2xl border border-border card-hover cursor-pointer"
            >
              {/* Icon */}
              <div className={`w-14 h-14 rounded-xl ${feature.bgColor} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                <feature.icon className="w-7 h-7" style={{ color: feature.color }} />
              </div>

              {/* Content */}
              <h3 className="text-lg font-semibold mb-2 group-hover:text-primary transition-colors">
                {feature.title}
              </h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>

        {/* Bottom CTA */}
        <div className="mt-16 text-center">
          <div className="inline-flex flex-col sm:flex-row items-center gap-4 p-6 bg-gradient-to-r from-violet-500/10 via-purple-500/10 to-fuchsia-500/10 rounded-2xl border border-primary/20">
            <div className="text-center sm:text-left">
              <h3 className="font-semibold mb-1">Ready to experience SonZo AI?</h3>
              <p className="text-sm text-muted-foreground">
                Start recognizing sign language sentences in minutes
              </p>
            </div>
            <a
              href="#recognition"
              className="btn-shine px-6 py-3 bg-primary text-primary-foreground rounded-xl font-medium hover:bg-primary/90 transition-colors whitespace-nowrap"
            >
              Try It Now
            </a>
          </div>
        </div>
      </div>
    </section>
  );
};

export default FeatureCards;
