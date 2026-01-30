import React, { useState, useEffect } from 'react';
import {
  Activity, Cpu, Database, Cloud, Zap, 
  Mic, FileText, Video,
  ArrowRight, CheckCircle, Clock,
  Layers, GitBranch, Box, Workflow
} from 'lucide-react';

interface AWSMetricsProps {
  isProcessing: boolean;
  currentStep: 'idle' | 'transcribe' | 'gloss' | 'lookup' | 'render';
  inputText?: string;
  aslGloss?: string;
  signsFound?: number;
  totalSigns?: number;
  processingTime?: number;
}

const AWSMetrics: React.FC<AWSMetricsProps> = ({
  isProcessing,
  currentStep,
  inputText = '',
  aslGloss = '',
  signsFound = 0,
  totalSigns = 107,
  processingTime = 0
}) => {
  const [metrics, setMetrics] = useState({
    apiCalls: 0,
    latency: 0,
    signLookups: 0,
    cacheHits: 0
  });

  useEffect(() => {
    if (isProcessing) {
      const interval = setInterval(() => {
        setMetrics(prev => ({
          apiCalls: prev.apiCalls + Math.floor(Math.random() * 3),
          latency: 50 + Math.floor(Math.random() * 100),
          signLookups: prev.signLookups + 1,
          cacheHits: prev.cacheHits + (Math.random() > 0.3 ? 1 : 0)
        }));
      }, 500);
      return () => clearInterval(interval);
    }
  }, [isProcessing]);

  const steps = [
    { id: 'transcribe', label: 'Speech-to-Text', icon: Mic, service: 'Amazon Transcribe' },
    { id: 'gloss', label: 'ASL Gloss', icon: FileText, service: 'Amazon Bedrock' },
    { id: 'lookup', label: 'Sign Lookup', icon: Database, service: 'DynamoDB' },
    { id: 'render', label: 'Avatar Render', icon: Video, service: 'RTMPose' }
  ];

  const getStepStatus = (stepId: string) => {
    const stepOrder = ['idle', 'transcribe', 'gloss', 'lookup', 'render'];
    const currentIndex = stepOrder.indexOf(currentStep);
    const stepIndex = stepOrder.indexOf(stepId);
    
    if (stepIndex < currentIndex) return 'complete';
    if (stepIndex === currentIndex) return 'active';
    return 'pending';
  };

  return (
    <div className="bg-card rounded-xl border border-border p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-orange-500 to-orange-600 flex items-center justify-center">
            <Cloud className="w-4 h-4 text-white" />
          </div>
          <div>
            <h3 className="text-sm font-bold">GenASL Pipeline</h3>
            <p className="text-[10px] text-muted-foreground">AWS Architecture Metrics</p>
          </div>
        </div>
        <div className={`px-2 py-1 rounded-full text-[10px] font-semibold ${
          isProcessing 
            ? 'bg-green-500/20 text-green-500 border border-green-500/30' 
            : 'bg-muted text-muted-foreground'
        }`}>
          {isProcessing ? 'Processing' : 'Ready'}
        </div>
      </div>

      {/* Pipeline Visualization */}
      <div className="bg-muted/30 rounded-lg p-3">
        <p className="text-[10px] font-medium text-muted-foreground mb-3 uppercase tracking-wider">Processing Pipeline</p>
        <div className="flex items-center justify-between">
          {steps.map((step, index) => {
            const status = getStepStatus(step.id);
            const Icon = step.icon;
            
            return (
              <React.Fragment key={step.id}>
                <div className="flex flex-col items-center gap-1.5">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-300 ${
                    status === 'complete' 
                      ? 'bg-green-500/20 text-green-500 border border-green-500/30' 
                      : status === 'active'
                        ? 'bg-primary/20 text-primary border border-primary/30 animate-pulse'
                        : 'bg-muted text-muted-foreground border border-border'
                  }`}>
                    {status === 'complete' ? (
                      <CheckCircle className="w-5 h-5" />
                    ) : (
                      <Icon className="w-5 h-5" />
                    )}
                  </div>
                  <div className="text-center">
                    <p className="text-[9px] font-medium">{step.label}</p>
                    <p className="text-[8px] text-muted-foreground">{step.service}</p>
                  </div>
                </div>
                {index < steps.length - 1 && (
                  <ArrowRight className={`w-4 h-4 mx-1 transition-colors ${
                    getStepStatus(steps[index + 1].id) !== 'pending'
                      ? 'text-primary'
                      : 'text-muted-foreground/30'
                  }`} />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Text Transformation */}
      {(inputText || aslGloss) && (
        <div className="bg-muted/30 rounded-lg p-3 space-y-2">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Text Transformation</p>
          
          {inputText && (
            <div className="flex items-start gap-2">
              <div className="w-16 shrink-0">
                <span className="text-[9px] font-medium text-blue-400 bg-blue-500/10 px-1.5 py-0.5 rounded">English</span>
              </div>
              <p className="text-xs text-foreground/80">{inputText}</p>
            </div>
          )}
          
          {aslGloss && (
            <div className="flex items-start gap-2">
              <div className="w-16 shrink-0">
                <span className="text-[9px] font-medium text-green-400 bg-green-500/10 px-1.5 py-0.5 rounded">ASL Gloss</span>
              </div>
              <p className="text-xs font-mono text-primary">{aslGloss}</p>
            </div>
          )}
        </div>
      )}

      {/* Real-time Metrics */}
      <div className="grid grid-cols-4 gap-2">
        <div className="bg-muted/30 rounded-lg p-2 text-center">
          <Zap className="w-4 h-4 mx-auto mb-1 text-yellow-500" />
          <p className="text-lg font-bold">{metrics.latency}</p>
          <p className="text-[9px] text-muted-foreground">Latency (ms)</p>
        </div>
        <div className="bg-muted/30 rounded-lg p-2 text-center">
          <Database className="w-4 h-4 mx-auto mb-1 text-blue-500" />
          <p className="text-lg font-bold">{signsFound}</p>
          <p className="text-[9px] text-muted-foreground">Signs Found</p>
        </div>
        <div className="bg-muted/30 rounded-lg p-2 text-center">
          <Activity className="w-4 h-4 mx-auto mb-1 text-green-500" />
          <p className="text-lg font-bold">{metrics.cacheHits}</p>
          <p className="text-[9px] text-muted-foreground">Cache Hits</p>
        </div>
        <div className="bg-muted/30 rounded-lg p-2 text-center">
          <Clock className="w-4 h-4 mx-auto mb-1 text-purple-500" />
          <p className="text-lg font-bold">{processingTime.toFixed(1)}</p>
          <p className="text-[9px] text-muted-foreground">Time (s)</p>
        </div>
      </div>

      {/* Architecture Diagram Mini */}
      <div className="bg-gradient-to-r from-slate-900 to-slate-800 rounded-lg p-3 text-white">
        <div className="flex items-center gap-2 mb-3">
          <Workflow className="w-4 h-4 text-orange-400" />
          <p className="text-[10px] font-medium">AWS Architecture</p>
        </div>
        
        <div className="flex items-center justify-between text-[8px]">
          {/* User Input */}
          <div className="flex flex-col items-center gap-1">
            <div className="w-8 h-8 rounded bg-blue-600/30 border border-blue-500/50 flex items-center justify-center">
              <Mic className="w-4 h-4 text-blue-400" />
            </div>
            <span className="text-blue-300">Input</span>
          </div>
          
          <ArrowRight className="w-3 h-3 text-white/30" />
          
          {/* API Gateway */}
          <div className="flex flex-col items-center gap-1">
            <div className="w-8 h-8 rounded bg-purple-600/30 border border-purple-500/50 flex items-center justify-center">
              <GitBranch className="w-4 h-4 text-purple-400" />
            </div>
            <span className="text-purple-300">API GW</span>
          </div>
          
          <ArrowRight className="w-3 h-3 text-white/30" />
          
          {/* Step Functions */}
          <div className="flex flex-col items-center gap-1">
            <div className="w-8 h-8 rounded bg-pink-600/30 border border-pink-500/50 flex items-center justify-center">
              <Layers className="w-4 h-4 text-pink-400" />
            </div>
            <span className="text-pink-300">Step Fn</span>
          </div>
          
          <ArrowRight className="w-3 h-3 text-white/30" />
          
          {/* Bedrock */}
          <div className="flex flex-col items-center gap-1">
            <div className="w-8 h-8 rounded bg-green-600/30 border border-green-500/50 flex items-center justify-center">
              <Cpu className="w-4 h-4 text-green-400" />
            </div>
            <span className="text-green-300">Bedrock</span>
          </div>
          
          <ArrowRight className="w-3 h-3 text-white/30" />
          
          {/* DynamoDB */}
          <div className="flex flex-col items-center gap-1">
            <div className="w-8 h-8 rounded bg-yellow-600/30 border border-yellow-500/50 flex items-center justify-center">
              <Database className="w-4 h-4 text-yellow-400" />
            </div>
            <span className="text-yellow-300">DynamoDB</span>
          </div>
          
          <ArrowRight className="w-3 h-3 text-white/30" />
          
          {/* S3 Output */}
          <div className="flex flex-col items-center gap-1">
            <div className="w-8 h-8 rounded bg-orange-600/30 border border-orange-500/50 flex items-center justify-center">
              <Box className="w-4 h-4 text-orange-400" />
            </div>
            <span className="text-orange-300">S3</span>
          </div>
        </div>
      </div>

      {/* Sign Database Stats */}
      <div className="flex items-center justify-between text-[10px] text-muted-foreground border-t border-border pt-3">
        <div className="flex items-center gap-4">
          <span>Sign Database: <strong className="text-foreground">{totalSigns}</strong> signs</span>
          <span>RTMPose: <strong className="text-foreground">133</strong> keypoints</span>
        </div>
        <a 
          href="https://github.com/aws-samples/genai-asl-avatar-generator" 
          target="_blank" 
          rel="noopener noreferrer"
          className="text-primary hover:underline flex items-center gap-1"
        >
          View on GitHub
        </a>
      </div>
    </div>
  );
};

export default AWSMetrics;
