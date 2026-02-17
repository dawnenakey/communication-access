import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Play, Pause, Volume2, VolumeX, Maximize2, Minimize2, 
  Loader2, AlertCircle, RefreshCw, User, Users,
  SkipForward, SkipBack, Film, Subtitles, Settings,
  ChevronDown, Check, Camera
} from 'lucide-react';
import { supabase } from '@/lib/supabase';

// API base - use same origin on demo.sonzo.io for GenASL/avatar, else external
const getAPI = () => {
  if (import.meta.env?.VITE_API_URL) return import.meta.env.VITE_API_URL;
  if (typeof window !== 'undefined' && window.location.hostname === 'demo.sonzo.io') return '';
  return 'https://api.sonzo.io';
};

// Realistic human signer images
const SIGNER_AVATARS = {
  female: 'https://d64gsuwffb70l.cloudfront.net/69757acc7df661eddaca039f_1769662049269_f23c9fd3.jpg',
  male: 'https://d64gsuwffb70l.cloudfront.net/69757acc7df661eddaca039f_1769662070071_b9b6621d.png'
};

interface VideoData {
  sign: string;
  videoUrl: string;
  thumbnailUrl: string;
  duration: number;
  signer: string;
  category: string;
  difficulty: string;
  available: boolean;
}

interface VideoAvatarProps {
  currentSentence: string;
  isResponding: boolean;
  language: string;
  onResponseComplete?: () => void;
  recognizedSign?: string | null;
  onVideoStart?: () => void;
  onVideoEnd?: () => void;
  showSubtitles?: boolean;
  onOpenMarketplace?: () => void;
  onOpenLearning?: () => void;
}

const VideoAvatar: React.FC<VideoAvatarProps> = ({
  currentSentence,
  isResponding,
  language,
  onResponseComplete,
  recognizedSign,
  onVideoStart,
  onVideoEnd,
  showSubtitles = true,
  onOpenMarketplace,
  onOpenLearning,
}) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isMuted, setIsMuted] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentVideoUrl, setCurrentVideoUrl] = useState<string | null>(null);
  const [videoQueue, setVideoQueue] = useState<VideoData[]>([]);
  const [currentVideoIndex, setCurrentVideoIndex] = useState(0);
  const [currentSignName, setCurrentSignName] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [selectedSigner, setSelectedSigner] = useState<'female' | 'male'>('female');
  const [showSignerMenu, setShowSignerMenu] = useState(false);
  const [subtitlesEnabled, setSubtitlesEnabled] = useState(showSubtitles);
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationFrame, setAnimationFrame] = useState(0);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number | null>(null);

  // Fetch video for a single sign
  const fetchSignVideo = useCallback(async (sign: string): Promise<VideoData | null> => {
    try {
      const { data, error } = await supabase.functions.invoke('asl-video-manager', {
        body: { action: 'get_video', sign }
      });

      if (error) throw error;
      return data;
    } catch (err) {
      console.error('Error fetching video:', err);
      return null;
    }
  }, []);

  // Fetch videos for a sequence of signs
  // On demo.sonzo.io: try /api/generate-sequence (avatar) first for GenASL-style videos
  const fetchVideoSequence = useCallback(async (signs: string[]): Promise<VideoData[]> => {
    const api = getAPI();
    if (api === '' || api === undefined) {
      // On demo.sonzo.io - try avatar generate-sequence first
      try {
        const glossSigns = signs.map(s => s.toUpperCase().replace(/\s+/g, '_'));
        const res = await fetch('/api/generate-sequence', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ signs: glossSigns }),
        });
        if (res.ok) {
          const data = await res.json();
          const url = data.video_url || data.url;
          if (url) {
            const fullUrl = url.startsWith('http') ? url : `${window.location.origin}${url.startsWith('/') ? '' : '/'}${url}`;
            return [{
              sign: signs.join(' '),
              videoUrl: fullUrl,
              thumbnailUrl: '',
              duration: 5000,
              signer: 'genasl',
              category: 'sequence',
              difficulty: 'beginner',
              available: true,
            }];
          }
        }
      } catch (e) {
        console.log('GenASL/avatar generate-sequence fallback:', e);
      }
    }
    try {
      const { data, error } = await supabase.functions.invoke('asl-video-manager', {
        body: { action: 'get_sequence', signs }
      });

      if (error) throw error;
      return data?.sequence || [];
    } catch (err) {
      console.error('Error fetching video sequence:', err);
      return [];
    }
  }, []);

  // Parse sentence into signs
  const parseSentenceToSigns = useCallback((sentence: string): string[] => {
    const words = sentence
      .toLowerCase()
      .replace(/[.,!?;:'"]/g, '')
      .split(/\s+/)
      .filter(w => w.length > 0);
    
    const skipWords = ['a', 'an', 'the', 'is', 'are', 'am', 'was', 'were', 'be', 'been', 'being'];
    return words.filter(w => !skipWords.includes(w));
  }, []);

  // Simulate signing animation with the realistic avatar
  const simulateSigning = useCallback((sentence: string) => {
    setIsAnimating(true);
    setIsPlaying(true);
    onVideoStart?.();
    
    const signs = parseSentenceToSigns(sentence);
    const totalDuration = signs.length * 1500; // 1.5 seconds per sign
    let currentSign = 0;
    
    const animate = () => {
      const elapsed = Date.now() - startTime;
      const signIndex = Math.floor(elapsed / 1500);
      
      if (signIndex < signs.length) {
        setCurrentSignName(signs[signIndex].toUpperCase());
        setCurrentVideoIndex(signIndex);
        setProgress((elapsed / totalDuration) * 100);
        setAnimationFrame(prev => prev + 1);
        animationRef.current = requestAnimationFrame(animate);
      } else {
        // Animation complete
        setIsAnimating(false);
        setIsPlaying(false);
        setCurrentSignName(null);
        setProgress(0);
        onVideoEnd?.();
        onResponseComplete?.();
      }
    };
    
    const startTime = Date.now();
    setVideoQueue(signs.map(s => ({ 
      sign: s, 
      videoUrl: '', 
      thumbnailUrl: '', 
      duration: 1500, 
      signer: selectedSigner,
      category: 'common',
      difficulty: 'beginner',
      available: true 
    })));
    
    animationRef.current = requestAnimationFrame(animate);
  }, [parseSentenceToSigns, selectedSigner, onVideoStart, onVideoEnd, onResponseComplete]);

  // Play video sequence
  const playVideoSequence = useCallback(async (sentence: string) => {
    setIsLoading(true);
    setError(null);
    onVideoStart?.();

    try {
      const signs = parseSentenceToSigns(sentence);
      const videos = await fetchVideoSequence(signs);
      
      // If no real videos available, use simulated signing with realistic avatar
      if (videos.length === 0 || !videos.some(v => v.available && v.videoUrl)) {
        setIsLoading(false);
        simulateSigning(sentence);
        return;
      }

      setVideoQueue(videos);
      setCurrentVideoIndex(0);
      
      const firstVideo = videos[0];
      if (firstVideo.available && firstVideo.videoUrl) {
        setCurrentVideoUrl(firstVideo.videoUrl);
        setCurrentSignName(firstVideo.sign);
      } else {
        simulateSigning(sentence);
      }
    } catch (err) {
      setError('Failed to load video sequence');
      simulateSigning(sentence);
    } finally {
      setIsLoading(false);
    }
  }, [parseSentenceToSigns, fetchVideoSequence, onVideoStart, simulateSigning]);

  // Play single sign
  const playSingleSign = useCallback(async (sign: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const video = await fetchSignVideo(sign);
      
      if (video?.available && video.videoUrl) {
        setCurrentVideoUrl(video.videoUrl);
        setCurrentSignName(video.sign);
        setVideoQueue([video]);
        setCurrentVideoIndex(0);
      } else {
        // Simulate single sign
        setIsLoading(false);
        setIsAnimating(true);
        setIsPlaying(true);
        setCurrentSignName(sign.toUpperCase());
        setVideoQueue([{ 
          sign, 
          videoUrl: '', 
          thumbnailUrl: '', 
          duration: 1500, 
          signer: selectedSigner,
          category: 'common',
          difficulty: 'beginner',
          available: true 
        }]);
        
        setTimeout(() => {
          setIsAnimating(false);
          setIsPlaying(false);
          setCurrentSignName(null);
          setVideoQueue([]);
        }, 1500);
      }
    } catch (err) {
      setError('Failed to load video');
    } finally {
      setIsLoading(false);
    }
  }, [fetchSignVideo, selectedSigner]);

  // Handle video ended
  const handleVideoEnded = useCallback(() => {
    const nextIndex = currentVideoIndex + 1;
    
    if (nextIndex < videoQueue.length) {
      setCurrentVideoIndex(nextIndex);
      const nextVideo = videoQueue[nextIndex];
      
      if (nextVideo.available && nextVideo.videoUrl) {
        setCurrentVideoUrl(nextVideo.videoUrl);
        setCurrentSignName(nextVideo.sign);
      } else {
        handleVideoEnded();
      }
    } else {
      setIsPlaying(false);
      setCurrentVideoUrl(null);
      setCurrentSignName(null);
      setVideoQueue([]);
      setCurrentVideoIndex(0);
      onVideoEnd?.();
      onResponseComplete?.();
    }
  }, [currentVideoIndex, videoQueue, onVideoEnd, onResponseComplete]);

  // Handle video time update
  const handleTimeUpdate = useCallback(() => {
    if (videoRef.current) {
      const progress = (videoRef.current.currentTime / videoRef.current.duration) * 100;
      setProgress(progress);
    }
  }, []);

  // Effect: Handle sentence signing
  useEffect(() => {
    if (isResponding && currentSentence) {
      playVideoSequence(currentSentence);
    }
  }, [isResponding, currentSentence, playVideoSequence]);

  // Effect: Handle recognized sign
  useEffect(() => {
    if (recognizedSign && !isResponding) {
      playSingleSign(recognizedSign);
    }
  }, [recognizedSign, isResponding, playSingleSign]);

  // Effect: Auto-play when video URL changes
  useEffect(() => {
    if (currentVideoUrl && videoRef.current) {
      videoRef.current.load();
      videoRef.current.play()
        .then(() => setIsPlaying(true))
        .catch(err => {
          console.error('Video play error:', err);
          // Fallback to simulation
          if (currentSentence) {
            simulateSigning(currentSentence);
          }
        });
    }
  }, [currentVideoUrl, currentSentence, simulateSigning]);

  // Cleanup animation on unmount
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  // Toggle fullscreen
  const toggleFullscreen = useCallback(() => {
    if (!containerRef.current) return;
    
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  // Skip to next/previous video
  const skipVideo = useCallback((direction: 'next' | 'prev') => {
    const newIndex = direction === 'next' 
      ? Math.min(currentVideoIndex + 1, videoQueue.length - 1)
      : Math.max(currentVideoIndex - 1, 0);
    
    if (newIndex !== currentVideoIndex) {
      setCurrentVideoIndex(newIndex);
      const video = videoQueue[newIndex];
      if (video.available && video.videoUrl) {
        setCurrentVideoUrl(video.videoUrl);
        setCurrentSignName(video.sign);
      }
    }
  }, [currentVideoIndex, videoQueue]);

  return (
    <div 
      ref={containerRef}
      className={`relative bg-card rounded-2xl border border-border overflow-hidden transition-all duration-300 ${
        isFullscreen ? 'fixed inset-4 z-50' : ''
      }`}
    >
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between p-3 bg-gradient-to-b from-black/80 to-transparent">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
            <Film className="w-4 h-4 text-white" />
          </div>
          <div>
            <p className="text-xs font-semibold text-white">SonZo Video Avatar</p>
            <p className="text-[10px] text-white/60">{language} • GenASL Pipeline</p>
          </div>
        </div>

        <div className="flex items-center gap-1">
          {/* Signer Selection */}
          <div className="relative">
            <button
              onClick={() => setShowSignerMenu(!showSignerMenu)}
              className="p-2 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 transition-all flex items-center gap-1"
            >
              <Users className="w-4 h-4" />
              <ChevronDown className="w-3 h-3" />
            </button>
            
            {showSignerMenu && (
              <div className="absolute right-0 top-full mt-1 w-40 bg-slate-800 rounded-lg border border-slate-700 shadow-xl overflow-hidden z-20">
                <button
                  onClick={() => { setSelectedSigner('female'); setShowSignerMenu(false); }}
                  className="w-full px-3 py-2 flex items-center gap-2 hover:bg-slate-700 transition-colors"
                >
                  <img src={SIGNER_AVATARS.female} className="w-8 h-8 rounded-full object-cover" alt="Female signer" />
                  <span className="text-sm text-white">Sarah</span>
                  {selectedSigner === 'female' && <Check className="w-4 h-4 text-green-400 ml-auto" />}
                </button>
                <button
                  onClick={() => { setSelectedSigner('male'); setShowSignerMenu(false); }}
                  className="w-full px-3 py-2 flex items-center gap-2 hover:bg-slate-700 transition-colors"
                >
                  <img src={SIGNER_AVATARS.male} className="w-8 h-8 rounded-full object-cover" alt="Male signer" />
                  <span className="text-sm text-white">Marcus</span>
                  {selectedSigner === 'male' && <Check className="w-4 h-4 text-green-400 ml-auto" />}
                </button>
              </div>
            )}
          </div>

          <button
            onClick={() => setSubtitlesEnabled(!subtitlesEnabled)}
            className={`p-2 rounded-lg transition-all ${
              subtitlesEnabled 
                ? 'bg-primary/20 text-primary' 
                : 'bg-white/10 text-white/70 hover:bg-white/20'
            }`}
          >
            <Subtitles className="w-4 h-4" />
          </button>
          
          <button
            onClick={() => setIsMuted(!isMuted)}
            className="p-2 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 transition-all"
          >
            {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
          </button>
          
          <button
            onClick={toggleFullscreen}
            className="p-2 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 transition-all"
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Video/Avatar Container */}
      <div className={`relative bg-gradient-to-b from-slate-100 via-slate-50 to-slate-100 ${
        isFullscreen ? 'h-full' : 'aspect-[3/4]'
      }`}>
        {/* Loading State */}
        {isLoading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/50 z-20">
            <Loader2 className="w-12 h-12 animate-spin text-primary mb-3" />
            <p className="text-sm text-white/70">Loading ASL video...</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/50 z-20">
            <AlertCircle className="w-12 h-12 text-red-500 mb-3" />
            <p className="text-sm text-white/70">{error}</p>
            <button 
              onClick={() => setError(null)}
              className="mt-3 px-4 py-2 bg-primary rounded-lg text-sm font-medium text-white"
            >
              Retry
            </button>
          </div>
        )}

        {/* Real Video Player (when video URL available) */}
        {currentVideoUrl && (
          <video
            ref={videoRef}
            className="w-full h-full object-cover"
            muted={isMuted}
            playsInline
            crossOrigin="anonymous"
            onEnded={handleVideoEnded}
            onTimeUpdate={handleTimeUpdate}
            onError={(e) => {
              const videoEl = e.currentTarget;
              const errorCode = videoEl.error?.code;
              const errorMessage = videoEl.error?.message || 'Unknown video error';
              console.error('Video playback error:', {
                code: errorCode,
                message: errorMessage,
                src: currentVideoUrl,
                networkState: videoEl.networkState,
                readyState: videoEl.readyState
              });
              setError(`Video failed to load: ${errorMessage}`);
              setCurrentVideoUrl(null);
              if (currentSentence) simulateSigning(currentSentence);
            }}
          >
            <source src={currentVideoUrl} type="video/mp4" />
          </video>
        )}

        {/* Realistic Human Avatar (when no video) */}
        {!currentVideoUrl && (
          <div className="relative w-full h-full">
            {/* Human Signer Image */}
            <img 
              src={SIGNER_AVATARS[selectedSigner]}
              alt="ASL Signer"
              className={`w-full h-full object-cover transition-all duration-300 ${
                isAnimating ? 'scale-[1.02]' : ''
              }`}
              style={{
                filter: isAnimating ? 'brightness(1.05)' : 'brightness(1)',
              }}
            />
            
            {/* Signing Animation Overlay */}
            {isAnimating && (
              <>
                {/* Hand movement indicators */}
                <div className="absolute inset-0 pointer-events-none">
                  {/* Left hand glow */}
                  <div 
                    className="absolute w-24 h-24 rounded-full bg-violet-500/20 blur-xl animate-pulse"
                    style={{
                      left: `${30 + Math.sin(animationFrame * 0.1) * 10}%`,
                      top: `${50 + Math.cos(animationFrame * 0.15) * 5}%`,
                      transform: 'translate(-50%, -50%)'
                    }}
                  />
                  {/* Right hand glow */}
                  <div 
                    className="absolute w-24 h-24 rounded-full bg-purple-500/20 blur-xl animate-pulse"
                    style={{
                      left: `${70 + Math.cos(animationFrame * 0.1) * 10}%`,
                      top: `${50 + Math.sin(animationFrame * 0.15) * 5}%`,
                      transform: 'translate(-50%, -50%)'
                    }}
                  />
                </div>
                
                {/* Active signing indicator */}
                <div className="absolute top-16 left-3 right-3">
                  <div className="bg-green-500/90 backdrop-blur-sm rounded-lg px-3 py-2 flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-white animate-pulse" />
                    <span className="text-xs font-semibold text-white">SIGNING</span>
                    <div className="flex-1 h-1 bg-white/30 rounded-full overflow-hidden ml-2">
                      <div 
                        className="h-full bg-white transition-all duration-100"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                  </div>
                </div>
              </>
            )}

            {/* Idle State Overlay */}
            {!isAnimating && !isLoading && (
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent flex items-end justify-center pb-20">
                <div className="text-center">
                  <p className="text-white/80 text-sm font-medium">Ready to sign in {language}</p>
                  <p className="text-white/50 text-xs mt-1">Type a message or use the camera</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Progress Bar */}
      {videoQueue.length > 0 && (
        <div className="absolute bottom-24 left-4 right-4 z-10">
          {/* Video Progress */}
          <div className="h-1.5 bg-black/30 backdrop-blur-sm rounded-full overflow-hidden mb-2">
            <div 
              className="h-full bg-gradient-to-r from-violet-500 to-purple-500 transition-all duration-100"
              style={{ width: `${progress}%` }}
            />
          </div>
          
          {/* Sequence Progress Dots */}
          <div className="flex gap-1 justify-center">
            {videoQueue.map((video, i) => (
              <div
                key={i}
                className={`w-2 h-2 rounded-full transition-all ${
                  i < currentVideoIndex 
                    ? 'bg-violet-500' 
                    : i === currentVideoIndex 
                      ? 'bg-white scale-125' 
                      : 'bg-white/30'
                }`}
              />
            ))}
          </div>
        </div>
      )}

      {/* Subtitles */}
      {subtitlesEnabled && currentSignName && (
        <div className="absolute bottom-32 left-4 right-4 z-10">
          <div className="bg-black/80 backdrop-blur-sm rounded-lg px-4 py-2 text-center">
            <p className="text-white font-bold text-lg tracking-wide">{currentSignName}</p>
            {videoQueue.length > 1 && (
              <p className="text-white/60 text-xs mt-1">
                Sign {currentVideoIndex + 1} of {videoQueue.length}
              </p>
            )}
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/90 via-black/70 to-transparent">
        {/* Current Status */}
        <div className="text-center mb-3">
          {!isPlaying && !isAnimating && (
            <p className="text-xs text-white/50">Ready to respond in {language}</p>
          )}
        </div>

        {/* Playback Controls */}
        <div className="flex items-center justify-center gap-4">
          <button
            onClick={() => skipVideo('prev')}
            disabled={currentVideoIndex === 0 || videoQueue.length === 0}
            className="p-2 rounded-full bg-white/10 text-white/70 hover:bg-white/20 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
          >
            <SkipBack className="w-4 h-4" />
          </button>
          
          <button
            onClick={() => {
              if (videoRef.current && currentVideoUrl) {
                if (isPlaying) {
                  videoRef.current.pause();
                  setIsPlaying(false);
                } else {
                  videoRef.current.play();
                  setIsPlaying(true);
                }
              }
            }}
            disabled={!currentVideoUrl && !isAnimating}
            className="p-4 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 text-white hover:from-violet-400 hover:to-purple-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg"
          >
            {isPlaying || isAnimating ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6 ml-0.5" />}
          </button>
          
          <button
            onClick={() => skipVideo('next')}
            disabled={currentVideoIndex >= videoQueue.length - 1 || videoQueue.length === 0}
            className="p-2 rounded-full bg-white/10 text-white/70 hover:bg-white/20 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
          >
            <SkipForward className="w-4 h-4" />
          </button>
        </div>

        {/* Playback Speed */}
        <div className="flex items-center justify-center gap-2 mt-3">
          <span className="text-[10px] text-white/40">Speed:</span>
          {[0.5, 0.75, 1, 1.25, 1.5].map(speed => (
            <button
              key={speed}
              onClick={() => {
                setPlaybackSpeed(speed);
                if (videoRef.current) {
                  videoRef.current.playbackRate = speed;
                }
              }}
              className={`px-2 py-0.5 rounded text-[10px] font-medium transition-all ${
                playbackSpeed === speed 
                  ? 'bg-violet-500 text-white' 
                  : 'bg-white/10 text-white/50 hover:bg-white/20'
              }`}
            >
              {speed}x
            </button>
          ))}
        </div>

        {/* Quick Actions */}
        <div className="flex items-center justify-center gap-2 mt-3">
          {onOpenMarketplace && (
            <button
              onClick={onOpenMarketplace}
              className="px-3 py-1.5 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 transition-all text-xs flex items-center gap-1.5"
            >
              <Users className="w-3.5 h-3.5" />
              More Signers
            </button>
          )}
          {onOpenLearning && (
            <button
              onClick={onOpenLearning}
              className="px-3 py-1.5 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 transition-all text-xs flex items-center gap-1.5"
            >
              <Camera className="w-3.5 h-3.5" />
              Learn Signs
            </button>
          )}
        </div>
      </div>

      {/* Status Badge */}
      <div className="absolute top-14 left-3">
        <div className={`flex items-center gap-1.5 px-2 py-1 rounded-full text-[10px] font-semibold backdrop-blur-sm ${
          isPlaying || isAnimating
            ? 'bg-green-500/20 text-green-400 border border-green-500/30' 
            : isLoading
              ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
              : 'bg-white/10 text-white/50 border border-white/10'
        }`}>
          <div className={`w-1.5 h-1.5 rounded-full ${
            isPlaying || isAnimating
              ? 'bg-green-500 animate-pulse' 
              : isLoading
                ? 'bg-yellow-500 animate-pulse'
                : 'bg-white/30'
          }`} />
          {isPlaying || isAnimating ? 'Signing' : isLoading ? 'Loading' : 'Idle'}
        </div>
      </div>
    </div>
  );
};

export default VideoAvatar;
