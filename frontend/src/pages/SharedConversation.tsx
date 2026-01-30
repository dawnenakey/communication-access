import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useConversations, Conversation } from '@/hooks/useConversations';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  MessageSquare, 
  Clock, 
  Download, 
  Play, 
  Pause,
  SkipBack,
  SkipForward,
  Home,
  Share2
} from 'lucide-react';

const SharedConversation: React.FC = () => {
  const { shareToken } = useParams<{ shareToken: string }>();
  const { 
    currentConversation, 
    isLoading, 
    error, 
    getSharedConversation,
    downloadExport 
  } = useConversations();
  
  const [isReplaying, setIsReplaying] = useState(false);
  const [replayIndex, setReplayIndex] = useState(0);
  const [replaySpeed, setReplaySpeed] = useState(1);

  useEffect(() => {
    if (shareToken) {
      getSharedConversation(shareToken);
    }
  }, [shareToken, getSharedConversation]);

  // Replay functionality
  useEffect(() => {
    if (!isReplaying || !currentConversation?.messages) return;

    const messages = currentConversation.messages;
    if (replayIndex >= messages.length) {
      setIsReplaying(false);
      return;
    }

    const currentMsg = messages[replayIndex];
    const nextMsg = messages[replayIndex + 1];
    
    let delay = 1500 / replaySpeed;
    if (nextMsg) {
      const timeDiff = nextMsg.timestamp_ms - currentMsg.timestamp_ms;
      delay = Math.min(Math.max(timeDiff / replaySpeed, 500), 3000);
    }

    const timer = setTimeout(() => {
      setReplayIndex(prev => prev + 1);
    }, delay);

    return () => clearTimeout(timer);
  }, [isReplaying, replayIndex, currentConversation, replaySpeed]);

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      month: 'long',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatTimestamp = (ms: number) => {
    const date = new Date(ms);
    return date.toISOString().substr(11, 8);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin h-12 w-12 border-4 border-primary border-t-transparent rounded-full mx-auto mb-4" />
          <p className="text-muted-foreground">Loading conversation...</p>
        </div>
      </div>
    );
  }

  if (error || !currentConversation) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Card className="max-w-md w-full mx-4">
          <CardContent className="pt-6 text-center">
            <div className="w-16 h-16 rounded-full bg-destructive/10 flex items-center justify-center mx-auto mb-4">
              <MessageSquare className="h-8 w-8 text-destructive" />
            </div>
            <h2 className="text-xl font-semibold mb-2">Conversation Not Found</h2>
            <p className="text-muted-foreground mb-6">
              {error || 'This conversation may have been deleted or the link is invalid.'}
            </p>
            <Link to="/">
              <Button className="gap-2">
                <Home className="h-4 w-4" />
                Go to Home
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-background/80 backdrop-blur-xl border-b border-border">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shadow-lg shadow-violet-500/30">
              <svg viewBox="0 0 24 24" className="w-6 h-6 text-white" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M7 11V7a5 5 0 0 1 10 0v4" strokeLinecap="round" />
                <path d="M12 11v6" strokeLinecap="round" />
                <path d="M8 15h8" strokeLinecap="round" />
                <circle cx="12" cy="18" r="1" fill="currentColor" />
                <path d="M5 11h2v8a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2v-8h2" strokeLinecap="round" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold gradient-text">SonZo AI</h1>
              <p className="text-[10px] text-muted-foreground -mt-1">Shared Conversation</p>
            </div>
          </Link>
          
          <div className="flex items-center gap-2">
            <Button 
              variant="outline" 
              size="sm" 
              className="gap-2"
              onClick={() => downloadExport(currentConversation.id, 'text')}
            >
              <Download className="h-4 w-4" />
              Export
            </Button>
            <Link to="/">
              <Button size="sm" className="gap-2">
                <Home className="h-4 w-4" />
                Try SonZo AI
              </Button>
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-5xl mx-auto px-4 py-8">
        {/* Conversation Info */}
        <Card className="mb-6">
          <CardHeader>
            <div className="flex items-start justify-between">
              <div>
                <CardTitle className="text-2xl mb-2">{currentConversation.title}</CardTitle>
                <p className="text-muted-foreground">{formatDate(currentConversation.created_at)}</p>
              </div>
              <Badge variant="secondary" className="gap-1">
                <Share2 className="h-3 w-3" />
                Shared
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center p-4 bg-muted/30 rounded-lg">
                <p className="text-3xl font-bold text-primary">{currentConversation.total_signs}</p>
                <p className="text-sm text-muted-foreground">Total Signs</p>
              </div>
              <div className="text-center p-4 bg-muted/30 rounded-lg">
                <p className="text-3xl font-bold text-primary">
                  {formatDuration(currentConversation.duration_seconds)}
                </p>
                <p className="text-sm text-muted-foreground">Duration</p>
              </div>
              <div className="text-center p-4 bg-muted/30 rounded-lg">
                <p className="text-3xl font-bold text-primary">
                  {currentConversation.messages ? new Set(currentConversation.messages.map(m => m.sign)).size : 0}
                </p>
                <p className="text-sm text-muted-foreground">Unique Signs</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Sign Timeline */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Sign Timeline</CardTitle>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Clock className="h-4 w-4" />
                {currentConversation.messages?.length || 0} signs recorded
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[400px] pr-4">
              <div className="space-y-3">
                {currentConversation.messages?.map((msg, index) => (
                  <div
                    key={msg.id}
                    className={`flex items-start gap-3 p-3 rounded-lg transition-colors ${
                      isReplaying && replayIndex === index
                        ? 'bg-primary/10 border border-primary/30'
                        : 'bg-muted/30 hover:bg-muted/50'
                    }`}
                  >
                    <div className="flex-shrink-0 w-16 text-xs text-muted-foreground font-mono">
                      {formatTimestamp(msg.timestamp_ms)}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary" className="font-medium">
                          {msg.sign}
                        </Badge>
                        {msg.confidence > 0 && (
                          <span className="text-xs text-muted-foreground">
                            {Math.round(msg.confidence * 100)}% confidence
                          </span>
                        )}
                      </div>
                      {msg.sentence && (
                        <p className="text-sm text-muted-foreground mt-1">
                          "{msg.sentence}"
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>

        {/* Replay Controls */}
        <div className="mt-6 p-4 bg-muted/30 rounded-xl">
          <div className="flex items-center justify-center gap-4">
            <Button
              variant="outline"
              size="icon"
              onClick={() => setReplayIndex(0)}
              disabled={!currentConversation.messages?.length}
            >
              <SkipBack className="h-4 w-4" />
            </Button>
            <Button
              variant="default"
              size="lg"
              className="gap-2 px-6"
              onClick={() => setIsReplaying(!isReplaying)}
              disabled={!currentConversation.messages?.length}
            >
              {isReplaying ? (
                <>
                  <Pause className="h-5 w-5" />
                  Pause
                </>
              ) : (
                <>
                  <Play className="h-5 w-5" />
                  Replay
                </>
              )}
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={() => setReplayIndex(prev => 
                Math.min(prev + 1, (currentConversation.messages?.length || 1) - 1)
              )}
              disabled={!currentConversation.messages?.length}
            >
              <SkipForward className="h-4 w-4" />
            </Button>
            
            <div className="ml-4 flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Speed:</span>
              <select
                value={replaySpeed}
                onChange={(e) => setReplaySpeed(Number(e.target.value))}
                className="bg-background border rounded px-2 py-1 text-sm"
              >
                <option value={0.5}>0.5x</option>
                <option value={1}>1x</option>
                <option value={1.5}>1.5x</option>
                <option value={2}>2x</option>
              </select>
            </div>
            
            <div className="ml-4 text-sm text-muted-foreground">
              Sign {replayIndex + 1} of {currentConversation.messages?.length || 0}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-12 py-8">
        <div className="max-w-5xl mx-auto px-4 text-center">
          <p className="text-muted-foreground">
            This conversation was shared via{' '}
            <Link to="/" className="text-primary hover:underline">
              SonZo AI
            </Link>
            {' '}- Real-time Sign Language Recognition Platform
          </p>
        </div>
      </footer>
    </div>
  );
};

export default SharedConversation;
