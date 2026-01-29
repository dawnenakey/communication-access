import React, { useState, useEffect, useCallback } from 'react';
import { useConversations, Conversation, ConversationMessage } from '@/hooks/useConversations';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuSeparator } from '@/components/ui/dropdown-menu';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  MessageSquare, 
  Clock, 
  Download, 
  Share2, 
  Trash2, 
  Play, 
  Pause,
  SkipBack,
  SkipForward,
  Search,
  MoreVertical,
  Copy,
  Check,
  Link,
  FileText,
  FileJson,
  Table,
  X,
  ChevronLeft,
  Edit2,
  Globe,
  Lock
} from 'lucide-react';

interface ConversationHistoryPageProps {
  isOpen: boolean;
  onClose: () => void;
  onReplaySign?: (sign: string) => void;
}

const ConversationHistoryPage: React.FC<ConversationHistoryPageProps> = ({
  isOpen,
  onClose,
  onReplaySign
}) => {
  const { isAuthenticated } = useAuth();
  const {
    conversations,
    currentConversation,
    isLoading,
    error,
    total,
    hasMore,
    getConversations,
    getConversation,
    updateConversation,
    deleteConversation,
    generateShareLink,
    revokeShareLink,
    downloadExport,
    clearError,
    setCurrentConversation
  } = useConversations();

  const [searchQuery, setSearchQuery] = useState('');
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null);
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [editTitle, setEditTitle] = useState('');
  const [shareUrl, setShareUrl] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [isReplaying, setIsReplaying] = useState(false);
  const [replayIndex, setReplayIndex] = useState(0);
  const [replaySpeed, setReplaySpeed] = useState(1);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  // Load conversations on mount
  useEffect(() => {
    if (isOpen && isAuthenticated) {
      getConversations();
    }
  }, [isOpen, isAuthenticated, getConversations]);

  // Filter conversations by search
  const filteredConversations = conversations.filter(conv =>
    conv.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Handle conversation selection
  const handleSelectConversation = useCallback(async (conv: Conversation) => {
    const fullConv = await getConversation(conv.id);
    if (fullConv) {
      setSelectedConversation(fullConv);
      setReplayIndex(0);
      setIsReplaying(false);
    }
  }, [getConversation]);

  // Handle title edit
  const handleSaveTitle = useCallback(async () => {
    if (selectedConversation && editTitle.trim()) {
      await updateConversation(selectedConversation.id, { title: editTitle.trim() });
      setSelectedConversation(prev => prev ? { ...prev, title: editTitle.trim() } : null);
      setIsEditingTitle(false);
    }
  }, [selectedConversation, editTitle, updateConversation]);

  // Handle share
  const handleShare = useCallback(async () => {
    if (!selectedConversation) return;
    
    const url = await generateShareLink(selectedConversation.id);
    if (url) {
      setShareUrl(url);
      setSelectedConversation(prev => prev ? { ...prev, is_public: true } : null);
    }
  }, [selectedConversation, generateShareLink]);

  // Handle copy share link
  const handleCopyLink = useCallback(() => {
    if (shareUrl) {
      navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [shareUrl]);

  // Handle revoke share
  const handleRevokeShare = useCallback(async () => {
    if (!selectedConversation) return;
    
    await revokeShareLink(selectedConversation.id);
    setShareUrl(null);
    setSelectedConversation(prev => prev ? { ...prev, is_public: false } : null);
  }, [selectedConversation, revokeShareLink]);

  // Handle delete
  const handleDelete = useCallback(async (convId: string) => {
    await deleteConversation(convId);
    if (selectedConversation?.id === convId) {
      setSelectedConversation(null);
    }
    setDeleteConfirmId(null);
  }, [deleteConversation, selectedConversation]);

  // Handle export
  const handleExport = useCallback(async (format: 'text' | 'json' | 'csv') => {
    if (selectedConversation) {
      await downloadExport(selectedConversation.id, format);
    }
  }, [selectedConversation, downloadExport]);

  // Replay functionality
  useEffect(() => {
    if (!isReplaying || !selectedConversation?.messages) return;

    const messages = selectedConversation.messages;
    if (replayIndex >= messages.length) {
      setIsReplaying(false);
      return;
    }

    const currentMsg = messages[replayIndex];
    const nextMsg = messages[replayIndex + 1];
    
    // Trigger sign animation
    if (onReplaySign) {
      onReplaySign(currentMsg.sign);
    }

    // Calculate delay to next message
    let delay = 1500 / replaySpeed;
    if (nextMsg) {
      const timeDiff = nextMsg.timestamp_ms - currentMsg.timestamp_ms;
      delay = Math.min(Math.max(timeDiff / replaySpeed, 500), 3000);
    }

    const timer = setTimeout(() => {
      setReplayIndex(prev => prev + 1);
    }, delay);

    return () => clearTimeout(timer);
  }, [isReplaying, replayIndex, selectedConversation, replaySpeed, onReplaySign]);

  // Format duration
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
  };

  // Format date
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Format timestamp
  const formatTimestamp = (ms: number) => {
    const date = new Date(ms);
    return date.toISOString().substr(11, 8);
  };

  if (!isOpen) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-6xl h-[85vh] p-0 gap-0">
        <DialogHeader className="p-6 pb-4 border-b">
          <div className="flex items-center justify-between">
            <div>
              <DialogTitle className="text-2xl font-bold">Conversation History</DialogTitle>
              <DialogDescription>
                View, replay, and share your past sign language conversations
              </DialogDescription>
            </div>
            <Badge variant="secondary" className="text-sm">
              {total} conversation{total !== 1 ? 's' : ''}
            </Badge>
          </div>
        </DialogHeader>

        <div className="flex flex-1 overflow-hidden">
          {/* Sidebar - Conversation List */}
          <div className="w-80 border-r flex flex-col">
            {/* Search */}
            <div className="p-4 border-b">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search conversations..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>

            {/* Conversation List */}
            <ScrollArea className="flex-1">
              {isLoading && conversations.length === 0 ? (
                <div className="p-8 text-center text-muted-foreground">
                  <div className="animate-spin h-8 w-8 border-2 border-primary border-t-transparent rounded-full mx-auto mb-4" />
                  Loading conversations...
                </div>
              ) : filteredConversations.length === 0 ? (
                <div className="p-8 text-center text-muted-foreground">
                  <MessageSquare className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p className="font-medium">No conversations found</p>
                  <p className="text-sm mt-1">Start a new conversation to see it here</p>
                </div>
              ) : (
                <div className="p-2 space-y-1">
                  {filteredConversations.map((conv) => (
                    <div
                      key={conv.id}
                      className={`p-3 rounded-lg cursor-pointer transition-colors group ${
                        selectedConversation?.id === conv.id
                          ? 'bg-primary/10 border border-primary/20'
                          : 'hover:bg-muted/50'
                      }`}
                      onClick={() => handleSelectConversation(conv)}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <h4 className="font-medium truncate">{conv.title}</h4>
                            {conv.is_public && (
                              <Globe className="h-3 w-3 text-green-500 flex-shrink-0" />
                            )}
                          </div>
                          <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
                            <span className="flex items-center gap-1">
                              <MessageSquare className="h-3 w-3" />
                              {conv.total_signs} signs
                            </span>
                            <span className="flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              {formatDuration(conv.duration_seconds)}
                            </span>
                          </div>
                          <p className="text-xs text-muted-foreground mt-1">
                            {formatDate(conv.created_at)}
                          </p>
                        </div>
                        
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <MoreVertical className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onClick={() => handleSelectConversation(conv)}>
                              <Play className="h-4 w-4 mr-2" />
                              View & Replay
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem 
                              onClick={(e) => {
                                e.stopPropagation();
                                setDeleteConfirmId(conv.id);
                              }}
                              className="text-destructive"
                            >
                              <Trash2 className="h-4 w-4 mr-2" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </div>
                  ))}
                  
                  {hasMore && (
                    <Button
                      variant="ghost"
                      className="w-full mt-2"
                      onClick={() => getConversations(50, conversations.length)}
                      disabled={isLoading}
                    >
                      Load More
                    </Button>
                  )}
                </div>
              )}
            </ScrollArea>
          </div>

          {/* Main Content - Conversation Detail */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {selectedConversation ? (
              <>
                {/* Conversation Header */}
                <div className="p-4 border-b flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="lg:hidden"
                      onClick={() => setSelectedConversation(null)}
                    >
                      <ChevronLeft className="h-5 w-5" />
                    </Button>
                    
                    {isEditingTitle ? (
                      <div className="flex items-center gap-2">
                        <Input
                          value={editTitle}
                          onChange={(e) => setEditTitle(e.target.value)}
                          className="w-64"
                          autoFocus
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleSaveTitle();
                            if (e.key === 'Escape') setIsEditingTitle(false);
                          }}
                        />
                        <Button size="sm" onClick={handleSaveTitle}>Save</Button>
                        <Button size="sm" variant="ghost" onClick={() => setIsEditingTitle(false)}>
                          Cancel
                        </Button>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2">
                        <h3 className="text-lg font-semibold">{selectedConversation.title}</h3>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          onClick={() => {
                            setEditTitle(selectedConversation.title);
                            setIsEditingTitle(true);
                          }}
                        >
                          <Edit2 className="h-3.5 w-3.5" />
                        </Button>
                        {selectedConversation.is_public ? (
                          <Badge variant="secondary" className="gap-1">
                            <Globe className="h-3 w-3" />
                            Public
                          </Badge>
                        ) : (
                          <Badge variant="outline" className="gap-1">
                            <Lock className="h-3 w-3" />
                            Private
                          </Badge>
                        )}
                      </div>
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    {/* Share Button */}
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="outline" size="sm" className="gap-2">
                          <Share2 className="h-4 w-4" />
                          Share
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end" className="w-80">
                        <div className="p-3">
                          <h4 className="font-medium mb-2">Share Conversation</h4>
                          {selectedConversation.is_public && selectedConversation.share_token ? (
                            <div className="space-y-3">
                              <div className="flex items-center gap-2">
                                <Input
                                  value={shareUrl || `${window.location.origin}/shared/${selectedConversation.share_token}`}
                                  readOnly
                                  className="text-xs"
                                />
                                <Button size="icon" variant="outline" onClick={handleCopyLink}>
                                  {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                                </Button>
                              </div>
                              <Button
                                variant="destructive"
                                size="sm"
                                className="w-full"
                                onClick={handleRevokeShare}
                              >
                                Revoke Share Link
                              </Button>
                            </div>
                          ) : (
                            <Button className="w-full" onClick={handleShare}>
                              <Link className="h-4 w-4 mr-2" />
                              Generate Share Link
                            </Button>
                          )}
                        </div>
                      </DropdownMenuContent>
                    </DropdownMenu>

                    {/* Export Button */}
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="outline" size="sm" className="gap-2">
                          <Download className="h-4 w-4" />
                          Export
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onClick={() => handleExport('text')}>
                          <FileText className="h-4 w-4 mr-2" />
                          Export as Text (.txt)
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={() => handleExport('json')}>
                          <FileJson className="h-4 w-4 mr-2" />
                          Export as JSON (.json)
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={() => handleExport('csv')}>
                          <Table className="h-4 w-4 mr-2" />
                          Export as CSV (.csv)
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>

                {/* Conversation Content */}
                <Tabs defaultValue="timeline" className="flex-1 flex flex-col overflow-hidden">
                  <TabsList className="mx-4 mt-4 w-fit">
                    <TabsTrigger value="timeline">Timeline</TabsTrigger>
                    <TabsTrigger value="sentences">Sentences</TabsTrigger>
                    <TabsTrigger value="stats">Statistics</TabsTrigger>
                  </TabsList>

                  <TabsContent value="timeline" className="flex-1 overflow-hidden mt-0 p-4">
                    <Card className="h-full flex flex-col">
                      <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-base">Sign Timeline</CardTitle>
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <Clock className="h-4 w-4" />
                            {formatDuration(selectedConversation.duration_seconds)}
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent className="flex-1 overflow-hidden">
                        <ScrollArea className="h-full pr-4">
                          <div className="space-y-3">
                            {selectedConversation.messages?.map((msg, index) => (
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
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-8 w-8"
                                  onClick={() => {
                                    setReplayIndex(index);
                                    if (onReplaySign) onReplaySign(msg.sign);
                                  }}
                                >
                                  <Play className="h-4 w-4" />
                                </Button>
                              </div>
                            ))}
                          </div>
                        </ScrollArea>
                      </CardContent>
                    </Card>
                  </TabsContent>

                  <TabsContent value="sentences" className="flex-1 overflow-hidden mt-0 p-4">
                    <Card className="h-full flex flex-col">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-base">Recognized Sentences</CardTitle>
                      </CardHeader>
                      <CardContent className="flex-1 overflow-hidden">
                        <ScrollArea className="h-full pr-4">
                          <div className="space-y-3">
                            {(() => {
                              const sentences = new Map<string, { count: number; firstTime: number }>();
                              selectedConversation.messages?.forEach(msg => {
                                if (msg.sentence) {
                                  const existing = sentences.get(msg.sentence);
                                  if (existing) {
                                    existing.count++;
                                  } else {
                                    sentences.set(msg.sentence, { count: 1, firstTime: msg.timestamp_ms });
                                  }
                                }
                              });
                              
                              return Array.from(sentences.entries()).map(([sentence, data], index) => (
                                <div key={index} className="p-4 rounded-lg bg-muted/30">
                                  <p className="font-medium">"{sentence}"</p>
                                  <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                                    <span>First recognized: {formatTimestamp(data.firstTime)}</span>
                                    {data.count > 1 && (
                                      <Badge variant="outline" className="text-xs">
                                        Repeated {data.count}x
                                      </Badge>
                                    )}
                                  </div>
                                </div>
                              ));
                            })()}
                          </div>
                        </ScrollArea>
                      </CardContent>
                    </Card>
                  </TabsContent>

                  <TabsContent value="stats" className="flex-1 overflow-hidden mt-0 p-4">
                    <div className="grid grid-cols-2 gap-4">
                      <Card>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm font-medium text-muted-foreground">
                            Total Signs
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="text-3xl font-bold">{selectedConversation.total_signs}</p>
                        </CardContent>
                      </Card>
                      <Card>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm font-medium text-muted-foreground">
                            Duration
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="text-3xl font-bold">
                            {formatDuration(selectedConversation.duration_seconds)}
                          </p>
                        </CardContent>
                      </Card>
                      <Card>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm font-medium text-muted-foreground">
                            Unique Signs
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="text-3xl font-bold">
                            {new Set(selectedConversation.messages?.map(m => m.sign)).size}
                          </p>
                        </CardContent>
                      </Card>
                      <Card>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm font-medium text-muted-foreground">
                            Avg Confidence
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="text-3xl font-bold">
                            {selectedConversation.messages && selectedConversation.messages.length > 0
                              ? Math.round(
                                  (selectedConversation.messages.reduce((sum, m) => sum + (m.confidence || 0), 0) /
                                    selectedConversation.messages.length) *
                                    100
                                )
                              : 0}%
                          </p>
                        </CardContent>
                      </Card>
                    </div>
                  </TabsContent>
                </Tabs>

                {/* Replay Controls */}
                <div className="p-4 border-t bg-muted/30">
                  <div className="flex items-center justify-center gap-4">
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() => setReplayIndex(0)}
                      disabled={!selectedConversation.messages?.length}
                    >
                      <SkipBack className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="default"
                      size="lg"
                      className="gap-2 px-6"
                      onClick={() => setIsReplaying(!isReplaying)}
                      disabled={!selectedConversation.messages?.length}
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
                        Math.min(prev + 1, (selectedConversation.messages?.length || 1) - 1)
                      )}
                      disabled={!selectedConversation.messages?.length}
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
                      Sign {replayIndex + 1} of {selectedConversation.messages?.length || 0}
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center text-muted-foreground">
                <div className="text-center">
                  <MessageSquare className="h-16 w-16 mx-auto mb-4 opacity-30" />
                  <p className="text-lg font-medium">Select a conversation</p>
                  <p className="text-sm mt-1">Choose a conversation from the list to view details</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Delete Confirmation Dialog */}
        <Dialog open={!!deleteConfirmId} onOpenChange={() => setDeleteConfirmId(null)}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Delete Conversation</DialogTitle>
              <DialogDescription>
                Are you sure you want to delete this conversation? This action cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <div className="flex justify-end gap-3 mt-4">
              <Button variant="outline" onClick={() => setDeleteConfirmId(null)}>
                Cancel
              </Button>
              <Button 
                variant="destructive" 
                onClick={() => deleteConfirmId && handleDelete(deleteConfirmId)}
              >
                Delete
              </Button>
            </div>
          </DialogContent>
        </Dialog>

        {/* Error Toast */}
        {error && (
          <div className="absolute bottom-4 right-4 bg-destructive text-destructive-foreground px-4 py-2 rounded-lg flex items-center gap-2">
            <span>{error}</span>
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={clearError}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default ConversationHistoryPage;
