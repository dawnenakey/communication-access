import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, Mic, MicOff, Trash2, Download, 
  MessageSquare, User, Bot, Clock, Save, History
} from 'lucide-react';

interface Message {
  id: string;
  type: 'user' | 'avatar';
  content: string;
  timestamp: Date;
  confidence?: number;
  language: string;
}

interface ConversationPanelProps {
  messages: Message[];
  onSendMessage: (message: string) => void;
  onClearHistory: () => void;
  language: string;
  isAvatarResponding: boolean;
  onSaveConversation?: () => void;
  onOpenHistory?: () => void;
  isSaving?: boolean;
}

const ConversationPanel: React.FC<ConversationPanelProps> = ({
  messages,
  onSendMessage,
  onClearHistory,
  language,
  isAvatarResponding,
  onSaveConversation,
  onOpenHistory,
  isSaving = false
}) => {

  const [inputText, setInputText] = useState('');
  const [isListening, setIsListening] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputText.trim() && !isAvatarResponding) {
      onSendMessage(inputText.trim());
      setInputText('');
    }
  };

  const toggleListening = () => {
    setIsListening(!isListening);
    // Simulated voice input
    if (!isListening) {
      setTimeout(() => {
        setInputText('What features does this app have?');
        setIsListening(false);
      }, 2000);
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const exportConversation = () => {
    const text = messages.map(m => 
      `[${formatTime(m.timestamp)}] ${m.type === 'user' ? 'You' : 'SonZo'}: ${m.content}`
    ).join('\n');
    
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sonzo-conversation-${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Quick reply suggestions
  const quickReplies = [
    "Hello, how are you?",
    "What can you do?",
    "Teach me a sign",
    "How does this work?",
    "Thank you for helping"
  ];

  return (
    <div className="flex flex-col h-full bg-card rounded-2xl border border-border overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-muted/30">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
            <MessageSquare className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="font-semibold">Conversation</h3>
            <p className="text-xs text-muted-foreground">
              {messages.length} messages • {language}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {onOpenHistory && (
            <button
              onClick={onOpenHistory}
              className="p-2 rounded-lg hover:bg-muted transition-colors"
              title="View conversation history"
            >
              <History className="w-4 h-4" />
            </button>
          )}
          {onSaveConversation && messages.length > 0 && (
            <button
              onClick={onSaveConversation}
              disabled={isSaving}
              className="p-2 rounded-lg hover:bg-muted transition-colors disabled:opacity-50"
              title="Save conversation"
            >
              <Save className={`w-4 h-4 ${isSaving ? 'animate-pulse' : ''}`} />
            </button>
          )}
          <button
            onClick={exportConversation}
            className="p-2 rounded-lg hover:bg-muted transition-colors"
            title="Export conversation"
          >
            <Download className="w-4 h-4" />
          </button>
          <button
            onClick={onClearHistory}
            className="p-2 rounded-lg hover:bg-muted text-red-500 transition-colors"
            title="Clear history"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>


      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center py-8">
            <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-4">
              <MessageSquare className="w-8 h-8 text-primary" />
            </div>
            <h4 className="font-semibold mb-2">Start a Conversation</h4>
            <p className="text-sm text-muted-foreground max-w-xs mb-4">
              Sign a sentence using the camera or type a message below. 
              The avatar will respond in {language}.
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {quickReplies.slice(0, 3).map((reply, i) => (
                <button
                  key={i}
                  onClick={() => onSendMessage(reply)}
                  className="px-3 py-1.5 rounded-full text-xs bg-muted hover:bg-muted/80 transition-colors"
                >
                  {reply}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.type === 'user' ? 'flex-row-reverse' : ''}`}
              >
                {/* Avatar */}
                <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                  message.type === 'user' 
                    ? 'bg-primary' 
                    : 'bg-gradient-to-br from-violet-500 to-purple-600'
                }`}>
                  {message.type === 'user' ? (
                    <User className="w-4 h-4 text-white" />
                  ) : (
                    <Bot className="w-4 h-4 text-white" />
                  )}
                </div>

                {/* Message Bubble */}
                <div className={`flex-1 max-w-[80%] ${message.type === 'user' ? 'text-right' : ''}`}>
                  <div className={`inline-block px-4 py-3 rounded-2xl ${
                    message.type === 'user'
                      ? 'conversation-bubble-user'
                      : 'conversation-bubble-avatar'
                  }`}>
                    <p className="text-sm">{message.content}</p>
                  </div>
                  
                  <div className={`flex items-center gap-2 mt-1 text-xs text-muted-foreground ${
                    message.type === 'user' ? 'justify-end' : ''
                  }`}>
                    <Clock className="w-3 h-3" />
                    <span>{formatTime(message.timestamp)}</span>
                    {message.confidence && (
                      <>
                        <span>•</span>
                        <span>{(message.confidence * 100).toFixed(0)}% confidence</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            ))}

            {/* Typing Indicator */}
            {isAvatarResponding && (
              <div className="flex gap-3">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                  <Bot className="w-4 h-4 text-white" />
                </div>
                <div className="px-4 py-3 rounded-2xl conversation-bubble-avatar">
                  <div className="flex gap-1">
                    <span className="w-2 h-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-2 h-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-2 h-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Quick Replies */}
      {messages.length > 0 && (
        <div className="px-4 py-2 border-t border-border bg-muted/20">
          <div className="flex items-center gap-2 overflow-x-auto pb-1">
            <span className="text-xs text-muted-foreground flex-shrink-0">Quick:</span>
            {quickReplies.map((reply, i) => (
              <button
                key={i}
                onClick={() => onSendMessage(reply)}
                disabled={isAvatarResponding}
                className="flex-shrink-0 px-3 py-1 rounded-full text-xs bg-muted hover:bg-primary hover:text-primary-foreground transition-colors disabled:opacity-50"
              >
                {reply}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input Area */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-border bg-muted/30">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={toggleListening}
            className={`p-3 rounded-xl transition-colors ${
              isListening 
                ? 'bg-red-500 text-white animate-pulse' 
                : 'bg-muted hover:bg-muted/80'
            }`}
          >
            {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
          </button>

          <div className="flex-1 relative">
            <input
              ref={inputRef}
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Type a message or use voice..."
              disabled={isAvatarResponding}
              className="w-full px-4 py-3 rounded-xl bg-background border border-border focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none transition-all disabled:opacity-50"
            />
          </div>

          <button
            type="submit"
            disabled={!inputText.trim() || isAvatarResponding}
            className="p-3 rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>

        <p className="text-xs text-muted-foreground mt-2 text-center">
          Sign language input from camera will appear here automatically
        </p>
      </form>
    </div>
  );
};

export default ConversationPanel;
