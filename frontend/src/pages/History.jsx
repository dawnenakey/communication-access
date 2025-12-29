import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { toast } from "sonner";
import { useAuth, API } from "../App";
import { Button } from "../components/ui/button";
import { ScrollArea } from "../components/ui/scroll-area";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "../components/ui/alert-dialog";
import { Avatar, AvatarFallback, AvatarImage } from "../components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../components/ui/dropdown-menu";
import { 
  Hand, Trash2, Volume2, Clock, ArrowRight, ArrowLeft,
  BookOpen, Camera, History as HistoryIcon, LogOut, User, Settings, 
  Loader2, ChevronLeft, Calendar
} from "lucide-react";
import { format, formatDistanceToNow } from "date-fns";

export default function History() {
  const { user } = useAuth();
  const navigate = useNavigate();
  
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    setIsLoading(true);
    try {
      const response = await axios.get(`${API}/history`);
      setHistory(response.data);
    } catch (error) {
      toast.error("Failed to load history");
    } finally {
      setIsLoading(false);
    }
  };

  const deleteHistoryItem = async (historyId) => {
    try {
      await axios.delete(`${API}/history/${historyId}`);
      toast.success("Entry deleted");
      loadHistory();
    } catch (error) {
      toast.error("Failed to delete entry");
    }
  };

  const clearAllHistory = async () => {
    try {
      await axios.delete(`${API}/history`);
      toast.success("History cleared");
      setHistory([]);
    } catch (error) {
      toast.error("Failed to clear history");
    }
  };

  const speakText = (text) => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
    toast.success("Speaking...");
  };

  const handleLogout = async () => {
    try {
      await axios.post(`${API}/auth/logout`);
      navigate("/");
    } catch (error) {
      navigate("/");
    }
  };

  const formatDate = (dateString) => {
    try {
      const date = new Date(dateString);
      return formatDistanceToNow(date, { addSuffix: true });
    } catch {
      return dateString;
    }
  };

  const getTypeIcon = (type) => {
    if (type === "asl_to_text") {
      return <ArrowRight className="w-4 h-4" />;
    }
    return <ArrowLeft className="w-4 h-4" />;
  };

  const getTypeLabel = (type) => {
    if (type === "asl_to_text") {
      return "ASL → Text";
    }
    return "Text → ASL";
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 glass border-b border-white/5">
        <div className="flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-500 flex items-center justify-center">
              <Hand className="w-5 h-5 text-black" />
            </div>
            <span className="font-heading font-bold text-xl tracking-tight">SignSync AI</span>
          </div>

          <nav className="hidden md:flex items-center gap-2">
            <Button 
              variant="ghost" 
              onClick={() => navigate("/dashboard")}
              data-testid="nav-dashboard"
            >
              <Camera className="w-4 h-4 mr-2" />
              Translate
            </Button>
            <Button 
              variant="ghost" 
              onClick={() => navigate("/dictionary")}
              data-testid="nav-dictionary"
            >
              <BookOpen className="w-4 h-4 mr-2" />
              Dictionary
            </Button>
            <Button 
              variant="ghost" 
              onClick={() => navigate("/history")}
              data-testid="nav-history"
              className="text-foreground bg-white/5"
            >
              <HistoryIcon className="w-4 h-4 mr-2" />
              History
            </Button>
          </nav>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="gap-2" data-testid="user-menu-btn">
                <Avatar className="w-8 h-8">
                  <AvatarImage src={user?.picture} alt={user?.name} />
                  <AvatarFallback className="bg-primary/20 text-primary">
                    {user?.name?.charAt(0) || "U"}
                  </AvatarFallback>
                </Avatar>
                <span className="hidden md:inline font-medium">{user?.name}</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56 glass">
              <DropdownMenuItem className="cursor-pointer" disabled>
                <User className="w-4 h-4 mr-2" />
                Profile
              </DropdownMenuItem>
              <DropdownMenuItem className="cursor-pointer" disabled>
                <Settings className="w-4 h-4 mr-2" />
                Settings
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem 
                className="cursor-pointer text-destructive focus:text-destructive"
                onClick={handleLogout}
                data-testid="logout-btn"
              >
                <LogOut className="w-4 h-4 mr-2" />
                Logout
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      {/* Main Content */}
      <main className="p-6 lg:p-8">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Page Header */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={() => navigate("/dashboard")}
                className="mb-2 -ml-2"
              >
                <ChevronLeft className="w-4 h-4 mr-1" />
                Back to Dashboard
              </Button>
              <h1 className="font-heading font-bold text-3xl md:text-4xl tracking-tight">
                Translation History
              </h1>
              <p className="text-muted-foreground mt-1">
                View your past translations and speak them again
              </p>
            </div>
            
            {history.length > 0 && (
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button 
                    variant="outline"
                    className="rounded-full border-white/10 text-destructive hover:text-destructive"
                    data-testid="clear-all-btn"
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    Clear All
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent className="glass">
                  <AlertDialogHeader>
                    <AlertDialogTitle>Clear All History</AlertDialogTitle>
                    <AlertDialogDescription>
                      Are you sure you want to delete all translation history? This action cannot be undone.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel className="rounded-full">Cancel</AlertDialogCancel>
                    <AlertDialogAction 
                      onClick={clearAllHistory}
                      className="rounded-full bg-destructive text-destructive-foreground"
                    >
                      Clear All
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            )}
          </div>

          {/* History List */}
          {isLoading ? (
            <div className="flex items-center justify-center py-20">
              <Loader2 className="w-8 h-8 animate-spin text-cyan-400" />
            </div>
          ) : history.length > 0 ? (
            <div className="space-y-4">
              {history.map((item) => (
                <div 
                  key={item.history_id}
                  className="glass-card p-6 space-y-4"
                  data-testid={`history-item-${item.history_id}`}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                        item.input_type === "asl_to_text" 
                          ? "bg-cyan-400/10 text-cyan-400" 
                          : "bg-pink-400/10 text-pink-400"
                      }`}>
                        {getTypeIcon(item.input_type)}
                      </div>
                      <div>
                        <span className={`text-sm font-medium ${
                          item.input_type === "asl_to_text" 
                            ? "text-cyan-400" 
                            : "text-pink-400"
                        }`}>
                          {getTypeLabel(item.input_type)}
                        </span>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
                          <Clock className="w-3 h-3" />
                          {formatDate(item.created_at)}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      {item.confidence && (
                        <span className="font-mono text-sm text-cyan-400">
                          {item.confidence.toFixed(1)}%
                        </span>
                      )}
                      <Button 
                        variant="ghost"
                        size="icon"
                        onClick={() => speakText(item.output_content)}
                        className="rounded-full"
                        data-testid={`speak-${item.history_id}`}
                      >
                        <Volume2 className="w-4 h-4" />
                      </Button>
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button 
                            variant="ghost"
                            size="icon"
                            className="rounded-full text-muted-foreground hover:text-destructive"
                            data-testid={`delete-${item.history_id}`}
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent className="glass">
                          <AlertDialogHeader>
                            <AlertDialogTitle>Delete Entry</AlertDialogTitle>
                            <AlertDialogDescription>
                              Are you sure you want to delete this translation?
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel className="rounded-full">Cancel</AlertDialogCancel>
                            <AlertDialogAction 
                              onClick={() => deleteHistoryItem(item.history_id)}
                              className="rounded-full bg-destructive text-destructive-foreground"
                            >
                              Delete
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </div>
                  </div>

                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="p-4 rounded-xl bg-black/20 border border-white/5">
                      <p className="text-xs text-muted-foreground mb-2">Input</p>
                      <p className="text-sm">{item.input_content}</p>
                    </div>
                    <div className="p-4 rounded-xl bg-black/20 border border-white/5">
                      <p className="text-xs text-muted-foreground mb-2">Output</p>
                      <p className="text-sm">{item.output_content}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="glass-card p-12 text-center">
              <HistoryIcon className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
              <h3 className="font-heading font-bold text-xl mb-2">No history yet</h3>
              <p className="text-muted-foreground mb-6">
                Start translating ASL to build your history
              </p>
              <Button 
                onClick={() => navigate("/dashboard")}
                className="rounded-full bg-primary text-primary-foreground"
              >
                <Camera className="w-4 h-4 mr-2" />
                Start Translating
              </Button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
