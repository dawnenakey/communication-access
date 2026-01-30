import React, { useState, useEffect } from 'react';
import { 
  X, User, Mail, Globe, Camera, Edit2, Save,
  Trophy, Clock, MessageSquare, BookOpen, 
  BarChart3, Calendar, Loader2, CheckCircle
} from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

interface UserProfileModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface UserStats {
  completedLessons: number;
  totalTimeMinutes: number;
  totalSentencesPracticed: number;
  totalConversations: number;
}

interface LearningProgress {
  lessonId: string;
  language: string;
  completed: boolean;
  score?: number;
  sentencesPracticed: number;
  timeSpentSeconds: number;
  completedAt?: string;
}

const UserProfileModal: React.FC<UserProfileModalProps> = ({ isOpen, onClose }) => {
  const { user, updateProfile, getUserStats, getLearningProgress } = useAuth();
  const [activeTab, setActiveTab] = useState<'profile' | 'progress' | 'history'>('profile');
  const [isEditing, setIsEditing] = useState(false);
  const [displayName, setDisplayName] = useState(user?.displayName || '');
  const [preferredLanguage, setPreferredLanguage] = useState(user?.preferredLanguage || 'ASL');
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [stats, setStats] = useState<UserStats | null>(null);
  const [progress, setProgress] = useState<LearningProgress[]>([]);

  useEffect(() => {
    if (isOpen && user) {
      setDisplayName(user.displayName);
      setPreferredLanguage(user.preferredLanguage);
      loadUserData();
    }
  }, [isOpen, user]);

  const loadUserData = async () => {
    setIsLoading(true);
    try {
      const [userStats, userProgress] = await Promise.all([
        getUserStats(),
        getLearningProgress()
      ]);
      setStats(userStats);
      setProgress(userProgress);
    } catch (err) {
      console.error('Failed to load user data:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSaveProfile = async () => {
    setIsSaving(true);
    try {
      await updateProfile({
        displayName,
        preferredLanguage
      });
      setIsEditing(false);
    } catch (err) {
      console.error('Failed to save profile:', err);
    } finally {
      setIsSaving(false);
    }
  };

  if (!isOpen || !user) return null;

  const languages = [
    { code: 'ASL', name: 'American Sign Language' },
    { code: 'BSL', name: 'British Sign Language' },
    { code: 'ISL', name: 'Indian Sign Language' },
    { code: 'LSF', name: 'French Sign Language' },
    { code: 'DGS', name: 'German Sign Language' },
    { code: 'JSL', name: 'Japanese Sign Language' },
  ];

  const lessonNames: Record<string, string> = {
    'asl-1': 'Basic Greetings',
    'asl-2': 'Introducing Yourself',
    'asl-3': 'Asking Questions',
    'asl-4': 'Daily Activities',
    'asl-5': 'Expressing Feelings',
    'asl-6': 'Making Plans',
    'asl-7': 'Complex Conversations',
    'asl-8': 'Professional Communication'
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-2xl bg-card rounded-2xl border border-border shadow-2xl overflow-hidden max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center overflow-hidden">
              {user.avatarUrl ? (
                <img src={user.avatarUrl} alt={user.displayName} className="w-full h-full object-cover" />
              ) : (
                <span className="text-2xl font-bold text-white">
                  {user.displayName?.charAt(0).toUpperCase() || user.email.charAt(0).toUpperCase()}
                </span>
              )}
            </div>
            <div>
              <h2 className="text-xl font-bold">{user.displayName}</h2>
              <p className="text-sm text-muted-foreground">{user.email}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-muted transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-border">
          {[
            { id: 'profile' as const, label: 'Profile', icon: User },
            { id: 'progress' as const, label: 'Progress', icon: BarChart3 },
            { id: 'history' as const, label: 'Activity', icon: Calendar }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'text-primary border-b-2 border-primary bg-primary/5'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 animate-spin text-primary" />
            </div>
          ) : (
            <>
              {/* Profile Tab */}
              {activeTab === 'profile' && (
                <div className="space-y-6">
                  {/* Stats Overview */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <div className="p-4 bg-muted/50 rounded-xl text-center">
                      <Trophy className="w-6 h-6 text-yellow-500 mx-auto mb-2" />
                      <p className="text-2xl font-bold">{stats?.completedLessons || 0}</p>
                      <p className="text-xs text-muted-foreground">Lessons</p>
                    </div>
                    <div className="p-4 bg-muted/50 rounded-xl text-center">
                      <Clock className="w-6 h-6 text-blue-500 mx-auto mb-2" />
                      <p className="text-2xl font-bold">{stats?.totalTimeMinutes || 0}</p>
                      <p className="text-xs text-muted-foreground">Minutes</p>
                    </div>
                    <div className="p-4 bg-muted/50 rounded-xl text-center">
                      <BookOpen className="w-6 h-6 text-green-500 mx-auto mb-2" />
                      <p className="text-2xl font-bold">{stats?.totalSentencesPracticed || 0}</p>
                      <p className="text-xs text-muted-foreground">Sentences</p>
                    </div>
                    <div className="p-4 bg-muted/50 rounded-xl text-center">
                      <MessageSquare className="w-6 h-6 text-purple-500 mx-auto mb-2" />
                      <p className="text-2xl font-bold">{stats?.totalConversations || 0}</p>
                      <p className="text-xs text-muted-foreground">Chats</p>
                    </div>
                  </div>

                  {/* Profile Form */}
                  <div className="bg-muted/30 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-semibold">Profile Information</h3>
                      {!isEditing ? (
                        <button
                          onClick={() => setIsEditing(true)}
                          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm hover:bg-muted transition-colors"
                        >
                          <Edit2 className="w-4 h-4" />
                          Edit
                        </button>
                      ) : (
                        <button
                          onClick={handleSaveProfile}
                          disabled={isSaving}
                          className="flex items-center gap-2 px-4 py-1.5 bg-primary text-primary-foreground rounded-lg text-sm hover:bg-primary/90 transition-colors disabled:opacity-50"
                        >
                          {isSaving ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                          ) : (
                            <Save className="w-4 h-4" />
                          )}
                          Save
                        </button>
                      )}
                    </div>

                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium mb-1.5">Display Name</label>
                        {isEditing ? (
                          <div className="relative">
                            <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                            <input
                              type="text"
                              value={displayName}
                              onChange={(e) => setDisplayName(e.target.value)}
                              className="w-full pl-11 pr-4 py-2.5 rounded-lg bg-background border border-border focus:border-primary outline-none"
                            />
                          </div>
                        ) : (
                          <p className="flex items-center gap-2 px-3 py-2.5 rounded-lg bg-background">
                            <User className="w-5 h-5 text-muted-foreground" />
                            {user.displayName}
                          </p>
                        )}
                      </div>

                      <div>
                        <label className="block text-sm font-medium mb-1.5">Email</label>
                        <p className="flex items-center gap-2 px-3 py-2.5 rounded-lg bg-background text-muted-foreground">
                          <Mail className="w-5 h-5" />
                          {user.email}
                        </p>
                      </div>

                      <div>
                        <label className="block text-sm font-medium mb-1.5">Preferred Language</label>
                        {isEditing ? (
                          <div className="relative">
                            <Globe className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                            <select
                              value={preferredLanguage}
                              onChange={(e) => setPreferredLanguage(e.target.value)}
                              className="w-full pl-11 pr-4 py-2.5 rounded-lg bg-background border border-border focus:border-primary outline-none appearance-none"
                            >
                              {languages.map((lang) => (
                                <option key={lang.code} value={lang.code}>
                                  {lang.code} - {lang.name}
                                </option>
                              ))}
                            </select>
                          </div>
                        ) : (
                          <p className="flex items-center gap-2 px-3 py-2.5 rounded-lg bg-background">
                            <Globe className="w-5 h-5 text-muted-foreground" />
                            {user.preferredLanguage} - {languages.find(l => l.code === user.preferredLanguage)?.name}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Progress Tab */}
              {activeTab === 'progress' && (
                <div className="space-y-4">
                  <h3 className="font-semibold">Learning Progress</h3>
                  
                  {progress.length === 0 ? (
                    <div className="text-center py-12">
                      <BookOpen className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">No lessons completed yet</p>
                      <p className="text-sm text-muted-foreground mt-1">Start learning to track your progress!</p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {progress.map((item, index) => (
                        <div
                          key={index}
                          className="flex items-center gap-4 p-4 bg-muted/30 rounded-xl"
                        >
                          <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                            item.completed ? 'bg-green-500/10' : 'bg-muted'
                          }`}>
                            {item.completed ? (
                              <CheckCircle className="w-5 h-5 text-green-500" />
                            ) : (
                              <BookOpen className="w-5 h-5 text-muted-foreground" />
                            )}
                          </div>
                          <div className="flex-1">
                            <p className="font-medium">{lessonNames[item.lessonId] || item.lessonId}</p>
                            <p className="text-sm text-muted-foreground">
                              {item.language} • {item.sentencesPracticed} sentences • {Math.round(item.timeSpentSeconds / 60)} min
                            </p>
                          </div>
                          {item.score && (
                            <div className="text-right">
                              <p className="font-bold text-primary">{item.score}%</p>
                              <p className="text-xs text-muted-foreground">Score</p>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Activity Tab */}
              {activeTab === 'history' && (
                <div className="space-y-4">
                  <h3 className="font-semibold">Recent Activity</h3>
                  
                  <div className="text-center py-12">
                    <Calendar className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-muted-foreground">Activity history coming soon</p>
                    <p className="text-sm text-muted-foreground mt-1">Track your daily learning streaks and achievements</p>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default UserProfileModal;
