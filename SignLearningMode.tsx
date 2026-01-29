import React, { useState, useCallback, useEffect, useRef } from 'react';
import { 
  X, Play, Pause, SkipForward, RotateCcw, Trophy, 
  Target, Flame, Star, CheckCircle, XCircle, Clock,
  Camera, Volume2, ChevronRight, Award, Zap, BookOpen
} from 'lucide-react';
import { SIGN_DATABASE, SignData, getCategories, getSignsByCategory } from './SignDictionary';

interface LearningProgress {
  signsLearned: string[];
  currentStreak: number;
  longestStreak: number;
  totalPracticeTime: number;
  achievements: Achievement[];
  lessonProgress: Record<string, number>;
}

interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: string;
  unlockedAt?: Date;
}

interface SignLearningModeProps {
  isOpen: boolean;
  onClose: () => void;
  onPlaySign: (sign: SignData) => void;
  recognizedSign?: string | null;
  language: string;
}

// Achievement definitions
const ACHIEVEMENTS: Achievement[] = [
  { id: 'first_sign', name: 'First Steps', description: 'Learn your first sign', icon: '🎯' },
  { id: 'ten_signs', name: 'Getting Started', description: 'Learn 10 signs', icon: '📚' },
  { id: 'fifty_signs', name: 'Sign Scholar', description: 'Learn 50 signs', icon: '🎓' },
  { id: 'hundred_signs', name: 'Sign Master', description: 'Learn 100 signs', icon: '👑' },
  { id: 'streak_3', name: 'On Fire', description: '3 day streak', icon: '🔥' },
  { id: 'streak_7', name: 'Week Warrior', description: '7 day streak', icon: '⚡' },
  { id: 'streak_30', name: 'Monthly Master', description: '30 day streak', icon: '🏆' },
  { id: 'perfect_lesson', name: 'Perfect Score', description: 'Complete a lesson with 100% accuracy', icon: '💯' },
  { id: 'speed_demon', name: 'Speed Demon', description: 'Complete 10 signs in under 2 minutes', icon: '🚀' },
  { id: 'all_greetings', name: 'Social Butterfly', description: 'Learn all greeting signs', icon: '👋' },
  { id: 'all_emotions', name: 'Emotionally Intelligent', description: 'Learn all emotion signs', icon: '❤️' },
  { id: 'all_actions', name: 'Action Hero', description: 'Learn all action signs', icon: '💪' },
];

// Lesson structure
interface Lesson {
  id: string;
  title: string;
  description: string;
  category: string;
  signs: string[];
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedTime: number; // minutes
}

const LESSONS: Lesson[] = [
  {
    id: 'greetings-basics',
    title: 'Basic Greetings',
    description: 'Learn essential greeting signs for everyday conversations',
    category: 'Greetings',
    signs: ['hello', 'goodbye', 'thank_you', 'please', 'sorry'],
    difficulty: 'beginner',
    estimatedTime: 5
  },
  {
    id: 'yes-no-responses',
    title: 'Yes, No & Responses',
    description: 'Master common response signs',
    category: 'Responses',
    signs: ['yes', 'no', 'maybe', 'okay', 'good', 'bad'],
    difficulty: 'beginner',
    estimatedTime: 5
  },
  {
    id: 'pronouns-basics',
    title: 'Personal Pronouns',
    description: 'Learn to sign I, you, we, they and more',
    category: 'Pronouns',
    signs: ['i_me', 'you', 'he_she_it', 'we', 'they', 'my_mine', 'your_yours'],
    difficulty: 'beginner',
    estimatedTime: 7
  },
  {
    id: 'questions-basics',
    title: 'Asking Questions',
    description: 'Learn to ask who, what, where, when, why, how',
    category: 'Questions',
    signs: ['what', 'where', 'when', 'why', 'who', 'how', 'how_much'],
    difficulty: 'intermediate',
    estimatedTime: 8
  },
  {
    id: 'emotions-feelings',
    title: 'Emotions & Feelings',
    description: 'Express your emotions in sign language',
    category: 'Emotions',
    signs: ['happy', 'sad', 'angry', 'scared', 'surprised', 'tired', 'love', 'i_love_you', 'excited', 'worried'],
    difficulty: 'intermediate',
    estimatedTime: 10
  },
  {
    id: 'common-actions',
    title: 'Common Actions',
    description: 'Learn everyday action verbs',
    category: 'Actions',
    signs: ['help', 'want', 'need', 'have', 'go', 'come', 'eat', 'drink', 'sleep', 'work'],
    difficulty: 'intermediate',
    estimatedTime: 10
  },
  {
    id: 'time-concepts',
    title: 'Time Concepts',
    description: 'Express time-related concepts',
    category: 'Time',
    signs: ['now', 'later', 'before', 'after', 'today', 'tomorrow', 'yesterday', 'week', 'month', 'year'],
    difficulty: 'intermediate',
    estimatedTime: 10
  },
  {
    id: 'family-people',
    title: 'Family & People',
    description: 'Learn signs for family members and people',
    category: 'People',
    signs: ['person', 'friend', 'family', 'mother', 'father', 'child', 'baby', 'boy', 'girl', 'teacher'],
    difficulty: 'beginner',
    estimatedTime: 10
  }
];

const SignLearningMode: React.FC<SignLearningModeProps> = ({
  isOpen,
  onClose,
  onPlaySign,
  recognizedSign,
  language
}) => {
  // State
  const [activeTab, setActiveTab] = useState<'lessons' | 'practice' | 'achievements'>('lessons');
  const [selectedLesson, setSelectedLesson] = useState<Lesson | null>(null);
  const [currentSignIndex, setCurrentSignIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showAnswer, setShowAnswer] = useState(false);
  const [userAttempts, setUserAttempts] = useState<Record<string, 'correct' | 'incorrect' | null>>({});
  const [lessonStartTime, setLessonStartTime] = useState<number | null>(null);
  const [progress, setProgress] = useState<LearningProgress>({
    signsLearned: [],
    currentStreak: 0,
    longestStreak: 0,
    totalPracticeTime: 0,
    achievements: [],
    lessonProgress: {}
  });
  const [practiceMode, setPracticeMode] = useState<'watch' | 'practice'>('watch');
  const [feedback, setFeedback] = useState<{ type: 'correct' | 'incorrect' | 'hint'; message: string } | null>(null);

  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Get current sign in lesson
  const currentSign = selectedLesson 
    ? SIGN_DATABASE[selectedLesson.signs[currentSignIndex]]
    : null;

  // Handle sign recognition feedback
  useEffect(() => {
    if (recognizedSign && currentSign && practiceMode === 'practice') {
      const normalizedRecognized = recognizedSign.toLowerCase().replace(/[\s-]/g, '_');
      const normalizedCurrent = currentSign.name.toLowerCase().replace(/[\s-]/g, '_');
      
      if (normalizedRecognized === normalizedCurrent || 
          normalizedRecognized.includes(normalizedCurrent) ||
          normalizedCurrent.includes(normalizedRecognized)) {
        // Correct!
        setFeedback({ type: 'correct', message: 'Perfect! Great job!' });
        setUserAttempts(prev => ({ ...prev, [currentSign.name]: 'correct' }));
        
        // Auto advance after delay
        setTimeout(() => {
          if (currentSignIndex < (selectedLesson?.signs.length || 0) - 1) {
            setCurrentSignIndex(prev => prev + 1);
            setFeedback(null);
            setShowAnswer(false);
          }
        }, 1500);
      } else {
        // Incorrect
        setFeedback({ type: 'incorrect', message: `Not quite. You signed "${recognizedSign.replace(/_/g, ' ')}"` });
      }
    }
  }, [recognizedSign, currentSign, practiceMode, currentSignIndex, selectedLesson]);

  // Start lesson
  const handleStartLesson = useCallback((lesson: Lesson) => {
    setSelectedLesson(lesson);
    setCurrentSignIndex(0);
    setUserAttempts({});
    setLessonStartTime(Date.now());
    setShowAnswer(false);
    setFeedback(null);
    setPracticeMode('watch');
  }, []);

  // Play current sign
  const handlePlaySign = useCallback(() => {
    if (currentSign) {
      setIsPlaying(true);
      onPlaySign(currentSign);
      setTimeout(() => setIsPlaying(false), currentSign.duration + 200);
    }
  }, [currentSign, onPlaySign]);

  // Next sign
  const handleNextSign = useCallback(() => {
    if (selectedLesson && currentSignIndex < selectedLesson.signs.length - 1) {
      setCurrentSignIndex(prev => prev + 1);
      setShowAnswer(false);
      setFeedback(null);
    } else {
      // Lesson complete
      const correctCount = Object.values(userAttempts).filter(v => v === 'correct').length;
      const totalSigns = selectedLesson?.signs.length || 0;
      const accuracy = totalSigns > 0 ? (correctCount / totalSigns) * 100 : 0;
      
      // Update progress
      setProgress(prev => ({
        ...prev,
        signsLearned: [...new Set([...prev.signsLearned, ...(selectedLesson?.signs || [])])],
        lessonProgress: {
          ...prev.lessonProgress,
          [selectedLesson?.id || '']: Math.max(prev.lessonProgress[selectedLesson?.id || ''] || 0, accuracy)
        }
      }));
      
      // Show completion
      setFeedback({
        type: 'correct',
        message: `Lesson complete! Accuracy: ${accuracy.toFixed(0)}%`
      });
    }
  }, [selectedLesson, currentSignIndex, userAttempts]);

  // Previous sign
  const handlePrevSign = useCallback(() => {
    if (currentSignIndex > 0) {
      setCurrentSignIndex(prev => prev - 1);
      setShowAnswer(false);
      setFeedback(null);
    }
  }, [currentSignIndex]);

  // Reset lesson
  const handleResetLesson = useCallback(() => {
    setCurrentSignIndex(0);
    setUserAttempts({});
    setShowAnswer(false);
    setFeedback(null);
    setLessonStartTime(Date.now());
  }, []);

  // Calculate lesson progress
  const getLessonProgress = useCallback((lessonId: string) => {
    return progress.lessonProgress[lessonId] || 0;
  }, [progress.lessonProgress]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-5xl max-h-[90vh] bg-card rounded-2xl shadow-2xl border border-border overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border bg-gradient-to-r from-emerald-600/10 to-teal-600/10">
          <div>
            <h2 className="text-2xl font-bold flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
                <BookOpen className="w-5 h-5 text-white" />
              </div>
              Learn Sign Language
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
              Interactive lessons with real-time feedback • {language}
            </p>
          </div>
          <div className="flex items-center gap-4">
            {/* Stats */}
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-amber-500/10 text-amber-600">
                <Flame className="w-4 h-4" />
                <span className="font-medium">{progress.currentStreak} day streak</span>
              </div>
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-violet-500/10 text-violet-600">
                <Star className="w-4 h-4" />
                <span className="font-medium">{progress.signsLearned.length} signs learned</span>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-muted transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-border">
          {(['lessons', 'practice', 'achievements'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => {
                setActiveTab(tab);
                setSelectedLesson(null);
              }}
              className={`flex-1 px-6 py-4 text-sm font-medium transition-colors ${
                activeTab === tab
                  ? 'text-primary border-b-2 border-primary bg-primary/5'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {tab === 'lessons' && <BookOpen className="w-4 h-4 inline mr-2" />}
              {tab === 'practice' && <Target className="w-4 h-4 inline mr-2" />}
              {tab === 'achievements' && <Trophy className="w-4 h-4 inline mr-2" />}
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {/* Lessons Tab */}
          {activeTab === 'lessons' && !selectedLesson && (
            <div className="grid md:grid-cols-2 gap-4">
              {LESSONS.map(lesson => {
                const lessonProgress = getLessonProgress(lesson.id);
                return (
                  <div
                    key={lesson.id}
                    onClick={() => handleStartLesson(lesson)}
                    className="p-5 rounded-xl border border-border hover:border-primary cursor-pointer transition-all hover:shadow-lg group"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <h3 className="font-semibold group-hover:text-primary transition-colors">
                          {lesson.title}
                        </h3>
                        <p className="text-sm text-muted-foreground mt-1">
                          {lesson.description}
                        </p>
                      </div>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        lesson.difficulty === 'beginner' ? 'bg-green-500/10 text-green-600' :
                        lesson.difficulty === 'intermediate' ? 'bg-amber-500/10 text-amber-600' :
                        'bg-red-500/10 text-red-600'
                      }`}>
                        {lesson.difficulty}
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between text-sm text-muted-foreground">
                      <div className="flex items-center gap-4">
                        <span className="flex items-center gap-1">
                          <Target className="w-4 h-4" />
                          {lesson.signs.length} signs
                        </span>
                        <span className="flex items-center gap-1">
                          <Clock className="w-4 h-4" />
                          {lesson.estimatedTime} min
                        </span>
                      </div>
                      {lessonProgress > 0 && (
                        <span className="flex items-center gap-1 text-emerald-600">
                          <CheckCircle className="w-4 h-4" />
                          {lessonProgress.toFixed(0)}%
                        </span>
                      )}
                    </div>

                    {/* Progress bar */}
                    <div className="mt-3 h-1.5 rounded-full bg-muted overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-emerald-500 to-teal-500 transition-all"
                        style={{ width: `${lessonProgress}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Active Lesson */}
          {activeTab === 'lessons' && selectedLesson && currentSign && (
            <div className="max-w-3xl mx-auto">
              {/* Lesson Header */}
              <div className="flex items-center justify-between mb-6">
                <button
                  onClick={() => setSelectedLesson(null)}
                  className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
                >
                  <ChevronRight className="w-4 h-4 rotate-180" />
                  Back to Lessons
                </button>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">
                    {currentSignIndex + 1} / {selectedLesson.signs.length}
                  </span>
                  <div className="w-32 h-2 rounded-full bg-muted overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-emerald-500 to-teal-500 transition-all"
                      style={{ width: `${((currentSignIndex + 1) / selectedLesson.signs.length) * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Mode Toggle */}
              <div className="flex justify-center mb-6">
                <div className="inline-flex rounded-xl border border-border p-1 bg-muted/50">
                  <button
                    onClick={() => setPracticeMode('watch')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      practiceMode === 'watch'
                        ? 'bg-background shadow text-foreground'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <Play className="w-4 h-4 inline mr-2" />
                    Watch
                  </button>
                  <button
                    onClick={() => setPracticeMode('practice')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      practiceMode === 'practice'
                        ? 'bg-background shadow text-foreground'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <Camera className="w-4 h-4 inline mr-2" />
                    Practice
                  </button>
                </div>
              </div>

              {/* Sign Card */}
              <div className="p-8 rounded-2xl border border-border bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
                <div className="text-center mb-6">
                  <h3 className="text-3xl font-bold mb-2">{currentSign.name}</h3>
                  <p className="text-muted-foreground">{currentSign.description}</p>
                  <div className="flex items-center justify-center gap-3 mt-3">
                    <span className="px-3 py-1 rounded-full bg-primary/10 text-primary text-sm">
                      {currentSign.category}
                    </span>
                    <span className="px-3 py-1 rounded-full bg-muted text-sm">
                      {currentSign.handshape}
                    </span>
                  </div>
                </div>

                {/* Play Button */}
                <div className="flex justify-center mb-6">
                  <button
                    onClick={handlePlaySign}
                    disabled={isPlaying}
                    className="w-20 h-20 rounded-full bg-gradient-to-br from-emerald-500 to-teal-600 text-white flex items-center justify-center shadow-lg hover:shadow-xl transition-all hover:scale-105 disabled:opacity-50"
                  >
                    {isPlaying ? (
                      <div className="w-6 h-6 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <Play className="w-8 h-8 ml-1" />
                    )}
                  </button>
                </div>

                {/* Practice Mode Instructions */}
                {practiceMode === 'practice' && (
                  <div className="text-center p-4 rounded-xl bg-blue-500/10 border border-blue-500/20 mb-4">
                    <Camera className="w-6 h-6 text-blue-500 mx-auto mb-2" />
                    <p className="text-sm text-blue-700 dark:text-blue-300">
                      Watch the avatar, then try signing it yourself. The camera will check your sign!
                    </p>
                  </div>
                )}

                {/* Feedback */}
                {feedback && (
                  <div className={`text-center p-4 rounded-xl mb-4 ${
                    feedback.type === 'correct' 
                      ? 'bg-green-500/10 border border-green-500/20' 
                      : feedback.type === 'incorrect'
                        ? 'bg-red-500/10 border border-red-500/20'
                        : 'bg-amber-500/10 border border-amber-500/20'
                  }`}>
                    {feedback.type === 'correct' && <CheckCircle className="w-6 h-6 text-green-500 mx-auto mb-2" />}
                    {feedback.type === 'incorrect' && <XCircle className="w-6 h-6 text-red-500 mx-auto mb-2" />}
                    <p className={`text-sm font-medium ${
                      feedback.type === 'correct' ? 'text-green-700 dark:text-green-300' :
                      feedback.type === 'incorrect' ? 'text-red-700 dark:text-red-300' :
                      'text-amber-700 dark:text-amber-300'
                    }`}>
                      {feedback.message}
                    </p>
                  </div>
                )}

                {/* Hint */}
                {!showAnswer && (
                  <button
                    onClick={() => setShowAnswer(true)}
                    className="w-full text-center text-sm text-muted-foreground hover:text-foreground"
                  >
                    Need a hint? Click to show details
                  </button>
                )}

                {showAnswer && (
                  <div className="mt-4 p-4 rounded-xl bg-muted/50">
                    <h4 className="font-medium mb-2">How to sign "{currentSign.name}":</h4>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      <li>• Movement: {currentSign.movement}</li>
                      <li>• Location: {currentSign.location}</li>
                      <li>• Handshape: {currentSign.handshape}</li>
                    </ul>
                  </div>
                )}
              </div>

              {/* Navigation */}
              <div className="flex items-center justify-between mt-6">
                <button
                  onClick={handlePrevSign}
                  disabled={currentSignIndex === 0}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg border border-border hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <ChevronRight className="w-4 h-4 rotate-180" />
                  Previous
                </button>
                
                <button
                  onClick={handleResetLesson}
                  className="p-2 rounded-lg hover:bg-muted"
                  title="Reset Lesson"
                >
                  <RotateCcw className="w-5 h-5" />
                </button>

                <button
                  onClick={handleNextSign}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:opacity-90"
                >
                  {currentSignIndex === selectedLesson.signs.length - 1 ? 'Finish' : 'Next'}
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}

          {/* Practice Tab - Quick Practice */}
          {activeTab === 'practice' && (
            <div className="max-w-4xl mx-auto">
              <div className="text-center mb-8">
                <h3 className="text-xl font-semibold mb-2">Quick Practice</h3>
                <p className="text-muted-foreground">
                  Select a category and practice signs at your own pace
                </p>
              </div>

              {/* Category Selection */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                {getCategories().map(category => {
                  const signs = getSignsByCategory(category);
                  return (
                    <button
                      key={category}
                      onClick={() => {
                        const randomSign = signs[Math.floor(Math.random() * signs.length)];
                        onPlaySign(randomSign);
                      }}
                      className="p-4 rounded-xl border border-border hover:border-primary hover:shadow-lg transition-all text-center group"
                    >
                      <div className="w-12 h-12 rounded-full bg-gradient-to-br from-emerald-500/10 to-teal-500/10 flex items-center justify-center mx-auto mb-3 group-hover:scale-110 transition-transform">
                        <Zap className="w-6 h-6 text-emerald-600" />
                      </div>
                      <h4 className="font-medium">{category}</h4>
                      <p className="text-sm text-muted-foreground">{signs.length} signs</p>
                    </button>
                  );
                })}
              </div>

              {/* All Signs Grid */}
              <h4 className="font-semibold mb-4">All Signs ({Object.keys(SIGN_DATABASE).length})</h4>
              <div className="grid grid-cols-3 md:grid-cols-5 lg:grid-cols-6 gap-2">
                {Object.values(SIGN_DATABASE).map(sign => (
                  <button
                    key={sign.name}
                    onClick={() => onPlaySign(sign)}
                    className={`p-3 rounded-lg border text-sm text-left transition-all hover:border-primary hover:shadow ${
                      progress.signsLearned.includes(sign.name.toLowerCase().replace(/[\s-]/g, '_'))
                        ? 'border-emerald-500/50 bg-emerald-500/5'
                        : 'border-border'
                    }`}
                  >
                    <span className="font-medium truncate block">{sign.name}</span>
                    <span className="text-xs text-muted-foreground">{sign.category}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Achievements Tab */}
          {activeTab === 'achievements' && (
            <div className="max-w-3xl mx-auto">
              <div className="text-center mb-8">
                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center mx-auto mb-4">
                  <Trophy className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-2">Your Achievements</h3>
                <p className="text-muted-foreground">
                  {progress.achievements.length} / {ACHIEVEMENTS.length} unlocked
                </p>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                {ACHIEVEMENTS.map(achievement => {
                  const isUnlocked = progress.achievements.some(a => a.id === achievement.id);
                  return (
                    <div
                      key={achievement.id}
                      className={`p-4 rounded-xl border transition-all ${
                        isUnlocked
                          ? 'border-amber-500/50 bg-amber-500/5'
                          : 'border-border opacity-60'
                      }`}
                    >
                      <div className="flex items-center gap-4">
                        <div className={`w-12 h-12 rounded-full flex items-center justify-center text-2xl ${
                          isUnlocked ? 'bg-amber-500/20' : 'bg-muted'
                        }`}>
                          {achievement.icon}
                        </div>
                        <div className="flex-1">
                          <h4 className="font-medium">{achievement.name}</h4>
                          <p className="text-sm text-muted-foreground">{achievement.description}</p>
                        </div>
                        {isUnlocked && (
                          <CheckCircle className="w-5 h-5 text-amber-500" />
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SignLearningMode;
