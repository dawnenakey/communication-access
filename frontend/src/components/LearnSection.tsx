import React, { useState } from 'react';
import { 
  BookOpen, Play, CheckCircle, Circle, 
  ChevronRight, Star, Clock, Trophy,
  ArrowRight, Lock, Sparkles
} from 'lucide-react';

interface Lesson {
  id: string;
  title: string;
  description: string;
  duration: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  sentences: string[];
  completed: boolean;
  locked: boolean;
}

interface LearnSectionProps {
  language: string;
  onStartLesson: (lessonId: string, sentences: string[]) => void;
}

const lessons: Record<string, Lesson[]> = {
  ASL: [
    {
      id: 'asl-1',
      title: 'Basic Greetings',
      description: 'Learn essential greeting sentences in ASL',
      duration: '10 min',
      difficulty: 'Beginner',
      sentences: ['Hello, how are you?', 'My name is...', 'Nice to meet you', 'Good morning', 'Good night'],
      completed: true,
      locked: false
    },
    {
      id: 'asl-2',
      title: 'Introducing Yourself',
      description: 'Master self-introduction sentences',
      duration: '15 min',
      difficulty: 'Beginner',
      sentences: ['I am from...', 'I work as a...', 'I am learning sign language', 'I am deaf/hearing', 'This is my friend'],
      completed: true,
      locked: false
    },
    {
      id: 'asl-3',
      title: 'Asking Questions',
      description: 'Learn how to form questions in ASL',
      duration: '20 min',
      difficulty: 'Beginner',
      sentences: ['What is your name?', 'Where are you from?', 'How old are you?', 'Do you understand?', 'Can you help me?'],
      completed: false,
      locked: false
    },
    {
      id: 'asl-4',
      title: 'Daily Activities',
      description: 'Describe your daily routine',
      duration: '25 min',
      difficulty: 'Intermediate',
      sentences: ['I wake up early', 'I eat breakfast', 'I go to work', 'I come home late', 'I watch television'],
      completed: false,
      locked: false
    },
    {
      id: 'asl-5',
      title: 'Expressing Feelings',
      description: 'Communicate emotions and feelings',
      duration: '20 min',
      difficulty: 'Intermediate',
      sentences: ['I am happy today', 'I feel tired', 'I am excited', 'I am worried about...', 'I love this'],
      completed: false,
      locked: false
    },
    {
      id: 'asl-6',
      title: 'Making Plans',
      description: 'Discuss future plans and activities',
      duration: '25 min',
      difficulty: 'Intermediate',
      sentences: ['Let us meet tomorrow', 'I will go to the store', 'Do you want to come?', 'What time works for you?', 'See you later'],
      completed: false,
      locked: true
    },
    {
      id: 'asl-7',
      title: 'Complex Conversations',
      description: 'Advanced sentence structures',
      duration: '30 min',
      difficulty: 'Advanced',
      sentences: ['I have been learning for months', 'Could you repeat that please?', 'I did not understand', 'Let me explain differently', 'Thank you for your patience'],
      completed: false,
      locked: true
    },
    {
      id: 'asl-8',
      title: 'Professional Communication',
      description: 'Workplace and formal sentences',
      duration: '35 min',
      difficulty: 'Advanced',
      sentences: ['I have a meeting at noon', 'Please send me the document', 'I need more information', 'The deadline is Friday', 'I appreciate your help'],
      completed: false,
      locked: true
    }
  ]
};

const LearnSection: React.FC<LearnSectionProps> = ({ language, onStartLesson }) => {
  const [selectedDifficulty, setSelectedDifficulty] = useState<string>('All');
  const [expandedLesson, setExpandedLesson] = useState<string | null>(null);

  const currentLessons = lessons[language] || lessons.ASL;
  
  const filteredLessons = currentLessons.filter(lesson => 
    selectedDifficulty === 'All' || lesson.difficulty === selectedDifficulty
  );

  const completedCount = currentLessons.filter(l => l.completed).length;
  const totalCount = currentLessons.length;
  const progressPercent = (completedCount / totalCount) * 100;

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Beginner': return 'text-green-500 bg-green-500/10';
      case 'Intermediate': return 'text-yellow-500 bg-yellow-500/10';
      case 'Advanced': return 'text-red-500 bg-red-500/10';
      default: return 'text-muted-foreground bg-muted';
    }
  };

  return (
    <section id="learn" className="py-20">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 mb-4">
            <BookOpen className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium text-primary">Learn {language}</span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            Master <span className="gradient-text">Sentence Signing</span>
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Progress through structured lessons to learn complete sentences in {language}. 
            Practice with our AI and get real-time feedback.
          </p>
        </div>

        {/* Progress Overview */}
        <div className="grid sm:grid-cols-3 gap-6 mb-12">
          <div className="bg-card rounded-xl border border-border p-6 text-center">
            <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-3">
              <Trophy className="w-6 h-6 text-primary" />
            </div>
            <p className="text-3xl font-bold mb-1">{completedCount}/{totalCount}</p>
            <p className="text-sm text-muted-foreground">Lessons Completed</p>
          </div>

          <div className="bg-card rounded-xl border border-border p-6 text-center">
            <div className="w-12 h-12 rounded-full bg-yellow-500/10 flex items-center justify-center mx-auto mb-3">
              <Star className="w-6 h-6 text-yellow-500" />
            </div>
            <p className="text-3xl font-bold mb-1">{Math.floor(progressPercent)}%</p>
            <p className="text-sm text-muted-foreground">Overall Progress</p>
          </div>

          <div className="bg-card rounded-xl border border-border p-6 text-center">
            <div className="w-12 h-12 rounded-full bg-green-500/10 flex items-center justify-center mx-auto mb-3">
              <Sparkles className="w-6 h-6 text-green-500" />
            </div>
            <p className="text-3xl font-bold mb-1">
              {currentLessons.reduce((acc, l) => acc + l.sentences.length, 0)}
            </p>
            <p className="text-sm text-muted-foreground">Sentences to Learn</p>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Course Progress</span>
            <span className="text-sm text-muted-foreground">{Math.floor(progressPercent)}% complete</span>
          </div>
          <div className="h-3 bg-muted rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-violet-500 to-purple-500 rounded-full transition-all duration-500"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>

        {/* Difficulty Filter */}
        <div className="flex flex-wrap justify-center gap-2 mb-8">
          {['All', 'Beginner', 'Intermediate', 'Advanced'].map((difficulty) => (
            <button
              key={difficulty}
              onClick={() => setSelectedDifficulty(difficulty)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                selectedDifficulty === difficulty
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted hover:bg-muted/80'
              }`}
            >
              {difficulty}
            </button>
          ))}
        </div>

        {/* Lessons List */}
        <div className="space-y-4">
          {filteredLessons.map((lesson, index) => (
            <div
              key={lesson.id}
              className={`bg-card rounded-xl border border-border overflow-hidden transition-all ${
                lesson.locked ? 'opacity-60' : 'card-hover'
              }`}
            >
              {/* Lesson Header */}
              <button
                onClick={() => !lesson.locked && setExpandedLesson(expandedLesson === lesson.id ? null : lesson.id)}
                disabled={lesson.locked}
                className="w-full flex items-center gap-4 p-5 text-left"
              >
                {/* Status Icon */}
                <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${
                  lesson.completed 
                    ? 'bg-green-500/10' 
                    : lesson.locked 
                      ? 'bg-muted' 
                      : 'bg-primary/10'
                }`}>
                  {lesson.completed ? (
                    <CheckCircle className="w-5 h-5 text-green-500" />
                  ) : lesson.locked ? (
                    <Lock className="w-5 h-5 text-muted-foreground" />
                  ) : (
                    <span className="text-sm font-bold text-primary">{index + 1}</span>
                  )}
                </div>

                {/* Lesson Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <h3 className="font-semibold truncate">{lesson.title}</h3>
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getDifficultyColor(lesson.difficulty)}`}>
                      {lesson.difficulty}
                    </span>
                  </div>
                  <p className="text-sm text-muted-foreground truncate">{lesson.description}</p>
                </div>

                {/* Meta */}
                <div className="flex items-center gap-4 flex-shrink-0">
                  <div className="flex items-center gap-1 text-sm text-muted-foreground">
                    <Clock className="w-4 h-4" />
                    {lesson.duration}
                  </div>
                  {!lesson.locked && (
                    <ChevronRight className={`w-5 h-5 text-muted-foreground transition-transform ${
                      expandedLesson === lesson.id ? 'rotate-90' : ''
                    }`} />
                  )}
                </div>
              </button>

              {/* Expanded Content */}
              {expandedLesson === lesson.id && !lesson.locked && (
                <div className="px-5 pb-5 border-t border-border">
                  <div className="pt-4">
                    <h4 className="text-sm font-medium mb-3">Sentences in this lesson:</h4>
                    <div className="grid sm:grid-cols-2 gap-2 mb-4">
                      {lesson.sentences.map((sentence, i) => (
                        <div key={i} className="flex items-center gap-2 p-2 rounded-lg bg-muted/50">
                          <Circle className="w-3 h-3 text-muted-foreground flex-shrink-0" />
                          <span className="text-sm">{sentence}</span>
                        </div>
                      ))}
                    </div>

                    <button
                      onClick={() => onStartLesson(lesson.id, lesson.sentences)}
                      className="flex items-center gap-2 px-6 py-3 bg-primary text-primary-foreground rounded-xl font-medium hover:bg-primary/90 transition-colors"
                    >
                      <Play className="w-5 h-5" />
                      {lesson.completed ? 'Practice Again' : 'Start Lesson'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Unlock More CTA */}
        <div className="mt-12 text-center">
          <div className="inline-flex flex-col items-center p-6 bg-gradient-to-r from-violet-500/10 via-purple-500/10 to-fuchsia-500/10 rounded-2xl border border-primary/20">
            <Lock className="w-8 h-8 text-primary mb-3" />
            <h3 className="font-semibold mb-2">Unlock Advanced Lessons</h3>
            <p className="text-sm text-muted-foreground mb-4 max-w-sm">
              Complete the intermediate lessons to unlock advanced content and master complex sentence structures.
            </p>
            <button className="flex items-center gap-2 px-6 py-3 bg-muted hover:bg-muted/80 rounded-xl font-medium transition-colors">
              View All Lessons
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default LearnSection;
