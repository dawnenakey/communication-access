import React, { useState, useCallback, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { useConversations } from '@/hooks/useConversations';
import { supabase } from '@/lib/supabase';
import Header from './Header';
import HeroSection from './HeroSection';
import CameraFeed, { CameraType, FullBodyPose } from './CameraFeed';
import VideoAvatar from './VideoAvatar';
import ConversationPanel from './ConversationPanel';
import FeatureCards from './FeatureCards';
import FAQSection from './FAQSection';
import AWSMetrics from './AWSMetrics';
import LearnSection from './LearnSection';
import GlobalLanguagesSection from './GlobalLanguagesSection';
import PricingSection from './PricingSection';
import Footer from './Footer';
import SettingsModal, { AppSettings } from './SettingsModal';
import AuthModal from './AuthModal';
import UserProfileModal from './UserProfileModal';
import AdminDashboard from '@/pages/AdminDashboard';
import ConversationHistoryPage from '@/pages/ConversationHistoryPage';
import AvatarMarketplace from './AvatarMarketplace';
import SignLearningMode from '@/pages/SignLearningMode';
import { getSign, SignData, SIGN_DATABASE } from '@/data/SignDictionary';



interface Message {
  id: string;
  type: 'user' | 'avatar';
  content: string;
  timestamp: Date;
  confidence?: number;
  language: string;
}

interface RecognizedSignData {
  sign: string;
  sentence: string | null;
  timestampMs: number;
  confidence: number;
}

interface ASLGlossResult {
  success: boolean;
  originalText: string;
  gloss: string;
  signs: string[];
  nonManualMarkers: string[];
  notes: string;
  classifiers: string[];
}

const defaultSettings: AppSettings = {
  camera: {
    resolution: '1280x720',
    fps: 30,
    depthEnabled: true,
    landmarksVisible: true
  },
  model: {
    confidenceThreshold: 0.7,
    sentenceBufferSize: 90,
    autoCorrect: true
  },
  avatar: {
    signSpeed: 1,
    showSubtitles: true,
    voiceEnabled: false
  },
  display: {
    theme: 'system',
    notifications: true,
    soundEffects: true
  }
};

// AI-powered ASL Gloss Converter
const convertToASLGlossAI = async (text: string): Promise<ASLGlossResult> => {
  try {
    const { data, error } = await supabase.functions.invoke('asl-gloss-converter', {
      body: { text, context: 'conversation' }
    });

    if (error) throw error;
    return data;
  } catch (err) {
    console.error('ASL Gloss conversion error:', err);
    // Fallback to simple rule-based conversion
    const gloss = text.toUpperCase()
      .replace(/\b(A|AN|THE)\b/gi, '')
      .replace(/\b(AM|IS|ARE|WAS|WERE)\b/gi, '')
      .replace(/\s+/g, ' ')
      .trim();
    
    return {
      success: false,
      originalText: text,
      gloss,
      signs: gloss.split(' ').filter(Boolean),
      nonManualMarkers: [],
      notes: 'Fallback conversion used',
      classifiers: []
    };
  }
};


const AppLayout: React.FC = () => {
  const { 
    user, 
    settings: userSettings, 
    isAuthenticated, 
    saveSettings, 
    saveLearningProgress,
    saveConversation 
  } = useAuth();

  const {
    createConversation,
    addMessages,
    isLoading: isConversationLoading
  } = useConversations();

  // State
  const [currentLanguage, setCurrentLanguage] = useState('ASL');
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isOAKConnected, setIsOAKConnected] = useState(true);
  const [cameraType, setCameraType] = useState<CameraType>('webcam');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isAvatarResponding, setIsAvatarResponding] = useState(false);
  const [currentAvatarSentence, setCurrentAvatarSentence] = useState('');
  const [recognizedSign, setRecognizedSign] = useState<string | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [authModalMode, setAuthModalMode] = useState<'signin' | 'signup'>('signin');
  const [isProfileModalOpen, setIsProfileModalOpen] = useState(false);
  const [isAdminOpen, setIsAdminOpen] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [settings, setSettings] = useState<AppSettings>(defaultSettings);
  const [conversationSessionId] = useState(() => `session-${Date.now()}`);
  const [recognizedSigns, setRecognizedSigns] = useState<RecognizedSignData[]>([]);
  const [isSavingConversation, setIsSavingConversation] = useState(false);
  
  // Avatar marketplace and learning mode state
  const [isMarketplaceOpen, setIsMarketplaceOpen] = useState(false);
  const [isLearningModeOpen, setIsLearningModeOpen] = useState(false);
  const [selectedAvatarUrl, setSelectedAvatarUrl] = useState<string | undefined>();
  
  // AWS Pipeline metrics state
  const [pipelineStep, setPipelineStep] = useState<'idle' | 'transcribe' | 'gloss' | 'lookup' | 'render'>('idle');
  const [aslGloss, setAslGloss] = useState('');
  const [signsFound, setSignsFound] = useState(0);
  const [processingStartTime, setProcessingStartTime] = useState<number | null>(null);
  const [processingTime, setProcessingTime] = useState(0);

  // Sync settings from auth context
  useEffect(() => {
    if (userSettings) {
      setSettings(userSettings);
    }
  }, [userSettings]);

  // Sync preferred language from user profile
  useEffect(() => {
    if (user?.preferredLanguage) {
      setCurrentLanguage(user.preferredLanguage);
    }
  }, [user]);

  // Update connection status based on camera type
  useEffect(() => {
    if (cameraType === 'oak_ai' || cameraType === 'lumen') {
      setIsOAKConnected(true);
    } else {
      setIsOAKConnected(false);
    }
  }, [cameraType]);

  // Update processing time
  useEffect(() => {
    if (processingStartTime && isAvatarResponding) {
      const interval = setInterval(() => {
        setProcessingTime((Date.now() - processingStartTime) / 1000);
      }, 100);
      return () => clearInterval(interval);
    }
  }, [processingStartTime, isAvatarResponding]);

  // Avatar response sentences based on context
  const avatarResponses: Record<string, string[]> = {
    greeting: [
      "Hello! I am SonZo AI, your sign language assistant.",
      "Welcome! I can help you communicate in sign language.",
      "Hi there! Nice to meet you today."
    ],
    features: [
      "This app recognizes full sentences in sign language using AI.",
      "I can translate between twelve different sign languages globally.",
      "The OAK camera captures depth data for accurate recognition."
    ],
    help: [
      "I am here to help you learn and communicate in sign language.",
      "You can ask me questions and I will respond by signing.",
      "Try signing a sentence and I will translate it for you."
    ],
    thanks: [
      "You are welcome! I am happy to help you.",
      "Thank you for using SonZo AI today.",
      "It is my pleasure to assist you."
    ],
    default: [
      "I understand. Let me respond to that.",
      "That is a great question. Here is my answer.",
      "Thank you for sharing. I am processing your message."
    ]
  };

  // Get contextual avatar response
  const getAvatarResponse = useCallback((userMessage: string): string => {
    const lowerMessage = userMessage.toLowerCase();
    
    if (lowerMessage.includes('hello') || lowerMessage.includes('hi') || lowerMessage.includes('hey')) {
      return avatarResponses.greeting[Math.floor(Math.random() * avatarResponses.greeting.length)];
    }
    if (lowerMessage.includes('feature') || lowerMessage.includes('what can') || lowerMessage.includes('how does')) {
      return avatarResponses.features[Math.floor(Math.random() * avatarResponses.features.length)];
    }
    if (lowerMessage.includes('help') || lowerMessage.includes('assist')) {
      return avatarResponses.help[Math.floor(Math.random() * avatarResponses.help.length)];
    }
    if (lowerMessage.includes('thank') || lowerMessage.includes('thanks')) {
      return avatarResponses.thanks[Math.floor(Math.random() * avatarResponses.thanks.length)];
    }
    
    return avatarResponses.default[Math.floor(Math.random() * avatarResponses.default.length)];
  }, []);

  // Handle sign recognized from camera
  const handleSignRecognized = useCallback((sign: string, confidence: number) => {
    setRecognizedSign(sign);
    
    // Store recognized sign data for saving
    setRecognizedSigns(prev => [...prev, {
      sign,
      sentence: null,
      timestampMs: Date.now(),
      confidence
    }]);
    
    // Clear after a short delay
    setTimeout(() => setRecognizedSign(null), 2000);
  }, []);

  // Process sentence through pipeline with AI-powered ASL Gloss
  const processSentencePipeline = useCallback(async (sentence: string) => {
    setProcessingStartTime(Date.now());
    
    // Step 1: Transcribe (simulated - already have text)
    setPipelineStep('transcribe');
    await new Promise(r => setTimeout(r, 200));
    
    // Step 2: Convert to ASL Gloss using AI
    setPipelineStep('gloss');
    const glossResult = await convertToASLGlossAI(sentence);
    setAslGloss(glossResult.gloss);
    
    // Step 3: Lookup signs in database
    setPipelineStep('lookup');
    const words = glossResult.signs;
    let found = 0;
    for (const word of words) {
      const cleanWord = word.toLowerCase().replace(/[^a-z]/g, '');
      if (getSign(cleanWord)) found++;
    }
    setSignsFound(found);
    await new Promise(r => setTimeout(r, 200));
    
    // Step 4: Render avatar
    setPipelineStep('render');
    
    return glossResult.gloss;
  }, []);



  // Handle camera sentence recognition
  const handleSentenceRecognized = useCallback(async (sentence: string, confidence: number) => {
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: sentence,
      timestamp: new Date(),
      confidence,
      language: currentLanguage
    };
    
    setMessages(prev => [...prev, userMessage]);
    setRecognizedSign(null);

    // Update the last recognized sign with the sentence
    setRecognizedSigns(prev => {
      if (prev.length > 0) {
        const updated = [...prev];
        updated[updated.length - 1].sentence = sentence;
        return updated;
      }
      return prev;
    });

    // Process through pipeline and generate avatar response
    await processSentencePipeline(sentence);
    
    const response = getAvatarResponse(sentence);
    setCurrentAvatarSentence(response);
    setIsAvatarResponding(true);
  }, [currentLanguage, getAvatarResponse, processSentencePipeline]);

  // Handle text message from conversation panel
  const handleSendMessage = useCallback(async (message: string) => {
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: message,
      timestamp: new Date(),
      language: currentLanguage
    };
    
    setMessages(prev => [...prev, userMessage]);

    // Process through pipeline
    await processSentencePipeline(message);

    // Generate avatar response
    const response = getAvatarResponse(message);
    setCurrentAvatarSentence(response);
    setIsAvatarResponding(true);
  }, [currentLanguage, getAvatarResponse, processSentencePipeline]);

  // Handle avatar response completion
  const handleAvatarResponseComplete = useCallback(() => {
    const avatarMessage: Message = {
      id: `avatar-${Date.now()}`,
      type: 'avatar',
      content: currentAvatarSentence,
      timestamp: new Date(),
      language: currentLanguage
    };
    
    setMessages(prev => {
      const newMessages = [...prev, avatarMessage];
      
      // Save conversation if authenticated
      if (isAuthenticated) {
        saveConversation(conversationSessionId, newMessages.slice(-10).map(m => ({
          type: m.type,
          content: m.content,
          language: m.language,
          confidence: m.confidence
        })));
      }
      
      return newMessages;
    });
    
    setIsAvatarResponding(false);
    setCurrentAvatarSentence('');
    setPipelineStep('idle');
    setProcessingStartTime(null);
  }, [currentAvatarSentence, currentLanguage, isAuthenticated, saveConversation, conversationSessionId]);

  // Handle FAQ question
  const handleAskQuestion = useCallback(async (question: string, signResponse: string) => {
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: question,
      timestamp: new Date(),
      language: currentLanguage
    };
    
    setMessages(prev => [...prev, userMessage]);
    
    // Scroll to recognition section
    document.getElementById('recognition')?.scrollIntoView({ behavior: 'smooth' });

    await processSentencePipeline(signResponse);
    setCurrentAvatarSentence(signResponse);
    setIsAvatarResponding(true);
  }, [currentLanguage, processSentencePipeline]);

  // Handle lesson start
  const handleStartLesson = useCallback(async (lessonId: string, sentences: string[]) => {
    // Scroll to recognition section
    document.getElementById('recognition')?.scrollIntoView({ behavior: 'smooth' });
    
    // Save learning progress if authenticated
    if (isAuthenticated) {
      saveLearningProgress({
        lessonId,
        language: currentLanguage,
        sentencesPracticed: 1,
        timeSpentSeconds: 0
      });
    }
    
    // Process and start with first sentence
    await processSentencePipeline(sentences[0]);
    setCurrentAvatarSentence(sentences[0]);
    setIsAvatarResponding(true);
  }, [currentLanguage, isAuthenticated, saveLearningProgress, processSentencePipeline]);

  // Clear conversation history
  const handleClearHistory = useCallback(() => {
    setMessages([]);
    setRecognizedSigns([]);
    setAslGloss('');
    setSignsFound(0);
    setPipelineStep('idle');
  }, []);

  // Save conversation to database
  const handleSaveConversation = useCallback(async () => {
    if (!isAuthenticated || recognizedSigns.length === 0) return;

    setIsSavingConversation(true);
    try {
      const conversation = await createConversation(`Conversation - ${new Date().toLocaleDateString()}`);
      if (conversation) {
        await addMessages(conversation.id, recognizedSigns);
        setRecognizedSigns([]); // Clear after saving
      }
    } catch (err) {
      console.error('Failed to save conversation:', err);
    } finally {
      setIsSavingConversation(false);
    }
  }, [isAuthenticated, recognizedSigns, createConversation, addMessages]);

  // Start recognition
  const handleStartRecognition = useCallback(() => {
    setIsCameraActive(true);
    document.getElementById('recognition')?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  // Handle settings save
  const handleSaveSettings = useCallback((newSettings: AppSettings) => {
    setSettings(newSettings);
    saveSettings(newSettings);
  }, [saveSettings]);

  // Open auth modal
  const handleOpenAuth = useCallback((mode: 'signin' | 'signup') => {
    setAuthModalMode(mode);
    setIsAuthModalOpen(true);
  }, []);

  // Handle camera type change
  const handleCameraTypeChange = useCallback((type: CameraType) => {
    setCameraType(type);
    setIsCameraActive(false);
  }, []);

  // Handle replay sign from history
  const handleReplaySign = useCallback((sign: string) => {
    setRecognizedSign(sign);
    setTimeout(() => setRecognizedSign(null), 2000);
  }, []);

  // Handle avatar selection from marketplace
  const handleSelectAvatar = useCallback((url: string) => {
    setSelectedAvatarUrl(url);
    setIsMarketplaceOpen(false);
  }, []);

  // Handle play sign from learning mode
  const handlePlaySignFromLearning = useCallback((sign: SignData) => {
    setRecognizedSign(sign.name);
    setTimeout(() => setRecognizedSign(null), sign.duration + 500);
  }, []);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <Header
        currentLanguage={currentLanguage}
        onLanguageChange={setCurrentLanguage}
        isConnected={isOAKConnected}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenAuth={handleOpenAuth}
        onOpenProfile={() => setIsProfileModalOpen(true)}
        onOpenAdmin={() => setIsAdminOpen(true)}
        onOpenHistory={() => setIsHistoryOpen(true)}
      />

      {/* Main Content */}
      <main className="pt-16">
        {/* Hero Section */}
        <HeroSection onStartRecognition={handleStartRecognition} />

        {/* Recognition Section */}
        <section id="recognition" className="py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            {/* Section Header */}
            <div className="text-center mb-12">
              <h2 className="text-3xl sm:text-4xl font-bold mb-4">
                Real-time <span className="gradient-text">Sign Recognition</span>
              </h2>
              <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                Use your webcam, OAK AI, or Lumen camera to sign complete sentences. 
                Our realistic video avatar will respond in {currentLanguage} using GenASL technology.
              </p>

            </div>

            {/* Recognition Interface */}
            <div className="grid lg:grid-cols-3 gap-6">
              {/* Camera Feed */}
              <div className="lg:col-span-2">
                <CameraFeed
                  isActive={isCameraActive}
                  onToggle={() => setIsCameraActive(!isCameraActive)}
                  onSentenceRecognized={handleSentenceRecognized}
                  onSignRecognized={handleSignRecognized}
                  language={currentLanguage}
                  cameraType={cameraType}
                  onCameraTypeChange={handleCameraTypeChange}
                />
              </div>

              {/* Video Avatar (GenASL-style) */}
              <div>
                <VideoAvatar
                  currentSentence={currentAvatarSentence}
                  isResponding={isAvatarResponding}
                  language={currentLanguage}
                  onResponseComplete={handleAvatarResponseComplete}
                  recognizedSign={recognizedSign}
                  showSubtitles={settings.avatar.showSubtitles}
                  onOpenMarketplace={() => setIsMarketplaceOpen(true)}
                  onOpenLearning={() => setIsLearningModeOpen(true)}
                />
              </div>

            </div>

            {/* AWS Pipeline Metrics */}
            <div className="mt-6">
              <AWSMetrics
                isProcessing={isAvatarResponding}
                currentStep={pipelineStep}
                inputText={messages.length > 0 ? messages[messages.length - 1]?.content : ''}
                aslGloss={aslGloss}
                signsFound={signsFound}
                totalSigns={Object.keys(SIGN_DATABASE).length}
                processingTime={processingTime}
              />
            </div>

            {/* Conversation Panel */}
            <div className="mt-6">
              <div className="max-w-4xl mx-auto">
                <div className="h-[400px]">
                  <ConversationPanel
                    messages={messages}
                    onSendMessage={handleSendMessage}
                    onClearHistory={handleClearHistory}
                    language={currentLanguage}
                    isAvatarResponding={isAvatarResponding}
                    onSaveConversation={isAuthenticated ? handleSaveConversation : undefined}
                    onOpenHistory={isAuthenticated ? () => setIsHistoryOpen(true) : undefined}
                    isSaving={isSavingConversation}
                  />
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <FeatureCards />

        {/* Learn Section */}
        <LearnSection
          language={currentLanguage}
          onStartLesson={handleStartLesson}
        />

        {/* Global Languages Section */}
        <GlobalLanguagesSection />

        {/* Pricing Section */}
        <PricingSection />

        {/* FAQ Section */}
        <FAQSection onAskQuestion={handleAskQuestion} />
      </main>

      {/* Footer */}
      <Footer />

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onSaveSettings={handleSaveSettings}
      />

      {/* Auth Modal */}
      <AuthModal
        isOpen={isAuthModalOpen}
        onClose={() => setIsAuthModalOpen(false)}
        initialMode={authModalMode}
      />

      {/* Profile Modal */}
      <UserProfileModal
        isOpen={isProfileModalOpen}
        onClose={() => setIsProfileModalOpen(false)}
      />

      {/* Admin Dashboard */}
      <AdminDashboard
        isOpen={isAdminOpen}
        onClose={() => setIsAdminOpen(false)}
      />

      {/* Conversation History */}
      <ConversationHistoryPage
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        onReplaySign={handleReplaySign}
      />

      {/* Avatar Marketplace */}
      <AvatarMarketplace
        isOpen={isMarketplaceOpen}
        onClose={() => setIsMarketplaceOpen(false)}
        onSelectAvatar={handleSelectAvatar}
        currentAvatarUrl={selectedAvatarUrl}
      />

      {/* Sign Learning Mode */}
      <SignLearningMode
        isOpen={isLearningModeOpen}
        onClose={() => setIsLearningModeOpen(false)}
        onPlaySign={handlePlaySignFromLearning}
        recognizedSign={recognizedSign}
        language={currentLanguage}
      />
    </div>
  );
};

export default AppLayout;
