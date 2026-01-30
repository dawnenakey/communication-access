import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { supabase } from '@/lib/supabase';

interface User {
  id: string;
  email: string;
  displayName: string;
  avatarUrl?: string;
  preferredLanguage: string;
}

interface UserSettings {
  camera: {
    resolution: string;
    fps: number;
    depthEnabled: boolean;
    landmarksVisible: boolean;
  };
  model: {
    confidenceThreshold: number;
    sentenceBufferSize: number;
    autoCorrect: boolean;
  };
  avatar: {
    signSpeed: number;
    showSubtitles: boolean;
    voiceEnabled: boolean;
  };
  display: {
    theme: 'light' | 'dark' | 'system';
    notifications: boolean;
    soundEffects: boolean;
  };
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

interface UserStats {
  completedLessons: number;
  totalTimeMinutes: number;
  totalSentencesPracticed: number;
  totalConversations: number;
}

interface AuthContextType {
  user: User | null;
  settings: UserSettings | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  signUp: (email: string, password: string, displayName?: string) => Promise<{ success: boolean; error?: string }>;
  signIn: (email: string, password: string) => Promise<{ success: boolean; error?: string }>;
  signOut: () => Promise<void>;
  updateProfile: (data: Partial<User>) => Promise<{ success: boolean; error?: string }>;
  saveSettings: (settings: UserSettings) => Promise<{ success: boolean; error?: string }>;
  saveLearningProgress: (progress: Partial<LearningProgress> & { lessonId: string; language: string }) => Promise<void>;
  getLearningProgress: (language?: string) => Promise<LearningProgress[]>;
  saveConversation: (sessionId: string, messages: any[]) => Promise<void>;
  getUserStats: () => Promise<UserStats | null>;
}

const defaultSettings: UserSettings = {
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

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [settings, setSettings] = useState<UserSettings | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [token, setToken] = useState<string | null>(null);

  // Load session from localStorage on mount
  useEffect(() => {
    const storedToken = localStorage.getItem('sonzo_token');
    if (storedToken) {
      validateSession(storedToken);
    } else {
      setIsLoading(false);
    }
  }, []);

  const validateSession = async (sessionToken: string) => {
    try {
      const { data, error } = await supabase.functions.invoke('sonzo-auth', {
        body: { action: 'validate', token: sessionToken }
      });

      if (error || !data.valid) {
        localStorage.removeItem('sonzo_token');
        setIsLoading(false);
        return;
      }

      setUser(data.user);
      setToken(sessionToken);
      
      if (data.settings) {
        setSettings({
          camera: {
            resolution: data.settings.camera_resolution || defaultSettings.camera.resolution,
            fps: data.settings.camera_fps || defaultSettings.camera.fps,
            depthEnabled: data.settings.depth_enabled ?? defaultSettings.camera.depthEnabled,
            landmarksVisible: data.settings.landmarks_visible ?? defaultSettings.camera.landmarksVisible
          },
          model: {
            confidenceThreshold: data.settings.confidence_threshold || defaultSettings.model.confidenceThreshold,
            sentenceBufferSize: data.settings.sentence_buffer_size || defaultSettings.model.sentenceBufferSize,
            autoCorrect: data.settings.auto_correct ?? defaultSettings.model.autoCorrect
          },
          avatar: {
            signSpeed: data.settings.avatar_sign_speed || defaultSettings.avatar.signSpeed,
            showSubtitles: data.settings.show_subtitles ?? defaultSettings.avatar.showSubtitles,
            voiceEnabled: data.settings.voice_enabled ?? defaultSettings.avatar.voiceEnabled
          },
          display: {
            theme: data.settings.theme || defaultSettings.display.theme,
            notifications: data.settings.notifications ?? defaultSettings.display.notifications,
            soundEffects: data.settings.sound_effects ?? defaultSettings.display.soundEffects
          }
        });
      } else {
        setSettings(defaultSettings);
      }
    } catch (err) {
      localStorage.removeItem('sonzo_token');
    } finally {
      setIsLoading(false);
    }
  };

  const signUp = useCallback(async (email: string, password: string, displayName?: string) => {
    try {
      const { data, error } = await supabase.functions.invoke('sonzo-auth', {
        body: { action: 'signup', email, password, displayName }
      });

      if (error || data.error) {
        return { success: false, error: data?.error || 'Sign up failed' };
      }

      localStorage.setItem('sonzo_token', data.token);
      setToken(data.token);
      setUser(data.user);
      setSettings(defaultSettings);

      return { success: true };
    } catch (err) {
      return { success: false, error: 'Network error' };
    }
  }, []);

  const signIn = useCallback(async (email: string, password: string) => {
    try {
      const { data, error } = await supabase.functions.invoke('sonzo-auth', {
        body: { action: 'signin', email, password }
      });

      if (error || data.error) {
        return { success: false, error: data?.error || 'Sign in failed' };
      }

      localStorage.setItem('sonzo_token', data.token);
      setToken(data.token);
      setUser(data.user);

      if (data.settings) {
        setSettings({
          camera: {
            resolution: data.settings.camera_resolution || defaultSettings.camera.resolution,
            fps: data.settings.camera_fps || defaultSettings.camera.fps,
            depthEnabled: data.settings.depth_enabled ?? defaultSettings.camera.depthEnabled,
            landmarksVisible: data.settings.landmarks_visible ?? defaultSettings.camera.landmarksVisible
          },
          model: {
            confidenceThreshold: data.settings.confidence_threshold || defaultSettings.model.confidenceThreshold,
            sentenceBufferSize: data.settings.sentence_buffer_size || defaultSettings.model.sentenceBufferSize,
            autoCorrect: data.settings.auto_correct ?? defaultSettings.model.autoCorrect
          },
          avatar: {
            signSpeed: data.settings.avatar_sign_speed || defaultSettings.avatar.signSpeed,
            showSubtitles: data.settings.show_subtitles ?? defaultSettings.avatar.showSubtitles,
            voiceEnabled: data.settings.voice_enabled ?? defaultSettings.avatar.voiceEnabled
          },
          display: {
            theme: data.settings.theme || defaultSettings.display.theme,
            notifications: data.settings.notifications ?? defaultSettings.display.notifications,
            soundEffects: data.settings.sound_effects ?? defaultSettings.display.soundEffects
          }
        });
      } else {
        setSettings(defaultSettings);
      }

      return { success: true };
    } catch (err) {
      return { success: false, error: 'Network error' };
    }
  }, []);

  const signOut = useCallback(async () => {
    if (token) {
      await supabase.functions.invoke('sonzo-auth', {
        body: { action: 'signout', token }
      });
    }

    localStorage.removeItem('sonzo_token');
    setToken(null);
    setUser(null);
    setSettings(null);
  }, [token]);

  const updateProfile = useCallback(async (data: Partial<User>) => {
    if (!token) return { success: false, error: 'Not authenticated' };

    try {
      const { data: result, error } = await supabase.functions.invoke('sonzo-user-data', {
        body: { 
          action: 'updateProfile', 
          token,
          displayName: data.displayName,
          avatarUrl: data.avatarUrl,
          preferredLanguage: data.preferredLanguage
        }
      });

      if (error || result.error) {
        return { success: false, error: result?.error || 'Update failed' };
      }

      setUser(result.user);
      return { success: true };
    } catch (err) {
      return { success: false, error: 'Network error' };
    }
  }, [token]);

  const saveSettings = useCallback(async (newSettings: UserSettings) => {
    setSettings(newSettings);

    if (!token) return { success: true }; // Save locally only if not authenticated

    try {
      const { data, error } = await supabase.functions.invoke('sonzo-user-data', {
        body: { action: 'saveSettings', token, settings: newSettings }
      });

      if (error || data.error) {
        return { success: false, error: data?.error || 'Save failed' };
      }

      return { success: true };
    } catch (err) {
      return { success: false, error: 'Network error' };
    }
  }, [token]);

  const saveLearningProgress = useCallback(async (progress: Partial<LearningProgress> & { lessonId: string; language: string }) => {
    if (!token) return;

    try {
      await supabase.functions.invoke('sonzo-user-data', {
        body: { 
          action: 'saveLearningProgress', 
          token,
          ...progress
        }
      });
    } catch (err) {
      console.error('Failed to save learning progress:', err);
    }
  }, [token]);

  const getLearningProgress = useCallback(async (language?: string): Promise<LearningProgress[]> => {
    if (!token) return [];

    try {
      const { data, error } = await supabase.functions.invoke('sonzo-user-data', {
        body: { action: 'getLearningProgress', token, language }
      });

      if (error || data.error) return [];
      return data.progress || [];
    } catch (err) {
      return [];
    }
  }, [token]);

  const saveConversation = useCallback(async (sessionId: string, messages: any[]) => {
    if (!token || messages.length === 0) return;

    try {
      await supabase.functions.invoke('sonzo-user-data', {
        body: { action: 'saveConversation', token, sessionId, messages }
      });
    } catch (err) {
      console.error('Failed to save conversation:', err);
    }
  }, [token]);

  const getUserStats = useCallback(async (): Promise<UserStats | null> => {
    if (!token) return null;

    try {
      const { data, error } = await supabase.functions.invoke('sonzo-user-data', {
        body: { action: 'getUserStats', token }
      });

      if (error || data.error) return null;
      return data.stats;
    } catch (err) {
      return null;
    }
  }, [token]);

  return (
    <AuthContext.Provider
      value={{
        user,
        settings,
        isLoading,
        isAuthenticated: !!user,
        signUp,
        signIn,
        signOut,
        updateProfile,
        saveSettings,
        saveLearningProgress,
        getLearningProgress,
        saveConversation,
        getUserStats
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};
