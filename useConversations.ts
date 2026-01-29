import { useState, useCallback } from 'react';
import { supabase } from '@/lib/supabase';

export interface ConversationMessage {
  id: string;
  sign: string;
  sentence: string | null;
  timestamp_ms: number;
  confidence: number;
  landmarks_data?: any;
  created_at: string;
}

export interface Conversation {
  id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  is_public: boolean;
  share_token: string | null;
  total_signs: number;
  duration_seconds: number;
  messages?: ConversationMessage[];
}

export interface ConversationsState {
  conversations: Conversation[];
  currentConversation: Conversation | null;
  isLoading: boolean;
  error: string | null;
  total: number;
  hasMore: boolean;
}

export function useConversations() {
  const [state, setState] = useState<ConversationsState>({
    conversations: [],
    currentConversation: null,
    isLoading: false,
    error: null,
    total: 0,
    hasMore: false
  });

  const getToken = () => localStorage.getItem('sonzo_token');

  const createConversation = useCallback(async (title?: string): Promise<Conversation | null> => {
    const token = getToken();
    if (!token) {
      setState(prev => ({ ...prev, error: 'Authentication required' }));
      return null;
    }

    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const { data, error } = await supabase.functions.invoke('sonzo-conversations', {
        body: { action: 'createConversation', token, title }
      });

      if (error || data.error) {
        setState(prev => ({ ...prev, isLoading: false, error: data?.error || 'Failed to create conversation' }));
        return null;
      }

      setState(prev => ({
        ...prev,
        isLoading: false,
        conversations: [data.conversation, ...prev.conversations],
        currentConversation: data.conversation
      }));

      return data.conversation;
    } catch (err) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Network error' }));
      return null;
    }
  }, []);

  const getConversations = useCallback(async (limit = 50, offset = 0) => {
    const token = getToken();
    if (!token) {
      setState(prev => ({ ...prev, error: 'Authentication required' }));
      return;
    }

    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const { data, error } = await supabase.functions.invoke('sonzo-conversations', {
        body: { action: 'getConversations', token, limit, offset }
      });

      if (error || data.error) {
        setState(prev => ({ ...prev, isLoading: false, error: data?.error || 'Failed to load conversations' }));
        return;
      }

      setState(prev => ({
        ...prev,
        isLoading: false,
        conversations: offset === 0 ? data.conversations : [...prev.conversations, ...data.conversations],
        total: data.total,
        hasMore: data.hasMore
      }));
    } catch (err) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Network error' }));
    }
  }, []);

  const getConversation = useCallback(async (conversationId: string): Promise<Conversation | null> => {
    const token = getToken();
    if (!token) {
      setState(prev => ({ ...prev, error: 'Authentication required' }));
      return null;
    }

    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const { data, error } = await supabase.functions.invoke('sonzo-conversations', {
        body: { action: 'getConversation', token, conversationId }
      });

      if (error || data.error) {
        setState(prev => ({ ...prev, isLoading: false, error: data?.error || 'Failed to load conversation' }));
        return null;
      }

      setState(prev => ({
        ...prev,
        isLoading: false,
        currentConversation: data.conversation
      }));

      return data.conversation;
    } catch (err) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Network error' }));
      return null;
    }
  }, []);

  const getSharedConversation = useCallback(async (shareToken: string): Promise<Conversation | null> => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const { data, error } = await supabase.functions.invoke('sonzo-conversations', {
        body: { action: 'getSharedConversation', shareToken }
      });

      if (error || data.error) {
        setState(prev => ({ ...prev, isLoading: false, error: data?.error || 'Conversation not found' }));
        return null;
      }

      setState(prev => ({
        ...prev,
        isLoading: false,
        currentConversation: data.conversation
      }));

      return data.conversation;
    } catch (err) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Network error' }));
      return null;
    }
  }, []);

  const addMessage = useCallback(async (
    conversationId: string,
    sign: string,
    sentence: string | null,
    timestampMs: number,
    confidence: number,
    landmarksData?: any
  ) => {
    const token = getToken();
    if (!token) return;

    try {
      await supabase.functions.invoke('sonzo-conversations', {
        body: { 
          action: 'addMessage', 
          token,
          conversationId, 
          sign, 
          sentence, 
          timestampMs, 
          confidence,
          landmarksData 
        }
      });
    } catch (err) {
      console.error('Failed to add message:', err);
    }
  }, []);

  const addMessages = useCallback(async (
    conversationId: string,
    messages: Array<{
      sign: string;
      sentence?: string | null;
      timestampMs: number;
      confidence?: number;
      landmarksData?: any;
    }>
  ) => {
    const token = getToken();
    if (!token) return;

    try {
      await supabase.functions.invoke('sonzo-conversations', {
        body: { action: 'addMessages', token, conversationId, messages }
      });
    } catch (err) {
      console.error('Failed to add messages:', err);
    }
  }, []);

  const updateConversation = useCallback(async (
    conversationId: string,
    updates: { title?: string; isPublic?: boolean }
  ) => {
    const token = getToken();
    if (!token) {
      setState(prev => ({ ...prev, error: 'Authentication required' }));
      return false;
    }

    try {
      const { data, error } = await supabase.functions.invoke('sonzo-conversations', {
        body: { action: 'updateConversation', token, conversationId, ...updates }
      });

      if (error || data.error) {
        setState(prev => ({ ...prev, error: data?.error || 'Failed to update conversation' }));
        return false;
      }

      setState(prev => ({
        ...prev,
        conversations: prev.conversations.map(c => 
          c.id === conversationId ? data.conversation : c
        ),
        currentConversation: prev.currentConversation?.id === conversationId 
          ? data.conversation 
          : prev.currentConversation
      }));

      return true;
    } catch (err) {
      setState(prev => ({ ...prev, error: 'Network error' }));
      return false;
    }
  }, []);

  const deleteConversation = useCallback(async (conversationId: string) => {
    const token = getToken();
    if (!token) {
      setState(prev => ({ ...prev, error: 'Authentication required' }));
      return false;
    }

    try {
      const { data, error } = await supabase.functions.invoke('sonzo-conversations', {
        body: { action: 'deleteConversation', token, conversationId }
      });

      if (error || data.error) {
        setState(prev => ({ ...prev, error: data?.error || 'Failed to delete conversation' }));
        return false;
      }

      setState(prev => ({
        ...prev,
        conversations: prev.conversations.filter(c => c.id !== conversationId),
        currentConversation: prev.currentConversation?.id === conversationId 
          ? null 
          : prev.currentConversation,
        total: prev.total - 1
      }));

      return true;
    } catch (err) {
      setState(prev => ({ ...prev, error: 'Network error' }));
      return false;
    }
  }, []);

  const generateShareLink = useCallback(async (conversationId: string): Promise<string | null> => {
    const token = getToken();
    if (!token) {
      setState(prev => ({ ...prev, error: 'Authentication required' }));
      return null;
    }

    try {
      const { data, error } = await supabase.functions.invoke('sonzo-conversations', {
        body: { action: 'generateShareLink', token, conversationId }
      });

      if (error || data.error) {
        setState(prev => ({ ...prev, error: data?.error || 'Failed to generate share link' }));
        return null;
      }

      // Update local state
      setState(prev => ({
        ...prev,
        conversations: prev.conversations.map(c => 
          c.id === conversationId ? { ...c, is_public: true, share_token: data.shareToken } : c
        ),
        currentConversation: prev.currentConversation?.id === conversationId 
          ? { ...prev.currentConversation, is_public: true, share_token: data.shareToken }
          : prev.currentConversation
      }));

      return `${window.location.origin}/shared/${data.shareToken}`;
    } catch (err) {
      setState(prev => ({ ...prev, error: 'Network error' }));
      return null;
    }
  }, []);

  const revokeShareLink = useCallback(async (conversationId: string) => {
    const token = getToken();
    if (!token) {
      setState(prev => ({ ...prev, error: 'Authentication required' }));
      return false;
    }

    try {
      const { data, error } = await supabase.functions.invoke('sonzo-conversations', {
        body: { action: 'revokeShareLink', token, conversationId }
      });

      if (error || data.error) {
        setState(prev => ({ ...prev, error: data?.error || 'Failed to revoke share link' }));
        return false;
      }

      setState(prev => ({
        ...prev,
        conversations: prev.conversations.map(c => 
          c.id === conversationId ? { ...c, is_public: false } : c
        ),
        currentConversation: prev.currentConversation?.id === conversationId 
          ? { ...prev.currentConversation, is_public: false }
          : prev.currentConversation
      }));

      return true;
    } catch (err) {
      setState(prev => ({ ...prev, error: 'Network error' }));
      return false;
    }
  }, []);

  const exportConversation = useCallback(async (
    conversationId: string,
    format: 'text' | 'json' | 'csv' = 'text'
  ): Promise<{ content: string; filename: string } | null> => {
    const token = getToken();

    try {
      const { data, error } = await supabase.functions.invoke('sonzo-conversations', {
        body: { action: 'exportConversation', token, conversationId, format }
      });

      if (error || data.error) {
        setState(prev => ({ ...prev, error: data?.error || 'Failed to export conversation' }));
        return null;
      }

      return { content: data.content, filename: data.filename };
    } catch (err) {
      setState(prev => ({ ...prev, error: 'Network error' }));
      return null;
    }
  }, []);

  const downloadExport = useCallback(async (
    conversationId: string,
    format: 'text' | 'json' | 'csv' = 'text'
  ) => {
    const result = await exportConversation(conversationId, format);
    if (!result) return;

    const blob = new Blob([result.content], { 
      type: format === 'json' ? 'application/json' : 'text/plain' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = result.filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [exportConversation]);

  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  const setCurrentConversation = useCallback((conversation: Conversation | null) => {
    setState(prev => ({ ...prev, currentConversation: conversation }));
  }, []);

  return {
    ...state,
    createConversation,
    getConversations,
    getConversation,
    getSharedConversation,
    addMessage,
    addMessages,
    updateConversation,
    deleteConversation,
    generateShareLink,
    revokeShareLink,
    exportConversation,
    downloadExport,
    clearError,
    setCurrentConversation
  };
}
