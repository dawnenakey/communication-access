/**
 * Auth Store (Zustand)
 * Manages authentication state across the app
 */

import { create } from 'zustand';
import { api, AuthAPI } from '../services/api';
import type { User, LoginCredentials, RegisterData, AuthState } from '../types';

interface AuthStore extends AuthState {
  // Actions
  initialize: () => Promise<void>;
  login: (credentials: LoginCredentials) => Promise<{ success: boolean; error?: string }>;
  register: (data: RegisterData) => Promise<{ success: boolean; error?: string }>;
  logout: () => Promise<void>;
  updateConsent: (consent: boolean) => Promise<boolean>;
  refreshUser: () => Promise<void>;
}

export const useAuthStore = create<AuthStore>((set, get) => ({
  // Initial state
  user: null,
  token: null,
  isLoading: true,
  isAuthenticated: false,

  // Initialize - check for existing token on app start
  initialize: async () => {
    set({ isLoading: true });

    try {
      await api.init();
      const response = await AuthAPI.getCurrentUser();

      if (response.success && response.data) {
        set({
          user: response.data,
          isAuthenticated: true,
          isLoading: false,
        });
      } else {
        set({
          user: null,
          isAuthenticated: false,
          isLoading: false,
        });
      }
    } catch (error) {
      set({
        user: null,
        isAuthenticated: false,
        isLoading: false,
      });
    }
  },

  // Login
  login: async (credentials) => {
    set({ isLoading: true });

    const response = await AuthAPI.login(credentials);

    if (response.success && response.data) {
      set({
        user: response.data.user,
        token: response.data.token,
        isAuthenticated: true,
        isLoading: false,
      });
      return { success: true };
    }

    set({ isLoading: false });
    return {
      success: false,
      error: response.error?.message || 'Login failed',
    };
  },

  // Register
  register: async (data) => {
    set({ isLoading: true });

    const response = await AuthAPI.register(data);

    if (response.success && response.data) {
      set({
        user: response.data.user,
        token: response.data.token,
        isAuthenticated: true,
        isLoading: false,
      });
      return { success: true };
    }

    set({ isLoading: false });
    return {
      success: false,
      error: response.error?.message || 'Registration failed',
    };
  },

  // Logout
  logout: async () => {
    await AuthAPI.logout();
    set({
      user: null,
      token: null,
      isAuthenticated: false,
    });
  },

  // Update consent
  updateConsent: async (consent) => {
    const response = await AuthAPI.updateConsent(consent);

    if (response.success && response.data) {
      set({ user: response.data });
      return true;
    }

    return false;
  },

  // Refresh user data
  refreshUser: async () => {
    const response = await AuthAPI.getCurrentUser();

    if (response.success && response.data) {
      set({ user: response.data });
    }
  },
}));

// =============================================================================
// HOOKS
// =============================================================================

export const useAuth = () => {
  const store = useAuthStore();
  return {
    user: store.user,
    isLoading: store.isLoading,
    isAuthenticated: store.isAuthenticated,
    login: store.login,
    register: store.register,
    logout: store.logout,
  };
};

export const useUser = () => {
  return useAuthStore((state) => state.user);
};

export const useConsent = () => {
  const user = useAuthStore((state) => state.user);
  const updateConsent = useAuthStore((state) => state.updateConsent);

  return {
    hasConsented: user?.consent.trainingDataConsent ?? false,
    consentedAt: user?.consent.consentedAt,
    updateConsent,
  };
};
