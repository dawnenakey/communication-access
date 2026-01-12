/**
 * API Service
 * Handles all HTTP requests to SignedWord backend
 */

import { Platform } from 'react-native';
import * as SecureStore from 'expo-secure-store';
import AsyncStorage from '@react-native-async-storage/async-storage';
import type {
  ApiResponse,
  User,
  LoginCredentials,
  RegisterData,
  Devotional,
  Recording,
  PresignedUrlResponse,
  PaginatedResponse,
} from '../types';

// =============================================================================
// CONFIGURATION
// =============================================================================

const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || 'https://api.signedword.sonzo.io';
const TOKEN_KEY = 'signedword_auth_token';

// =============================================================================
// TOKEN STORAGE (Platform-specific)
// =============================================================================

const TokenStorage = {
  async get(): Promise<string | null> {
    if (Platform.OS === 'web') {
      return AsyncStorage.getItem(TOKEN_KEY);
    }
    return SecureStore.getItemAsync(TOKEN_KEY);
  },

  async set(token: string): Promise<void> {
    if (Platform.OS === 'web') {
      await AsyncStorage.setItem(TOKEN_KEY, token);
    } else {
      await SecureStore.setItemAsync(TOKEN_KEY, token);
    }
  },

  async remove(): Promise<void> {
    if (Platform.OS === 'web') {
      await AsyncStorage.removeItem(TOKEN_KEY);
    } else {
      await SecureStore.deleteItemAsync(TOKEN_KEY);
    }
  },
};

// =============================================================================
// HTTP CLIENT
// =============================================================================

class ApiClient {
  private baseUrl: string;
  private token: string | null = null;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async init(): Promise<void> {
    this.token = await TokenStorage.get();
  }

  setToken(token: string | null): void {
    this.token = token;
    if (token) {
      TokenStorage.set(token);
    } else {
      TokenStorage.remove();
    }
  }

  private async request<T>(
    method: string,
    endpoint: string,
    body?: unknown,
    customHeaders?: Record<string, string>
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseUrl}${endpoint}`;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...customHeaders,
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    try {
      const response = await fetch(url, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
      });

      const data = await response.json();

      if (!response.ok) {
        return {
          success: false,
          error: {
            code: data.code || 'API_ERROR',
            message: data.message || 'An error occurred',
            details: data.details,
          },
        };
      }

      return { success: true, data };
    } catch (error) {
      return {
        success: false,
        error: {
          code: 'NETWORK_ERROR',
          message: error instanceof Error ? error.message : 'Network error',
        },
      };
    }
  }

  get<T>(endpoint: string): Promise<ApiResponse<T>> {
    return this.request<T>('GET', endpoint);
  }

  post<T>(endpoint: string, body?: unknown): Promise<ApiResponse<T>> {
    return this.request<T>('POST', endpoint, body);
  }

  put<T>(endpoint: string, body?: unknown): Promise<ApiResponse<T>> {
    return this.request<T>('PUT', endpoint, body);
  }

  delete<T>(endpoint: string): Promise<ApiResponse<T>> {
    return this.request<T>('DELETE', endpoint);
  }
}

// =============================================================================
// API INSTANCE
// =============================================================================

export const api = new ApiClient(API_BASE_URL);

// =============================================================================
// AUTH API
// =============================================================================

export const AuthAPI = {
  async login(credentials: LoginCredentials): Promise<ApiResponse<{ user: User; token: string }>> {
    const response = await api.post<{ user: User; token: string }>('/auth/login', credentials);
    if (response.success && response.data) {
      api.setToken(response.data.token);
    }
    return response;
  },

  async register(data: RegisterData): Promise<ApiResponse<{ user: User; token: string }>> {
    const response = await api.post<{ user: User; token: string }>('/auth/register', data);
    if (response.success && response.data) {
      api.setToken(response.data.token);
    }
    return response;
  },

  async logout(): Promise<void> {
    await api.post('/auth/logout');
    api.setToken(null);
  },

  async getCurrentUser(): Promise<ApiResponse<User>> {
    return api.get<User>('/auth/me');
  },

  async refreshToken(): Promise<ApiResponse<{ token: string }>> {
    const response = await api.post<{ token: string }>('/auth/refresh');
    if (response.success && response.data) {
      api.setToken(response.data.token);
    }
    return response;
  },

  async updateConsent(consent: boolean): Promise<ApiResponse<User>> {
    return api.put<User>('/auth/consent', { trainingDataConsent: consent });
  },

  async deleteAccount(): Promise<ApiResponse<void>> {
    const response = await api.delete<void>('/auth/account');
    if (response.success) {
      api.setToken(null);
    }
    return response;
  },
};

// =============================================================================
// DEVOTIONALS API
// =============================================================================

export const DevotionalsAPI = {
  async getToday(): Promise<ApiResponse<Devotional>> {
    return api.get<Devotional>('/devotionals/today');
  },

  async getById(id: string): Promise<ApiResponse<Devotional>> {
    return api.get<Devotional>(`/devotionals/${id}`);
  },

  async getByDay(day: number): Promise<ApiResponse<Devotional>> {
    return api.get<Devotional>(`/devotionals/day/${day}`);
  },

  async list(page = 1, pageSize = 10): Promise<ApiResponse<PaginatedResponse<Devotional>>> {
    return api.get<PaginatedResponse<Devotional>>(`/devotionals?page=${page}&pageSize=${pageSize}`);
  },

  async getCompleted(): Promise<ApiResponse<Devotional[]>> {
    return api.get<Devotional[]>('/devotionals/completed');
  },
};

// =============================================================================
// RECORDINGS API
// =============================================================================

export const RecordingsAPI = {
  async getPresignedUrl(
    devotionalId: string,
    contentType: string
  ): Promise<ApiResponse<PresignedUrlResponse>> {
    return api.post<PresignedUrlResponse>('/recordings/presigned-url', {
      devotionalId,
      contentType,
    });
  },

  async confirmUpload(
    s3Key: string,
    metadata: Recording['metadata']
  ): Promise<ApiResponse<Recording>> {
    return api.post<Recording>('/recordings/confirm', { s3Key, metadata });
  },

  async list(): Promise<ApiResponse<Recording[]>> {
    return api.get<Recording[]>('/recordings');
  },

  async delete(id: string): Promise<ApiResponse<void>> {
    return api.delete<void>(`/recordings/${id}`);
  },

  async deleteAll(): Promise<ApiResponse<void>> {
    return api.delete<void>('/recordings');
  },
};

// =============================================================================
// USER API
// =============================================================================

export const UserAPI = {
  async updatePreferences(
    preferences: Partial<User['preferences']>
  ): Promise<ApiResponse<User>> {
    return api.put<User>('/user/preferences', preferences);
  },

  async getProgress(): Promise<ApiResponse<User['progress']>> {
    return api.get<User['progress']>('/user/progress');
  },

  async updateProfile(data: { displayName?: string }): Promise<ApiResponse<User>> {
    return api.put<User>('/user/profile', data);
  },
};
