/**
 * SignedWord Type Definitions
 */

// =============================================================================
// USER & AUTH
// =============================================================================

export interface User {
  id: string;
  email: string;
  displayName?: string;
  avatarUrl?: string;
  createdAt: string;
  preferences: UserPreferences;
  progress: UserProgress;
  consent: ConsentStatus;
}

export interface UserPreferences {
  captionsEnabled: boolean;
  playbackSpeed: number;
  theme: 'light' | 'dark' | 'system';
  notificationsEnabled: boolean;
  reminderTime?: string; // HH:MM format
}

export interface UserProgress {
  currentDay: number;
  completedDevotionals: string[];
  streak: number;
  lastCompletedAt?: string;
  totalRecordings: number;
}

export interface ConsentStatus {
  trainingDataConsent: boolean;
  consentedAt?: string;
  canWithdraw: boolean;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  isAuthenticated: boolean;
}

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface RegisterData {
  email: string;
  password: string;
  displayName?: string;
}

// =============================================================================
// DEVOTIONALS
// =============================================================================

export interface Devotional {
  id: string;
  day: number;
  title: string;
  scripture: ScriptureReference;
  videoUrl: string;
  videoDuration: number; // seconds
  thumbnailUrl?: string;
  aslGlosses: string[]; // Signs used in the video
  prompt: ReflectionPrompt;
  createdAt: string;
}

export interface ScriptureReference {
  book: string;
  chapter: number;
  verseStart: number;
  verseEnd?: number;
  text: string;
  translation: string; // e.g., "NIV", "ESV"
}

export interface ReflectionPrompt {
  question: string;
  suggestedSigns?: string[]; // Suggested ASL vocabulary
  maxDuration: number; // seconds (30-60)
  exampleResponse?: string;
}

// =============================================================================
// RECORDINGS
// =============================================================================

export interface Recording {
  id: string;
  devotionalId: string;
  userId: string;
  videoUri: string; // Local URI before upload
  s3Key?: string; // After upload
  duration: number;
  status: RecordingStatus;
  metadata: RecordingMetadata;
  createdAt: string;
  uploadedAt?: string;
}

export type RecordingStatus =
  | 'recording'
  | 'preview'
  | 'uploading'
  | 'uploaded'
  | 'failed';

export interface RecordingMetadata {
  promptId: string;
  consentGranted: boolean;
  deviceType: 'ios' | 'android' | 'web';
  resolution: string;
  fps: number;
}

// =============================================================================
// API RESPONSES
// =============================================================================

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: ApiError;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

// =============================================================================
// UPLOAD
// =============================================================================

export interface UploadProgress {
  loaded: number;
  total: number;
  percentage: number;
}

export interface PresignedUrlResponse {
  uploadUrl: string;
  s3Key: string;
  expiresAt: string;
}

// =============================================================================
// UI STATE
// =============================================================================

export interface CameraState {
  hasPermission: boolean | null;
  isRecording: boolean;
  recordingDuration: number;
  maxDuration: number;
  countdown: number | null;
}

export interface VideoPlayerState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  isBuffering: boolean;
  playbackSpeed: number;
}
