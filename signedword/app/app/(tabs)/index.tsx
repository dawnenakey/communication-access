/**
 * Today Screen
 * Main devotional experience - watch video, record response
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
  ActivityIndicator,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { VideoPlayer } from '../../src/components/VideoPlayer';
import { ConsentModal } from '../../src/components/ConsentModal';
import { useAuth, useConsent } from '../../src/context/auth';
import { DevotionalsAPI } from '../../src/services/api';
import { colors, spacing, borderRadius } from '../../src/components/ui/theme';
import type { Devotional } from '../../src/types';

type DevotionalStep = 'watch' | 'prompt' | 'complete';

export default function TodayScreen() {
  const router = useRouter();
  const { user, isAuthenticated } = useAuth();
  const { hasConsented, updateConsent } = useConsent();

  const [devotional, setDevotional] = useState<Devotional | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState<DevotionalStep>('watch');
  const [videoCompleted, setVideoCompleted] = useState(false);
  const [showConsentModal, setShowConsentModal] = useState(false);

  // Fetch today's devotional
  const fetchDevotional = useCallback(async () => {
    try {
      setError(null);
      const response = await DevotionalsAPI.getToday();

      if (response.success && response.data) {
        setDevotional(response.data);

        // Check if already completed today
        if (user?.progress.completedDevotionals.includes(response.data.id)) {
          setStep('complete');
        }
      } else {
        setError(response.error?.message || 'Failed to load devotional');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, [user]);

  useEffect(() => {
    if (isAuthenticated) {
      fetchDevotional();
    }
  }, [isAuthenticated, fetchDevotional]);

  // Handle video completion
  const handleVideoComplete = () => {
    setVideoCompleted(true);
  };

  // Move to prompt step
  const handleContinueToPrompt = () => {
    setStep('prompt');
  };

  // Start recording
  const handleStartRecording = () => {
    // Check consent first
    if (!hasConsented) {
      setShowConsentModal(true);
      return;
    }

    router.push({
      pathname: '/record',
      params: {
        devotionalId: devotional?.id,
        prompt: devotional?.prompt.question,
        maxDuration: devotional?.prompt.maxDuration || 60,
      },
    });
  };

  // Handle consent
  const handleConsentAccept = async () => {
    await updateConsent(true);
    setShowConsentModal(false);
    // Now start recording
    router.push({
      pathname: '/record',
      params: {
        devotionalId: devotional?.id,
        prompt: devotional?.prompt.question,
        maxDuration: devotional?.prompt.maxDuration || 60,
      },
    });
  };

  const handleConsentDecline = () => {
    setShowConsentModal(false);
    // Still allow recording, but won't contribute to training
    router.push({
      pathname: '/record',
      params: {
        devotionalId: devotional?.id,
        prompt: devotional?.prompt.question,
        maxDuration: devotional?.prompt.maxDuration || 60,
        noConsent: 'true',
      },
    });
  };

  // Skip response (complete without recording)
  const handleSkipResponse = () => {
    setStep('complete');
  };

  // Refresh handler
  const handleRefresh = () => {
    setIsRefreshing(true);
    fetchDevotional();
  };

  // Redirect to auth if not authenticated
  if (!isAuthenticated) {
    return (
      <View style={styles.container}>
        <View style={styles.authPrompt}>
          <Ionicons name="book" size={64} color={colors.primary[500]} />
          <Text style={styles.authTitle}>Welcome to SignedWord</Text>
          <Text style={styles.authSubtitle}>
            Sign in to start your daily ASL devotional journey
          </Text>
          <TouchableOpacity
            style={styles.authButton}
            onPress={() => router.push('/(auth)/login')}
          >
            <Text style={styles.authButtonText}>Sign In</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // Loading state
  if (isLoading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color={colors.primary[500]} />
        <Text style={styles.loadingText}>Loading today's devotional...</Text>
      </View>
    );
  }

  // Error state
  if (error || !devotional) {
    return (
      <View style={styles.centerContainer}>
        <Ionicons name="cloud-offline" size={64} color={colors.gray[500]} />
        <Text style={styles.errorText}>{error || 'No devotional available'}</Text>
        <TouchableOpacity style={styles.retryButton} onPress={handleRefresh}>
          <Text style={styles.retryButtonText}>Try Again</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl
          refreshing={isRefreshing}
          onRefresh={handleRefresh}
          tintColor={colors.primary[500]}
        />
      }
    >
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.dayLabel}>Day {devotional.day}</Text>
        <Text style={styles.title}>{devotional.title}</Text>
      </View>

      {/* Scripture */}
      <View style={styles.scriptureCard}>
        <Text style={styles.scriptureReference}>
          {devotional.scripture.book} {devotional.scripture.chapter}:
          {devotional.scripture.verseStart}
          {devotional.scripture.verseEnd
            ? `-${devotional.scripture.verseEnd}`
            : ''}
        </Text>
        <Text style={styles.scriptureText}>{devotional.scripture.text}</Text>
        <Text style={styles.scriptureTranslation}>
          — {devotional.scripture.translation}
        </Text>
      </View>

      {/* Step: Watch Video */}
      {step === 'watch' && (
        <>
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Watch in ASL</Text>
            <VideoPlayer
              uri={devotional.videoUrl}
              posterUri={devotional.thumbnailUrl}
              showCaptions
              captionsText={devotional.scripture.text}
              onComplete={handleVideoComplete}
            />
          </View>

          {/* Continue button */}
          <TouchableOpacity
            style={[
              styles.primaryButton,
              !videoCompleted && styles.primaryButtonDisabled,
            ]}
            onPress={handleContinueToPrompt}
            disabled={!videoCompleted}
          >
            <Text style={styles.primaryButtonText}>
              {videoCompleted ? 'Continue' : 'Watch the video to continue'}
            </Text>
            <Ionicons name="arrow-forward" size={20} color="white" />
          </TouchableOpacity>
        </>
      )}

      {/* Step: Response Prompt */}
      {step === 'prompt' && (
        <>
          <View style={styles.promptSection}>
            <Ionicons
              name="chatbubble-ellipses"
              size={32}
              color={colors.primary[500]}
            />
            <Text style={styles.promptTitle}>Reflect & Respond</Text>
            <Text style={styles.promptQuestion}>{devotional.prompt.question}</Text>

            {devotional.prompt.suggestedSigns && (
              <View style={styles.suggestedSigns}>
                <Text style={styles.suggestedLabel}>Suggested signs:</Text>
                <Text style={styles.suggestedText}>
                  {devotional.prompt.suggestedSigns.join(' • ')}
                </Text>
              </View>
            )}
          </View>

          {/* Record button */}
          <TouchableOpacity
            style={styles.recordButton}
            onPress={handleStartRecording}
          >
            <View style={styles.recordButtonInner}>
              <Ionicons name="videocam" size={28} color="white" />
            </View>
            <Text style={styles.recordButtonText}>Record Your Response</Text>
            <Text style={styles.recordButtonSubtext}>
              Up to {devotional.prompt.maxDuration} seconds
            </Text>
          </TouchableOpacity>

          {/* Skip option */}
          <TouchableOpacity style={styles.skipButton} onPress={handleSkipResponse}>
            <Text style={styles.skipButtonText}>Skip for today</Text>
          </TouchableOpacity>
        </>
      )}

      {/* Step: Complete */}
      {step === 'complete' && (
        <View style={styles.completeSection}>
          <View style={styles.completeIcon}>
            <Ionicons name="checkmark-circle" size={80} color={colors.success} />
          </View>
          <Text style={styles.completeTitle}>Devotional Complete!</Text>
          <Text style={styles.completeText}>
            Great job! Come back tomorrow for your next devotional.
          </Text>

          {/* Streak display */}
          {user?.progress.streak && user.progress.streak > 0 && (
            <View style={styles.streakCard}>
              <Ionicons name="flame" size={32} color={colors.secondary[500]} />
              <Text style={styles.streakNumber}>{user.progress.streak}</Text>
              <Text style={styles.streakLabel}>day streak!</Text>
            </View>
          )}
        </View>
      )}

      {/* Consent Modal */}
      <ConsentModal
        visible={showConsentModal}
        onAccept={handleConsentAccept}
        onDecline={handleConsentDecline}
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.dark,
  },
  content: {
    padding: spacing.lg,
    paddingBottom: spacing['3xl'],
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.xl,
    backgroundColor: colors.background.dark,
  },
  loadingText: {
    marginTop: spacing.md,
    color: colors.text.dark.secondary,
    fontSize: 16,
  },
  errorText: {
    marginTop: spacing.md,
    color: colors.text.dark.secondary,
    fontSize: 16,
    textAlign: 'center',
  },
  retryButton: {
    marginTop: spacing.lg,
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.md,
    backgroundColor: colors.primary[500],
    borderRadius: borderRadius.lg,
  },
  retryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  authPrompt: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.xl,
  },
  authTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: colors.text.dark.primary,
    marginTop: spacing.lg,
  },
  authSubtitle: {
    fontSize: 16,
    color: colors.text.dark.secondary,
    textAlign: 'center',
    marginTop: spacing.sm,
    marginBottom: spacing.xl,
  },
  authButton: {
    backgroundColor: colors.primary[500],
    paddingHorizontal: spacing['2xl'],
    paddingVertical: spacing.md,
    borderRadius: borderRadius.lg,
  },
  authButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
  },
  header: {
    marginBottom: spacing.lg,
  },
  dayLabel: {
    fontSize: 14,
    color: colors.primary[400],
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: colors.text.dark.primary,
    marginTop: spacing.xs,
  },
  scriptureCard: {
    backgroundColor: colors.gray[800],
    padding: spacing.lg,
    borderRadius: borderRadius.lg,
    marginBottom: spacing.xl,
    borderLeftWidth: 4,
    borderLeftColor: colors.primary[500],
  },
  scriptureReference: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.primary[400],
    marginBottom: spacing.sm,
  },
  scriptureText: {
    fontSize: 18,
    lineHeight: 28,
    color: colors.text.dark.primary,
    fontStyle: 'italic',
  },
  scriptureTranslation: {
    fontSize: 12,
    color: colors.text.dark.muted,
    marginTop: spacing.sm,
    textAlign: 'right',
  },
  section: {
    marginBottom: spacing.xl,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: colors.text.dark.primary,
    marginBottom: spacing.md,
  },
  primaryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[500],
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.xl,
    borderRadius: borderRadius.lg,
    gap: spacing.sm,
  },
  primaryButtonDisabled: {
    backgroundColor: colors.gray[700],
  },
  primaryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  promptSection: {
    alignItems: 'center',
    padding: spacing.xl,
    backgroundColor: colors.gray[800],
    borderRadius: borderRadius.xl,
    marginBottom: spacing.xl,
  },
  promptTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: colors.text.dark.primary,
    marginTop: spacing.md,
    marginBottom: spacing.sm,
  },
  promptQuestion: {
    fontSize: 18,
    color: colors.text.dark.secondary,
    textAlign: 'center',
    lineHeight: 26,
  },
  suggestedSigns: {
    marginTop: spacing.lg,
    padding: spacing.md,
    backgroundColor: colors.gray[700],
    borderRadius: borderRadius.md,
    width: '100%',
  },
  suggestedLabel: {
    fontSize: 12,
    color: colors.text.dark.muted,
    marginBottom: spacing.xs,
  },
  suggestedText: {
    fontSize: 14,
    color: colors.primary[300],
    fontWeight: '500',
  },
  recordButton: {
    alignItems: 'center',
    padding: spacing.xl,
    backgroundColor: colors.primary[900],
    borderRadius: borderRadius.xl,
    borderWidth: 2,
    borderColor: colors.primary[500],
    marginBottom: spacing.lg,
  },
  recordButtonInner: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: colors.primary[500],
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  recordButtonText: {
    fontSize: 18,
    fontWeight: '600',
    color: colors.text.dark.primary,
  },
  recordButtonSubtext: {
    fontSize: 14,
    color: colors.text.dark.muted,
    marginTop: spacing.xs,
  },
  skipButton: {
    alignItems: 'center',
    padding: spacing.md,
  },
  skipButtonText: {
    fontSize: 14,
    color: colors.text.dark.muted,
  },
  completeSection: {
    alignItems: 'center',
    padding: spacing.xl,
  },
  completeIcon: {
    marginBottom: spacing.lg,
  },
  completeTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: colors.text.dark.primary,
    marginBottom: spacing.sm,
  },
  completeText: {
    fontSize: 16,
    color: colors.text.dark.secondary,
    textAlign: 'center',
  },
  streakCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.gray[800],
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.xl,
    borderRadius: borderRadius.full,
    marginTop: spacing.xl,
    gap: spacing.sm,
  },
  streakNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: colors.secondary[500],
  },
  streakLabel: {
    fontSize: 16,
    color: colors.text.dark.secondary,
  },
});
