/**
 * Devotional Detail Screen
 * View past devotional with recorded response
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { VideoPlayer } from '../../src/components/VideoPlayer';
import { DevotionalsAPI, RecordingsAPI } from '../../src/services/api';
import { useAuth } from '../../src/context/auth';
import { colors, spacing, borderRadius } from '../../src/components/ui/theme';
import type { Devotional, Recording } from '../../src/types';

export default function DevotionalDetailScreen() {
  const router = useRouter();
  const { id } = useLocalSearchParams<{ id: string }>();
  const { user, isAuthenticated } = useAuth();

  const [devotional, setDevotional] = useState<Devotional | null>(null);
  const [recording, setRecording] = useState<Recording | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch devotional and recording
  const fetchData = useCallback(async () => {
    if (!id) return;

    try {
      setError(null);

      // Fetch devotional
      const devResponse = await DevotionalsAPI.getById(id);
      if (devResponse.success && devResponse.data) {
        setDevotional(devResponse.data);
      } else {
        setError('Devotional not found');
        return;
      }

      // Check if user has a recording for this devotional
      const recResponse = await RecordingsAPI.getByDevotional(id);
      if (recResponse.success && recResponse.data) {
        setRecording(recResponse.data);
      }
    } catch (err) {
      setError('Failed to load devotional');
    } finally {
      setIsLoading(false);
    }
  }, [id]);

  useEffect(() => {
    if (isAuthenticated) {
      fetchData();
    }
  }, [isAuthenticated, fetchData]);

  const isCompleted = devotional && user?.progress.completedDevotionals.includes(devotional.id);

  if (isLoading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color={colors.primary[500]} />
      </View>
    );
  }

  if (error || !devotional) {
    return (
      <View style={styles.centerContainer}>
        <Ionicons name="alert-circle" size={48} color={colors.gray[500]} />
        <Text style={styles.errorText}>{error || 'Devotional not found'}</Text>
        <TouchableOpacity style={styles.backButton} onPress={() => router.back()}>
          <Text style={styles.backButtonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Header Info */}
      <View style={styles.header}>
        <View style={styles.headerTop}>
          <Text style={styles.dayLabel}>Day {devotional.day}</Text>
          {isCompleted && (
            <View style={styles.completedBadge}>
              <Ionicons name="checkmark-circle" size={16} color={colors.success} />
              <Text style={styles.completedText}>Completed</Text>
            </View>
          )}
        </View>
        <Text style={styles.title}>{devotional.title}</Text>
        <Text style={styles.date}>
          {new Date(devotional.date).toLocaleDateString('en-US', {
            weekday: 'long',
            month: 'long',
            day: 'numeric',
            year: 'numeric',
          })}
        </Text>
      </View>

      {/* Scripture */}
      <View style={styles.scriptureCard}>
        <Text style={styles.scriptureReference}>
          {devotional.scripture.book} {devotional.scripture.chapter}:
          {devotional.scripture.verseStart}
          {devotional.scripture.verseEnd ? `-${devotional.scripture.verseEnd}` : ''}
        </Text>
        <Text style={styles.scriptureText}>{devotional.scripture.text}</Text>
        <Text style={styles.scriptureTranslation}>
          — {devotional.scripture.translation}
        </Text>
      </View>

      {/* Devotional Video */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Devotional Video</Text>
        <VideoPlayer
          uri={devotional.videoUrl}
          posterUri={devotional.thumbnailUrl}
          showCaptions
          captionsText={devotional.scripture.text}
        />
      </View>

      {/* Prompt */}
      <View style={styles.promptCard}>
        <Ionicons name="chatbubble-ellipses" size={24} color={colors.primary[500]} />
        <Text style={styles.promptTitle}>Reflection Prompt</Text>
        <Text style={styles.promptText}>{devotional.prompt.question}</Text>
      </View>

      {/* User Recording */}
      {recording ? (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Your Response</Text>
          <VideoPlayer uri={recording.videoUrl} showControls />
          <View style={styles.recordingMeta}>
            <Text style={styles.recordingDate}>
              Recorded {new Date(recording.createdAt).toLocaleDateString()}
            </Text>
            {recording.hasConsent && (
              <View style={styles.consentBadge}>
                <Ionicons name="heart" size={14} color={colors.success} />
                <Text style={styles.consentBadgeText}>Contributing to ASL training</Text>
              </View>
            )}
          </View>
        </View>
      ) : isCompleted ? (
        <View style={styles.noRecording}>
          <Ionicons name="videocam-off" size={32} color={colors.gray[500]} />
          <Text style={styles.noRecordingText}>You skipped recording for this devotional</Text>
        </View>
      ) : (
        <TouchableOpacity
          style={styles.recordButton}
          onPress={() =>
            router.push({
              pathname: '/record',
              params: {
                devotionalId: devotional.id,
                prompt: devotional.prompt.question,
                maxDuration: devotional.prompt.maxDuration || 60,
              },
            })
          }
        >
          <View style={styles.recordButtonIcon}>
            <Ionicons name="videocam" size={24} color="white" />
          </View>
          <Text style={styles.recordButtonText}>Record Your Response</Text>
        </TouchableOpacity>
      )}
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
    backgroundColor: colors.background.dark,
    padding: spacing.xl,
  },
  errorText: {
    marginTop: spacing.md,
    fontSize: 16,
    color: colors.text.dark.secondary,
    textAlign: 'center',
  },
  backButton: {
    marginTop: spacing.lg,
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.md,
    backgroundColor: colors.primary[500],
    borderRadius: borderRadius.lg,
  },
  backButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  header: {
    marginBottom: spacing.xl,
  },
  headerTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.xs,
  },
  dayLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.primary[400],
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  completedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  completedText: {
    fontSize: 12,
    color: colors.success,
    fontWeight: '500',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: colors.text.dark.primary,
  },
  date: {
    fontSize: 14,
    color: colors.text.dark.muted,
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
  promptCard: {
    alignItems: 'center',
    backgroundColor: colors.gray[800],
    padding: spacing.xl,
    borderRadius: borderRadius.xl,
    marginBottom: spacing.xl,
  },
  promptTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.dark.primary,
    marginTop: spacing.sm,
    marginBottom: spacing.sm,
  },
  promptText: {
    fontSize: 16,
    color: colors.text.dark.secondary,
    textAlign: 'center',
    lineHeight: 24,
  },
  recordingMeta: {
    marginTop: spacing.md,
    alignItems: 'center',
  },
  recordingDate: {
    fontSize: 14,
    color: colors.text.dark.muted,
  },
  consentBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    marginTop: spacing.sm,
  },
  consentBadgeText: {
    fontSize: 12,
    color: colors.success,
  },
  noRecording: {
    alignItems: 'center',
    padding: spacing.xl,
    backgroundColor: colors.gray[800],
    borderRadius: borderRadius.lg,
  },
  noRecordingText: {
    marginTop: spacing.md,
    fontSize: 14,
    color: colors.text.dark.muted,
    textAlign: 'center',
  },
  recordButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.lg,
    backgroundColor: colors.primary[900],
    borderRadius: borderRadius.xl,
    borderWidth: 2,
    borderColor: colors.primary[500],
    gap: spacing.md,
  },
  recordButtonIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: colors.primary[500],
    justifyContent: 'center',
    alignItems: 'center',
  },
  recordButtonText: {
    fontSize: 18,
    fontWeight: '600',
    color: colors.text.dark.primary,
  },
});
