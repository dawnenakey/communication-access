/**
 * Record Screen
 * Full-screen camera recording for devotional responses
 */

import React, { useState, useRef, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  SafeAreaView,
  Alert,
  ActivityIndicator,
  Platform,
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { CameraCapture } from '../src/components/CameraCapture';
import { RecordingsAPI } from '../src/services/api';
import { colors, spacing, borderRadius } from '../src/components/ui/theme';

type RecordingState = 'ready' | 'countdown' | 'recording' | 'processing' | 'done';

export default function RecordScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{
    devotionalId: string;
    prompt: string;
    maxDuration: string;
    noConsent?: string;
  }>();

  const [recordingState, setRecordingState] = useState<RecordingState>('ready');
  const [recordedUri, setRecordedUri] = useState<string | null>(null);
  const cameraRef = useRef<{ startRecording: () => void; stopRecording: () => void }>(null);

  const maxDuration = parseInt(params.maxDuration || '60', 10);
  const hasConsent = params.noConsent !== 'true';

  // Handle recording complete
  const handleRecordingComplete = useCallback((uri: string) => {
    setRecordedUri(uri);
    setRecordingState('done');

    // Navigate to preview with the recorded video
    router.push({
      pathname: '/preview',
      params: {
        videoUri: uri,
        devotionalId: params.devotionalId,
        hasConsent: hasConsent.toString(),
      },
    });
  }, [router, params.devotionalId, hasConsent]);

  // Handle recording error
  const handleRecordingError = useCallback((error: string) => {
    Alert.alert('Recording Error', error, [
      { text: 'OK', onPress: () => setRecordingState('ready') },
    ]);
  }, []);

  // Handle close
  const handleClose = () => {
    if (recordingState === 'recording') {
      Alert.alert(
        'Stop Recording?',
        'Your current recording will be lost.',
        [
          { text: 'Continue Recording', style: 'cancel' },
          { text: 'Discard', style: 'destructive', onPress: () => router.back() },
        ]
      );
    } else {
      router.back();
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.closeButton} onPress={handleClose}>
          <Ionicons name="close" size={28} color="white" />
        </TouchableOpacity>

        {recordingState !== 'recording' && (
          <View style={styles.promptContainer}>
            <Text style={styles.promptLabel}>Respond to:</Text>
            <Text style={styles.promptText} numberOfLines={2}>
              {params.prompt || 'Record your response'}
            </Text>
          </View>
        )}
      </View>

      {/* Camera View */}
      <View style={styles.cameraContainer}>
        <CameraCapture
          maxDuration={maxDuration}
          onRecordingComplete={handleRecordingComplete}
          onError={handleRecordingError}
          showPreview
        />
      </View>

      {/* Instructions */}
      {recordingState === 'ready' && (
        <View style={styles.instructions}>
          <View style={styles.instructionItem}>
            <Ionicons name="hand-left" size={20} color={colors.primary[400]} />
            <Text style={styles.instructionText}>
              Sign your response to the prompt above
            </Text>
          </View>
          <View style={styles.instructionItem}>
            <Ionicons name="time" size={20} color={colors.primary[400]} />
            <Text style={styles.instructionText}>
              Maximum {maxDuration} seconds
            </Text>
          </View>
          <View style={styles.instructionItem}>
            <Ionicons name="eye" size={20} color={colors.primary[400]} />
            <Text style={styles.instructionText}>
              You'll preview before submitting
            </Text>
          </View>

          {!hasConsent && (
            <View style={styles.noConsentBadge}>
              <Ionicons name="lock-closed" size={16} color={colors.text.dark.muted} />
              <Text style={styles.noConsentText}>
                Private recording (not used for training)
              </Text>
            </View>
          )}
        </View>
      )}

      {/* Processing indicator */}
      {recordingState === 'processing' && (
        <View style={styles.processingOverlay}>
          <ActivityIndicator size="large" color={colors.primary[500]} />
          <Text style={styles.processingText}>Processing video...</Text>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  header: {
    paddingHorizontal: spacing.lg,
    paddingTop: Platform.OS === 'android' ? spacing.xl : spacing.md,
    paddingBottom: spacing.md,
    zIndex: 10,
  },
  closeButton: {
    position: 'absolute',
    top: Platform.OS === 'android' ? spacing.xl : spacing.md,
    left: spacing.lg,
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 20,
  },
  promptContainer: {
    marginLeft: 60,
    marginRight: spacing.md,
    backgroundColor: 'rgba(0,0,0,0.6)',
    padding: spacing.md,
    borderRadius: borderRadius.lg,
  },
  promptLabel: {
    fontSize: 12,
    color: colors.primary[400],
    fontWeight: '600',
    textTransform: 'uppercase',
    marginBottom: spacing.xs,
  },
  promptText: {
    fontSize: 16,
    color: 'white',
    lineHeight: 22,
  },
  cameraContainer: {
    flex: 1,
  },
  instructions: {
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.lg,
    backgroundColor: 'rgba(0,0,0,0.8)',
  },
  instructionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    marginBottom: spacing.sm,
  },
  instructionText: {
    fontSize: 14,
    color: colors.text.dark.secondary,
  },
  noConsentBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginTop: spacing.md,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    backgroundColor: colors.gray[800],
    borderRadius: borderRadius.md,
    alignSelf: 'flex-start',
  },
  noConsentText: {
    fontSize: 12,
    color: colors.text.dark.muted,
  },
  processingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.8)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  processingText: {
    marginTop: spacing.md,
    fontSize: 16,
    color: 'white',
  },
});
