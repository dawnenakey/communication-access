/**
 * Preview Screen
 * Review recorded video before uploading
 */

import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  SafeAreaView,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { VideoPlayer } from '../src/components/VideoPlayer';
import { RecordingsAPI } from '../src/services/api';
import { useAuthStore } from '../src/context/auth';
import { colors, spacing, borderRadius } from '../src/components/ui/theme';

export default function PreviewScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{
    videoUri: string;
    devotionalId: string;
    hasConsent: string;
  }>();
  const refreshUser = useAuthStore((state) => state.refreshUser);

  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const hasConsent = params.hasConsent === 'true';

  // Handle upload
  const handleUpload = useCallback(async () => {
    if (!params.videoUri || !params.devotionalId) {
      Alert.alert('Error', 'Missing video or devotional information');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    try {
      const response = await RecordingsAPI.upload({
        videoUri: params.videoUri,
        devotionalId: params.devotionalId,
        hasConsent,
        onProgress: (progress) => setUploadProgress(progress),
      });

      if (response.success) {
        // Refresh user data to update progress
        await refreshUser();

        // Navigate back to main screen with success
        Alert.alert(
          'Response Submitted!',
          'Your signed response has been saved.',
          [
            {
              text: 'Continue',
              onPress: () => {
                // Navigate to tabs and mark devotional as complete
                router.replace('/(tabs)');
              },
            },
          ]
        );
      } else {
        throw new Error(response.error?.message || 'Upload failed');
      }
    } catch (error) {
      Alert.alert(
        'Upload Failed',
        error instanceof Error ? error.message : 'Please try again.',
        [
          { text: 'Retry', onPress: handleUpload },
          { text: 'Cancel', style: 'cancel' },
        ]
      );
    } finally {
      setIsUploading(false);
    }
  }, [params.videoUri, params.devotionalId, hasConsent, refreshUser, router]);

  // Handle re-record
  const handleReRecord = () => {
    Alert.alert(
      'Record Again?',
      'Your current recording will be discarded.',
      [
        { text: 'Keep Recording', style: 'cancel' },
        {
          text: 'Re-record',
          onPress: () => router.back(),
        },
      ]
    );
  };

  // Handle discard
  const handleDiscard = () => {
    Alert.alert(
      'Discard Recording?',
      'This recording will be permanently deleted.',
      [
        { text: 'Keep', style: 'cancel' },
        {
          text: 'Discard',
          style: 'destructive',
          onPress: () => {
            // Navigate back to devotional without saving
            router.replace('/(tabs)');
          },
        },
      ]
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.headerButton} onPress={handleDiscard}>
          <Ionicons name="close" size={24} color={colors.text.dark.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Preview</Text>
        <View style={styles.headerButton} />
      </View>

      {/* Video Preview */}
      <View style={styles.videoContainer}>
        {params.videoUri ? (
          <VideoPlayer
            uri={params.videoUri}
            autoPlay={false}
            showControls
            style={styles.video}
          />
        ) : (
          <View style={styles.noVideo}>
            <Ionicons name="videocam-off" size={48} color={colors.gray[500]} />
            <Text style={styles.noVideoText}>No video to preview</Text>
          </View>
        )}
      </View>

      {/* Consent indicator */}
      <View style={styles.consentInfo}>
        <Ionicons
          name={hasConsent ? 'heart' : 'lock-closed'}
          size={20}
          color={hasConsent ? colors.success : colors.text.dark.muted}
        />
        <Text style={styles.consentText}>
          {hasConsent
            ? 'This recording will help improve ASL recognition'
            : 'Private recording (not used for training)'}
        </Text>
      </View>

      {/* Action Buttons */}
      <View style={styles.actions}>
        {/* Re-record button */}
        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={handleReRecord}
          disabled={isUploading}
        >
          <Ionicons name="refresh" size={24} color={colors.text.dark.primary} />
          <Text style={styles.secondaryButtonText}>Re-record</Text>
        </TouchableOpacity>

        {/* Submit button */}
        <TouchableOpacity
          style={[styles.primaryButton, isUploading && styles.primaryButtonDisabled]}
          onPress={handleUpload}
          disabled={isUploading}
        >
          {isUploading ? (
            <View style={styles.uploadingContent}>
              <ActivityIndicator color="white" size="small" />
              <Text style={styles.primaryButtonText}>
                Uploading {Math.round(uploadProgress * 100)}%
              </Text>
            </View>
          ) : (
            <>
              <Ionicons name="cloud-upload" size={24} color="white" />
              <Text style={styles.primaryButtonText}>Submit Response</Text>
            </>
          )}
        </TouchableOpacity>
      </View>

      {/* Upload progress bar */}
      {isUploading && (
        <View style={styles.progressContainer}>
          <View style={[styles.progressBar, { width: `${uploadProgress * 100}%` }]} />
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.dark,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
  },
  headerButton: {
    width: 44,
    height: 44,
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: colors.text.dark.primary,
  },
  videoContainer: {
    flex: 1,
    margin: spacing.lg,
    borderRadius: borderRadius.xl,
    overflow: 'hidden',
    backgroundColor: '#000',
  },
  video: {
    flex: 1,
  },
  noVideo: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  noVideoText: {
    marginTop: spacing.md,
    color: colors.text.dark.muted,
    fontSize: 16,
  },
  consentInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.lg,
    gap: spacing.sm,
  },
  consentText: {
    fontSize: 14,
    color: colors.text.dark.secondary,
  },
  actions: {
    flexDirection: 'row',
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.xl,
    gap: spacing.md,
  },
  secondaryButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.md,
    backgroundColor: colors.gray[800],
    borderRadius: borderRadius.lg,
    gap: spacing.sm,
  },
  secondaryButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.dark.primary,
  },
  primaryButton: {
    flex: 2,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.md,
    backgroundColor: colors.primary[500],
    borderRadius: borderRadius.lg,
    gap: spacing.sm,
  },
  primaryButtonDisabled: {
    backgroundColor: colors.primary[700],
  },
  primaryButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: 'white',
  },
  uploadingContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  progressContainer: {
    height: 4,
    backgroundColor: colors.gray[700],
    marginHorizontal: spacing.lg,
    marginBottom: spacing.lg,
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: colors.primary[500],
  },
});
