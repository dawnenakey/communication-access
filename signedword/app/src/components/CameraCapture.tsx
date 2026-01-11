/**
 * CameraCapture Component
 * Cross-platform camera recording for sign language responses
 *
 * - Mobile: expo-camera
 * - Web: MediaRecorder API
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Platform,
  Alert,
} from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius } from './ui/theme';

interface CameraCaptureProps {
  maxDuration?: number; // seconds
  onRecordingComplete: (uri: string, duration: number) => void;
  onCancel?: () => void;
  prompt?: string;
}

export function CameraCapture({
  maxDuration = 60,
  onRecordingComplete,
  onCancel,
  prompt,
}: CameraCaptureProps) {
  const [permission, requestPermission] = useCameraPermissions();
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [facing, setFacing] = useState<CameraType>('front');

  const cameraRef = useRef<CameraView>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const recordingStartRef = useRef<number>(0);

  // Web-specific refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (Platform.OS === 'web' && streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Initialize web camera
  useEffect(() => {
    if (Platform.OS === 'web') {
      initWebCamera();
    }
  }, []);

  const initWebCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: true,
      });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      Alert.alert('Camera Error', 'Could not access camera. Please check permissions.');
    }
  };

  // Start countdown then recording
  const startCountdown = useCallback(() => {
    setCountdown(3);

    const countdownInterval = setInterval(() => {
      setCountdown(prev => {
        if (prev === null || prev <= 1) {
          clearInterval(countdownInterval);
          startRecording();
          return null;
        }
        return prev - 1;
      });
    }, 1000);
  }, []);

  // Start recording
  const startRecording = async () => {
    setIsRecording(true);
    setRecordingDuration(0);
    recordingStartRef.current = Date.now();

    // Start duration timer
    timerRef.current = setInterval(() => {
      const elapsed = Math.floor((Date.now() - recordingStartRef.current) / 1000);
      setRecordingDuration(elapsed);

      // Auto-stop at max duration
      if (elapsed >= maxDuration) {
        stopRecording();
      }
    }, 100);

    if (Platform.OS === 'web') {
      // Web recording
      if (streamRef.current) {
        chunksRef.current = [];
        const mediaRecorder = new MediaRecorder(streamRef.current, {
          mimeType: 'video/webm;codecs=vp9',
        });

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            chunksRef.current.push(event.data);
          }
        };

        mediaRecorder.onstop = () => {
          const blob = new Blob(chunksRef.current, { type: 'video/webm' });
          const url = URL.createObjectURL(blob);
          const duration = Math.floor((Date.now() - recordingStartRef.current) / 1000);
          onRecordingComplete(url, duration);
        };

        mediaRecorderRef.current = mediaRecorder;
        mediaRecorder.start(1000); // Collect data every second
      }
    } else {
      // Mobile recording
      if (cameraRef.current) {
        try {
          const video = await cameraRef.current.recordAsync({
            maxDuration,
          });
          const duration = Math.floor((Date.now() - recordingStartRef.current) / 1000);
          onRecordingComplete(video.uri, duration);
        } catch (error) {
          console.error('Recording error:', error);
          setIsRecording(false);
        }
      }
    }
  };

  // Stop recording
  const stopRecording = async () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    setIsRecording(false);

    if (Platform.OS === 'web') {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
    } else {
      if (cameraRef.current) {
        await cameraRef.current.stopRecording();
      }
    }
  };

  // Toggle camera facing
  const toggleFacing = () => {
    setFacing(current => (current === 'back' ? 'front' : 'back'));

    if (Platform.OS === 'web' && streamRef.current) {
      // Restart web camera with new facing mode
      streamRef.current.getTracks().forEach(track => track.stop());
      initWebCamera();
    }
  };

  // Format duration display
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Permission handling
  if (!permission) {
    return (
      <View style={styles.container}>
        <Text style={styles.messageText}>Requesting camera permission...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <View style={styles.permissionContainer}>
          <Ionicons name="camera-outline" size={64} color={colors.gray[400]} />
          <Text style={styles.permissionTitle}>Camera Access Required</Text>
          <Text style={styles.permissionText}>
            SignedWord needs camera access to record your sign language responses.
          </Text>
          <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
            <Text style={styles.permissionButtonText}>Grant Permission</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Camera view */}
      {Platform.OS === 'web' ? (
        <video
          ref={videoRef as any}
          autoPlay
          playsInline
          muted
          style={styles.webVideo as any}
        />
      ) : (
        <CameraView
          ref={cameraRef}
          style={styles.camera}
          facing={facing}
          mode="video"
        />
      )}

      {/* Prompt overlay */}
      {prompt && !isRecording && countdown === null && (
        <View style={styles.promptOverlay}>
          <Text style={styles.promptText}>{prompt}</Text>
        </View>
      )}

      {/* Countdown overlay */}
      {countdown !== null && (
        <View style={styles.countdownOverlay}>
          <Text style={styles.countdownText}>{countdown}</Text>
          <Text style={styles.countdownLabel}>Get ready...</Text>
        </View>
      )}

      {/* Recording indicator */}
      {isRecording && (
        <View style={styles.recordingIndicator}>
          <View style={styles.recordingDot} />
          <Text style={styles.recordingTime}>
            {formatDuration(recordingDuration)} / {formatDuration(maxDuration)}
          </Text>
        </View>
      )}

      {/* Progress bar */}
      {isRecording && (
        <View style={styles.progressBar}>
          <View
            style={[
              styles.progressFill,
              { width: `${(recordingDuration / maxDuration) * 100}%` },
            ]}
          />
        </View>
      )}

      {/* Controls */}
      <View style={styles.controls}>
        {/* Cancel button */}
        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={onCancel}
          disabled={isRecording}
        >
          <Ionicons
            name="close"
            size={28}
            color={isRecording ? colors.gray[600] : colors.gray[300]}
          />
        </TouchableOpacity>

        {/* Record button */}
        <TouchableOpacity
          style={[
            styles.recordButton,
            isRecording && styles.recordButtonActive,
          ]}
          onPress={isRecording ? stopRecording : startCountdown}
        >
          {isRecording ? (
            <View style={styles.stopIcon} />
          ) : (
            <View style={styles.recordIcon} />
          )}
        </TouchableOpacity>

        {/* Flip camera button */}
        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={toggleFacing}
          disabled={isRecording}
        >
          <Ionicons
            name="camera-reverse"
            size={28}
            color={isRecording ? colors.gray[600] : colors.gray[300]}
          />
        </TouchableOpacity>
      </View>

      {/* Instructions */}
      {!isRecording && countdown === null && (
        <View style={styles.instructions}>
          <Text style={styles.instructionsText}>
            Tap the record button when ready. Max {maxDuration} seconds.
          </Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.gray[900],
  },
  camera: {
    flex: 1,
  },
  webVideo: {
    flex: 1,
    width: '100%',
    height: '100%',
    objectFit: 'cover',
    transform: [{ scaleX: -1 }], // Mirror for selfie
  },
  messageText: {
    color: colors.gray[300],
    fontSize: 16,
    textAlign: 'center',
    marginTop: 100,
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.xl,
  },
  permissionTitle: {
    color: colors.gray[100],
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: spacing.lg,
    marginBottom: spacing.sm,
  },
  permissionText: {
    color: colors.gray[400],
    fontSize: 16,
    textAlign: 'center',
    marginBottom: spacing.xl,
  },
  permissionButton: {
    backgroundColor: colors.primary[500],
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.lg,
  },
  permissionButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  promptOverlay: {
    position: 'absolute',
    top: spacing.xl,
    left: spacing.md,
    right: spacing.md,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: spacing.md,
    borderRadius: borderRadius.lg,
  },
  promptText: {
    color: 'white',
    fontSize: 18,
    textAlign: 'center',
    fontStyle: 'italic',
  },
  countdownOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
  },
  countdownText: {
    color: 'white',
    fontSize: 120,
    fontWeight: 'bold',
  },
  countdownLabel: {
    color: colors.gray[300],
    fontSize: 24,
    marginTop: spacing.md,
  },
  recordingIndicator: {
    position: 'absolute',
    top: spacing.xl,
    right: spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.full,
  },
  recordingDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: colors.error,
    marginRight: spacing.sm,
  },
  recordingTime: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  progressBar: {
    position: 'absolute',
    bottom: 140,
    left: spacing.md,
    right: spacing.md,
    height: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    borderRadius: 2,
  },
  progressFill: {
    height: '100%',
    backgroundColor: colors.primary[500],
    borderRadius: 2,
  },
  controls: {
    position: 'absolute',
    bottom: spacing.xl,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: spacing.xl,
    paddingHorizontal: spacing.xl,
  },
  recordButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: 'white',
  },
  recordButtonActive: {
    backgroundColor: 'rgba(239, 68, 68, 0.2)',
    borderColor: colors.error,
  },
  recordIcon: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: colors.error,
  },
  stopIcon: {
    width: 28,
    height: 28,
    borderRadius: 4,
    backgroundColor: colors.error,
  },
  secondaryButton: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  instructions: {
    position: 'absolute',
    bottom: 130,
    left: spacing.md,
    right: spacing.md,
    alignItems: 'center',
  },
  instructionsText: {
    color: colors.gray[400],
    fontSize: 14,
    textAlign: 'center',
  },
});
