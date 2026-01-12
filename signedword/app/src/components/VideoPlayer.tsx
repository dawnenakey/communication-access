/**
 * VideoPlayer Component
 * Cross-platform video player for devotional content
 */

import React, { useState, useRef, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Platform,
} from 'react-native';
import { Video, ResizeMode, AVPlaybackStatus } from 'expo-av';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius } from './ui/theme';

interface VideoPlayerProps {
  uri: string;
  posterUri?: string;
  autoPlay?: boolean;
  showControls?: boolean;
  showCaptions?: boolean;
  captionsText?: string;
  onComplete?: () => void;
  onProgress?: (progress: number) => void;
  style?: object;
}

export function VideoPlayer({
  uri,
  posterUri,
  autoPlay = false,
  showControls = true,
  showCaptions = false,
  captionsText,
  onComplete,
  onProgress,
  style,
}: VideoPlayerProps) {
  const videoRef = useRef<Video>(null);
  const [status, setStatus] = useState<AVPlaybackStatus | null>(null);
  const [isBuffering, setIsBuffering] = useState(false);
  const [showControlsOverlay, setShowControlsOverlay] = useState(true);

  const isPlaying = status?.isLoaded && status.isPlaying;
  const duration = status?.isLoaded ? status.durationMillis || 0 : 0;
  const position = status?.isLoaded ? status.positionMillis || 0 : 0;
  const progress = duration > 0 ? position / duration : 0;

  const handlePlaybackStatusUpdate = useCallback(
    (newStatus: AVPlaybackStatus) => {
      setStatus(newStatus);

      if (newStatus.isLoaded) {
        setIsBuffering(newStatus.isBuffering);

        if (onProgress && newStatus.durationMillis) {
          onProgress(newStatus.positionMillis / newStatus.durationMillis);
        }

        if (newStatus.didJustFinish && onComplete) {
          onComplete();
        }
      }
    },
    [onProgress, onComplete]
  );

  const togglePlayPause = async () => {
    if (!videoRef.current) return;

    if (isPlaying) {
      await videoRef.current.pauseAsync();
    } else {
      await videoRef.current.playAsync();
    }
  };

  const seekTo = async (seconds: number) => {
    if (!videoRef.current) return;
    await videoRef.current.setPositionAsync(seconds * 1000);
  };

  const skip = async (seconds: number) => {
    if (!videoRef.current || !status?.isLoaded) return;
    const newPosition = Math.max(0, Math.min(position + seconds * 1000, duration));
    await videoRef.current.setPositionAsync(newPosition);
  };

  const formatTime = (ms: number) => {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const toggleControls = () => {
    setShowControlsOverlay(!showControlsOverlay);
  };

  return (
    <View style={[styles.container, style]}>
      <TouchableOpacity
        activeOpacity={1}
        onPress={toggleControls}
        style={styles.videoContainer}
      >
        <Video
          ref={videoRef}
          source={{ uri }}
          posterSource={posterUri ? { uri: posterUri } : undefined}
          usePoster={!!posterUri}
          posterStyle={styles.poster}
          style={styles.video}
          resizeMode={ResizeMode.CONTAIN}
          shouldPlay={autoPlay}
          isLooping={false}
          onPlaybackStatusUpdate={handlePlaybackStatusUpdate}
        />

        {/* Buffering indicator */}
        {isBuffering && (
          <View style={styles.bufferingOverlay}>
            <ActivityIndicator size="large" color={colors.primary[500]} />
          </View>
        )}

        {/* Controls overlay */}
        {showControls && showControlsOverlay && (
          <View style={styles.controlsOverlay}>
            {/* Center controls */}
            <View style={styles.centerControls}>
              <TouchableOpacity
                onPress={() => skip(-10)}
                style={styles.skipButton}
              >
                <Ionicons name="play-back" size={28} color="white" />
                <Text style={styles.skipText}>10</Text>
              </TouchableOpacity>

              <TouchableOpacity
                onPress={togglePlayPause}
                style={styles.playButton}
              >
                <Ionicons
                  name={isPlaying ? 'pause' : 'play'}
                  size={40}
                  color="white"
                />
              </TouchableOpacity>

              <TouchableOpacity
                onPress={() => skip(10)}
                style={styles.skipButton}
              >
                <Ionicons name="play-forward" size={28} color="white" />
                <Text style={styles.skipText}>10</Text>
              </TouchableOpacity>
            </View>

            {/* Bottom controls */}
            <View style={styles.bottomControls}>
              {/* Progress bar */}
              <View style={styles.progressContainer}>
                <View style={styles.progressBar}>
                  <View
                    style={[styles.progressFill, { width: `${progress * 100}%` }]}
                  />
                </View>
              </View>

              {/* Time */}
              <View style={styles.timeContainer}>
                <Text style={styles.timeText}>{formatTime(position)}</Text>
                <Text style={styles.timeText}>{formatTime(duration)}</Text>
              </View>
            </View>
          </View>
        )}

        {/* Captions */}
        {showCaptions && captionsText && (
          <View style={styles.captionsContainer}>
            <Text style={styles.captionsText}>{captionsText}</Text>
          </View>
        )}
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
    aspectRatio: 16 / 9,
    backgroundColor: colors.gray[900],
    borderRadius: borderRadius.lg,
    overflow: 'hidden',
  },
  videoContainer: {
    flex: 1,
  },
  video: {
    flex: 1,
  },
  poster: {
    resizeMode: 'cover',
  },
  bufferingOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
  },
  controlsOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.4)',
    justifyContent: 'space-between',
  },
  centerControls: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: spacing.xl,
  },
  playButton: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: 'rgba(139, 92, 246, 0.9)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  skipButton: {
    alignItems: 'center',
  },
  skipText: {
    color: 'white',
    fontSize: 12,
    marginTop: 2,
  },
  bottomControls: {
    padding: spacing.md,
  },
  progressContainer: {
    marginBottom: spacing.sm,
  },
  progressBar: {
    height: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    borderRadius: 2,
  },
  progressFill: {
    height: '100%',
    backgroundColor: colors.primary[500],
    borderRadius: 2,
  },
  timeContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  timeText: {
    color: 'white',
    fontSize: 12,
  },
  captionsContainer: {
    position: 'absolute',
    bottom: 80,
    left: spacing.md,
    right: spacing.md,
    backgroundColor: 'rgba(0, 0, 0, 0.75)',
    padding: spacing.sm,
    borderRadius: borderRadius.sm,
  },
  captionsText: {
    color: 'white',
    fontSize: 16,
    textAlign: 'center',
  },
});
