/**
 * ConsentModal Component
 * Privacy consent for training data collection
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Modal,
  Switch,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius } from './ui/theme';

interface ConsentModalProps {
  visible: boolean;
  onAccept: () => void;
  onDecline: () => void;
}

export function ConsentModal({ visible, onAccept, onDecline }: ConsentModalProps) {
  const [understood, setUnderstood] = useState(false);

  return (
    <Modal
      visible={visible}
      animationType="slide"
      presentationStyle="pageSheet"
      onRequestClose={onDecline}
    >
      <View style={styles.container}>
        <ScrollView style={styles.scrollView} contentContainerStyle={styles.content}>
          {/* Header */}
          <View style={styles.header}>
            <View style={styles.iconContainer}>
              <Ionicons name="hand-left" size={48} color={colors.primary[500]} />
            </View>
            <Text style={styles.title}>Help Us Improve ASL Recognition</Text>
            <Text style={styles.subtitle}>
              Your signing can help train AI to better understand American Sign Language
            </Text>
          </View>

          {/* What we collect */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>What We Collect</Text>
            <View style={styles.bulletList}>
              <BulletPoint
                icon="videocam"
                text="Video recordings of your signed responses to devotional prompts"
              />
              <BulletPoint
                icon="document-text"
                text="The prompt/question you were responding to"
              />
              <BulletPoint
                icon="time"
                text="Timestamp and device type (for quality purposes)"
              />
            </View>
          </View>

          {/* How we use it */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>How We Use Your Data</Text>
            <View style={styles.bulletList}>
              <BulletPoint
                icon="hardware-chip"
                iconColor={colors.success}
                text="Train our SonZo AI sign language recognition models"
              />
              <BulletPoint
                icon="lock-closed"
                iconColor={colors.success}
                text="Your videos are NEVER shown to other users"
              />
              <BulletPoint
                icon="eye-off"
                iconColor={colors.success}
                text="Your videos are NEVER shared publicly"
              />
              <BulletPoint
                icon="shield-checkmark"
                iconColor={colors.success}
                text="Data is stored securely and encrypted"
              />
            </View>
          </View>

          {/* Your rights */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Your Rights</Text>
            <View style={styles.bulletList}>
              <BulletPoint
                icon="trash"
                iconColor={colors.secondary[500]}
                text="Delete all your contributed data at any time"
              />
              <BulletPoint
                icon="close-circle"
                iconColor={colors.secondary[500]}
                text="Withdraw consent and stop contributing"
              />
              <BulletPoint
                icon="download"
                iconColor={colors.secondary[500]}
                text="Request a copy of your data"
              />
            </View>
          </View>

          {/* Not required notice */}
          <View style={styles.noticeBox}>
            <Ionicons name="information-circle" size={24} color={colors.info} />
            <Text style={styles.noticeText}>
              Contributing is optional. You can still use SignedWord without sharing your
              recordings for training.
            </Text>
          </View>

          {/* Confirmation checkbox */}
          <View style={styles.confirmationRow}>
            <Switch
              value={understood}
              onValueChange={setUnderstood}
              trackColor={{ false: colors.gray[600], true: colors.primary[400] }}
              thumbColor={understood ? colors.primary[500] : colors.gray[400]}
            />
            <Text style={styles.confirmationText}>
              I understand how my data will be used
            </Text>
          </View>
        </ScrollView>

        {/* Actions */}
        <View style={styles.actions}>
          <TouchableOpacity style={styles.declineButton} onPress={onDecline}>
            <Text style={styles.declineButtonText}>No Thanks</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.acceptButton, !understood && styles.acceptButtonDisabled]}
            onPress={onAccept}
            disabled={!understood}
          >
            <Text style={styles.acceptButtonText}>I Agree to Help</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
}

interface BulletPointProps {
  icon: keyof typeof Ionicons.glyphMap;
  text: string;
  iconColor?: string;
}

function BulletPoint({ icon, text, iconColor = colors.gray[400] }: BulletPointProps) {
  return (
    <View style={styles.bulletItem}>
      <Ionicons name={icon} size={20} color={iconColor} style={styles.bulletIcon} />
      <Text style={styles.bulletText}>{text}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.dark,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: spacing.lg,
    paddingBottom: spacing['2xl'],
  },
  header: {
    alignItems: 'center',
    marginBottom: spacing.xl,
  },
  iconContainer: {
    width: 96,
    height: 96,
    borderRadius: 48,
    backgroundColor: colors.primary[900],
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.lg,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: colors.text.dark.primary,
    textAlign: 'center',
    marginBottom: spacing.sm,
  },
  subtitle: {
    fontSize: 16,
    color: colors.text.dark.secondary,
    textAlign: 'center',
    lineHeight: 24,
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
  bulletList: {
    gap: spacing.md,
  },
  bulletItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  bulletIcon: {
    marginRight: spacing.sm,
    marginTop: 2,
  },
  bulletText: {
    flex: 1,
    fontSize: 15,
    color: colors.text.dark.secondary,
    lineHeight: 22,
  },
  noticeBox: {
    flexDirection: 'row',
    backgroundColor: colors.gray[800],
    padding: spacing.md,
    borderRadius: borderRadius.lg,
    marginBottom: spacing.xl,
    alignItems: 'flex-start',
  },
  noticeText: {
    flex: 1,
    marginLeft: spacing.sm,
    fontSize: 14,
    color: colors.text.dark.secondary,
    lineHeight: 20,
  },
  confirmationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.md,
  },
  confirmationText: {
    flex: 1,
    marginLeft: spacing.md,
    fontSize: 16,
    color: colors.text.dark.primary,
  },
  actions: {
    flexDirection: 'row',
    padding: spacing.lg,
    gap: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.gray[800],
  },
  declineButton: {
    flex: 1,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.lg,
    backgroundColor: colors.gray[800],
    alignItems: 'center',
  },
  declineButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.dark.secondary,
  },
  acceptButton: {
    flex: 1,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.lg,
    backgroundColor: colors.primary[500],
    alignItems: 'center',
  },
  acceptButtonDisabled: {
    backgroundColor: colors.gray[700],
  },
  acceptButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: 'white',
  },
});
