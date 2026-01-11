/**
 * Profile Screen
 * User settings, consent management, and account options
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Switch,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useAuth, useConsent } from '../../src/context/auth';
import { UserAPI } from '../../src/services/api';
import { colors, spacing, borderRadius } from '../../src/components/ui/theme';

export default function ProfileScreen() {
  const router = useRouter();
  const { user, isAuthenticated, logout } = useAuth();
  const { hasConsented, updateConsent } = useConsent();

  const [isUpdatingConsent, setIsUpdatingConsent] = useState(false);
  const [isDeletingData, setIsDeletingData] = useState(false);

  // Handle consent toggle
  const handleConsentToggle = async (value: boolean) => {
    setIsUpdatingConsent(true);
    try {
      await updateConsent(value);
    } catch (error) {
      Alert.alert('Error', 'Failed to update consent setting');
    } finally {
      setIsUpdatingConsent(false);
    }
  };

  // Handle data deletion request
  const handleDeleteData = () => {
    Alert.alert(
      'Delete My Data',
      'This will permanently delete all your recorded responses. This action cannot be undone. Your account and progress will remain.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            setIsDeletingData(true);
            try {
              const response = await UserAPI.deleteRecordings();
              if (response.success) {
                Alert.alert('Success', 'Your recorded responses have been deleted.');
              } else {
                Alert.alert('Error', response.error?.message || 'Failed to delete data');
              }
            } catch (error) {
              Alert.alert('Error', 'Failed to delete data');
            } finally {
              setIsDeletingData(false);
            }
          },
        },
      ]
    );
  };

  // Handle account deletion
  const handleDeleteAccount = () => {
    Alert.alert(
      'Delete Account',
      'This will permanently delete your account and all associated data. This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete Account',
          style: 'destructive',
          onPress: async () => {
            try {
              const response = await UserAPI.deleteAccount();
              if (response.success) {
                await logout();
                Alert.alert('Account Deleted', 'Your account has been permanently deleted.');
              } else {
                Alert.alert('Error', response.error?.message || 'Failed to delete account');
              }
            } catch (error) {
              Alert.alert('Error', 'Failed to delete account');
            }
          },
        },
      ]
    );
  };

  // Handle logout
  const handleLogout = () => {
    Alert.alert('Sign Out', 'Are you sure you want to sign out?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Sign Out',
        onPress: async () => {
          await logout();
        },
      },
    ]);
  };

  if (!isAuthenticated || !user) {
    return (
      <View style={styles.centerContainer}>
        <Ionicons name="person-circle" size={80} color={colors.gray[500]} />
        <Text style={styles.emptyTitle}>Sign In Required</Text>
        <Text style={styles.emptyText}>Sign in to manage your profile and settings</Text>
        <TouchableOpacity
          style={styles.signInButton}
          onPress={() => router.push('/(auth)/login')}
        >
          <Text style={styles.signInButtonText}>Sign In</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* User Info */}
      <View style={styles.userCard}>
        <View style={styles.avatar}>
          <Text style={styles.avatarText}>
            {user.name.charAt(0).toUpperCase()}
          </Text>
        </View>
        <Text style={styles.userName}>{user.name}</Text>
        <Text style={styles.userEmail}>{user.email}</Text>
      </View>

      {/* Progress Stats */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Your Progress</Text>
        <View style={styles.statsGrid}>
          <View style={styles.statItem}>
            <Ionicons name="flame" size={24} color={colors.secondary[500]} />
            <Text style={styles.statValue}>{user.progress.streak}</Text>
            <Text style={styles.statLabel}>Day Streak</Text>
          </View>
          <View style={styles.statItem}>
            <Ionicons name="checkmark-circle" size={24} color={colors.success} />
            <Text style={styles.statValue}>{user.progress.completedDevotionals.length}</Text>
            <Text style={styles.statLabel}>Completed</Text>
          </View>
          <View style={styles.statItem}>
            <Ionicons name="videocam" size={24} color={colors.primary[500]} />
            <Text style={styles.statValue}>{user.progress.recordingsCount}</Text>
            <Text style={styles.statLabel}>Recordings</Text>
          </View>
        </View>
      </View>

      {/* Privacy & Data */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Privacy & Data</Text>

        {/* Training Data Consent */}
        <View style={styles.settingRow}>
          <View style={styles.settingInfo}>
            <Text style={styles.settingLabel}>Contribute to ASL Training</Text>
            <Text style={styles.settingDescription}>
              Allow your signed responses to help improve ASL recognition for the Deaf community
            </Text>
          </View>
          {isUpdatingConsent ? (
            <ActivityIndicator size="small" color={colors.primary[500]} />
          ) : (
            <Switch
              value={hasConsented}
              onValueChange={handleConsentToggle}
              trackColor={{ false: colors.gray[600], true: colors.primary[700] }}
              thumbColor={hasConsented ? colors.primary[500] : colors.gray[400]}
            />
          )}
        </View>

        {/* Privacy Info */}
        <View style={styles.privacyNote}>
          <Ionicons name="shield-checkmark" size={20} color={colors.success} />
          <Text style={styles.privacyText}>
            Your responses are always private and never shared publicly. Training data is
            anonymized and used only to improve sign language recognition.
          </Text>
        </View>

        {/* Delete Recordings */}
        <TouchableOpacity
          style={styles.dangerButton}
          onPress={handleDeleteData}
          disabled={isDeletingData}
        >
          {isDeletingData ? (
            <ActivityIndicator size="small" color={colors.error} />
          ) : (
            <>
              <Ionicons name="trash-outline" size={20} color={colors.error} />
              <Text style={styles.dangerButtonText}>Delete My Recordings</Text>
            </>
          )}
        </TouchableOpacity>
      </View>

      {/* Account Actions */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Account</Text>

        <TouchableOpacity style={styles.menuItem} onPress={handleLogout}>
          <Ionicons name="log-out-outline" size={22} color={colors.text.dark.primary} />
          <Text style={styles.menuItemText}>Sign Out</Text>
          <Ionicons name="chevron-forward" size={20} color={colors.text.dark.muted} />
        </TouchableOpacity>

        <TouchableOpacity style={styles.menuItem} onPress={handleDeleteAccount}>
          <Ionicons name="person-remove-outline" size={22} color={colors.error} />
          <Text style={[styles.menuItemText, { color: colors.error }]}>Delete Account</Text>
          <Ionicons name="chevron-forward" size={20} color={colors.error} />
        </TouchableOpacity>
      </View>

      {/* App Info */}
      <View style={styles.footer}>
        <Text style={styles.footerText}>SignedWord v1.0.0</Text>
        <Text style={styles.footerText}>A SonZo AI Project</Text>
        <Text style={styles.footerSubtext}>
          Empowering the Deaf community through ASL Bible devotions
        </Text>
      </View>
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
  emptyTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: colors.text.dark.primary,
    marginTop: spacing.lg,
  },
  emptyText: {
    fontSize: 16,
    color: colors.text.dark.secondary,
    textAlign: 'center',
    marginTop: spacing.sm,
    marginBottom: spacing.xl,
  },
  signInButton: {
    backgroundColor: colors.primary[500],
    paddingHorizontal: spacing['2xl'],
    paddingVertical: spacing.md,
    borderRadius: borderRadius.lg,
  },
  signInButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  userCard: {
    alignItems: 'center',
    padding: spacing.xl,
    backgroundColor: colors.gray[800],
    borderRadius: borderRadius.xl,
    marginBottom: spacing.xl,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: colors.primary[500],
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  avatarText: {
    fontSize: 32,
    fontWeight: 'bold',
    color: 'white',
  },
  userName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: colors.text.dark.primary,
  },
  userEmail: {
    fontSize: 14,
    color: colors.text.dark.muted,
    marginTop: spacing.xs,
  },
  section: {
    marginBottom: spacing.xl,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text.dark.muted,
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: spacing.md,
  },
  statsGrid: {
    flexDirection: 'row',
    backgroundColor: colors.gray[800],
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: colors.text.dark.primary,
    marginTop: spacing.sm,
  },
  statLabel: {
    fontSize: 12,
    color: colors.text.dark.muted,
    marginTop: spacing.xs,
  },
  settingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.gray[800],
    padding: spacing.lg,
    borderRadius: borderRadius.lg,
    marginBottom: spacing.md,
  },
  settingInfo: {
    flex: 1,
    marginRight: spacing.md,
  },
  settingLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.dark.primary,
  },
  settingDescription: {
    fontSize: 13,
    color: colors.text.dark.muted,
    marginTop: spacing.xs,
    lineHeight: 18,
  },
  privacyNote: {
    flexDirection: 'row',
    backgroundColor: colors.gray[900],
    padding: spacing.md,
    borderRadius: borderRadius.md,
    gap: spacing.sm,
    marginBottom: spacing.md,
  },
  privacyText: {
    flex: 1,
    fontSize: 13,
    color: colors.text.dark.secondary,
    lineHeight: 18,
  },
  dangerButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.md,
    borderRadius: borderRadius.lg,
    borderWidth: 1,
    borderColor: colors.error,
    gap: spacing.sm,
  },
  dangerButtonText: {
    fontSize: 14,
    color: colors.error,
    fontWeight: '500',
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.gray[800],
    padding: spacing.lg,
    borderRadius: borderRadius.lg,
    marginBottom: spacing.sm,
    gap: spacing.md,
  },
  menuItemText: {
    flex: 1,
    fontSize: 16,
    color: colors.text.dark.primary,
  },
  footer: {
    alignItems: 'center',
    paddingTop: spacing.xl,
    borderTopWidth: 1,
    borderTopColor: colors.gray[800],
  },
  footerText: {
    fontSize: 14,
    color: colors.text.dark.muted,
  },
  footerSubtext: {
    fontSize: 12,
    color: colors.text.dark.muted,
    marginTop: spacing.sm,
    textAlign: 'center',
  },
});
