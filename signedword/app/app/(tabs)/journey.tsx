/**
 * Journey Screen
 * Calendar view of completed devotionals and progress tracking
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
import { useAuth } from '../../src/context/auth';
import { DevotionalsAPI } from '../../src/services/api';
import { colors, spacing, borderRadius } from '../../src/components/ui/theme';
import type { Devotional } from '../../src/types';

const DAYS_IN_WEEK = 7;
const WEEKS_TO_SHOW = 6;

interface CalendarDay {
  date: Date;
  devotional?: Devotional;
  isCompleted: boolean;
  isToday: boolean;
  isCurrentMonth: boolean;
}

export default function JourneyScreen() {
  const router = useRouter();
  const { user, isAuthenticated } = useAuth();

  const [currentMonth, setCurrentMonth] = useState(new Date());
  const [devotionals, setDevotionals] = useState<Devotional[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Fetch devotionals for the current month
  const fetchDevotionals = useCallback(async () => {
    try {
      const startDate = new Date(currentMonth.getFullYear(), currentMonth.getMonth(), 1);
      const endDate = new Date(currentMonth.getFullYear(), currentMonth.getMonth() + 1, 0);

      const response = await DevotionalsAPI.getAll({
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      });

      if (response.success && response.data) {
        setDevotionals(response.data);
      }
    } catch (err) {
      console.error('Failed to fetch devotionals:', err);
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, [currentMonth]);

  useEffect(() => {
    if (isAuthenticated) {
      fetchDevotionals();
    }
  }, [isAuthenticated, fetchDevotionals]);

  // Generate calendar grid
  const generateCalendarDays = (): CalendarDay[] => {
    const days: CalendarDay[] = [];
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const firstDayOfMonth = new Date(currentMonth.getFullYear(), currentMonth.getMonth(), 1);
    const startingDayOfWeek = firstDayOfMonth.getDay();
    const daysInMonth = new Date(currentMonth.getFullYear(), currentMonth.getMonth() + 1, 0).getDate();

    // Previous month days
    const prevMonthDays = new Date(currentMonth.getFullYear(), currentMonth.getMonth(), 0).getDate();
    for (let i = startingDayOfWeek - 1; i >= 0; i--) {
      const date = new Date(currentMonth.getFullYear(), currentMonth.getMonth() - 1, prevMonthDays - i);
      days.push({
        date,
        isCompleted: false,
        isToday: false,
        isCurrentMonth: false,
      });
    }

    // Current month days
    for (let day = 1; day <= daysInMonth; day++) {
      const date = new Date(currentMonth.getFullYear(), currentMonth.getMonth(), day);
      date.setHours(0, 0, 0, 0);

      const devotional = devotionals.find((d) => {
        const devDate = new Date(d.date);
        devDate.setHours(0, 0, 0, 0);
        return devDate.getTime() === date.getTime();
      });

      const isCompleted = devotional
        ? user?.progress.completedDevotionals.includes(devotional.id) ?? false
        : false;

      days.push({
        date,
        devotional,
        isCompleted,
        isToday: date.getTime() === today.getTime(),
        isCurrentMonth: true,
      });
    }

    // Fill remaining days
    const remainingDays = DAYS_IN_WEEK * WEEKS_TO_SHOW - days.length;
    for (let i = 1; i <= remainingDays; i++) {
      const date = new Date(currentMonth.getFullYear(), currentMonth.getMonth() + 1, i);
      days.push({
        date,
        isCompleted: false,
        isToday: false,
        isCurrentMonth: false,
      });
    }

    return days;
  };

  // Navigate months
  const goToPreviousMonth = () => {
    setCurrentMonth(new Date(currentMonth.getFullYear(), currentMonth.getMonth() - 1, 1));
  };

  const goToNextMonth = () => {
    setCurrentMonth(new Date(currentMonth.getFullYear(), currentMonth.getMonth() + 1, 1));
  };

  // Handle day press
  const handleDayPress = (day: CalendarDay) => {
    if (day.devotional && day.isCurrentMonth) {
      router.push({
        pathname: '/devotional/[id]',
        params: { id: day.devotional.id },
      });
    }
  };

  // Refresh handler
  const handleRefresh = () => {
    setIsRefreshing(true);
    fetchDevotionals();
  };

  // Format month name
  const monthName = currentMonth.toLocaleDateString('en-US', {
    month: 'long',
    year: 'numeric',
  });

  // Calculate stats
  const completedCount = user?.progress.completedDevotionals.length ?? 0;
  const currentStreak = user?.progress.streak ?? 0;

  if (!isAuthenticated) {
    return (
      <View style={styles.centerContainer}>
        <Ionicons name="calendar" size={64} color={colors.gray[500]} />
        <Text style={styles.emptyText}>Sign in to view your journey</Text>
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
      {/* Stats Cards */}
      <View style={styles.statsRow}>
        <View style={styles.statCard}>
          <Ionicons name="flame" size={24} color={colors.secondary[500]} />
          <Text style={styles.statNumber}>{currentStreak}</Text>
          <Text style={styles.statLabel}>Day Streak</Text>
        </View>
        <View style={styles.statCard}>
          <Ionicons name="checkmark-circle" size={24} color={colors.success} />
          <Text style={styles.statNumber}>{completedCount}</Text>
          <Text style={styles.statLabel}>Completed</Text>
        </View>
      </View>

      {/* Calendar Header */}
      <View style={styles.calendarHeader}>
        <TouchableOpacity onPress={goToPreviousMonth} style={styles.navButton}>
          <Ionicons name="chevron-back" size={24} color={colors.text.dark.primary} />
        </TouchableOpacity>
        <Text style={styles.monthTitle}>{monthName}</Text>
        <TouchableOpacity onPress={goToNextMonth} style={styles.navButton}>
          <Ionicons name="chevron-forward" size={24} color={colors.text.dark.primary} />
        </TouchableOpacity>
      </View>

      {/* Day Headers */}
      <View style={styles.weekDayRow}>
        {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map((day) => (
          <Text key={day} style={styles.weekDayLabel}>
            {day}
          </Text>
        ))}
      </View>

      {/* Calendar Grid */}
      {isLoading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary[500]} />
        </View>
      ) : (
        <View style={styles.calendarGrid}>
          {generateCalendarDays().map((day, index) => (
            <TouchableOpacity
              key={index}
              style={[
                styles.dayCell,
                day.isToday && styles.todayCell,
                !day.isCurrentMonth && styles.otherMonthCell,
              ]}
              onPress={() => handleDayPress(day)}
              disabled={!day.devotional || !day.isCurrentMonth}
            >
              <Text
                style={[
                  styles.dayNumber,
                  day.isToday && styles.todayNumber,
                  !day.isCurrentMonth && styles.otherMonthNumber,
                ]}
              >
                {day.date.getDate()}
              </Text>
              {day.isCompleted && (
                <View style={styles.completedDot}>
                  <Ionicons name="checkmark" size={12} color="white" />
                </View>
              )}
              {day.devotional && !day.isCompleted && day.isCurrentMonth && (
                <View style={styles.availableDot} />
              )}
            </TouchableOpacity>
          ))}
        </View>
      )}

      {/* Legend */}
      <View style={styles.legend}>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: colors.success }]} />
          <Text style={styles.legendText}>Completed</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: colors.primary[500] }]} />
          <Text style={styles.legendText}>Available</Text>
        </View>
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
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: colors.background.dark,
    padding: spacing.xl,
  },
  emptyText: {
    marginTop: spacing.md,
    color: colors.text.dark.secondary,
    fontSize: 16,
  },
  loadingContainer: {
    height: 300,
    justifyContent: 'center',
    alignItems: 'center',
  },
  statsRow: {
    flexDirection: 'row',
    gap: spacing.md,
    marginBottom: spacing.xl,
  },
  statCard: {
    flex: 1,
    backgroundColor: colors.gray[800],
    padding: spacing.lg,
    borderRadius: borderRadius.lg,
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 32,
    fontWeight: 'bold',
    color: colors.text.dark.primary,
    marginTop: spacing.sm,
  },
  statLabel: {
    fontSize: 14,
    color: colors.text.dark.muted,
    marginTop: spacing.xs,
  },
  calendarHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.lg,
  },
  navButton: {
    padding: spacing.sm,
  },
  monthTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: colors.text.dark.primary,
  },
  weekDayRow: {
    flexDirection: 'row',
    marginBottom: spacing.sm,
  },
  weekDayLabel: {
    flex: 1,
    textAlign: 'center',
    fontSize: 12,
    fontWeight: '600',
    color: colors.text.dark.muted,
    textTransform: 'uppercase',
  },
  calendarGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  dayCell: {
    width: `${100 / 7}%`,
    aspectRatio: 1,
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: borderRadius.md,
  },
  todayCell: {
    backgroundColor: colors.primary[900],
    borderWidth: 2,
    borderColor: colors.primary[500],
  },
  otherMonthCell: {
    opacity: 0.3,
  },
  dayNumber: {
    fontSize: 16,
    color: colors.text.dark.primary,
  },
  todayNumber: {
    fontWeight: 'bold',
    color: colors.primary[400],
  },
  otherMonthNumber: {
    color: colors.text.dark.muted,
  },
  completedDot: {
    position: 'absolute',
    bottom: 4,
    width: 16,
    height: 16,
    borderRadius: 8,
    backgroundColor: colors.success,
    justifyContent: 'center',
    alignItems: 'center',
  },
  availableDot: {
    position: 'absolute',
    bottom: 6,
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.primary[500],
  },
  legend: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: spacing.xl,
    marginTop: spacing.xl,
    paddingTop: spacing.lg,
    borderTopWidth: 1,
    borderTopColor: colors.gray[800],
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  legendDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  legendText: {
    fontSize: 14,
    color: colors.text.dark.secondary,
  },
});
