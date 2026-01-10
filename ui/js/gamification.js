/**
 * Gamification Module - SonZo AI
 * Handles streaks, achievements, XP, and progress tracking
 */

const Gamification = {
    // Achievement definitions
    ACHIEVEMENTS: {
        first_sign: {
            id: 'first_sign',
            name: 'First Sign',
            description: 'Learn your first sign',
            icon: '🎯',
            xp: 10
        },
        alphabet_master: {
            id: 'alphabet_master',
            name: 'Alphabet Master',
            description: 'Learn all 26 letter signs',
            icon: '🔤',
            xp: 100
        },
        number_ninja: {
            id: 'number_ninja',
            name: 'Number Ninja',
            description: 'Learn all number signs (0-9)',
            icon: '🔢',
            xp: 50
        },
        streak_5: {
            id: 'streak_5',
            name: '5 Day Streak',
            description: 'Practice for 5 days in a row',
            icon: '🔥',
            xp: 50
        },
        streak_10: {
            id: 'streak_10',
            name: '10 Day Streak',
            description: 'Practice for 10 days in a row',
            icon: '🔥',
            xp: 100
        },
        streak_30: {
            id: 'streak_30',
            name: '30 Day Streak',
            description: 'Practice for 30 days in a row',
            icon: '⭐',
            xp: 300
        },
        signs_10: {
            id: 'signs_10',
            name: 'Getting Started',
            description: 'Learn 10 signs',
            icon: '🌱',
            xp: 30
        },
        signs_50: {
            id: 'signs_50',
            name: 'Half Century',
            description: 'Learn 50 signs',
            icon: '💯',
            xp: 150
        },
        signs_100: {
            id: 'signs_100',
            name: 'Century Club',
            description: 'Learn 100 signs',
            icon: '🏆',
            xp: 300
        },
        conversation_first: {
            id: 'conversation_first',
            name: 'First Conversation',
            description: 'Complete your first conversation',
            icon: '💬',
            xp: 25
        },
        conversation_pro: {
            id: 'conversation_pro',
            name: 'Conversation Pro',
            description: 'Complete 50 conversations',
            icon: '🗣️',
            xp: 200
        },
        perfect_day: {
            id: 'perfect_day',
            name: 'Perfect Day',
            description: '100% accuracy in a practice session',
            icon: '⭐',
            xp: 50
        },
        speed_demon: {
            id: 'speed_demon',
            name: 'Speed Demon',
            description: 'Recognize 10 signs in under 30 seconds',
            icon: '⚡',
            xp: 75
        },
        early_bird: {
            id: 'early_bird',
            name: 'Early Bird',
            description: 'Practice before 8 AM',
            icon: '🌅',
            xp: 20
        },
        night_owl: {
            id: 'night_owl',
            name: 'Night Owl',
            description: 'Practice after 10 PM',
            icon: '🦉',
            xp: 20
        },
        social_butterfly: {
            id: 'social_butterfly',
            name: 'Social Butterfly',
            description: 'Share a signed video',
            icon: '🦋',
            xp: 25
        }
    },

    // Level thresholds
    LEVELS: [
        { level: 1, xpRequired: 0, title: 'Newcomer' },
        { level: 2, xpRequired: 100, title: 'Beginner' },
        { level: 3, xpRequired: 250, title: 'Learner' },
        { level: 4, xpRequired: 500, title: 'Student' },
        { level: 5, xpRequired: 1000, title: 'Practitioner' },
        { level: 6, xpRequired: 2000, title: 'Intermediate' },
        { level: 7, xpRequired: 3500, title: 'Advanced' },
        { level: 8, xpRequired: 5000, title: 'Expert' },
        { level: 9, xpRequired: 7500, title: 'Master' },
        { level: 10, xpRequired: 10000, title: 'Grandmaster' }
    ],

    /**
     * Initialize gamification
     */
    init() {
        // Check for daily streak on app load
        this.checkDailyStreak();
    },

    /**
     * Check and update daily streak
     */
    checkDailyStreak() {
        const progress = Storage.getProgress();
        const lastPractice = progress.lastPractice ? new Date(progress.lastPractice) : null;
        const today = new Date();

        if (!lastPractice) return;

        // Check if streak should be reset
        const daysSinceLastPractice = Math.floor(
            (today - lastPractice) / (1000 * 60 * 60 * 24)
        );

        if (daysSinceLastPractice > 1) {
            // Streak broken
            progress.streak = 0;
            Storage.setProgress(progress);
            app?.showToast('Your streak was reset. Start a new one today!', 'info');
        }
    },

    /**
     * Record practice session
     */
    recordPractice() {
        const streak = Storage.updateStreak();
        this.checkStreakAchievements(streak);

        // Check time-based achievements
        const hour = new Date().getHours();
        if (hour < 8) this.unlockAchievement('early_bird');
        if (hour >= 22) this.unlockAchievement('night_owl');

        return streak;
    },

    /**
     * Record sign learned
     */
    recordSignLearned(sign) {
        Storage.addLearnedSign(sign);

        const progress = Storage.getProgress();
        const signCount = progress.signsLearned.length;

        // Check achievements
        if (signCount === 1) this.unlockAchievement('first_sign');
        if (signCount >= 10) this.unlockAchievement('signs_10');
        if (signCount >= 50) this.unlockAchievement('signs_50');
        if (signCount >= 100) this.unlockAchievement('signs_100');

        // Check alphabet completion
        const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
        if (alphabet.every(letter => progress.signsLearned.includes(letter))) {
            this.unlockAchievement('alphabet_master');
        }

        // Check number completion
        const numbers = '0123456789'.split('');
        if (numbers.every(num => progress.signsLearned.includes(num))) {
            this.unlockAchievement('number_ninja');
        }

        // Award XP
        this.addXP(10);

        return signCount;
    },

    /**
     * Record sign recognition accuracy
     */
    recordAccuracy(sign, correct) {
        Storage.updateAccuracy(sign, correct);

        if (correct) {
            this.addXP(5);
        }
    },

    /**
     * Check streak achievements
     */
    checkStreakAchievements(streak) {
        if (streak >= 5) this.unlockAchievement('streak_5');
        if (streak >= 10) this.unlockAchievement('streak_10');
        if (streak >= 30) this.unlockAchievement('streak_30');
    },

    /**
     * Unlock achievement
     */
    unlockAchievement(achievementId) {
        const achievement = this.ACHIEVEMENTS[achievementId];
        if (!achievement) return false;

        const unlocked = Storage.unlockAchievement(achievementId);

        if (unlocked) {
            // Award XP
            this.addXP(achievement.xp);

            // Show notification
            app?.showToast(
                `Achievement Unlocked: ${achievement.icon} ${achievement.name}`,
                'success'
            );

            // Haptic feedback
            Utils.vibrate([100, 50, 100]);

            return true;
        }

        return false;
    },

    /**
     * Add XP
     */
    addXP(amount) {
        const oldProgress = Storage.getProgress();
        const oldLevel = this.getLevel(oldProgress.xp);

        const newProgress = Storage.addXP(amount);
        const newLevel = this.getLevel(newProgress.xp);

        // Level up!
        if (newLevel.level > oldLevel.level) {
            app?.showToast(
                `Level Up! You're now ${newLevel.title} (Level ${newLevel.level})`,
                'success'
            );
            Utils.vibrate([100, 50, 100, 50, 100]);
        }

        return newProgress.xp;
    },

    /**
     * Get level info for XP amount
     */
    getLevel(xp) {
        for (let i = this.LEVELS.length - 1; i >= 0; i--) {
            if (xp >= this.LEVELS[i].xpRequired) {
                return this.LEVELS[i];
            }
        }
        return this.LEVELS[0];
    },

    /**
     * Get XP progress to next level
     */
    getLevelProgress(xp) {
        const currentLevel = this.getLevel(xp);
        const currentLevelIndex = this.LEVELS.findIndex(l => l.level === currentLevel.level);
        const nextLevel = this.LEVELS[currentLevelIndex + 1];

        if (!nextLevel) {
            return { current: xp, required: xp, progress: 100 };
        }

        const xpInLevel = xp - currentLevel.xpRequired;
        const xpForLevel = nextLevel.xpRequired - currentLevel.xpRequired;

        return {
            current: xpInLevel,
            required: xpForLevel,
            progress: Math.floor((xpInLevel / xpForLevel) * 100)
        };
    },

    /**
     * Get all achievements with unlock status
     */
    getAllAchievements() {
        const progress = Storage.getProgress();
        const unlockedIds = progress.achievements || [];

        return Object.values(this.ACHIEVEMENTS).map(achievement => ({
            ...achievement,
            unlocked: unlockedIds.includes(achievement.id)
        }));
    },

    /**
     * Get progress summary
     */
    getProgressSummary() {
        const progress = Storage.getProgress();
        const level = this.getLevel(progress.xp);
        const levelProgress = this.getLevelProgress(progress.xp);

        return {
            signsLearned: progress.signsLearned.length,
            streak: progress.streak,
            xp: progress.xp,
            level: level.level,
            levelTitle: level.title,
            levelProgress: levelProgress,
            practiceTime: progress.practiceTime,
            achievementsUnlocked: progress.achievements.length,
            totalAchievements: Object.keys(this.ACHIEVEMENTS).length
        };
    },

    /**
     * Get weekly activity data
     */
    getWeeklyActivity() {
        // This would ideally come from more detailed tracking
        // For now, return placeholder data
        const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
        return days.map(day => ({
            day,
            minutes: Math.floor(Math.random() * 60), // Placeholder
            signs: Math.floor(Math.random() * 20)
        }));
    },

    /**
     * Get confused signs (signs with low accuracy)
     */
    getConfusedSigns(limit = 5) {
        const progress = Storage.getProgress();
        const accuracy = progress.accuracy || {};

        const confusedSigns = Object.entries(accuracy)
            .map(([sign, data]) => ({
                sign,
                accuracy: data.total > 0 ? data.correct / data.total : 0,
                attempts: data.total
            }))
            .filter(item => item.attempts >= 3 && item.accuracy < 0.7)
            .sort((a, b) => a.accuracy - b.accuracy)
            .slice(0, limit);

        return confusedSigns;
    },

    /**
     * Generate streak calendar data
     */
    getStreakCalendar() {
        // This would ideally track daily practice
        // For now, return recent streak data
        const progress = Storage.getProgress();
        const streak = progress.streak;
        const calendar = [];

        for (let i = 6; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            calendar.push({
                date: date.toDateString(),
                practiced: i < streak
            });
        }

        return calendar;
    },

    /**
     * Get daily challenge
     */
    getDailyChallenge() {
        // Generate a daily challenge based on progress
        const progress = Storage.getProgress();
        const signsLearned = progress.signsLearned.length;

        const challenges = [
            {
                type: 'learn_new',
                title: 'Learn 3 new question signs',
                target: 3,
                reward: 50
            },
            {
                type: 'practice',
                title: 'Practice 5 signs you know',
                target: 5,
                reward: 30
            },
            {
                type: 'conversation',
                title: 'Have a conversation with your avatar',
                target: 1,
                reward: 40
            },
            {
                type: 'speed',
                title: 'Recognize 10 signs in under 1 minute',
                target: 10,
                reward: 60
            }
        ];

        // Return a "random" challenge based on day
        const dayOfYear = Math.floor(
            (new Date() - new Date(new Date().getFullYear(), 0, 0)) / (1000 * 60 * 60 * 24)
        );

        return challenges[dayOfYear % challenges.length];
    },

    /**
     * Record conversation completion
     */
    recordConversation() {
        // Track conversations for achievements
        const progress = Storage.getProgress();
        const conversations = (progress.conversationsCompleted || 0) + 1;

        progress.conversationsCompleted = conversations;
        Storage.setProgress(progress);

        if (conversations === 1) this.unlockAchievement('conversation_first');
        if (conversations >= 50) this.unlockAchievement('conversation_pro');

        this.addXP(15);
    },

    /**
     * Record video share
     */
    recordShare() {
        this.unlockAchievement('social_butterfly');
        this.addXP(10);
    }
};

// Export for use in modules
if (typeof module !== 'undefined') {
    module.exports = Gamification;
}
