/**
 * Storage Manager - SonZo AI
 * Handles persistent storage for user data, preferences, and progress
 */

const Storage = {
    PREFIX: 'sonzo_',

    /**
     * Get item from storage
     */
    get(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(this.PREFIX + key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (e) {
            console.error('Storage get error:', e);
            return defaultValue;
        }
    },

    /**
     * Set item in storage
     */
    set(key, value) {
        try {
            localStorage.setItem(this.PREFIX + key, JSON.stringify(value));
            return true;
        } catch (e) {
            console.error('Storage set error:', e);
            return false;
        }
    },

    /**
     * Remove item from storage
     */
    remove(key) {
        try {
            localStorage.removeItem(this.PREFIX + key);
            return true;
        } catch (e) {
            console.error('Storage remove error:', e);
            return false;
        }
    },

    /**
     * Clear all app storage
     */
    clear() {
        try {
            const keys = Object.keys(localStorage).filter(k => k.startsWith(this.PREFIX));
            keys.forEach(k => localStorage.removeItem(k));
            return true;
        } catch (e) {
            console.error('Storage clear error:', e);
            return false;
        }
    },

    // =========================================================================
    // User Profile
    // =========================================================================

    getUserProfile() {
        return this.get('user_profile', {
            id: Utils.generateId(),
            name: 'User',
            avatarId: null,
            avatarUrl: null,
            createdAt: new Date().toISOString(),
            goal: null,
            level: 'beginner',
            signingSpeed: 1,
            leftHanded: false
        });
    },

    setUserProfile(profile) {
        return this.set('user_profile', profile);
    },

    updateUserProfile(updates) {
        const profile = this.getUserProfile();
        return this.setUserProfile({ ...profile, ...updates });
    },

    // =========================================================================
    // Settings
    // =========================================================================

    getSettings() {
        return this.get('settings', {
            darkMode: window.matchMedia('(prefers-color-scheme: dark)').matches,
            highContrast: false,
            reduceMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
            largeText: false,
            haptic: true,
            showSkeleton: true,
            practiceReminders: true,
            achievementAlerts: true
        });
    },

    setSettings(settings) {
        return this.set('settings', settings);
    },

    updateSetting(key, value) {
        const settings = this.getSettings();
        settings[key] = value;
        return this.setSettings(settings);
    },

    // =========================================================================
    // Progress & Gamification
    // =========================================================================

    getProgress() {
        return this.get('progress', {
            signsLearned: [],
            accuracy: {},
            practiceTime: 0,
            streak: 0,
            lastPractice: null,
            xp: 0,
            level: 1,
            achievements: []
        });
    },

    setProgress(progress) {
        return this.set('progress', progress);
    },

    addLearnedSign(sign) {
        const progress = this.getProgress();
        if (!progress.signsLearned.includes(sign)) {
            progress.signsLearned.push(sign);
            this.setProgress(progress);
        }
    },

    updateAccuracy(sign, correct) {
        const progress = this.getProgress();
        if (!progress.accuracy[sign]) {
            progress.accuracy[sign] = { correct: 0, total: 0 };
        }
        progress.accuracy[sign].total++;
        if (correct) progress.accuracy[sign].correct++;
        this.setProgress(progress);
    },

    addPracticeTime(seconds) {
        const progress = this.getProgress();
        progress.practiceTime += seconds;
        this.setProgress(progress);
    },

    updateStreak() {
        const progress = this.getProgress();
        const today = new Date().toDateString();
        const lastPractice = progress.lastPractice ? new Date(progress.lastPractice).toDateString() : null;

        if (lastPractice === today) {
            // Already practiced today
            return progress.streak;
        }

        const yesterday = new Date();
        yesterday.setDate(yesterday.getDate() - 1);

        if (lastPractice === yesterday.toDateString()) {
            // Continue streak
            progress.streak++;
        } else if (lastPractice !== today) {
            // Reset streak
            progress.streak = 1;
        }

        progress.lastPractice = new Date().toISOString();
        this.setProgress(progress);
        return progress.streak;
    },

    addXP(amount) {
        const progress = this.getProgress();
        progress.xp += amount;

        // Level up every 500 XP
        const newLevel = Math.floor(progress.xp / 500) + 1;
        if (newLevel > progress.level) {
            progress.level = newLevel;
            // Trigger level up notification
        }

        this.setProgress(progress);
        return progress;
    },

    unlockAchievement(achievementId) {
        const progress = this.getProgress();
        if (!progress.achievements.includes(achievementId)) {
            progress.achievements.push(achievementId);
            this.setProgress(progress);
            return true;
        }
        return false;
    },

    // =========================================================================
    // Favorites
    // =========================================================================

    getFavorites() {
        return this.get('favorites', ['I_LOVE_YOU', 'HELLO', 'THANK_YOU']);
    },

    addFavorite(phrase) {
        const favorites = this.getFavorites();
        if (!favorites.includes(phrase)) {
            favorites.push(phrase);
            this.set('favorites', favorites);
        }
    },

    removeFavorite(phrase) {
        const favorites = this.getFavorites();
        const index = favorites.indexOf(phrase);
        if (index > -1) {
            favorites.splice(index, 1);
            this.set('favorites', favorites);
        }
    },

    // =========================================================================
    // Conversations
    // =========================================================================

    getConversations() {
        return this.get('conversations', []);
    },

    saveConversation(conversation) {
        const conversations = this.getConversations();
        const existing = conversations.findIndex(c => c.id === conversation.id);

        if (existing > -1) {
            conversations[existing] = conversation;
        } else {
            conversations.unshift(conversation);
        }

        // Keep only last 50 conversations
        if (conversations.length > 50) {
            conversations.splice(50);
        }

        this.set('conversations', conversations);
    },

    deleteConversation(id) {
        const conversations = this.getConversations();
        const index = conversations.findIndex(c => c.id === id);
        if (index > -1) {
            conversations.splice(index, 1);
            this.set('conversations', conversations);
        }
    },

    clearConversations() {
        this.set('conversations', []);
    },

    // =========================================================================
    // Custom Signs
    // =========================================================================

    getCustomSigns() {
        return this.get('custom_signs', []);
    },

    addCustomSign(sign) {
        const signs = this.getCustomSigns();
        signs.push({
            id: Utils.generateId(),
            ...sign,
            createdAt: new Date().toISOString()
        });
        this.set('custom_signs', signs);
    },

    deleteCustomSign(id) {
        const signs = this.getCustomSigns();
        const index = signs.findIndex(s => s.id === id);
        if (index > -1) {
            signs.splice(index, 1);
            this.set('custom_signs', signs);
        }
    },

    // =========================================================================
    // Onboarding
    // =========================================================================

    hasCompletedOnboarding() {
        return this.get('onboarding_complete', false);
    },

    setOnboardingComplete() {
        this.set('onboarding_complete', true);
    },

    resetOnboarding() {
        this.set('onboarding_complete', false);
    },

    // =========================================================================
    // Export / Import
    // =========================================================================

    exportData() {
        const data = {
            profile: this.getUserProfile(),
            settings: this.getSettings(),
            progress: this.getProgress(),
            favorites: this.getFavorites(),
            conversations: this.getConversations(),
            customSigns: this.getCustomSigns(),
            exportedAt: new Date().toISOString()
        };
        return data;
    },

    importData(data) {
        try {
            if (data.profile) this.setUserProfile(data.profile);
            if (data.settings) this.setSettings(data.settings);
            if (data.progress) this.setProgress(data.progress);
            if (data.favorites) this.set('favorites', data.favorites);
            if (data.conversations) this.set('conversations', data.conversations);
            if (data.customSigns) this.set('custom_signs', data.customSigns);
            return true;
        } catch (e) {
            console.error('Import error:', e);
            return false;
        }
    }
};

// Export for use in modules
if (typeof module !== 'undefined') {
    module.exports = Storage;
}
