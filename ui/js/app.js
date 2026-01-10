/**
 * Main Application - SonZo AI
 * Sign Language Communication Assistant
 */

const app = {
    // State
    currentScreen: 'loading-screen',
    onboardingStep: 0,
    selectedGoal: null,
    selectedLevel: null,
    isMenuOpen: false,

    // =========================================================================
    // Initialization
    // =========================================================================

    async init() {
        console.log('SonZo AI - Initializing...');

        // Apply saved settings
        this.applySettings();

        // Initialize modules
        Avatar.init();
        Gamification.init();

        // Check onboarding status
        if (Storage.hasCompletedOnboarding()) {
            await this.initMainApp();
        } else {
            this.showScreen('welcome-screen');
        }

        // Setup event listeners
        this.setupEventListeners();

        console.log('SonZo AI - Ready!');
    },

    async initMainApp() {
        // Load user data
        const profile = Storage.getUserProfile();
        this.updateUIWithProfile(profile);

        // Update progress display
        this.updateProgressUI();

        // Update avatar UI
        Avatar.updateAvatarUI();

        // Show main app
        this.showScreen('main-app');
    },

    // =========================================================================
    // Screen Management
    // =========================================================================

    showScreen(screenId) {
        // Hide all screens
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });

        // Show target screen
        const screen = Utils.$(screenId);
        if (screen) {
            screen.classList.add('active');
            this.currentScreen = screenId;
        }
    },

    // =========================================================================
    // Onboarding
    // =========================================================================

    startOnboarding() {
        this.onboardingStep = 1;
        this.showScreen('onboard-avatar');
    },

    skipOnboarding() {
        // Go to main app without full onboarding
        Storage.setOnboardingComplete();
        this.initMainApp();
    },

    onboardingBack() {
        if (this.onboardingStep === 1) {
            this.showScreen('welcome-screen');
        } else if (this.onboardingStep === 2) {
            this.showScreen('onboard-avatar');
        } else if (this.onboardingStep === 3) {
            this.showScreen('onboard-calibration');
        }
        this.onboardingStep = Math.max(0, this.onboardingStep - 1);
    },

    onboardingNext() {
        if (this.onboardingStep === 1) {
            this.onboardingStep = 2;
            this.showScreen('onboard-calibration');
            this.startCalibration();
        } else if (this.onboardingStep === 2) {
            this.onboardingStep = 3;
            this.showScreen('onboard-preferences');
        }
    },

    skipCalibration() {
        Camera.stopStream();
        this.onboardingStep = 3;
        this.showScreen('onboard-preferences');
    },

    async completeOnboarding() {
        // Save preferences
        const profile = Storage.getUserProfile();
        profile.goal = this.selectedGoal;
        profile.level = this.selectedLevel || 'beginner';
        profile.signingSpeed = parseFloat(Utils.$('speed-slider')?.value || 1);
        profile.leftHanded = Utils.$('left-handed-toggle')?.checked || false;
        Storage.setUserProfile(profile);

        // Update settings
        Storage.updateSetting('leftHanded', profile.leftHanded);

        // Mark onboarding complete
        Storage.setOnboardingComplete();

        // Show main app
        await this.initMainApp();
    },

    // =========================================================================
    // Avatar Creation
    // =========================================================================

    openCamera() {
        const modal = Utils.$('camera-modal');
        modal?.classList.add('open');
        Camera.init('camera-preview');
    },

    closeCameraModal() {
        const modal = Utils.$('camera-modal');
        modal?.classList.remove('open');
        Camera.stopStream();
    },

    async switchCamera() {
        await Camera.switchCamera();
    },

    async capturePhoto() {
        const photoData = Camera.capturePhoto();
        if (photoData) {
            await this.processAvatarPhoto(photoData);
        }
        this.closeCameraModal();
    },

    uploadPhoto() {
        Utils.$('photo-input')?.click();
    },

    async handlePhotoSelect(event) {
        const file = event.target.files?.[0];
        if (!file) return;

        try {
            const photoData = await Utils.loadImageAsBase64(file);
            await this.processAvatarPhoto(photoData);
        } catch (error) {
            this.showToast('Failed to load photo', 'error');
        }
    },

    async processAvatarPhoto(photoData) {
        try {
            // Show preview
            const preview = Utils.$('avatar-image');
            const placeholder = Utils.$('avatar-preview')?.querySelector('.placeholder-avatar');

            if (preview) {
                preview.src = photoData;
                preview.classList.remove('hidden');
            }
            if (placeholder) {
                placeholder.classList.add('hidden');
            }

            // Show customization options
            Utils.$('avatar-customization')?.classList.remove('hidden');

            // Enable next button
            const nextBtn = Utils.$('avatar-next-btn');
            if (nextBtn) nextBtn.disabled = false;

            // Create avatar via API
            this.showToast('Creating your avatar...', 'info');
            const result = await Avatar.createAvatar(photoData);

            if (result.quality_score < 0.7) {
                this.showToast('Photo quality is low. You may want to try another photo.', 'warning');
            } else {
                this.showToast('Avatar created successfully!', 'success');
            }

        } catch (error) {
            this.showToast(error.message || 'Failed to create avatar', 'error');
        }
    },

    // =========================================================================
    // Calibration
    // =========================================================================

    async startCalibration() {
        await Camera.init('calibration-camera');
        HandTracking.init('hand-overlay');
        HandTracking.start();

        Recognition.init({
            onRecognition: (result) => this.handleCalibrationRecognition(result),
            onHandsDetected: (hands) => HandTracking.updateLandmarks(hands)
        });
        Recognition.start();
    },

    handleCalibrationRecognition(result) {
        const calibrationSigns = ['HELLO', 'THANK_YOU', 'YES'];
        const signIndex = calibrationSigns.indexOf(result.sign);

        if (signIndex !== -1) {
            const signElement = Utils.$(`cal-sign-${signIndex}`);
            if (signElement && !signElement.classList.contains('completed')) {
                signElement.classList.add('completed');
                signElement.querySelector('.check-icon')?.classList.remove('hidden');
                Utils.vibrate(50);

                // Check if all signs completed
                const completedCount = document.querySelectorAll('.calibration-sign.completed').length;
                if (completedCount === 3) {
                    this.finishCalibration();
                }
            }
        }
    },

    finishCalibration() {
        Camera.stopStream();
        Recognition.stop();
        HandTracking.stop();

        this.showToast('Calibration complete!', 'success');

        setTimeout(() => {
            this.onboardingNext();
        }, 1000);
    },

    // =========================================================================
    // Preferences
    // =========================================================================

    selectGoal(goal) {
        this.selectedGoal = goal;
        document.querySelectorAll('.goal-option').forEach(el => {
            el.classList.toggle('selected', el.dataset.goal === goal);
        });
    },

    selectLevel(level) {
        this.selectedLevel = level;
        document.querySelectorAll('.level-option').forEach(el => {
            el.classList.toggle('selected', el.dataset.level === level);
        });
    },

    // =========================================================================
    // Main App Functions
    // =========================================================================

    async quickSign() {
        this.showScreen('signing-interface');
        await Camera.init('user-camera');
        HandTracking.init('hand-tracking-overlay');
        HandTracking.start();

        Recognition.init({
            onRecognition: (result) => this.handleRecognition(result),
            onProcessing: () => this.setRecognitionState('processing'),
            onHandsDetected: (hands) => HandTracking.updateLandmarks(hands)
        });
        Recognition.start();

        Gamification.recordPractice();
    },

    closeSigning() {
        Camera.stopStream();
        Recognition.stop();
        HandTracking.stop();
        this.showScreen('main-app');
    },

    handleRecognition(result) {
        if (result.isUnknown) {
            this.setRecognitionState('error');
            this.updateRecognizedText('Unknown sign - try again');
            return;
        }

        this.setRecognitionState('recognized');
        this.updateRecognizedText(Utils.formatSignName(result.sign));

        // Record for gamification
        Gamification.recordSignLearned(result.sign);

        Utils.vibrate(50);

        // Animate success
        HandTracking.animateSuccess(result.hands?.[0]?.landmarks);

        // Reset after delay
        setTimeout(() => {
            this.setRecognitionState('ready');
        }, 2000);
    },

    setRecognitionState(state) {
        const indicator = Utils.$('signing-status');
        if (indicator) {
            indicator.className = 'status-indicator ' + state;
            indicator.textContent = state === 'processing' ? 'Processing...' :
                                   state === 'recognized' ? 'Recognized!' :
                                   state === 'error' ? 'Try again' : 'Ready';
        }

        // Update ring
        const ring = Utils.$('ring-progress');
        if (ring) {
            ring.className = 'ring-progress ' + state;
        }
    },

    updateRecognizedText(text) {
        const bubble = Utils.$('recognized-text');
        if (bubble) {
            bubble.querySelector('.bubble-content').textContent = text;
            Utils.animate(bubble, 'pop');
        }
    },

    // =========================================================================
    // Avatar Signing
    // =========================================================================

    async signPhrase(phrase) {
        try {
            if (!Avatar.hasAvatar()) {
                this.showToast('Please create an avatar first', 'warning');
                return;
            }

            const avatarText = Utils.$('avatar-text');
            if (avatarText) {
                avatarText.querySelector('.bubble-content').textContent = Utils.formatSignName(phrase);
            }

            await Avatar.signPhrase(phrase);
        } catch (error) {
            this.showToast(error.message, 'error');
        }
    },

    handleTextInput(event) {
        if (event.key === 'Enter') {
            this.sendMessage();
        }
    },

    async sendMessage() {
        const input = Utils.$('text-input');
        const message = input?.value.trim();

        if (!message) return;

        input.value = '';

        // Convert to sign phrase (simplified)
        const phrase = message.toUpperCase().replace(/\s+/g, '_');
        await this.signPhrase(phrase);
    },

    quickSend(phrase) {
        this.signPhrase(phrase);
    },

    // =========================================================================
    // Navigation
    // =========================================================================

    switchTab(tab) {
        // Update nav
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        event.currentTarget.classList.add('active');

        // Handle tab switch
        switch (tab) {
            case 'home':
                this.showMainContent();
                break;
            case 'learn':
                this.showDictionary();
                break;
            case 'chat':
                this.showAllConversations();
                break;
            case 'profile':
                this.showSettings();
                break;
        }
    },

    showMainContent() {
        // Reset to home view
        this.showScreen('main-app');
    },

    // =========================================================================
    // Menu
    // =========================================================================

    toggleMenu() {
        this.isMenuOpen = !this.isMenuOpen;
        const menu = Utils.$('side-menu');
        const overlay = Utils.$('menu-overlay');

        menu?.classList.toggle('open', this.isMenuOpen);
        overlay?.classList.toggle('visible', this.isMenuOpen);
    },

    // =========================================================================
    // Conversations
    // =========================================================================

    startConversation() {
        const conversation = {
            id: Utils.generateId(),
            messages: [],
            createdAt: new Date().toISOString()
        };
        Storage.saveConversation(conversation);
        this.openConversation(conversation.id);
    },

    openConversation(id) {
        this.currentConversationId = id;
        this.showScreen('conversation-screen');
        this.loadConversationMessages(id);
    },

    closeConversation() {
        this.showScreen('main-app');
        Gamification.recordConversation();
    },

    loadConversationMessages(id) {
        const conversations = Storage.getConversations();
        const conversation = conversations.find(c => c.id === id);
        // Render messages...
    },

    showAllConversations() {
        // Show conversations list
    },

    // =========================================================================
    // Dashboard
    // =========================================================================

    showDashboard() {
        this.toggleMenu();
        this.showScreen('dashboard-screen');
        this.updateDashboardUI();
    },

    closeDashboard() {
        this.showScreen('main-app');
    },

    updateDashboardUI() {
        const summary = Gamification.getProgressSummary();

        Utils.$('total-signs').textContent = summary.signsLearned;
        // Update other stats...
    },

    // =========================================================================
    // Settings
    // =========================================================================

    showSettings() {
        this.toggleMenu();
        this.showScreen('settings-screen');
    },

    closeSettings() {
        this.showScreen('main-app');
    },

    toggleDarkMode() {
        const enabled = Utils.$('dark-mode-toggle')?.checked;
        Storage.updateSetting('darkMode', enabled);
        document.documentElement.setAttribute('data-theme', enabled ? 'dark' : 'light');
    },

    toggleHighContrast() {
        const enabled = Utils.$('high-contrast-toggle')?.checked;
        Storage.updateSetting('highContrast', enabled);
        document.documentElement.setAttribute('data-contrast', enabled ? 'high' : 'normal');
    },

    toggleReduceMotion() {
        const enabled = Utils.$('reduce-motion-toggle')?.checked;
        Storage.updateSetting('reduceMotion', enabled);
        document.documentElement.setAttribute('data-reduce-motion', enabled);
    },

    toggleLargeText() {
        const enabled = Utils.$('large-text-toggle')?.checked;
        Storage.updateSetting('largeText', enabled);
        document.documentElement.setAttribute('data-text-size', enabled ? 'large' : 'normal');
    },

    toggleHaptic() {
        const enabled = Utils.$('haptic-toggle')?.checked;
        Storage.updateSetting('haptic', enabled);
    },

    toggleLeftHanded() {
        const enabled = Utils.$('left-handed-setting')?.checked;
        Storage.updateSetting('leftHanded', enabled);
        Storage.updateUserProfile({ leftHanded: enabled });
        document.documentElement.setAttribute('data-hand', enabled ? 'left' : 'right');
    },

    applySettings() {
        const settings = Storage.getSettings();

        document.documentElement.setAttribute('data-theme', settings.darkMode ? 'dark' : 'light');
        document.documentElement.setAttribute('data-contrast', settings.highContrast ? 'high' : 'normal');
        document.documentElement.setAttribute('data-reduce-motion', settings.reduceMotion);
        document.documentElement.setAttribute('data-text-size', settings.largeText ? 'large' : 'normal');

        const profile = Storage.getUserProfile();
        document.documentElement.setAttribute('data-hand', profile.leftHanded ? 'left' : 'right');
    },

    // =========================================================================
    // Other Screens
    // =========================================================================

    showStreak() {
        const modal = Utils.$('streak-modal');
        modal?.classList.add('open');

        const progress = Storage.getProgress();
        Utils.$('streak-number').textContent = progress.streak;
    },

    closeStreakModal() {
        Utils.$('streak-modal')?.classList.remove('open');
    },

    showProfile() {
        this.showSettings();
    },

    showDictionary() {
        // Show sign dictionary
    },

    showCustomSigns() {
        // Show custom signs
    },

    showAchievements() {
        this.toggleMenu();
        this.showDashboard();
    },

    showCommunity() {
        this.showToast('Community features coming soon!', 'info');
    },

    showHelp() {
        this.showToast('Help & Support', 'info');
    },

    // =========================================================================
    // Utilities
    // =========================================================================

    showToast(message, type = 'info') {
        const container = Utils.$('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <span class="toast-icon">${type === 'success' ? '✓' : type === 'error' ? '✕' : type === 'warning' ? '⚠' : 'ℹ'}</span>
            <span class="toast-message">${message}</span>
            <button class="toast-close btn-icon" onclick="this.parentElement.remove()">✕</button>
        `;

        container.appendChild(toast);

        // Auto remove
        setTimeout(() => toast.remove(), 5000);
    },

    updateUIWithProfile(profile) {
        const menuName = Utils.$('menu-name');
        if (menuName) menuName.textContent = profile.name || 'User';

        const displayName = Utils.$('display-name');
        if (displayName) displayName.value = profile.name || 'User';
    },

    updateProgressUI() {
        const summary = Gamification.getProgressSummary();

        Utils.$('signs-learned').textContent = summary.signsLearned;
        Utils.$('accuracy-score').textContent = '89%'; // Calculate from data
        Utils.$('practice-time').textContent = Utils.formatHours(summary.practiceTime / 3600);
        Utils.$('streak-count').textContent = `🔥 ${summary.streak}`;
    },

    // =========================================================================
    // Event Listeners
    // =========================================================================

    setupEventListeners() {
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                if (this.currentScreen === 'signing-interface') {
                    this.closeSigning();
                } else if (this.isMenuOpen) {
                    this.toggleMenu();
                }
            }
        });

        // Visibility change (pause recognition when hidden)
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                Recognition.stop();
            } else if (this.currentScreen === 'signing-interface') {
                Recognition.start();
            }
        });

        // Prevent zoom on double tap (mobile)
        document.addEventListener('touchstart', (e) => {
            if (e.touches.length > 1) {
                e.preventDefault();
            }
        }, { passive: false });
    },

    // =========================================================================
    // Sharing
    // =========================================================================

    async shareApp() {
        const shared = await Utils.share({
            title: 'SonZo AI',
            text: 'Check out SonZo - Your Sign Language Companion!',
            url: window.location.origin
        });

        if (!shared) {
            Utils.copyToClipboard(window.location.origin);
            this.showToast('Link copied to clipboard!', 'success');
        }
    },

    exportData() {
        const data = Storage.exportData();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        Utils.downloadFile(blob, 'sonzo-backup.json');
        this.showToast('Data exported successfully!', 'success');
    },

    clearHistory() {
        if (confirm('Are you sure you want to clear all conversation history?')) {
            Storage.clearConversations();
            this.showToast('History cleared', 'success');
        }
    },

    deleteAccount() {
        if (confirm('Are you sure you want to delete your account? This cannot be undone.')) {
            Storage.clear();
            window.location.reload();
        }
    }
};

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    // Show loading screen briefly
    setTimeout(() => {
        app.init();
    }, 1500);
});

// Export for use in modules
if (typeof module !== 'undefined') {
    module.exports = app;
}
