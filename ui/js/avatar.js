/**
 * Avatar Module - SonZo AI
 * Handles avatar creation, customization, and video generation
 */

const Avatar = {
    apiUrl: '/api/avatar',
    currentAvatarId: null,
    currentVideoUrl: null,
    videoElement: null,
    placeholderElement: null,

    // Customization options
    skinTone: 'medium',
    background: 'gradient-purple',
    signingSpeed: 1,

    /**
     * Initialize avatar module
     */
    init(options = {}) {
        this.apiUrl = options.apiUrl || this.apiUrl;
        this.videoElement = Utils.$(options.videoElementId || 'avatar-video');
        this.placeholderElement = Utils.$(options.placeholderElementId || 'avatar-placeholder');

        // Load saved avatar
        const profile = Storage.getUserProfile();
        if (profile.avatarId) {
            this.currentAvatarId = profile.avatarId;
        }

        return true;
    },

    /**
     * Create avatar from photo
     */
    async createAvatar(photoData, name = null) {
        try {
            // Remove data URL prefix if present
            let imageData = photoData;
            if (photoData.startsWith('data:image')) {
                imageData = photoData.split(',')[1];
            }

            const response = await fetch(`${this.apiUrl}/create`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    photo: imageData,
                    name: name
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail?.message || error.detail || 'Failed to create avatar');
            }

            const result = await response.json();

            // Save avatar ID
            this.currentAvatarId = result.avatar_id;
            Storage.updateUserProfile({
                avatarId: result.avatar_id,
                avatarUrl: result.preview_url
            });

            return result;

        } catch (error) {
            console.error('Create avatar error:', error);
            throw error;
        }
    },

    /**
     * Generate signing video for phrase
     */
    async signPhrase(phrase) {
        if (!this.currentAvatarId) {
            throw new Error('No avatar created. Please create an avatar first.');
        }

        try {
            const response = await fetch(`${this.apiUrl}/${this.currentAvatarId}/sign`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    phrase: phrase
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail?.message || error.detail || 'Failed to generate sign');
            }

            const result = await response.json();

            // Play video
            if (result.video_url) {
                this.currentVideoUrl = result.video_url;
                await this.playVideo(result.video_url);
            }

            return result;

        } catch (error) {
            console.error('Sign phrase error:', error);
            throw error;
        }
    },

    /**
     * Play avatar video
     */
    async playVideo(videoUrl, options = {}) {
        if (!this.videoElement) {
            console.error('Video element not initialized');
            return;
        }

        // Hide placeholder
        if (this.placeholderElement) {
            this.placeholderElement.classList.add('hidden');
        }

        // Set video source
        this.videoElement.src = videoUrl;
        this.videoElement.playbackRate = options.speed || this.signingSpeed;

        // Show video
        this.videoElement.classList.remove('hidden');

        // Play
        try {
            await this.videoElement.play();

            // Handle video end
            return new Promise((resolve) => {
                this.videoElement.onended = () => {
                    if (!options.loop) {
                        this.showPlaceholder();
                    }
                    resolve();
                };
            });
        } catch (error) {
            console.error('Video play error:', error);
            this.showPlaceholder();
        }
    },

    /**
     * Show avatar placeholder
     */
    showPlaceholder() {
        if (this.videoElement) {
            this.videoElement.classList.add('hidden');
            this.videoElement.pause();
        }
        if (this.placeholderElement) {
            this.placeholderElement.classList.remove('hidden');
        }
    },

    /**
     * Get avatar info
     */
    async getAvatarInfo() {
        if (!this.currentAvatarId) return null;

        try {
            const response = await fetch(`${this.apiUrl}/${this.currentAvatarId}`);
            if (!response.ok) return null;
            return await response.json();
        } catch {
            return null;
        }
    },

    /**
     * Get generated videos for avatar
     */
    async getGeneratedVideos() {
        if (!this.currentAvatarId) return [];

        try {
            const response = await fetch(`${this.apiUrl}/${this.currentAvatarId}/videos`);
            if (!response.ok) return [];
            const data = await response.json();
            return data.videos || [];
        } catch {
            return [];
        }
    },

    /**
     * Delete avatar
     */
    async deleteAvatar() {
        if (!this.currentAvatarId) return;

        try {
            await fetch(`${this.apiUrl}/${this.currentAvatarId}`, {
                method: 'DELETE'
            });

            this.currentAvatarId = null;
            Storage.updateUserProfile({
                avatarId: null,
                avatarUrl: null
            });
        } catch (error) {
            console.error('Delete avatar error:', error);
        }
    },

    /**
     * Get available phrases
     */
    async getAvailablePhrases() {
        try {
            const response = await fetch('/api/phrases');
            if (!response.ok) return [];
            return await response.json();
        } catch {
            return [];
        }
    },

    /**
     * Set signing speed
     */
    setSpeed(speed) {
        this.signingSpeed = Math.max(0.5, Math.min(2, speed));
        if (this.videoElement) {
            this.videoElement.playbackRate = this.signingSpeed;
        }
    },

    /**
     * Set skin tone
     */
    setSkinTone(tone) {
        this.skinTone = tone;
        // This would be used when generating new avatar
    },

    /**
     * Set background
     */
    setBackground(bg) {
        this.background = bg;
        // Apply background to video container
        const container = this.videoElement?.parentElement;
        if (container) {
            container.dataset.bg = bg;
        }
    },

    /**
     * Export video
     */
    async exportVideo(phrase) {
        if (!this.currentAvatarId) return null;

        try {
            const response = await fetch(
                `${this.apiUrl}/${this.currentAvatarId}/video/${phrase}`
            );
            if (!response.ok) return null;

            const blob = await response.blob();
            return blob;
        } catch {
            return null;
        }
    },

    /**
     * Get avatar preview URL
     */
    getPreviewUrl() {
        const profile = Storage.getUserProfile();
        return profile.avatarUrl || null;
    },

    /**
     * Check if avatar exists
     */
    hasAvatar() {
        return !!this.currentAvatarId;
    },

    /**
     * Update avatar image in UI elements
     */
    updateAvatarUI() {
        const url = this.getPreviewUrl();
        if (!url) return;

        // Update all avatar images
        const avatarImages = [
            'header-avatar',
            'menu-avatar',
            'settings-avatar',
            'avatar-image',
            'avatar-still'
        ];

        avatarImages.forEach(id => {
            const img = Utils.$(id);
            if (img) {
                img.src = url;
            }
        });
    },

    /**
     * Preload common phrase videos
     */
    async preloadCommonPhrases() {
        const commonPhrases = ['HELLO', 'THANK_YOU', 'YES', 'NO', 'HELP'];

        for (const phrase of commonPhrases) {
            try {
                // Just trigger generation, don't wait
                fetch(`${this.apiUrl}/${this.currentAvatarId}/sign`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ phrase })
                });
            } catch {
                // Ignore errors during preload
            }
        }
    }
};

// Export for use in modules
if (typeof module !== 'undefined') {
    module.exports = Avatar;
}
