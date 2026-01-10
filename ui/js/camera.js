/**
 * Camera Manager - SonZo AI
 * Handles webcam access, photo capture, and video recording
 */

const Camera = {
    stream: null,
    videoElement: null,
    facingMode: 'user',
    isRecording: false,
    mediaRecorder: null,
    recordedChunks: [],

    /**
     * Initialize camera
     */
    async init(videoElementId, options = {}) {
        this.videoElement = Utils.$(videoElementId);
        if (!this.videoElement) {
            console.error('Video element not found:', videoElementId);
            return false;
        }

        return this.startStream(options);
    },

    /**
     * Start camera stream
     */
    async startStream(options = {}) {
        try {
            // Stop existing stream
            this.stopStream();

            const constraints = {
                video: {
                    facingMode: options.facingMode || this.facingMode,
                    width: { ideal: options.width || 1280 },
                    height: { ideal: options.height || 720 },
                    frameRate: { ideal: options.frameRate || 30 }
                },
                audio: options.audio || false
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);

            if (this.videoElement) {
                this.videoElement.srcObject = this.stream;
                await this.videoElement.play();
            }

            // Apply mirror for user-facing camera
            if (this.facingMode === 'user' && !Storage.get('settings.leftHanded', false)) {
                this.videoElement.style.transform = 'scaleX(-1)';
            } else {
                this.videoElement.style.transform = 'none';
            }

            return true;
        } catch (error) {
            console.error('Camera error:', error);
            this.handleCameraError(error);
            return false;
        }
    },

    /**
     * Stop camera stream
     */
    stopStream() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.videoElement) {
            this.videoElement.srcObject = null;
        }
    },

    /**
     * Switch camera (front/back)
     */
    async switchCamera() {
        this.facingMode = this.facingMode === 'user' ? 'environment' : 'user';
        return this.startStream({ facingMode: this.facingMode });
    },

    /**
     * Capture photo from video
     */
    capturePhoto(options = {}) {
        if (!this.videoElement || !this.stream) {
            console.error('Camera not initialized');
            return null;
        }

        const video = this.videoElement;
        const canvas = document.createElement('canvas');
        const width = options.width || video.videoWidth;
        const height = options.height || video.videoHeight;

        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext('2d');

        // Mirror if user-facing camera
        if (this.facingMode === 'user') {
            ctx.translate(width, 0);
            ctx.scale(-1, 1);
        }

        ctx.drawImage(video, 0, 0, width, height);

        // Return as data URL or blob
        if (options.asBlob) {
            return new Promise(resolve => {
                canvas.toBlob(resolve, 'image/jpeg', options.quality || 0.9);
            });
        }

        return canvas.toDataURL('image/jpeg', options.quality || 0.9);
    },

    /**
     * Start video recording
     */
    startRecording(options = {}) {
        if (!this.stream || this.isRecording) return false;

        this.recordedChunks = [];
        const mimeType = this.getSupportedMimeType();

        try {
            this.mediaRecorder = new MediaRecorder(this.stream, {
                mimeType,
                videoBitsPerSecond: options.videoBitsPerSecond || 2500000
            });

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.recordedChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstop = () => {
                this.isRecording = false;
                if (options.onStop) {
                    const blob = new Blob(this.recordedChunks, { type: mimeType });
                    options.onStop(blob);
                }
            };

            this.mediaRecorder.start(options.timeslice || 100);
            this.isRecording = true;

            // Auto-stop after max duration
            if (options.maxDuration) {
                setTimeout(() => this.stopRecording(), options.maxDuration);
            }

            return true;
        } catch (error) {
            console.error('Recording error:', error);
            return false;
        }
    },

    /**
     * Stop video recording
     */
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            return true;
        }
        return false;
    },

    /**
     * Get recorded video blob
     */
    getRecordedBlob() {
        if (this.recordedChunks.length === 0) return null;
        return new Blob(this.recordedChunks, { type: this.getSupportedMimeType() });
    },

    /**
     * Get supported mime type for recording
     */
    getSupportedMimeType() {
        const types = [
            'video/webm;codecs=vp9',
            'video/webm;codecs=vp8',
            'video/webm',
            'video/mp4'
        ];

        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                return type;
            }
        }

        return 'video/webm';
    },

    /**
     * Check camera permissions
     */
    async checkPermissions() {
        try {
            const result = await navigator.permissions.query({ name: 'camera' });
            return result.state;
        } catch {
            return 'prompt';
        }
    },

    /**
     * Handle camera errors
     */
    handleCameraError(error) {
        let message = 'Unable to access camera';

        switch (error.name) {
            case 'NotAllowedError':
                message = 'Camera permission denied. Please allow camera access in your browser settings.';
                break;
            case 'NotFoundError':
                message = 'No camera found on this device.';
                break;
            case 'NotReadableError':
                message = 'Camera is already in use by another application.';
                break;
            case 'OverconstrainedError':
                message = 'Camera does not support the requested settings.';
                break;
            case 'SecurityError':
                message = 'Camera access blocked due to security restrictions.';
                break;
        }

        app?.showToast(message, 'error');
    },

    /**
     * Get current frame as ImageData
     */
    getCurrentFrame() {
        if (!this.videoElement || !this.stream) return null;

        const canvas = document.createElement('canvas');
        canvas.width = this.videoElement.videoWidth;
        canvas.height = this.videoElement.videoHeight;

        const ctx = canvas.getContext('2d');

        // Mirror if user-facing
        if (this.facingMode === 'user') {
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
        }

        ctx.drawImage(this.videoElement, 0, 0);
        return ctx.getImageData(0, 0, canvas.width, canvas.height);
    },

    /**
     * Set mirror mode (for left-handed users)
     */
    setMirror(mirror) {
        if (this.videoElement) {
            this.videoElement.style.transform = mirror ? 'scaleX(-1)' : 'none';
        }
    },

    /**
     * Get video dimensions
     */
    getDimensions() {
        if (!this.videoElement) return { width: 0, height: 0 };
        return {
            width: this.videoElement.videoWidth,
            height: this.videoElement.videoHeight
        };
    },

    /**
     * Check if camera is active
     */
    isActive() {
        return this.stream && this.stream.active;
    },

    /**
     * Apply video constraints
     */
    async applyConstraints(constraints) {
        if (!this.stream) return false;

        try {
            const videoTrack = this.stream.getVideoTracks()[0];
            await videoTrack.applyConstraints(constraints);
            return true;
        } catch (error) {
            console.error('Apply constraints error:', error);
            return false;
        }
    },

    /**
     * Get camera capabilities
     */
    getCapabilities() {
        if (!this.stream) return null;

        const videoTrack = this.stream.getVideoTracks()[0];
        return videoTrack.getCapabilities();
    }
};

// Export for use in modules
if (typeof module !== 'undefined') {
    module.exports = Camera;
}
