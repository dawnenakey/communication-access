/**
 * Hand Tracking Module - SonZo AI
 * Renders hand skeleton overlay and provides visual feedback
 */

const HandTracking = {
    canvas: null,
    ctx: null,
    isRunning: false,
    lastLandmarks: null,
    animationFrame: null,

    // Hand landmark connections for drawing skeleton
    HAND_CONNECTIONS: [
        [0, 1], [1, 2], [2, 3], [3, 4],      // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8],      // Index
        [0, 9], [9, 10], [10, 11], [11, 12], // Middle
        [0, 13], [13, 14], [14, 15], [15, 16], // Ring
        [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
        [5, 9], [9, 13], [13, 17], [0, 17]   // Palm
    ],

    // Colors for visual feedback
    COLORS: {
        skeleton: 'rgba(99, 102, 241, 0.8)',  // Primary purple
        landmark: 'rgba(236, 72, 153, 0.9)',   // Pink
        processing: 'rgba(251, 191, 36, 0.8)', // Yellow
        recognized: 'rgba(34, 197, 94, 0.8)',  // Green
        error: 'rgba(239, 68, 68, 0.8)'        // Red
    },

    /**
     * Initialize hand tracking overlay
     */
    init(canvasId) {
        this.canvas = Utils.$(canvasId);
        if (!this.canvas) {
            console.error('Canvas not found:', canvasId);
            return false;
        }

        this.ctx = this.canvas.getContext('2d');
        this.resizeCanvas();

        // Listen for resize
        window.addEventListener('resize', Utils.debounce(() => this.resizeCanvas(), 100));

        return true;
    },

    /**
     * Resize canvas to match parent
     */
    resizeCanvas() {
        if (!this.canvas) return;

        const parent = this.canvas.parentElement;
        this.canvas.width = parent.clientWidth;
        this.canvas.height = parent.clientHeight;
    },

    /**
     * Start hand tracking visualization
     */
    start() {
        this.isRunning = true;
        this.render();
    },

    /**
     * Stop hand tracking visualization
     */
    stop() {
        this.isRunning = false;
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
        this.clear();
    },

    /**
     * Update landmarks from recognition
     */
    updateLandmarks(landmarks, state = 'normal') {
        this.lastLandmarks = landmarks;
        this.currentState = state;
    },

    /**
     * Main render loop
     */
    render() {
        if (!this.isRunning) return;

        this.clear();

        if (this.lastLandmarks && Storage.get('settings.showSkeleton', true)) {
            this.drawHands(this.lastLandmarks, this.currentState);
        }

        this.animationFrame = requestAnimationFrame(() => this.render());
    },

    /**
     * Clear canvas
     */
    clear() {
        if (!this.ctx) return;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    },

    /**
     * Draw hand landmarks and skeleton
     */
    drawHands(handsData, state = 'normal') {
        if (!handsData || !this.ctx) return;

        const hands = Array.isArray(handsData) ? handsData : [handsData];

        hands.forEach(hand => {
            if (!hand.landmarks) return;

            const landmarks = hand.landmarks;
            const color = this.getStateColor(state);

            // Draw connections
            this.drawConnections(landmarks, color);

            // Draw landmarks
            this.drawLandmarks(landmarks, color);

            // Draw label if available
            if (hand.label) {
                this.drawLabel(landmarks[0], hand.label, color);
            }
        });
    },

    /**
     * Draw skeleton connections
     */
    drawConnections(landmarks, color) {
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 3;
        this.ctx.lineCap = 'round';

        this.HAND_CONNECTIONS.forEach(([start, end]) => {
            const p1 = landmarks[start];
            const p2 = landmarks[end];

            if (p1 && p2) {
                this.ctx.beginPath();
                this.ctx.moveTo(p1.x * this.canvas.width, p1.y * this.canvas.height);
                this.ctx.lineTo(p2.x * this.canvas.width, p2.y * this.canvas.height);
                this.ctx.stroke();
            }
        });
    },

    /**
     * Draw landmark points
     */
    drawLandmarks(landmarks, color) {
        landmarks.forEach((landmark, index) => {
            const x = landmark.x * this.canvas.width;
            const y = landmark.y * this.canvas.height;

            // Larger points for fingertips
            const isFingertip = [4, 8, 12, 16, 20].includes(index);
            const radius = isFingertip ? 8 : 5;

            // Draw outer glow
            this.ctx.beginPath();
            this.ctx.arc(x, y, radius + 2, 0, Math.PI * 2);
            this.ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
            this.ctx.fill();

            // Draw main point
            this.ctx.beginPath();
            this.ctx.arc(x, y, radius, 0, Math.PI * 2);
            this.ctx.fillStyle = isFingertip ? this.COLORS.landmark : color;
            this.ctx.fill();
        });
    },

    /**
     * Draw label near wrist
     */
    drawLabel(wristLandmark, label, color) {
        const x = wristLandmark.x * this.canvas.width;
        const y = wristLandmark.y * this.canvas.height + 30;

        this.ctx.font = 'bold 16px Inter, sans-serif';
        this.ctx.textAlign = 'center';

        // Background
        const metrics = this.ctx.measureText(label);
        const padding = 8;
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(
            x - metrics.width / 2 - padding,
            y - 12 - padding,
            metrics.width + padding * 2,
            20 + padding
        );

        // Text
        this.ctx.fillStyle = color;
        this.ctx.fillText(label, x, y);
    },

    /**
     * Get color based on state
     */
    getStateColor(state) {
        switch (state) {
            case 'processing':
                return this.COLORS.processing;
            case 'recognized':
                return this.COLORS.recognized;
            case 'error':
                return this.COLORS.error;
            default:
                return this.COLORS.skeleton;
        }
    },

    /**
     * Draw recognition ring
     */
    drawRecognitionRing(progress, state = 'processing') {
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        const radius = Math.min(this.canvas.width, this.canvas.height) * 0.3;

        // Background ring
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        this.ctx.lineWidth = 6;
        this.ctx.stroke();

        // Progress ring
        const startAngle = -Math.PI / 2;
        const endAngle = startAngle + (progress * Math.PI * 2);

        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, radius, startAngle, endAngle);
        this.ctx.strokeStyle = this.getStateColor(state);
        this.ctx.lineWidth = 6;
        this.ctx.lineCap = 'round';
        this.ctx.stroke();
    },

    /**
     * Draw bounding box around hands
     */
    drawBoundingBox(landmarks, label = '') {
        if (!landmarks || landmarks.length === 0) return;

        let minX = 1, minY = 1, maxX = 0, maxY = 0;

        landmarks.forEach(landmark => {
            minX = Math.min(minX, landmark.x);
            minY = Math.min(minY, landmark.y);
            maxX = Math.max(maxX, landmark.x);
            maxY = Math.max(maxY, landmark.y);
        });

        const padding = 0.02;
        const x = (minX - padding) * this.canvas.width;
        const y = (minY - padding) * this.canvas.height;
        const width = (maxX - minX + padding * 2) * this.canvas.width;
        const height = (maxY - minY + padding * 2) * this.canvas.height;

        // Draw box
        this.ctx.strokeStyle = this.COLORS.recognized;
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        this.ctx.strokeRect(x, y, width, height);
        this.ctx.setLineDash([]);

        // Draw label
        if (label) {
            this.ctx.font = 'bold 14px Inter, sans-serif';
            this.ctx.fillStyle = this.COLORS.recognized;
            this.ctx.fillText(label, x, y - 5);
        }
    },

    /**
     * Animate recognition success
     */
    animateSuccess(landmarks) {
        let frame = 0;
        const totalFrames = 20;

        const animate = () => {
            if (frame >= totalFrames) return;

            this.clear();

            const scale = 1 + Math.sin(frame / totalFrames * Math.PI) * 0.1;
            const alpha = 1 - (frame / totalFrames) * 0.5;

            // Save context
            this.ctx.save();

            // Scale from center
            const centerX = this.canvas.width / 2;
            const centerY = this.canvas.height / 2;
            this.ctx.translate(centerX, centerY);
            this.ctx.scale(scale, scale);
            this.ctx.translate(-centerX, -centerY);

            // Draw with animation
            this.ctx.globalAlpha = alpha;
            this.drawHands({ landmarks }, 'recognized');

            this.ctx.restore();

            frame++;
            requestAnimationFrame(animate);
        };

        animate();
    },

    /**
     * Show "slow down" visual indicator
     */
    showSlowDownIndicator() {
        const indicator = Utils.$('slow-down');
        if (indicator) {
            indicator.classList.remove('hidden');
            setTimeout(() => indicator.classList.add('hidden'), 2000);
        }
    }
};

// Export for use in modules
if (typeof module !== 'undefined') {
    module.exports = HandTracking;
}
