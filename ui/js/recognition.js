/**
 * Sign Language Recognition Module - SonZo AI
 * Handles real-time sign recognition via API
 */

const Recognition = {
    isRunning: false,
    frameInterval: null,
    apiUrl: '/api/recognize',
    lastRecognition: null,
    recognitionBuffer: [],
    bufferSize: 5,
    confidenceThreshold: 0.7,
    unknownThreshold: 0.4,
    callbacks: {
        onRecognition: null,
        onProcessing: null,
        onError: null,
        onHandsDetected: null
    },

    // Timing
    frameRate: 10, // Frames per second to send
    lastFrameTime: 0,
    minFrameInterval: 100, // ms between frames

    // Stats
    stats: {
        framesProcessed: 0,
        averageLatency: 0,
        recognitionCount: 0
    },

    /**
     * Initialize recognition
     */
    init(options = {}) {
        this.apiUrl = options.apiUrl || this.apiUrl;
        this.confidenceThreshold = options.confidenceThreshold || this.confidenceThreshold;
        this.frameRate = options.frameRate || this.frameRate;

        if (options.onRecognition) this.callbacks.onRecognition = options.onRecognition;
        if (options.onProcessing) this.callbacks.onProcessing = options.onProcessing;
        if (options.onError) this.callbacks.onError = options.onError;
        if (options.onHandsDetected) this.callbacks.onHandsDetected = options.onHandsDetected;

        return true;
    },

    /**
     * Start recognition loop
     */
    start() {
        if (this.isRunning) return;

        this.isRunning = true;
        this.processFrames();

        console.log('Recognition started');
    },

    /**
     * Stop recognition
     */
    stop() {
        this.isRunning = false;
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }

        console.log('Recognition stopped');
    },

    /**
     * Process video frames
     */
    async processFrames() {
        while (this.isRunning) {
            const now = Date.now();

            if (now - this.lastFrameTime >= this.minFrameInterval) {
                this.lastFrameTime = now;
                await this.processFrame();
            }

            // Small delay to prevent blocking
            await Utils.sleep(10);
        }
    },

    /**
     * Process single frame
     */
    async processFrame() {
        if (!Camera.isActive()) return;

        try {
            // Get frame
            const frameData = Camera.capturePhoto({ quality: 0.8 });
            if (!frameData) return;

            // Notify processing
            this.callbacks.onProcessing?.();

            // Send to API
            const startTime = Date.now();
            const result = await this.sendFrame(frameData);
            const latency = Date.now() - startTime;

            // Update stats
            this.stats.framesProcessed++;
            this.stats.averageLatency = (this.stats.averageLatency * 0.9) + (latency * 0.1);

            // Process result
            if (result) {
                this.handleRecognitionResult(result);
            }

        } catch (error) {
            console.error('Frame processing error:', error);
            this.callbacks.onError?.(error);
        }
    },

    /**
     * Send frame to API
     */
    async sendFrame(frameData) {
        try {
            const response = await fetch(this.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: frameData,
                    timestamp: Date.now()
                })
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            // Don't throw for network errors during recognition
            console.warn('Recognition API error:', error.message);
            return null;
        }
    },

    /**
     * Handle recognition result
     */
    handleRecognitionResult(result) {
        // Update hand tracking
        if (result.hands) {
            this.callbacks.onHandsDetected?.(result.hands);
            HandTracking.updateLandmarks(result.hands, result.state || 'normal');
        }

        // Check if we have a valid recognition
        if (!result.sign || !result.confidence) {
            return;
        }

        // Add to buffer for stability
        this.recognitionBuffer.push({
            sign: result.sign,
            confidence: result.confidence,
            timestamp: Date.now()
        });

        // Keep buffer size limited
        if (this.recognitionBuffer.length > this.bufferSize) {
            this.recognitionBuffer.shift();
        }

        // Get stable recognition
        const stableRecognition = this.getStableRecognition();

        if (stableRecognition) {
            this.lastRecognition = stableRecognition;
            this.stats.recognitionCount++;
            this.callbacks.onRecognition?.(stableRecognition);
        }
    },

    /**
     * Get stable recognition from buffer
     */
    getStableRecognition() {
        if (this.recognitionBuffer.length < 3) return null;

        // Count occurrences of each sign
        const counts = {};
        let totalConfidence = 0;

        this.recognitionBuffer.forEach(item => {
            counts[item.sign] = (counts[item.sign] || 0) + 1;
            totalConfidence += item.confidence;
        });

        // Find most common sign
        let maxCount = 0;
        let mostCommon = null;

        for (const [sign, count] of Object.entries(counts)) {
            if (count > maxCount) {
                maxCount = count;
                mostCommon = sign;
            }
        }

        // Require majority agreement
        if (maxCount < Math.ceil(this.bufferSize / 2)) {
            return null;
        }

        // Calculate average confidence for this sign
        const avgConfidence = this.recognitionBuffer
            .filter(item => item.sign === mostCommon)
            .reduce((sum, item) => sum + item.confidence, 0) / maxCount;

        // Check confidence threshold
        if (avgConfidence < this.unknownThreshold) {
            return { sign: 'UNKNOWN', confidence: avgConfidence, isUnknown: true };
        }

        if (avgConfidence < this.confidenceThreshold) {
            return null; // Not confident enough
        }

        // Clear buffer after stable recognition
        this.recognitionBuffer = [];

        return {
            sign: mostCommon,
            confidence: avgConfidence,
            topPredictions: this.getTopPredictions(this.recognitionBuffer),
            isUnknown: false
        };
    },

    /**
     * Get top N predictions
     */
    getTopPredictions(buffer, n = 3) {
        const predictions = {};

        buffer.forEach(item => {
            if (!predictions[item.sign]) {
                predictions[item.sign] = { sign: item.sign, confidence: 0, count: 0 };
            }
            predictions[item.sign].confidence += item.confidence;
            predictions[item.sign].count++;
        });

        return Object.values(predictions)
            .map(p => ({
                sign: p.sign,
                confidence: p.confidence / p.count
            }))
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, n);
    },

    /**
     * Reset recognition state
     */
    reset() {
        this.recognitionBuffer = [];
        this.lastRecognition = null;
    },

    /**
     * Set confidence threshold
     */
    setConfidenceThreshold(threshold) {
        this.confidenceThreshold = Math.max(0, Math.min(1, threshold));
    },

    /**
     * Get recognition stats
     */
    getStats() {
        return {
            ...this.stats,
            isRunning: this.isRunning,
            bufferLength: this.recognitionBuffer.length
        };
    },

    /**
     * Recognize single image (not streaming)
     */
    async recognizeImage(imageData) {
        try {
            const response = await fetch(this.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: imageData,
                    single: true
                })
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Recognition error:', error);
            return null;
        }
    },

    /**
     * Check if recognition API is available
     */
    async checkAPI() {
        try {
            const response = await fetch(this.apiUrl.replace('/recognize', '/health'));
            return response.ok;
        } catch {
            return false;
        }
    }
};

// Export for use in modules
if (typeof module !== 'undefined') {
    module.exports = Recognition;
}
