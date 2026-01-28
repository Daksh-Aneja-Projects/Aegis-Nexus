/**
 * Chart Data Processing Web Worker
 * 
 * Offloads heavy data processing from the main thread to prevent UI jank.
 * Uses Comlink for typed, Promise-based communication.
 * 
 * PRODUCTION HARDENING:
 * - Main thread stays responsive during data storms
 * - Batch processing for WebSocket message floods
 * - Memory-efficient rolling window calculations
 */

// Welford's Online Algorithm for running statistics (same as backend)
class WelfordStats {
    constructor() {
        this.count = 0;
        this.mean = 0;
        this.m2 = 0;
    }

    update(value) {
        this.count += 1;
        const delta = value - this.mean;
        this.mean += delta / this.count;
        const delta2 = value - this.mean;
        this.m2 += delta * delta2;
    }

    get variance() {
        if (this.count < 2) return 0;
        return this.m2 / (this.count - 1);
    }

    get stdDev() {
        return Math.sqrt(this.variance);
    }
}

// Rolling window buffer for time-series data
class RollingBuffer {
    constructor(maxSize = 100) {
        this.maxSize = maxSize;
        this.data = [];
        this.stats = new WelfordStats();
    }

    push(point) {
        this.data.push(point);
        this.stats.update(point.value);

        // Trim to max size
        if (this.data.length > this.maxSize) {
            this.data.shift();
        }
    }

    getAll() {
        return this.data;
    }

    getLast(n) {
        return this.data.slice(-n);
    }

    getStats() {
        return {
            count: this.stats.count,
            mean: this.stats.mean,
            variance: this.stats.variance,
            stdDev: this.stats.stdDev,
            min: Math.min(...this.data.map(d => d.value)),
            max: Math.max(...this.data.map(d => d.value))
        };
    }
}

// Main worker state
const buffers = new Map();

/**
 * Process incoming metric data point
 */
function processMetric(metricName, value, timestamp) {
    if (!buffers.has(metricName)) {
        buffers.set(metricName, new RollingBuffer(200));
    }

    const buffer = buffers.get(metricName);
    buffer.push({
        timestamp: timestamp || Date.now(),
        value: value
    });

    return {
        metricName,
        currentValue: value,
        stats: buffer.getStats(),
        chartData: buffer.getLast(100)
    };
}

/**
 * Batch process multiple metrics
 */
function processBatch(metrics) {
    return metrics.map(({ name, value, timestamp }) =>
        processMetric(name, value, timestamp)
    );
}

/**
 * Compute anomaly score using Mahalanobis-like distance
 */
function computeAnomalyScore(metricName, value) {
    if (!buffers.has(metricName)) {
        return { score: 0, isAnomaly: false };
    }

    const stats = buffers.get(metricName).getStats();
    if (stats.stdDev === 0) {
        return { score: 0, isAnomaly: false };
    }

    const zScore = Math.abs(value - stats.mean) / stats.stdDev;
    const isAnomaly = zScore > 3; // 3-sigma rule

    return {
        score: zScore,
        isAnomaly,
        threshold: 3,
        mean: stats.mean,
        stdDev: stats.stdDev
    };
}

/**
 * Get all buffered data for a metric
 */
function getMetricData(metricName, limit = 100) {
    if (!buffers.has(metricName)) {
        return { chartData: [], stats: null };
    }

    const buffer = buffers.get(metricName);
    return {
        chartData: buffer.getLast(limit),
        stats: buffer.getStats()
    };
}

/**
 * Clear all buffers (for testing)
 */
function reset() {
    buffers.clear();
    return true;
}

// Export functions for Comlink
const workerApi = {
    processMetric,
    processBatch,
    computeAnomalyScore,
    getMetricData,
    reset
};

// Self message handler for native Worker usage
self.onmessage = (event) => {
    const { type, payload, id } = event.data;

    try {
        let result;
        switch (type) {
            case 'processMetric':
                result = processMetric(payload.name, payload.value, payload.timestamp);
                break;
            case 'processBatch':
                result = processBatch(payload.metrics);
                break;
            case 'computeAnomalyScore':
                result = computeAnomalyScore(payload.name, payload.value);
                break;
            case 'getMetricData':
                result = getMetricData(payload.name, payload.limit);
                break;
            case 'reset':
                result = reset();
                break;
            default:
                throw new Error(`Unknown message type: ${type}`);
        }

        self.postMessage({ id, success: true, result });
    } catch (error) {
        self.postMessage({ id, success: false, error: error.message });
    }
};

// Comlink exposure (if using Comlink)
if (typeof Comlink !== 'undefined') {
    Comlink.expose(workerApi);
}
