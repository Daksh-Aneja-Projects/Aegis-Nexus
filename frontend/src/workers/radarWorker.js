/**
 * RadarWorker - Semantic Drift Calculations (Level 5 Optimized)
 * 
 * Production Hardening:
 * - TypedArray-based vector operations for SIMD-friendly patterns
 * - Efficient memory usage with Float32Array
 * - Batch processing for high-frequency updates
 */

/* eslint-disable no-restricted-globals */

// TypedArray-based vector normalization
function normalizeVector(vector) {
    const arr = new Float32Array(vector);
    let magnitude = 0;
    for (let i = 0; i < arr.length; i++) {
        magnitude += arr[i] * arr[i];
    }
    magnitude = Math.sqrt(magnitude);

    if (magnitude === 0) return arr;

    for (let i = 0; i < arr.length; i++) {
        arr[i] /= magnitude;
    }
    return arr;
}

// Euclidean distance using TypedArray
function euclideanDistance(a, b) {
    const arrA = new Float32Array(a);
    const arrB = new Float32Array(b);
    const len = Math.min(arrA.length, arrB.length);

    let sum = 0;
    for (let i = 0; i < len; i++) {
        const diff = arrA[i] - arrB[i];
        sum += diff * diff;
    }
    return Math.sqrt(sum);
}

// Cosine similarity for embedding comparison
function cosineSimilarity(a, b) {
    const arrA = new Float32Array(a);
    const arrB = new Float32Array(b);
    const len = Math.min(arrA.length, arrB.length);

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < len; i++) {
        dotProduct += arrA[i] * arrB[i];
        normA += arrA[i] * arrA[i];
        normB += arrB[i] * arrB[i];
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator === 0 ? 0 : dotProduct / denominator;
}

// Batch process embeddings for efficiency
function batchProcessEmbeddings(embeddings, config) {
    const { threshold, maxPoints, referencePoint } = config;
    const results = new Array(embeddings.length);

    // Use Float32Array for distance cache
    const distances = new Float32Array(embeddings.length);

    // Calculate distances in batch
    for (let i = 0; i < embeddings.length; i++) {
        const e = embeddings[i];
        if (e.vector && referencePoint) {
            distances[i] = euclideanDistance(e.vector, referencePoint);
        } else {
            distances[i] = e.distance || 0;
        }
    }

    // Process results
    for (let i = 0; i < embeddings.length; i++) {
        const e = embeddings[i];
        const distance = distances[i];

        // Threat categorization
        let threatLevel = 'safe';
        if (distance >= threshold) {
            threatLevel = 'danger';
        } else if (distance >= threshold * 0.5) {
            threatLevel = 'warning';
        }

        // Polar to Cartesian conversion
        const angle = e.angle || (i * 0.1);
        const x = Math.cos(angle) * distance;
        const y = Math.sin(angle) * distance;

        results[i] = {
            ...e,
            distance,
            threatLevel,
            x,
            y,
            age: i / embeddings.length,
            normalized: e.vector ? Array.from(normalizeVector(e.vector)) : null
        };
    }

    return results;
}

// Message handler
self.onmessage = (e) => {
    const { type, embeddings, threshold, maxPoints, referencePoint } = e.data;

    if (type === 'BATCH_PROCESS') {
        // Batch processing mode for high-frequency updates
        const results = batchProcessEmbeddings(embeddings, { threshold, maxPoints, referencePoint });
        self.postMessage({ type: 'BATCH_RESULT', displayData: results });
        return;
    }

    if (!embeddings) return;

    // Process recent embeddings (standard mode)
    const recentEmbeddings = embeddings.slice(-maxPoints);

    const displayData = recentEmbeddings.map((e, i) => {
        // Calculate threat level
        let threatLevel = 'safe';
        if (e.distance >= threshold) {
            threatLevel = 'danger';
        } else if (e.distance >= threshold * 0.5) {
            threatLevel = 'warning';
        }

        // Polar to Cartesian
        const angle = e.angle || (i * 0.1);
        const x = Math.cos(angle) * e.distance;
        const y = Math.sin(angle) * e.distance;

        return {
            ...e,
            threatLevel,
            x,
            y,
            age: i / recentEmbeddings.length
        };
    });

    self.postMessage({ displayData });
};

// Expose utility functions for testing
self.radarUtils = {
    normalizeVector,
    euclideanDistance,
    cosineSimilarity,
    batchProcessEmbeddings
};
