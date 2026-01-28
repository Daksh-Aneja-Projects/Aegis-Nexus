/**
 * SemanticRadar - Semantic Telemetry & Drift Visualization
 * 
 * Visualizes prompt embeddings in vector space to detect prompt injection campaigns.
 * Safe prompts cluster in the center (green), attacks drift to the edges (red).
 * 
 * PRODUCTION FEATURES:
 * - Real-time embedding distance visualization
 * - Threshold-based coloring for threat detection
 * - Cluster detection for campaign identification
 * - Historical trail for pattern analysis
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, Shield, Activity, Target, Zap } from 'lucide-react';

/**
 * Calculate 2D position from embedding distance and angle
 */
const polarToCartesian = (distance, angle) => ({
    x: Math.cos(angle) * distance,
    y: Math.sin(angle) * distance,
});

/**
 * Determine threat level based on distance from safety constitution
 */
const getThreatLevel = (distance, threshold) => {
    if (distance < threshold * 0.5) return 'safe';
    if (distance < threshold) return 'warning';
    return 'danger';
};

const THREAT_COLORS = {
    safe: { fill: '#10b981', glow: 'rgba(16, 185, 129, 0.5)' },      // emerald
    warning: { fill: '#fbbf24', glow: 'rgba(251, 191, 36, 0.5)' },   // amber
    danger: { fill: '#f43f5e', glow: 'rgba(244, 63, 94, 0.5)' },     // rose
};

/**
 * SemanticRadar Component
 * @param {Object[]} embeddings - Array of {id, distance, angle, label, timestamp}
 * @param {number} threshold - Distance threshold for danger zone (default: 0.7)
 * @param {Function} onPointClick - Callback when a point is clicked
 */
const SemanticRadar = ({
    embeddings = [],
    threshold = 0.7,
    onPointClick,
    showLabels = false,
    maxPoints = 200,
}) => {
    const [selectedPoint, setSelectedPoint] = useState(null);
    const [hoveredPoint, setHoveredPoint] = useState(null);

    const [displayData, setDisplayData] = useState([]);
    const workerRef = useRef(null);

    useEffect(() => {
        // Level 5 Optimization: Offload distance & threat calc to Web Worker
        workerRef.current = new Worker(new URL('../workers/radarWorker.js', import.meta.url));

        workerRef.current.onmessage = (e) => {
            setDisplayData(e.data.displayData);
        };

        return () => workerRef.current.terminate();
    }, []);

    useEffect(() => {
        if (workerRef.current) {
            workerRef.current.postMessage({
                embeddings,
                threshold,
                maxPoints
            });
        }
    }, [embeddings, threshold, maxPoints]);

    // Statistics
    const stats = useMemo(() => {
        const total = displayData.length;
        const safe = displayData.filter(d => d.threatLevel === 'safe').length;
        const warning = displayData.filter(d => d.threatLevel === 'warning').length;
        const danger = displayData.filter(d => d.threatLevel === 'danger').length;

        return { total, safe, warning, danger };
    }, [displayData]);

    const handlePointClick = useCallback((point) => {
        setSelectedPoint(point.id === selectedPoint?.id ? null : point);
        onPointClick?.(point);
    }, [selectedPoint, onPointClick]);

    return (
        <div className="relative w-full aspect-square max-w-lg mx-auto">
            {/* Background Container */}
            <div className="absolute inset-0 rounded-full bg-black/60 backdrop-blur-xl border border-white/10 overflow-hidden">

                {/* Concentric Threshold Rings */}
                <svg className="absolute inset-0 w-full h-full" viewBox="-1.2 -1.2 2.4 2.4">
                    {/* Safe zone (center) */}
                    <circle
                        cx="0" cy="0"
                        r={threshold * 0.5}
                        fill="rgba(16, 185, 129, 0.05)"
                        stroke="rgba(16, 185, 129, 0.2)"
                        strokeWidth="0.01"
                    />

                    {/* Warning zone */}
                    <circle
                        cx="0" cy="0"
                        r={threshold}
                        fill="none"
                        stroke="rgba(251, 191, 36, 0.3)"
                        strokeWidth="0.01"
                        strokeDasharray="0.05 0.05"
                    />

                    {/* Danger zone boundary */}
                    <circle
                        cx="0" cy="0"
                        r="1"
                        fill="none"
                        stroke="rgba(244, 63, 94, 0.2)"
                        strokeWidth="0.02"
                    />

                    {/* Grid lines */}
                    {[0, 45, 90, 135].map(angle => (
                        <line
                            key={angle}
                            x1={Math.cos(angle * Math.PI / 180) * -1}
                            y1={Math.sin(angle * Math.PI / 180) * -1}
                            x2={Math.cos(angle * Math.PI / 180) * 1}
                            y2={Math.sin(angle * Math.PI / 180) * 1}
                            stroke="rgba(255,255,255,0.05)"
                            strokeWidth="0.005"
                        />
                    ))}

                    {/* Embedding Points */}
                    {displayData.map((point, i) => (
                        <g key={point.id || i}>
                            {/* Glow effect for recent points */}
                            {point.age > 0.8 && (
                                <circle
                                    cx={point.x}
                                    cy={point.y}
                                    r={0.04}
                                    fill={point.colors.glow}
                                    opacity={0.5}
                                />
                            )}

                            {/* Main point */}
                            <circle
                                cx={point.x}
                                cy={point.y}
                                r={point.id === selectedPoint?.id ? 0.04 : 0.02}
                                fill={point.colors.fill}
                                opacity={0.3 + point.age * 0.7}
                                style={{ cursor: 'pointer' }}
                                onClick={() => handlePointClick(point)}
                                onMouseEnter={() => setHoveredPoint(point)}
                                onMouseLeave={() => setHoveredPoint(null)}
                            />
                        </g>
                    ))}

                    {/* Center crosshair (safety constitution) */}
                    <circle cx="0" cy="0" r="0.03" fill="#00f2ff" />
                    <circle cx="0" cy="0" r="0.05" fill="none" stroke="#00f2ff" strokeWidth="0.005" />
                </svg>
            </div>

            {/* Zone Labels */}
            <div className="absolute inset-0 pointer-events-none">
                <div className="absolute top-4 left-1/2 -translate-x-1/2 text-xs font-mono text-slate-500 uppercase tracking-wider">
                    Danger Zone
                </div>
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-xs font-mono text-emerald-500/50 uppercase tracking-wider">
                    Safety Constitution
                </div>
            </div>

            {/* Hover Tooltip */}
            <AnimatePresence>
                {hoveredPoint && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 10 }}
                        className="absolute top-4 right-4 bg-black/80 backdrop-blur-md rounded-lg p-3 border border-white/10 text-xs font-mono"
                    >
                        <div className="text-slate-400">Distance:</div>
                        <div className="text-lg font-bold" style={{ color: hoveredPoint.colors.fill }}>
                            {hoveredPoint.distance.toFixed(3)}
                        </div>
                        {hoveredPoint.label && (
                            <div className="mt-2 text-slate-500 max-w-32 truncate">
                                {hoveredPoint.label}
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Stats Panel */}
            <div className="absolute bottom-2 left-2 right-2 flex justify-around text-xs font-mono">
                <div className="flex items-center gap-1">
                    <Shield size={12} className="text-emerald-400" />
                    <span className="text-emerald-400">{stats.safe}</span>
                </div>
                <div className="flex items-center gap-1">
                    <AlertTriangle size={12} className="text-amber-400" />
                    <span className="text-amber-400">{stats.warning}</span>
                </div>
                <div className="flex items-center gap-1">
                    <Zap size={12} className="text-rose-400" />
                    <span className="text-rose-400">{stats.danger}</span>
                </div>
            </div>
        </div>
    );
};

/**
 * SemanticRadarPanel - Full panel with controls
 */
export const SemanticRadarPanel = ({
    embeddings = [],
    threshold = 0.7,
    title = "Semantic Drift Monitor"
}) => {
    const [isLive, setIsLive] = useState(true);
    const [selectedPoint, setSelectedPoint] = useState(null);

    // Calculate threat score
    const threatScore = useMemo(() => {
        if (embeddings.length === 0) return 0;
        const recentDanger = embeddings.slice(-50).filter(e => e.distance > threshold).length;
        return Math.min(100, (recentDanger / 50) * 100);
    }, [embeddings, threshold]);

    return (
        <div className="glass-depth-1 rounded-3xl p-6 border border-white/10">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-neon-accent/10 border border-neon-accent/30">
                        <Target size={20} className="text-neon-accent" />
                    </div>
                    <div>
                        <h3 className="text-lg font-bold text-white">{title}</h3>
                        <p className="text-xs text-slate-500 font-mono">
                            {embeddings.length} samples tracked
                        </p>
                    </div>
                </div>

                {/* Live indicator */}
                <div className="flex items-center gap-2">
                    <motion.div
                        className={`w-2 h-2 rounded-full ${isLive ? 'bg-emerald-400' : 'bg-slate-500'}`}
                        animate={isLive ? { opacity: [1, 0.5, 1] } : {}}
                        transition={{ duration: 1, repeat: Infinity }}
                    />
                    <span className="text-xs font-mono text-slate-400">
                        {isLive ? 'LIVE' : 'PAUSED'}
                    </span>
                </div>
            </div>

            {/* Threat Score Bar */}
            <div className="mb-4">
                <div className="flex justify-between text-xs font-mono mb-1">
                    <span className="text-slate-400">Threat Level</span>
                    <span className={threatScore > 50 ? 'text-rose-400' : 'text-emerald-400'}>
                        {threatScore.toFixed(1)}%
                    </span>
                </div>
                <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                    <motion.div
                        className="h-full rounded-full"
                        style={{
                            background: threatScore > 50
                                ? 'linear-gradient(90deg, #fbbf24, #f43f5e)'
                                : 'linear-gradient(90deg, #10b981, #fbbf24)',
                        }}
                        initial={{ width: 0 }}
                        animate={{ width: `${threatScore}%` }}
                        transition={{ duration: 0.5 }}
                    />
                </div>
            </div>

            {/* Radar */}
            <SemanticRadar
                embeddings={embeddings}
                threshold={threshold}
                onPointClick={setSelectedPoint}
            />

            {/* Selected Point Details */}
            <AnimatePresence>
                {selectedPoint && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-4 p-3 rounded-xl bg-white/5 border border-white/10"
                    >
                        <div className="text-xs font-mono text-slate-400 mb-2">
                            Selected Sample
                        </div>
                        <div className="text-sm text-white truncate">
                            {selectedPoint.label || selectedPoint.id}
                        </div>
                        <div className="flex gap-4 mt-2 text-xs">
                            <span className="text-slate-500">
                                Distance: <span className="text-white">{selectedPoint.distance?.toFixed(4)}</span>
                            </span>
                            <span className="text-slate-500">
                                Status: <span style={{ color: selectedPoint.colors?.fill }}>
                                    {selectedPoint.threatLevel?.toUpperCase()}
                                </span>
                            </span>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

/**
 * Demo/Test Data Generator
 */
export const generateTestEmbeddings = (count = 100) => {
    const embeddings = [];

    for (let i = 0; i < count; i++) {
        // Mostly safe with some anomalies
        const isSafe = Math.random() > 0.15;
        const distance = isSafe
            ? Math.random() * 0.4  // Safe range
            : 0.6 + Math.random() * 0.4; // Danger range

        embeddings.push({
            id: `embed_${i}`,
            distance,
            angle: Math.random() * Math.PI * 2,
            label: isSafe ? 'Normal query' : 'Potential attack',
            timestamp: new Date(Date.now() - (count - i) * 1000),
        });
    }

    return embeddings;
};

export default React.memo(SemanticRadar);
