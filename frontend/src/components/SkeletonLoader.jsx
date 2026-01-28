/**
 * SkeletonLoader - Premium Loading States for Aegis Nexus
 * 
 * Provides graceful loading placeholders that maintain the "Obsidian Glass" aesthetic
 * while data is being fetched. Critical for Google-level UX.
 * 
 * PRODUCTION FEATURES:
 * - Multiple variants for different UI contexts
 * - Animated pulse effect
 * - Maintains layout stability (no CLS)
 */

import React from 'react';
import { motion } from 'framer-motion';

/**
 * SkeletonLoader component for loading states
 * @param {Object} props
 * @param {'card'|'line'|'circle'|'graph'|'panel'} props.variant - Type of skeleton
 * @param {number} props.count - Number of skeleton items to render
 * @param {string} props.className - Additional CSS classes
 */
const SkeletonLoader = ({ variant = 'card', count = 1, className = '' }) => {
    const variants = {
        card: 'h-32 rounded-xl',
        line: 'h-4 rounded',
        circle: 'w-12 h-12 rounded-full',
        graph: 'h-64 rounded-2xl',
        panel: 'h-48 rounded-3xl',
    };

    const baseClass = 'bg-gradient-to-r from-white/5 via-white/10 to-white/5 bg-[length:200%_100%]';

    return (
        <div className={`space-y-3 ${className}`}>
            {Array(count).fill(0).map((_, i) => (
                <motion.div
                    key={i}
                    className={`${baseClass} ${variants[variant]}`}
                    animate={{
                        backgroundPosition: ['200% 0', '-200% 0'],
                    }}
                    transition={{
                        duration: 1.5,
                        repeat: Infinity,
                        ease: 'linear',
                    }}
                />
            ))}
        </div>
    );
};

/**
 * SkeletonCard - Card-shaped skeleton for dashboard widgets
 */
export const SkeletonCard = ({ className = '' }) => (
    <div className={`glass-depth-1 rounded-3xl p-6 border border-white/10 ${className}`}>
        <SkeletonLoader variant="line" count={1} className="w-1/3 mb-4" />
        <SkeletonLoader variant="line" count={3} />
    </div>
);

/**
 * SkeletonGraph - Graph-shaped skeleton for visualizations
 */
export const SkeletonGraph = ({ className = '' }) => (
    <div className={`glass-depth-1 rounded-3xl p-6 border border-white/10 ${className}`}>
        <div className="flex items-center justify-between mb-4">
            <SkeletonLoader variant="line" count={1} className="w-1/4" />
            <SkeletonLoader variant="circle" count={1} />
        </div>
        <SkeletonLoader variant="graph" count={1} />
    </div>
);

/**
 * SkeletonWarRoom - Full WarRoom skeleton
 */
export const SkeletonWarRoom = () => (
    <div className="h-[800px] flex flex-col lg:flex-row gap-6 p-6 animate-pulse">
        {/* Main Graph Area */}
        <div className="flex-grow lg:w-2/3 rounded-3xl glass-depth-1 border border-white/10 relative overflow-hidden">
            {/* Background Grid Pattern */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:40px_40px]" />

            {/* Fake nodes */}
            <div className="absolute inset-0 flex items-center justify-center">
                {[0, 60, 120, 180, 240].map((angle, i) => (
                    <div
                        key={i}
                        className="absolute w-16 h-16 rounded-full bg-white/5 border border-white/10"
                        style={{
                            left: `${50 + 30 * Math.cos((angle * Math.PI) / 180)}%`,
                            top: `${50 + 30 * Math.sin((angle * Math.PI) / 180)}%`,
                            transform: 'translate(-50%, -50%)',
                        }}
                    />
                ))}
                {/* Center hub */}
                <div className="absolute w-20 h-20 rounded-full bg-white/10 border-2 border-white/20" />
            </div>

            {/* Input skeleton */}
            <div className="absolute bottom-6 left-6 right-6 h-14 rounded-xl bg-white/5 border border-white/10" />
        </div>

        {/* Detail Panel */}
        <div className="lg:w-1/3 glass-depth-2 p-6 rounded-3xl border border-white/5">
            <div className="space-y-6">
                <div className="flex items-center gap-4 pb-6 border-b border-white/10">
                    <div className="w-12 h-12 rounded-xl bg-white/5" />
                    <div className="flex-1 space-y-2">
                        <div className="h-5 w-3/4 rounded bg-white/5" />
                        <div className="h-3 w-1/2 rounded bg-white/5" />
                    </div>
                </div>
                <SkeletonLoader variant="line" count={5} />
            </div>
        </div>
    </div>
);

/**
 * SkeletonTable - Table skeleton for audit logs
 */
export const SkeletonTable = ({ rows = 5 }) => (
    <div className="glass-depth-1 rounded-2xl border border-white/10 overflow-hidden">
        {/* Header */}
        <div className="flex gap-4 p-4 border-b border-white/10 bg-white/5">
            <SkeletonLoader variant="line" count={1} className="w-20" />
            <SkeletonLoader variant="line" count={1} className="flex-1" />
            <SkeletonLoader variant="line" count={1} className="w-24" />
            <SkeletonLoader variant="line" count={1} className="w-16" />
        </div>
        {/* Rows */}
        {Array(rows).fill(0).map((_, i) => (
            <div key={i} className="flex gap-4 p-4 border-b border-white/5">
                <SkeletonLoader variant="line" count={1} className="w-20" />
                <SkeletonLoader variant="line" count={1} className="flex-1" />
                <SkeletonLoader variant="line" count={1} className="w-24" />
                <SkeletonLoader variant="line" count={1} className="w-16" />
            </div>
        ))}
    </div>
);

/**
 * Connection Status Indicator with skeleton fallback
 */
export const ConnectionSkeleton = () => (
    <div className="flex items-center gap-2 text-slate-500">
        <motion.div
            className="w-2 h-2 rounded-full bg-slate-500"
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 1.5, repeat: Infinity }}
        />
        <span className="text-xs font-mono uppercase">Connecting...</span>
    </div>
);

/**
 * Stale Data Indicator - Shows when data might be outdated
 */
export const StaleDataIndicator = ({ lastUpdated }) => {
    const secondsAgo = lastUpdated ? Math.floor((Date.now() - new Date(lastUpdated).getTime()) / 1000) : null;

    if (!secondsAgo || secondsAgo < 30) return null;

    return (
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-400 text-xs font-mono">
            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <span>Stale ({secondsAgo}s ago)</span>
        </div>
    );
};

export default SkeletonLoader;
