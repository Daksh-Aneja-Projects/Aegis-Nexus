/**
 * Glass Fortress Design System
 * Reusable style constants for the premium Aegis Nexus UI
 */

import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Utility function to merge Tailwind classes without conflicts
 */
export function cn(...inputs) {
    return twMerge(clsx(inputs));
}

/* ===== Glass Styles ===== */
export const glassPanel = "bg-slate-900/60 backdrop-blur-xl border border-slate-700/50 rounded-2xl shadow-glass";
export const glassCard = "bg-white/5 backdrop-blur-lg border border-white/10 rounded-xl shadow-glass-sm";
export const glassButton = "bg-slate-800/60 hover:bg-slate-700/60 backdrop-blur-sm border border-slate-600/50 rounded-lg px-4 py-2 font-medium transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]";

/* ===== Status Colors ===== */
export const statusColors = {
    safe: {
        bg: 'bg-emerald-500/20',
        text: 'text-emerald-400',
        border: 'border-emerald-500/30',
        glow: 'shadow-status-safe',
    },
    warning: {
        bg: 'bg-amber-500/20',
        text: 'text-amber-400',
        border: 'border-amber-500/30',
        glow: 'shadow-status-warning',
    },
    danger: {
        bg: 'bg-rose-500/20',
        text: 'text-rose-400',
        border: 'border-rose-500/30',
        glow: 'shadow-status-danger',
    },
    info: {
        bg: 'bg-sky-500/20',
        text: 'text-sky-400',
        border: 'border-sky-500/30',
        glow: '',
    },
    neutral: {
        bg: 'bg-slate-500/20',
        text: 'text-slate-400',
        border: 'border-slate-500/30',
        glow: '',
    },
};

/**
 * Get status styling based on value thresholds
 * @param {number} value - Current value (0-1 for percentages)
 * @param {Object} thresholds - { warning: number, danger: number }
 */
export function getStatusFromValue(value, thresholds = { warning: 0.6, danger: 0.8 }) {
    if (value >= thresholds.danger) return 'danger';
    if (value >= thresholds.warning) return 'warning';
    return 'safe';
}

/* ===== Animation Variants (Framer Motion) ===== */
export const motionVariants = {
    // Fade in with slight upward movement
    fadeInUp: {
        initial: { opacity: 0, y: 20 },
        animate: { opacity: 1, y: 0 },
        exit: { opacity: 0, y: -10 },
    },

    // Slide in from left with spring physics
    slideIn: {
        initial: { opacity: 0, x: -20 },
        animate: {
            opacity: 1,
            x: 0,
            transition: { type: 'spring', stiffness: 300, damping: 24 }
        },
        exit: { opacity: 0, x: 20 },
    },

    // Scale in for cards/modals
    scaleIn: {
        initial: { opacity: 0, scale: 0.95 },
        animate: {
            opacity: 1,
            scale: 1,
            transition: { type: 'spring', stiffness: 400, damping: 25 }
        },
        exit: { opacity: 0, scale: 0.95 },
    },

    // Stagger children animations
    staggerContainer: {
        animate: {
            transition: { staggerChildren: 0.05 }
        }
    },

    // Micro-interaction for buttons
    buttonTap: {
        whileTap: { scale: 0.98 },
        whileHover: { scale: 1.02 },
    },

    // Breathing/pulse effect for status indicators
    pulse: {
        animate: {
            scale: [1, 1.05, 1],
            opacity: [1, 0.8, 1],
            transition: { duration: 2, repeat: Infinity, ease: 'easeInOut' }
        }
    },
};

/* ===== Typography ===== */
export const typography = {
    h1: 'text-4xl font-bold tracking-tight',
    h2: 'text-2xl font-semibold tracking-tight',
    h3: 'text-lg font-semibold',
    body: 'text-sm text-slate-300',
    caption: 'text-xs text-slate-400',
    mono: 'font-mono text-sm',
    label: 'text-xs font-medium uppercase tracking-wider text-slate-400',
};
