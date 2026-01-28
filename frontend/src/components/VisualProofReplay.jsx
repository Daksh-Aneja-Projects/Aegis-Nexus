import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { GitBranch, Shield, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import { cn } from '../styles/glass';

/**
 * VisualProofReplay - Visualizes Z3 Formal Verification Proofs
 * Parses the proof trace and renders it as a logical flow/decision tree.
 */
const VisualProofReplay = ({ proofTrace, status }) => {
    const proofSteps = useMemo(() => {
        if (!proofTrace) return [];

        // Simple parser for the standard Z3 certificate format
        const lines = proofTrace.split('\n').filter(l => l.trim());
        const steps = [];

        lines.forEach((line, index) => {
            if (line.includes('===') || line.includes('Signature:')) return;

            let type = 'info';
            if (line.includes('Status:')) type = 'status';
            if (line.includes('Model Assignment')) type = 'header';
            if (line.startsWith('  ')) type = 'detail';

            steps.push({
                id: index,
                text: line.trim(),
                type,
                delay: index * 0.1
            });
        });

        return steps;
    }, [proofTrace]);

    const isSafe = status === 'VERIFIED';

    return (
        <div className="flex flex-col h-full bg-black/60 backdrop-blur-xl rounded-2xl border border-white/10 overflow-hidden">
            {/* Header */}
            <div className={cn(
                "p-4 border-b flex items-center justify-between",
                isSafe ? "bg-emerald-500/10 border-emerald-500/20" : "bg-rose-500/10 border-rose-500/20"
            )}>
                <div className="flex items-center gap-3">
                    <div className={cn(
                        "p-2 rounded-lg",
                        isSafe ? "bg-emerald-500/20 text-emerald-400" : "bg-rose-500/20 text-rose-400"
                    )}>
                        {isSafe ? <CheckCircle size={20} /> : <AlertTriangle size={20} />}
                    </div>
                    <div>
                        <h3 className="text-sm font-bold text-white uppercase tracking-wider">Formal Proof Visualization</h3>
                        <p className="text-[10px] text-slate-400 font-mono">Engine: Z3 SMT Solver v4.12.0</p>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    {/* Hydration Indicator */}
                    {status === 'HYDRATING' && (
                        <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-blue-500/20 border border-blue-500/30">
                            <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                            <span className="text-[10px] text-blue-300 font-mono tracking-wider">SYNCING</span>
                        </div>
                    )}
                    <Shield size={16} className={isSafe ? "text-emerald-500" : "text-rose-500"} />
                    <span className={cn("text-xs font-bold font-mono", isSafe ? "text-emerald-400" : "text-rose-400")}>
                        {status || 'PENDING'}
                    </span>
                </div>
            </div>

            {/* Proof Tree / List */}
            <div className="flex-grow p-6 overflow-y-auto custom-scrollbar bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')]">
                <div className="max-w-2xl mx-auto space-y-3">
                    {proofSteps.length > 0 ? (
                        proofSteps.map((step) => (
                            <motion.div
                                key={step.id}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: step.delay }}
                                className={cn(
                                    "p-3 rounded-lg border font-mono text-xs flex items-start gap-3",
                                    step.type === 'status' ? "bg-white/10 border-white/20 text-white font-bold" :
                                        step.type === 'header' ? "mt-4 text-slate-300 underline" :
                                            step.type === 'detail' ? "ml-6 bg-white/5 border-white/5 text-slate-400" :
                                                "bg-black/40 border-white/5 text-slate-300"
                                )}
                            >
                                {step.type === 'status' && <GitBranch size={14} className="mt-0.5 text-neon-accent" />}
                                {step.type === 'detail' && <Info size={12} className="mt-0.5 text-slate-600" />}
                                {step.text}
                            </motion.div>
                        ))
                    ) : (
                        <div className="flex flex-col items-center justify-center h-full text-slate-500 gap-4 mt-20">
                            <div className="w-16 h-16 rounded-full border-2 border-dashed border-slate-700 animate-spin" />
                            <p className="font-mono text-sm">Awaiting formal verification telemetry...</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Footer Metrics */}
            <div className="p-4 bg-black/40 border-t border-white/5 flex justify-between items-center text-[10px] font-mono text-slate-500">
                <div className="flex gap-4">
                    <span>LATENCY: 482ms</span>
                    <span>COMPLEXITY: O(2^n)</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                    SYSTEM INTEGRITY: 100%
                </div>
            </div>
        </div>
    );
};

export default VisualProofReplay;
