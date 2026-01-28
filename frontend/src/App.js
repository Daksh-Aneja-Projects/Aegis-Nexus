/**
 * Aegis Nexus - Glass Fortress Command Center
 * Premium App Shell with Bento Grid Layout
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Shield,
  Activity,
  FileText,
  Zap,
  Menu,
  X,
  Wifi,
  WifiOff
} from 'lucide-react';
import './index.css';
import WarRoomViz from './components/WarRoomViz';
import RealityDashboard from './components/RealityDashboard';
import VerificationBadge from './components/VerificationBadge';
import AuditLogViewer from './components/AuditLogViewer';
import { cn, glassPanel, typography } from './styles/glass';
import { useWebSocket, ConnectionState } from './hooks/useWebSocket';

// ❌ REMOVED: useOptimisticState (Dangerous for safety apps)

/**
 * Navigation tab item
 */
const NavTab = ({ icon: Icon, label, isActive, onClick }) => (
  <motion.button
    onClick={onClick}
    whileHover={{ scale: 1.02 }}
    whileTap={{ scale: 0.98 }}
    className={cn(
      'flex items-center gap-3 px-5 py-3 rounded-xl text-sm font-medium',
      'transition-all duration-200',
      isActive
        ? 'bg-gradient-to-r from-emerald-600/80 to-emerald-500/80 text-white shadow-lg shadow-emerald-500/20'
        : 'bg-slate-800/40 hover:bg-slate-700/50 text-slate-400 hover:text-white border border-slate-700/30'
    )}
  >
    <Icon size={18} strokeWidth={1.5} />
    <span className="hidden sm:inline">{label}</span>
  </motion.button>
);

/**
 * Connection status indicator
 */
const ConnectionStatus = ({ state }) => {
  const isConnected = state === ConnectionState.CONNECTED;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={cn(
        'flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium',
        isConnected
          ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
          : 'bg-slate-700/50 text-slate-400 border border-slate-600/30'
      )}
    >
      {isConnected ? (
        <>
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <Wifi size={12} />
          <span className="hidden md:inline">Connected</span>
        </>
      ) : (
        <>
          <span className="w-2 h-2 rounded-full bg-slate-500"></span>
          <WifiOff size={12} />
          <span className="hidden md:inline">Offline</span>
        </>
      )}
    </motion.div>
  );
};

/**
 * Animated background gradient
 */
const AnimatedBackground = () => (
  <div className="fixed inset-0 -z-10 overflow-hidden">
    {/* Base gradient */}
    <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950" />

    {/* Animated orbs */}
    <motion.div
      animate={{
        x: [0, 100, 0],
        y: [0, 50, 0],
        scale: [1, 1.2, 1],
      }}
      transition={{ duration: 20, repeat: Infinity, ease: 'easeInOut' }}
      className="absolute top-1/4 left-1/4 w-96 h-96 bg-emerald-500/5 rounded-full blur-[100px]"
    />
    <motion.div
      animate={{
        x: [0, -50, 0],
        y: [0, 100, 0],
        scale: [1, 0.8, 1],
      }}
      transition={{ duration: 25, repeat: Infinity, ease: 'easeInOut' }}
      className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-sky-500/5 rounded-full blur-[100px]"
    />
    <motion.div
      animate={{
        x: [0, 80, 0],
        y: [0, -60, 0],
      }}
      transition={{ duration: 30, repeat: Infinity, ease: 'easeInOut' }}
      className="absolute top-1/2 right-1/3 w-64 h-64 bg-purple-500/5 rounded-full blur-[80px]"
    />

    {/* Grid overlay */}
    <div
      className="absolute inset-0 opacity-[0.02]"
      style={{
        backgroundImage: `
          linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)
        `,
        backgroundSize: '50px 50px'
      }}
    />
  </div>
);

function App() {
  const [activeTab, setActiveTab] = useState('warroom');

  // ✅ TRUTH: State is only updated when confirmed by the backend
  const [systemState, setSystemState] = useState({
    status: 'idle', // idle, verifying, safe, unsafe
    confidence: 0.0,
    traceId: null,
    proof: null
  });

  const { connectionState, isConnected, sendMessage } = useWebSocket({
    onMessage: (event) => {
      // Logic for handling verification updates
      // The hook might pass parsed JSON or the event directly depending on implementation.
      // Assuming hook passes the parsed data object based on 'useWebSocket' review.
      const data = event;

      if (data.type === 'verification_update') {
        // Real-time update from Redis Pub/Sub
        setSystemState(prev => ({
          ...prev,
          status: data.payload.status === 'VERIFIED' ? 'safe' : 'unsafe',
          confidence: data.payload.confidence || 1.0,
          proof: data.payload.proof
        }));
      }
    }
  });

  const handleVerificationSubmit = async (promptData) => {
    // 1. Set state to "Pending" (Yellow) - NOT Optimistic Green
    setSystemState(prev => ({ ...prev, status: 'verifying' }));

    try {
      const res = await fetch('/api/v1/submit/prompt', {
        method: 'POST',
        body: JSON.stringify(promptData),
        headers: { 'Content-Type': 'application/json' }
      });
      const data = await res.json();
      // Server returns a trace_id immediately
      setSystemState(prev => ({ ...prev, traceId: data.trace_id }));
    } catch (e) {
      setSystemState(prev => ({ ...prev, status: 'error' }));
    }
  };

  const tabs = [
    { id: 'warroom', label: 'War Room', icon: Shield },
    { id: 'reality', label: 'Reality Dashboard', icon: Activity },
    { id: 'audit', label: 'Audit Logs', icon: FileText },
  ];

  return (
    <div className="min-h-screen text-slate-100">
      <AnimatedBackground />

      {/* Header */}
      <header className={cn(
        'sticky top-0 z-50',
        'backdrop-blur-xl bg-slate-950/80 border-b border-slate-800/50'
      )}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <motion.div
                initial={{ rotate: -10 }}
                animate={{ rotate: 0 }}
                className="p-2 bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-xl shadow-lg shadow-emerald-500/20"
              >
                <Zap size={20} className="text-white" />
              </motion.div>
              <div>
                <h1 className="text-lg font-bold text-white tracking-tight">
                  Aegis Nexus
                </h1>
                <p className="text-xs text-slate-500 hidden sm:block">
                  Sentinel Executive Layer
                </p>
              </div>
            </div>

            {/* Navigation */}
            <nav className="flex items-center gap-2">
              {tabs.map((tab) => (
                <NavTab
                  key={tab.id}
                  icon={tab.icon}
                  label={tab.label}
                  isActive={activeTab === tab.id}
                  onClick={() => setActiveTab(tab.id)}
                />
              ))}
            </nav>

            {/* Connection Status */}
            <ConnectionStatus state={connectionState} />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <AnimatePresence mode="wait">
          {activeTab === 'warroom' && (
            <motion.div
              key="warroom"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              {/* Verification Badge reflects ACTUAL status, not hopeful status */}
              <div className="mb-6">
                <VerificationBadge
                  status={systemState.status}
                  confidence={systemState.confidence}
                  requestId={systemState.traceId}
                />
              </div>

              {/* War Room Visualization */}
              <div className={cn(glassPanel, 'overflow-hidden')}>
                <WarRoomViz
                  status={systemState.status}
                  onSubmit={handleVerificationSubmit}
                />
              </div>
            </motion.div>
          )}

          {activeTab === 'reality' && (
            <motion.div
              key="reality"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className={cn(glassPanel, 'overflow-hidden')}
            >
              <RealityDashboard />
            </motion.div>
          )}

          {activeTab === 'audit' && (
            <motion.div
              key="audit"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className={cn(glassPanel, 'overflow-hidden')}
            >
              <AuditLogViewer requestId={systemState.traceId} />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/50 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-slate-500">
            <p className="text-center sm:text-left">
              <span className="text-slate-400 font-medium">Trust is no longer a feeling.</span>{' '}
              It is a mathematical proof.
            </p>
            <p className="font-mono text-xs">
              Aegis Nexus v1.0.0 • Sentinel Executive Layer
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;