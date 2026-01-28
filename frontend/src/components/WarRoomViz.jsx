/**
 * WarRoomViz - Premium AI Governance Visualization
 * "Obsidian Glass" design with Force-Directed Graph physics engine
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Sparkles,
  Shield,
  Sword,
  Scale,
  Send,
  AlertCircle,
  CheckCircle2,
  Clock,
  Zap,
  Lock,
  Search,
  Activity
} from 'lucide-react';
import { cn, typography } from '../styles/glass';
import { useOptimisticSubmit } from '../hooks/useOptimisticState';
import VisualProofReplay from './VisualProofReplay';

// --- Assets & Icons ---
const NODE_ICONS = {
  actor: Sparkles,
  judge: Scale,
  adversary: Sword,
  input: Search,
  hub: Shield,
};

const NODE_COLORS = {
  actor: '#38bdf8', // sky-400
  judge: '#fbbf24', // amber-400
  adversary: '#f43f5e', // rose-400
  input: '#a78bfa', // violet-400
  hub: '#10b981', // emerald-500
};

// --- Canvas Graph Component ---
const GraphCanvas = React.memo(({ nodes, width, height }) => {
  const canvasRef = useRef(null);
  const workerRef = useRef(null);
  const [displayNodes, setDisplayNodes] = useState([]);

  useEffect(() => {
    // Level 5 Optimization: Offload physics to Web Worker
    workerRef.current = new Worker(new URL('../workers/graphWorker.js', import.meta.url));

    workerRef.current.postMessage({
      type: 'INIT',
      payload: { nodes, width, height }
    });

    workerRef.current.onmessage = (e) => {
      if (e.data.type === 'TICK_RESULT') {
        setDisplayNodes(e.data.nodes);
      }
    };

    const ticker = setInterval(() => {
      workerRef.current.postMessage({ type: 'TICK' });
    }, 16); // ~60fps

    return () => {
      clearInterval(ticker);
      workerRef.current.terminate();
    };
  }, [width, height]); // Only re-init if dimensions change

  useEffect(() => {
    if (workerRef.current) {
      workerRef.current.postMessage({ type: 'UPDATE_NODES', payload: nodes });
    }
  }, [nodes]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || displayNodes.length === 0) return;
    const ctx = canvas.getContext('2d');

    // Draw Step
    ctx.clearRect(0, 0, width, height);

    // Draw Connections
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    displayNodes.forEach(node => {
      if (node.role !== 'hub') {
        const hub = displayNodes.find(n => n.role === 'hub') || { x: width / 2, y: height / 2 };
        ctx.moveTo(node.x, node.y);
        ctx.lineTo(hub.x, hub.y);
      }
    });
    ctx.stroke();

    // Draw Nodes
    displayNodes.forEach(node => {
      const color = NODE_COLORS[node.role] || '#fff';

      // Glow
      const gradient = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, 15);
      gradient.addColorStop(0, `${color}33`);
      gradient.addColorStop(1, 'transparent');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(node.x, node.y, 15, 0, Math.PI * 2);
      ctx.fill();

      // Core
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(node.x, node.y, 4, 0, Math.PI * 2);
      ctx.fill();

      // Label
      ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.font = '8px monospace';
      ctx.textAlign = 'center';
      ctx.fillText((node.name || node.id).toUpperCase(), node.x, node.y + 15);
    });
  }, [displayNodes, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="absolute inset-0 w-full h-full pointer-events-none"
    />
  );
});


// --- Components ---

// --- Components ---

const AgentCard = ({ agent, isActive }) => {
  const Icon = NODE_ICONS[agent.role] || Activity;
  const color = NODE_COLORS[agent.role] || '#fff';

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={cn(
        "relative p-4 rounded-2xl border transition-all duration-300 overflow-hidden group",
        isActive ? "bg-white/10 border-white/20 shadow-[0_0_30px_rgba(255,255,255,0.1)]" : "bg-black/40 border-white/5 hover:bg-white/5 hover:border-white/10"
      )}
      style={{ borderColor: isActive ? color : undefined }}
    >
      {/* Background Glow */}
      <div
        className="absolute inset-0 opacity-0 group-hover:opacity-20 transition-opacity duration-500"
        style={{ background: `radial-gradient(circle at center, ${color}, transparent 70%)` }}
      />

      <div className="relative flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div className="p-2 rounded-lg bg-black/40 border border-white/10">
            <Icon size={18} style={{ color }} />
          </div>
          <div className={cn(
            "text-[10px] font-mono px-2 py-1 rounded-full border",
            agent.status === 'active' ? "bg-emerald-500/10 border-emerald-500/50 text-emerald-400" : "bg-slate-500/10 border-slate-500/50 text-slate-400"
          )}>
            {agent.role.toUpperCase()}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-bold text-white">{agent.name}</h3>
          <p className="text-[10px] text-slate-400 font-mono mt-1 line-clamp-2">
            {agent.description || "Awaiting categorization..."}
          </p>
        </div>

        {/* Dynamic Telemetry */}
        <div className="pt-3 mt-1 border-t border-white/5 grid grid-cols-2 gap-2 text-[10px] font-mono text-slate-500">
          <div>
            <span className="block text-slate-600">STATE</span>
            <span className="text-slate-300">{agent.state || "IDLE"}</span>
          </div>
          <div>
            <span className="block text-slate-600">CONFIDENCE</span>
            <span className="text-slate-300">{(agent.confidence * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

// --- Main WarRoomViz ---

const WarRoomViz = ({ verificationStatus, onSubmit, isConnected }) => {
  const [prompt, setPrompt] = useState("");
  const [activeTab, setActiveTab] = useState('grid'); // 'grid' | 'graph'
  const containerRef = useRef(null);

  // Replay State
  const [replayMode, setReplayMode] = useState(false);
  const [replayData, setReplayData] = useState(null);
  const [replayIndex, setReplayIndex] = useState(0);

  // Hydration Logic: Fetch missing logs on reconnect
  useEffect(() => {
    if (isConnected) {
      console.log("🔗 Reconnected! Hydrating War Room state...");
      fetch("/api/v1/audit_trail/logs?limit=50")
        .then(res => res.json())
        .then(data => {
          if (data.success && data.data.entries) {
            // In a real app, we would merge these with current state
            // For the viz, we just ensure the 'agents' etc are refreshed
            console.log(`✅ Hydrated ${data.data.entries.length} audit entries.`);
          }
        })
        .catch(err => console.error("Hydration failed", err));
    }
  }, [isConnected]);

  // Handle Replay File Load
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const data = JSON.parse(event.target.result);
          setReplayData(data.steps || []); // Assume .replay file has 'steps' array
          setReplayMode(true);
          setReplayIndex(0);
        } catch (err) {
          console.error("Failed to parse replay file", err);
        }
      };
      reader.readAsText(file);
    }
  };

  // Derived state for viz
  const currentStatus = replayMode && replayData
    ? replayData[replayIndex]
    : verificationStatus;

  // Extract Agents for Grid
  // Prioritize new 'agents' array from backend, fallback to legacy nodes
  const agents = currentStatus?.phase_1_audit?.agents || [
    { id: 'actor', role: 'actor', name: 'Actor Agent', confidence: 0.8 },
    { id: 'adversary', role: 'adversary', name: 'Adversary', confidence: 0.9 },
    { id: 'judge', role: 'judge', name: 'Supreme Judge', confidence: 0.95 }
  ];

  const { isPending, submit } = useOptimisticSubmit({
    onSubmit // Assume passed from parent
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    submit({ prompt, priority: 1, agent_count: 5 }); // Default to 5 agents for demo
    setPrompt("");
  };

  return (
    <div className="h-[800px] flex flex-col lg:flex-row gap-6 p-6">

      {/* 2/3 Main Visualization Area */}
      <div
        className="flex-grow lg:w-2/3 relative rounded-3xl overflow-hidden glass-depth-1 border border-white/10 flex flex-col"
        ref={containerRef}
      >
        {/* Header / Tabs */}
        <div className="absolute top-0 left-0 right-0 p-6 flex justify-between items-start z-10">
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
              <Shield size={20} className="text-neon-accent" />
              WAR ROOM
            </h2>
            <p className="text-xs text-slate-400 font-mono mt-1">
              ACTIVE AGENTS: {agents.length} | NETWORK STATUS: SECURE |
              <span className={cn(
                "ml-2 px-2 py-0.5 rounded border font-bold",
                currentStatus?.defcon_level === 3 ? "bg-orange-500/20 border-orange-500 text-orange-400 animate-pulse" : "bg-emerald-500/10 border-emerald-500/50 text-emerald-400"
              )}>
                DEFCON {currentStatus?.defcon_level || 5}
              </span>
            </p>
          </div>
          <div className="flex bg-black/40 rounded-lg p-1 border border-white/10">
            <button
              onClick={() => setActiveTab('grid')}
              className={cn("px-3 py-1 rounded-md text-xs font-medium transition-all", activeTab === 'grid' ? "bg-white/10 text-white" : "text-slate-500 hover:text-white")}
            >
              GRID
            </button>
            <button
              onClick={() => setActiveTab('graph')}
              className={cn("px-3 py-1 rounded-md text-xs font-medium transition-all", activeTab === 'graph' ? "bg-white/10 text-white" : "text-slate-500 hover:text-white")}
            >
              GRAPH
            </button>
          </div>
        </div>

        {/* Dynamic Content Area */}
        <div className="flex-grow p-6 pt-24 overflow-y-auto custom-scrollbar">
          {activeTab === 'grid' ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <AnimatePresence>
                {agents.map((agent, idx) => (
                  <AgentCard
                    key={agent.agent_id || agent.id || idx}
                    agent={{
                      ...agent,
                      role: agent.role || (agent.agent_id?.includes('actor') ? 'actor' : agent.agent_id?.includes('judge') ? 'judge' : 'adversary'),
                      name: agent.profile || agent.name || "Unknown Agent",
                      confidence: agent.confidence_level || agent.confidence || 0.5
                    }}
                    isActive={true}
                  />
                ))}
              </AnimatePresence>
            </div>
          ) : (
            <div className="relative w-full h-full min-h-[400px]">
              <GraphCanvas
                nodes={agents.map((a, i) => ({ ...a, id: a.agent_id || i, role: a.role || 'actor' }))}
                width={800}
                height={500}
              />
              <div className="relative z-10 pointer-events-auto h-full">
                <VisualProofReplay
                  proofTrace={currentStatus?.proof}
                  status={currentStatus?.status}
                />
              </div>
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="p-6 bg-black/40 border-t border-white/5 backdrop-blur-md">
          <form onSubmit={handleSubmit} className="flex gap-4">
            <input
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              placeholder="Initialize System Audit..."
              className="flex-1 bg-black/50 backdrop-blur-md text-white px-6 py-4 rounded-xl border border-white/10 focus:border-neon-accent focus:outline-none transition-all font-mono text-sm"
            />
            <button
              type="submit"
              disabled={isPending || replayMode}
              className="px-8 py-4 bg-neon-accent/10 border border-neon-accent text-neon-accent font-bold rounded-xl hover:bg-neon-accent/20 transition-all uppercase tracking-wider text-xs flex items-center gap-2 disabled:opacity-50"
            >
              {isPending ? <Activity className="animate-spin" size={16} /> : <Zap size={16} />}
              {isPending ? 'Processing' : 'Execute'}
            </button>
          </form>
        </div>

        {/* Holographic Replay Overlay */}
        {replayMode && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute bottom-32 left-1/2 -translate-x-1/2 w-2/3 bg-black/60 backdrop-blur-xl p-6 rounded-2xl border border-neon-accent/30 shadow-[0_0_50px_rgba(var(--neon-accent-rgb),0.2)] z-30"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-full bg-neon-accent/20 border border-neon-accent/50 text-neon-accent">
                  <Clock size={18} className="animate-pulse" />
                </div>
                <div>
                  <h4 className="text-white text-xs font-bold uppercase tracking-wider">Holographic Replay Mode</h4>
                  <p className="text-[10px] text-slate-400 font-mono">Temporal Scrubbing: Step {replayIndex + 1} of {replayData?.length}</p>
                </div>
              </div>
              <button
                onClick={() => setReplayMode(false)}
                className="text-slate-500 hover:text-white transition-colors"
              >
                EXIT
              </button>
            </div>

            <div className="relative h-2 bg-white/5 rounded-full overflow-hidden">
              <motion.div
                className="absolute inset-y-0 left-0 bg-neon-accent shadow-[0_0_15px_rgba(var(--neon-accent-rgb),0.5)]"
                style={{ width: `${replayData?.length > 1 ? ((replayIndex) / (replayData.length - 1)) * 100 : 100}%` }}
              />
              <input
                type="range"
                min="0"
                max={(replayData?.length || 1) - 1}
                step="1"
                value={replayIndex}
                onChange={e => setReplayIndex(Number(e.target.value))}
                className="absolute inset-0 w-full opacity-0 cursor-pointer z-10"
              />
            </div>

            {/* Playback Controls (Level 5 UX) */}
            <div className="flex justify-center gap-4 mt-4">
              <button
                onClick={() => setReplayIndex(prev => Math.max(0, prev - 1))}
                className="text-slate-400 hover:text-white transition-colors"
              >
                <Clock size={16} className="rotate-180" />
              </button>
              <button
                onClick={() => {
                  const interval = setInterval(() => {
                    setReplayIndex(prev => {
                      if (prev >= (replayData?.length || 1) - 1) {
                        clearInterval(interval);
                        return prev;
                      }
                      return prev + 1;
                    });
                  }, 500);
                }}
                className="p-2 rounded-full bg-neon-accent/20 border border-neon-accent/50 text-neon-accent hover:bg-neon-accent/40"
              >
                <Zap size={16} />
              </button>
              <button
                onClick={() => setReplayIndex(prev => Math.min((replayData?.length || 1) - 1, prev + 1))}
                className="text-slate-400 hover:text-white transition-colors"
              >
                <Clock size={16} />
              </button>
            </div>
          </motion.div>
        )}
      </div>

      {/* 1/3 Detail Panel */}
      <div className="lg:w-1/3 glass-depth-2 p-6 rounded-3xl border border-white/5 relative overflow-hidden flex flex-col">
        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-6 border-b border-white/10 pb-4">System Telemetry</h3>

        <div className="space-y-4 flex-grow overflow-y-auto">
          {/* Detailed logs or selected agent info */}
          {/* Fallback to simple list for now */}
          {agents.map((agent, idx) => (
            <div key={idx} className="bg-white/5 p-3 rounded-lg border border-white/5 flex justify-between items-center">
              <span className="text-xs text-slate-300 font-mono">{agent.name || agent.id}</span>
              <span className={cn("w-2 h-2 rounded-full", agent.status === 'error' ? "bg-rose-500" : "bg-emerald-500")} />
            </div>
          ))}
        </div>

        <div className="mt-auto pt-6 border-t border-white/5">
          <div className="flex justify-between text-xs text-slate-500 font-mono">
            <span>LATENCY: 12ms</span>
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
              ONLINE
            </span>
          </div>
        </div>
      </div>

    </div>
  );
};

export default WarRoomViz;
