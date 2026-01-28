import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useLiveSystemState } from '../hooks/useLiveSystemState';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { ShieldCheck, Activity, Lock, AlertTriangle, Clock, Play, Pause, SkipBack, SkipForward, Film } from 'lucide-react';

const RealityDashboard = () => {
  // 1. LIVE DATA CONNECTION (Replaced brittle WebSocket with Robust Polling)
  // Connects to: http://localhost:8000/api/v1/system/status
  const { data: liveData, status } = useLiveSystemState('http://localhost:8000/api/v1/system/status');
  const isConnected = status === 'LIVE';

  // 2. STATE MANAGEMENT
  const [metrics, setMetrics] = useState({
    systemEntropy: 0,
    threatLevel: 'CALIBRATING...', // Initial state
    activeNodes: 0,
    auditLog: []
  });

  const [history, setHistory] = useState([]);

  // 3. HOLOGRAPHIC AUDIT REPLAY STATE
  const [replayMode, setReplayMode] = useState(false);
  const [replayData, setReplayData] = useState([]);
  const [replayIndex, setReplayIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const replayIntervalRef = useRef(null);

  // 4. DATA INGESTION LOGIC
  useEffect(() => {
    if (liveData && !replayMode) {
      setMetrics(prev => ({ ...prev, ...liveData }));
      setHistory(prev => {
        // Keep last 60 points for 5s of data @ 60fps equivalent (actually just event based)
        const timestamp = new Date();
        const newHistory = [...prev, {
          time: timestamp.toLocaleTimeString(),
          timestamp: timestamp.getTime(),
          val: liveData.systemEntropy || 0,
          threatLevel: liveData.threatLevel,
          activeNodes: liveData.activeNodes,
          snapshot: { ...liveData }
        }];
        return newHistory.slice(-300); // Keep 5 minutes of history for replay
      });
    }
  }, [liveData, replayMode]);

  // 5. REPLAY CONTROLS
  const loadHistoricalData = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/audit/timeline?minutes=60');
      if (response.ok) {
        const data = await response.json();
        setReplayData(data.events || []);
      } else {
        // Use local history as fallback
        setReplayData(history);
      }
    } catch (err) {
      console.warn('Failed to load historical data, using local cache');
      setReplayData(history);
    }
  }, [history]);

  const enterReplayMode = useCallback(async () => {
    setIsPlaying(false);
    await loadHistoricalData();
    setReplayMode(true);
    setReplayIndex(0);
  }, [loadHistoricalData]);

  const exitReplayMode = useCallback(() => {
    setReplayMode(false);
    setIsPlaying(false);
    setReplayIndex(0);
    if (replayIntervalRef.current) {
      clearInterval(replayIntervalRef.current);
    }
  }, []);

  const togglePlayback = useCallback(() => {
    setIsPlaying(prev => !prev);
  }, []);

  // Playback loop
  useEffect(() => {
    if (isPlaying && replayMode && replayData.length > 0) {
      replayIntervalRef.current = setInterval(() => {
        setReplayIndex(prev => {
          if (prev >= replayData.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 100 / playbackSpeed);
    } else {
      if (replayIntervalRef.current) {
        clearInterval(replayIntervalRef.current);
      }
    }
    return () => {
      if (replayIntervalRef.current) {
        clearInterval(replayIntervalRef.current);
      }
    };
  }, [isPlaying, replayMode, replayData.length, playbackSpeed]);

  // Update metrics during replay
  useEffect(() => {
    if (replayMode && replayData[replayIndex]) {
      const snapshot = replayData[replayIndex].snapshot || replayData[replayIndex];
      setMetrics(prev => ({
        ...prev,
        systemEntropy: snapshot.systemEntropy || snapshot.val || prev.systemEntropy,
        threatLevel: snapshot.threatLevel || prev.threatLevel,
        activeNodes: snapshot.activeNodes || prev.activeNodes
      }));
    }
  }, [replayIndex, replayData, replayMode]);

  // 6. PREMIUM UI COMPONENTS
  const StatCard = ({ label, value, icon: Icon, color }) => (
    <div className={`
      relative p-6 rounded-2xl overflow-hidden transition-all duration-500
      ${isConnected || replayMode ? 'glass-panel hover:bg-white/5 active-glow' : 'bg-gray-900/50 grayscale opacity-50'}
    `}>
      {/* Background Gradient Mesh */}
      <div className={`absolute top-0 right-0 w-32 h-32 bg-${color}-500/10 blur-[50px] rounded-full pointer-events-none`} />

      <div className="relative flex items-center space-x-4">
        <div className={`p-3 rounded-xl ${isConnected || replayMode ? `bg-${color}-500/20` : 'bg-slate-700/20'} transition-colors duration-500`}>
          <Icon className={`w-8 h-8 ${isConnected || replayMode ? `text-${color}-400` : 'text-slate-500'}`} />
        </div>
        <div>
          <p className="text-xs text-slate-400 font-bold uppercase tracking-widest mb-1">{label}</p>
          <h3 className={`text-2xl font-mono font-bold ${isConnected || replayMode ? 'text-white text-glow' : 'text-slate-500'}`}>
            {value}
          </h3>
        </div>
      </div>
    </div>
  );

  // 7. TIME SLIDER COMPONENT
  const TimeSlider = () => {
    const dataSource = replayMode ? replayData : history;
    const currentIndex = replayMode ? replayIndex : dataSource.length - 1;
    const currentTime = dataSource[currentIndex]?.time || '--:--:--';

    return (
      <div className="glass-panel p-6 rounded-2xl">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <Film className="w-5 h-5 text-purple-400" />
            Holographic Audit Replay
            {replayMode && (
              <span className="ml-2 px-2 py-0.5 text-xs bg-purple-500/20 text-purple-300 rounded-full animate-pulse">
                REPLAY MODE
              </span>
            )}
          </h3>

          <div className="flex items-center gap-2">
            {!replayMode ? (
              <button
                onClick={enterReplayMode}
                className="px-4 py-2 bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 rounded-lg transition-all flex items-center gap-2"
              >
                <Clock className="w-4 h-4" />
                Enter Replay
              </button>
            ) : (
              <button
                onClick={exitReplayMode}
                className="px-4 py-2 bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 rounded-lg transition-all"
              >
                Exit Replay
              </button>
            )}
          </div>
        </div>

        {replayMode && dataSource.length > 0 && (
          <>
            {/* Playback Controls */}
            <div className="flex items-center justify-center gap-4 mb-6">
              <button
                onClick={() => setReplayIndex(Math.max(0, replayIndex - 10))}
                className="p-2 bg-slate-700/50 hover:bg-slate-600/50 rounded-lg transition-all"
              >
                <SkipBack className="w-5 h-5 text-slate-300" />
              </button>

              <button
                onClick={togglePlayback}
                className={`p-4 rounded-full transition-all ${isPlaying
                    ? 'bg-amber-500/20 hover:bg-amber-500/30 text-amber-300'
                    : 'bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300'
                  }`}
              >
                {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6" />}
              </button>

              <button
                onClick={() => setReplayIndex(Math.min(dataSource.length - 1, replayIndex + 10))}
                className="p-2 bg-slate-700/50 hover:bg-slate-600/50 rounded-lg transition-all"
              >
                <SkipForward className="w-5 h-5 text-slate-300" />
              </button>

              {/* Speed Control */}
              <div className="ml-4 flex items-center gap-2">
                <span className="text-xs text-slate-400">Speed:</span>
                {[0.5, 1, 2, 4].map(speed => (
                  <button
                    key={speed}
                    onClick={() => setPlaybackSpeed(speed)}
                    className={`px-2 py-1 text-xs rounded transition-all ${playbackSpeed === speed
                        ? 'bg-cyan-500/30 text-cyan-300'
                        : 'bg-slate-700/30 text-slate-400 hover:text-slate-200'
                      }`}
                  >
                    {speed}x
                  </button>
                ))}
              </div>
            </div>

            {/* Time Slider */}
            <div className="relative">
              <input
                type="range"
                min={0}
                max={dataSource.length - 1}
                value={replayIndex}
                onChange={(e) => setReplayIndex(parseInt(e.target.value, 10))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
              />

              <div className="flex justify-between mt-2 text-xs text-slate-500">
                <span>{dataSource[0]?.time || '--:--:--'}</span>
                <span className="text-purple-400 font-mono text-sm">{currentTime}</span>
                <span>{dataSource[dataSource.length - 1]?.time || '--:--:--'}</span>
              </div>
            </div>

            {/* Current State Card */}
            <div className="mt-4 p-4 bg-slate-800/50 rounded-xl border border-slate-700">
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-slate-500">Timestamp</span>
                  <p className="text-white font-mono">{currentTime}</p>
                </div>
                <div>
                  <span className="text-slate-500">Entropy</span>
                  <p className="text-cyan-400 font-mono">{(metrics.systemEntropy || 0).toFixed(2)}%</p>
                </div>
                <div>
                  <span className="text-slate-500">Threat Level</span>
                  <p className="text-amber-400 font-mono">{metrics.threatLevel}</p>
                </div>
              </div>
            </div>
          </>
        )}

        {!replayMode && (
          <p className="text-slate-500 text-sm text-center py-4">
            Click "Enter Replay" to access the Black Box Recorder and replay historical decisions.
          </p>
        )}
      </div>
    );
  };

  return (
    <div className={`p-8 space-y-8 w-full max-w-7xl mx-auto transition-all duration-1000 ${isConnected || replayMode ? '' : 'filter grayscale contrast-125'}`}>
      {/* Header with Connection Status */}
      <div className="flex justify-between items-center">
        <h1 className="text-4xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-600">
          Aegis Nexus <span className="text-white text-lg font-light opacity-70">| Sentinel Dashboard</span>
        </h1>
        <div className={`flex items-center gap-2 px-4 py-1 rounded-full border ${replayMode
            ? 'border-purple-500/50 bg-purple-500/10'
            : isConnected
              ? 'border-emerald-500/50 bg-emerald-500/10'
              : 'border-rose-500/50 bg-rose-500/10'
          }`}>
          <div className={`w-2 h-2 rounded-full ${replayMode
              ? 'bg-purple-400 animate-pulse'
              : isConnected
                ? 'bg-emerald-400 animate-pulse'
                : 'bg-rose-400'
            }`} />
          <span className="text-xs font-mono font-bold tracking-widest text-slate-300">
            {replayMode ? 'REPLAY MODE' : isConnected ? 'SYSTEM ONLINE' : 'OFFLINE'}
          </span>
        </div>
      </div>

      {/* Metric Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <StatCard label="System Entropy" value={`${(metrics.systemEntropy || 0).toFixed(2)}%`} icon={Activity} color="cyan" />
        <StatCard label="Threat Level" value={metrics.threatLevel || 'UNKNOWN'} icon={AlertTriangle} color="amber" />
        <StatCard label="Active Nodes" value={metrics.activeNodes || 0} icon={ShieldCheck} color="emerald" />
        <StatCard label="PQC Ledger Height" value="#8904" icon={Lock} color="purple" />
      </div>

      {/* Live Entropy Chart with Replay Marker */}
      <div className="glass-panel p-6 rounded-2xl h-96">
        <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
          <Activity className="w-5 h-5 text-cyan-400" /> Real-Time Cognitive Stability
          {replayMode && <span className="text-xs text-purple-400 ml-2">(Historical View)</span>}
        </h3>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={replayMode ? replayData : history}>
            <defs>
              <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="time" stroke="#475569" fontSize={12} tickLine={false} axisLine={false} />
            <YAxis stroke="#475569" fontSize={12} tickLine={false} axisLine={false} domain={[0, 100]} />
            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px' }} />
            <Line type="monotone" dataKey="val" stroke="#22d3ee" strokeWidth={3} dot={false} activeDot={{ r: 8 }} />
            {replayMode && replayData[replayIndex] && (
              <ReferenceLine
                x={replayData[replayIndex].time}
                stroke="#a855f7"
                strokeWidth={2}
                label={{ value: '▼', position: 'top', fill: '#a855f7' }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Holographic Audit Replay Section */}
      <TimeSlider />
    </div>
  );
};

export default RealityDashboard;