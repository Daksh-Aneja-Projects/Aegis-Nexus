import React, { useState, useEffect } from 'react';
import { Settings, Save, RefreshCw, Lock, AlertTriangle } from 'lucide-react';
import { cn } from '../styles/glass';

const ConfigManager = ({ isOpen, onClose }) => {
    const [config, setConfig] = useState({});
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [adminKey, setAdminKey] = useState("");
    const [pendingChanges, setPendingChanges] = useState({});

    // Fetch initial config
    const fetchConfig = async () => {
        if (!adminKey) return;
        setLoading(true);
        try {
            const res = await fetch('/api/v1/system/config', {
                headers: { 'X-Admin-Key': adminKey }
            });
            if (!res.ok) throw new Error("Auth Failed");
            const data = await res.json();
            setConfig(data);
            setError(null);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleUpdate = async () => {
        if (!Object.keys(pendingChanges).length) return;
        setLoading(true);
        try {
            const res = await fetch('/api/v1/system/update-config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Admin-Key': adminKey
                },
                body: JSON.stringify({ updates: pendingChanges })
            });
            const data = await res.json();
            if (data.status === 'success') {
                setConfig(prev => ({ ...prev, ...pendingChanges }));
                setPendingChanges({});
                alert("Configuration Updated!");
            } else {
                throw new Error(data.message || "Update failed");
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleChange = (key, value) => {
        setPendingChanges(prev => ({
            ...prev,
            [key]: value
        }));
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
            <div className="w-full max-w-lg bg-black/80 border border-white/10 rounded-2xl glass-depth-2 overflow-hidden flex flex-col max-h-[80vh]">

                {/* Header */}
                <div className="p-6 border-b border-white/10 flex justify-between items-center">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-neon-accent/10 rounded-lg">
                            <Settings className="text-neon-accent" size={20} />
                        </div>
                        <h2 className="text-lg font-bold text-white tracking-wide">System Configuration</h2>
                    </div>
                    <button onClick={onClose} className="text-slate-400 hover:text-white">×</button>
                </div>

                {/* Content */}
                <div className="p-6 overflow-y-auto custom-scrollbar space-y-6">

                    {/* Admin Key Input */}
                    {!config.MAX_DEBATE_ROUNDS && (
                        <div className="space-y-2">
                            <label className="text-xs text-slate-400 font-mono uppercase">Admin Access Key</label>
                            <div className="flex gap-2">
                                <input
                                    type="password"
                                    value={adminKey}
                                    onChange={e => setAdminKey(e.target.value)}
                                    className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white font-mono text-sm focus:border-neon-accent outline-none"
                                    placeholder="Enter Admin Key..."
                                />
                                <button
                                    onClick={fetchConfig}
                                    className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-white"
                                >
                                    <Lock size={16} />
                                </button>
                            </div>
                            {error && <p className="text-rose-500 text-xs flex items-center gap-1"><AlertTriangle size={12} /> {error}</p>}
                        </div>
                    )}

                    {/* Config Form */}
                    {config.MAX_DEBATE_ROUNDS && (
                        <div className="space-y-4">
                            {Object.entries(config).map(([key, val]) => (
                                <div key={key} className="space-y-1">
                                    <label className="text-[10px] text-slate-500 font-mono uppercase">{key.replace(/_/g, ' ')}</label>
                                    <input
                                        value={pendingChanges[key] !== undefined ? pendingChanges[key] : val}
                                        onChange={e => handleChange(key, e.target.value)}
                                        className={cn(
                                            "w-full bg-white/5 border rounded-lg px-3 py-2 text-white font-mono text-sm outline-none transition-colors",
                                            pendingChanges[key] !== undefined ? "border-neon-accent/50 bg-neon-accent/5" : "border-white/10"
                                        )}
                                    />
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="p-6 border-t border-white/10 flex justify-end gap-3 bg-white/5">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-slate-400 hover:text-white text-sm"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleUpdate}
                        disabled={loading || !Object.keys(pendingChanges).length}
                        className="px-6 py-2 bg-neon-accent text-black font-bold rounded-lg hover:bg-neon-accent/90 disabled:opacity-50 flex items-center gap-2"
                    >
                        {loading ? <RefreshCw className="animate-spin" size={16} /> : <Save size={16} />}
                        Save Changes
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ConfigManager;
