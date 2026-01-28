/**
 * AuditLogViewer - Code Editor Styled Infinite Scroll
 * Premium audit trail with JetBrains Mono styling and spring animations
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FixedSizeList as List } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';
import {
  Search,
  Filter,
  Download,
  RefreshCw,
  ChevronDown,
  Check,
  X,
  Clock,
  Shield,
  Terminal,
  Hash,
  Calendar,
  Layers
} from 'lucide-react';
import { cn, glassPanel, glassCard, typography, motionVariants } from '../styles/glass';
import { useAuditStream } from '../hooks/useWebSocket';

/**
 * Log entry type badge
 */
const EventTypeBadge = ({ type }) => {
  const config = {
    prompt_submission_received: { bg: 'bg-sky-500/20', text: 'text-sky-400', icon: Terminal },
    adversarial_audit_started: { bg: 'bg-amber-500/20', text: 'text-amber-400', icon: Shield },
    formal_verification_completed: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', icon: Check },
    sensor_fusion_completed: { bg: 'bg-purple-500/20', text: 'text-purple-400', icon: Layers },
    verification_completed: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', icon: Shield },
    default: { bg: 'bg-slate-500/20', text: 'text-slate-400', icon: Clock },
  };

  const typeConfig = config[type] || config.default;
  const Icon = typeConfig.icon;

  return (
    <span className={cn(
      'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-mono',
      typeConfig.bg, typeConfig.text
    )}>
      <Icon size={12} />
      {type.replace(/_/g, ' ')}
    </span>
  );
};

/**
 * Component badge
 */
const ComponentBadge = ({ component }) => (
  <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-slate-700/50 rounded text-xs font-mono text-slate-400">
    <Layers size={10} />
    {component}
  </span>
);

/**
 * Code-styled JSON viewer
 */
const JsonViewer = ({ data }) => {
  const [expanded, setExpanded] = useState(false);
  const jsonString = JSON.stringify(data, null, 2);
  const lines = jsonString.split('\n');
  const previewLines = lines.slice(0, 3).join('\n');

  return (
    <div className="relative">
      <pre className={cn(
        'font-mono text-xs p-4 rounded-lg overflow-x-auto',
        'bg-slate-950/80 border border-slate-800',
        'text-slate-300'
      )}>
        <code>
          {expanded ? jsonString : previewLines}
          {!expanded && lines.length > 3 && '...'}
        </code>
      </pre>
      {lines.length > 3 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="absolute top-2 right-2 p-1 rounded bg-slate-800/80 hover:bg-slate-700 text-slate-400 text-xs"
        >
          {expanded ? 'Collapse' : `+${lines.length - 3} lines`}
        </button>
      )}
    </div>
  );
};

// Row component for react-window
const Row = ({ index, style, data }) => {
  const log = data[index];
  const timestamp = new Date(log.timestamp);

  // Apply a simplified motion effect or just static for performance in virtualized list
  return (
    <div style={style} className="px-2 py-1">
      <div className={cn(
        glassCard,
        'p-4 border-l-2 h-full',
        log.eventType.includes('completed') ? 'border-l-emerald-500' :
          log.eventType.includes('started') ? 'border-l-amber-500' :
            'border-l-sky-500'
      )}>
        {/* Header Row */}
        <div className="flex flex-wrap items-center gap-3 mb-3">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <Clock size={12} />
            <span className="font-mono">{timestamp.toLocaleTimeString()}</span>
            <span className="text-slate-600">•</span>
            <span className="font-mono">{timestamp.toLocaleDateString()}</span>
          </div>
          <EventTypeBadge type={log.eventType} />
          <ComponentBadge component={log.component} />
        </div>

        {/* Details */}
        <JsonViewer data={log.details} />

        {/* Footer with hash */}
        <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-800/50">
          <div className="flex items-center gap-2 text-xs text-slate-500 font-mono">
            <Hash size={12} />
            <span className="truncate max-w-[200px]">{log.hash}</span>
          </div>
          <button className={cn(
            'px-3 py-1.5 rounded-lg text-xs font-medium',
            'bg-slate-800/60 hover:bg-slate-700/60 text-slate-400 hover:text-white',
            'transition-all duration-200 hover:scale-105'
          )}>
            Verify Integrity
          </button>
        </div>
      </div>
    </div>
  );
};

/**
 * Filter dropdown
 */
const FilterDropdown = ({ label, options, value, onChange, icon: Icon }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          'flex items-center gap-2 px-3 py-2 rounded-lg text-sm',
          'bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50',
          'text-slate-300 transition-all duration-200'
        )}
      >
        <Icon size={14} className="text-slate-400" />
        <span>{value || label}</span>
        <ChevronDown size={14} className={cn(
          'text-slate-400 transition-transform',
          isOpen && 'rotate-180'
        )} />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className={cn(
              'absolute top-full mt-1 left-0 z-50 w-48',
              'bg-slate-900/95 backdrop-blur-xl border border-slate-700/50',
              'rounded-xl shadow-xl overflow-hidden'
            )}
          >
            <div className="p-1">
              <button
                onClick={() => { onChange(''); setIsOpen(false); }}
                className="w-full px-3 py-2 text-left text-sm text-slate-400 hover:bg-slate-800/50 rounded-lg"
              >
                All {label}s
              </button>
              {options.map((opt) => (
                <button
                  key={opt}
                  onClick={() => { onChange(opt); setIsOpen(false); }}
                  className={cn(
                    'w-full px-3 py-2 text-left text-sm rounded-lg',
                    value === opt ? 'bg-emerald-500/20 text-emerald-400' : 'text-slate-300 hover:bg-slate-800/50'
                  )}
                >
                  {opt}
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

/**
 * Main AuditLogViewer Component
 */
const AuditLogViewer = ({ requestId }) => {
  const [auditLogs, setAuditLogs] = useState([]);
  const [filteredLogs, setFilteredLogs] = useState([]);
  const [filters, setFilters] = useState({
    eventType: '',
    component: '',
    searchQuery: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const containerRef = useRef(null);

  // WebSocket for real-time updates
  const { auditEvents, isConnected } = useAuditStream();

  // Add new WebSocket events to logs
  useEffect(() => {
    if (auditEvents.length > 0) {
      setAuditLogs(prev => {
        const newLogs = auditEvents.filter(
          event => !prev.some(log => log.id === event.id)
        );
        return [...newLogs, ...prev].slice(0, 50000); // Allow larger buffer now that we have virtualization
      });
    }
  }, [auditEvents]);

  // Fetch initial audit logs
  const fetchAuditLogs = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Mock data - in production this would be an API call
      // Generate some dummy data to test virtualization
      const mockLogs = Array.from({ length: 100 }, (_, i) => ({
        id: `log_${i}`,
        timestamp: new Date(Date.now() - i * 60000).toISOString(),
        eventType: ['prompt_submission_received', 'adversarial_audit_started', 'formal_verification_completed'][i % 3],
        component: ['api_layer', 'battle_room', 'z3_verifier'][i % 3],
        details: {
          request_id: `req_${i}`,
          info: 'Mock log entry for testing virtualization'
        },
        hash: `hash_${i}`
      }));
      // Keep original mock data logic if needed, but ensure it's an array
      const originalMockLogs = [
        {
          id: 'log_001',
          timestamp: new Date(Date.now() - 300000).toISOString(),
          eventType: 'prompt_submission_received',
          component: 'api_layer',
          details: {
            request_id: 'req_abc123',
            prompt_length: 150,
            llm_provider: 'openai',
            priority: 5
          },
          hash: 'a1b2c3d4e5f67890abcdef1234567890'
        },
        // ... more logs
      ];

      setAuditLogs([...originalMockLogs, ...mockLogs]); // Combine
    } catch (err) {
      setError('Failed to fetch audit logs: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAuditLogs();
  }, [fetchAuditLogs]);

  // Apply filters
  useEffect(() => {
    let filtered = [...auditLogs];

    if (filters.eventType) {
      filtered = filtered.filter(log => log.eventType === filters.eventType);
    }

    if (filters.component) {
      filtered = filtered.filter(log => log.component === filters.component);
    }

    if (filters.searchQuery) {
      const query = filters.searchQuery.toLowerCase();
      filtered = filtered.filter(log =>
        log.eventType.toLowerCase().includes(query) ||
        log.component.toLowerCase().includes(query) ||
        JSON.stringify(log.details).toLowerCase().includes(query)
      );
    }

    setFilteredLogs(filtered);
  }, [auditLogs, filters]);

  const uniqueEventTypes = [...new Set(auditLogs.map(log => log.eventType))];
  const uniqueComponents = [...new Set(auditLogs.map(log => log.component))];

  const exportLogs = () => {
    const dataStr = JSON.stringify(filteredLogs, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `audit_logs_${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
  };

  return (
    <div className="p-6 space-y-6 h-screen flex flex-col box-border">
      {/* Header */}
      <div className="flex items-center justify-between flex-shrink-0">
        <div>
          <h2 className={cn(typography.h2, 'text-white')}>Audit Trail Explorer</h2>
          <p className={cn(typography.caption, 'mt-1')}>
            Cryptographically signed event history
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className={cn(
            'px-3 py-1.5 rounded-full text-xs font-medium flex items-center gap-2',
            isConnected ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-700/50 text-slate-400'
          )}>
            <span className={cn(
              'w-2 h-2 rounded-full',
              isConnected ? 'bg-emerald-500 animate-pulse' : 'bg-slate-500'
            )} />
            {isConnected ? 'Live Stream' : 'Polling'}
          </span>
        </div>
      </div>

      {/* Filters Bar */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className={cn(glassPanel, 'p-4 flex-shrink-0')}
      >
        <div className="flex flex-wrap items-center gap-3">
          {/* Search */}
          <div className="relative flex-1 min-w-[200px]">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="Search logs..."
              value={filters.searchQuery}
              onChange={(e) => setFilters(prev => ({ ...prev, searchQuery: e.target.value }))}
              className={cn(
                'w-full pl-10 pr-4 py-2 rounded-lg font-mono text-sm',
                'bg-slate-900/80 border border-slate-700/50',
                'text-slate-200 placeholder-slate-500',
                'focus:outline-none focus:border-emerald-500/50'
              )}
            />
          </div>

          {/* Filters */}
          <FilterDropdown
            label="Event Type"
            options={uniqueEventTypes}
            value={filters.eventType}
            onChange={(val) => setFilters(prev => ({ ...prev, eventType: val }))}
            icon={Terminal}
          />

          <FilterDropdown
            label="Component"
            options={uniqueComponents}
            value={filters.component}
            onChange={(val) => setFilters(prev => ({ ...prev, component: val }))}
            icon={Layers}
          />

          {/* Divider */}
          <div className="h-8 w-px bg-slate-700/50" />

          {/* Actions */}
          <button
            onClick={fetchAuditLogs}
            className={cn(
              'p-2 rounded-lg',
              'bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50',
              'text-slate-400 hover:text-white transition-all'
            )}
          >
            <RefreshCw size={16} className={isLoading ? 'animate-spin' : ''} />
          </button>

          <button
            onClick={exportLogs}
            className={cn(
              'flex items-center gap-2 px-3 py-2 rounded-lg text-sm',
              'bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50',
              'text-slate-300 hover:text-white transition-all'
            )}
          >
            <Download size={14} />
            Export
          </button>
        </div>

        {/* Time Travel Slider - Shadow Execution Replay */}
        <div className="mt-6 pt-4 border-t border-slate-800/50">
          <div className="flex justify-between text-xs text-slate-400 mb-2 uppercase font-mono tracking-widest">
            <span>Origin (t=0)</span>
            <span className="text-emerald-400">Time Travel Replay</span>
            <span>Now</span>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            defaultValue="100"
            className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500 hover:accent-emerald-400 transition-all"
            onChange={(e) => {
              const percent = parseInt(e.target.value, 10);
              // Logic to filter visible logs based on time (simulated by index for now)
              const count = Math.ceil((auditLogs.length * percent) / 100);
              const sliced = auditLogs.slice(0, count);
              // In a real replay, we'd also update the visualization state here
              setFilteredLogs(sliced);
            }}
          />
        </div>

        {/* Results count */}
        <div className="mt-3 text-sm text-slate-400">
          Showing <span className="text-white font-mono">{filteredLogs.length}</span> of{' '}
          <span className="text-white font-mono">{auditLogs.length}</span> entries
          {requestId && (
            <span className="ml-2 text-sky-400">• Request: {requestId}</span>
          )}
        </div>
      </motion.div>

      {/* Log Entries (Virtualized) */}
      <div className="flex-1 min-h-0 border border-slate-800/50 rounded-xl bg-slate-900/30 overflow-hidden">
        {error ? (
          <div className={cn(glassPanel, 'p-6 text-center m-4')}>
            <X size={24} className="mx-auto text-rose-400 mb-2" />
            <p className="text-rose-400">{error}</p>
            <button
              onClick={fetchAuditLogs}
              className="mt-4 px-4 py-2 bg-rose-500/20 text-rose-400 rounded-lg hover:bg-rose-500/30"
            >
              Retry
            </button>
          </div>
        ) : filteredLogs.length === 0 ? (
          <div className={cn(glassPanel, 'p-12 text-center m-4')}>
            <Search size={32} className="mx-auto text-slate-500 mb-3" />
            <p className="text-slate-400">No audit entries match your filters</p>
          </div>
        ) : (
          <AutoSizer>
            {({ height, width }) => (
              <List
                height={height}
                itemCount={filteredLogs.length}
                itemSize={250} // Approximate height of a card
                width={width}
                itemData={filteredLogs}
              >
                {Row}
              </List>
            )}
          </AutoSizer>
        )}
      </div>
    </div>
  );
};

export default AuditLogViewer;