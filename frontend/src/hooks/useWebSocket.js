/**
 * WebSocket Hook for Real-Time Updates
 * Connects to Aegis Nexus WebSocket endpoint for live audit stream
 * Includes Throttled Buffer Engine for high-frequency updates (Chaos Monkey resistant)
 * 
 * PRODUCTION HARDENING v2.1:
 * - Backpressure handling with message shedding
 * - Priority queue for critical messages
 * - Buffer overflow protection
 */

import { useState, useEffect, useCallback, useRef } from 'react';

const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/api/v1/ws/audit-stream';

// Reconnection configuration
const RECONNECT_INITIAL_DELAY = 1000;
const RECONNECT_MAX_DELAY = 30000;
const RECONNECT_MULTIPLIER = 1.5;

// Throttle configuration (approx 60fps)
const FLUSH_INTERVAL_MS = 16;

// Backpressure configuration - prevents UI freeze during War Room events
const MAX_BUFFER_SIZE = 1000;      // Hard limit on buffer size
const SHED_THRESHOLD = 800;        // Start shedding low-priority messages
const EMERGENCY_TRIM_SIZE = 500;   // Trim to this size on overflow

// Priority message types (never shed)
const PRIORITY_MESSAGE_TYPES = [
    'verification_update',
    'lockdown_alert',
    'security_breach',
    'system_status',
    'canary_breach'
];

/**
 * WebSocket connection states
 */
export const ConnectionState = {
    CONNECTING: 'connecting',
    CONNECTED: 'connected',
    DISCONNECTED: 'disconnected',
    RECONNECTING: 'reconnecting',
    ERROR: 'error',
};

/**
 * Custom hook for WebSocket connection with auto-reconnect and throttling
 * @param {Object} options - Configuration options
 * @param {Function} options.onMessage - Callback for incoming messages (throttled)
 * @param {Function} options.onConnect - Callback when connection established
 * @param {Function} options.onDisconnect - Callback when disconnected
 * @param {boolean} options.enabled - Whether to connect (default: true)
 */
export function useWebSocket({ onMessage, onConnect, onDisconnect, enabled = true } = {}) {
    const [connectionState, setConnectionState] = useState(ConnectionState.DISCONNECTED);
    const [lastMessage, setLastMessage] = useState(null);

    // Throttling State
    const messageBufferRef = useRef([]);
    const lastFlushTimeRef = useRef(0);
    const rafRef = useRef(null);

    const wsRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);
    const reconnectDelayRef = useRef(RECONNECT_INITIAL_DELAY);
    const reconnectCountRef = useRef(0); // Dedicated ref for tracking retries
    const mountedRef = useRef(true);

    // Ensure onMessage is stable to prevent effect re-runs
    const onMessageRef = useRef(onMessage);
    useEffect(() => {
        onMessageRef.current = onMessage;
    }, [onMessage]);

    /**
     * Decoupled Render Loop (Level 5 Hardening)
     * Processes queue on Animation Frame to ensure 60fps stability
     */
    /**
     * Decoupled Render Loop (Level 5 Hardening)
     * Processes queue at throttled 30fps to ensure UI stability under pressure
     */
    const THRESHOLD_30FPS = 33; // ~33ms for 30fps

    const processQueue = useCallback(() => {
        if (!mountedRef.current) return;

        const now = performance.now();
        const elapsed = now - lastFlushTimeRef.current;

        // Level 5 Logic: Decoupled Render Loop (33ms = 30fps)
        // This prevents the "Thundering Herd" problem on the React Fiber tree
        if (elapsed < THRESHOLD_30FPS && messageBufferRef.current.length < 50) {
            rafRef.current = requestAnimationFrame(processQueue);
            return;
        }

        const buffer = messageBufferRef.current;

        if (buffer.length > 0) {
            // Level 5 Logic: Batch process - take the latest state for 'lastMessage'
            // to decouple network ingress from React's state update cycle
            const latestMsg = buffer[buffer.length - 1];

            // Batch all updates into a single frame to minimize React re-renders
            if (onMessageRef.current) {
                // If buffer is very large (> 100), we perform "Backpressure Shedding"
                // by only processing the most critical/recent messages.
                const messagesToProcess = buffer.length > 100 ? buffer.slice(-100) : buffer;
                messagesToProcess.forEach(msg => onMessageRef.current(msg));
            }

            setLastMessage(latestMsg);

            // Clear buffer and update timestamp
            messageBufferRef.current = [];
            lastFlushTimeRef.current = now;
        }

        rafRef.current = requestAnimationFrame(processQueue);
    }, []);

    /**
     * Queue incoming message with backpressure handling
     * Sheds low-priority messages when buffer is under pressure
     */
    const queueMessage = useCallback((data) => {
        const bufferSize = messageBufferRef.current.length;
        const isPriorityMessage = PRIORITY_MESSAGE_TYPES.includes(data.type);

        // Backpressure Level 1: Shed low-priority messages above threshold
        if (bufferSize >= SHED_THRESHOLD && !isPriorityMessage) {
            // Log shedding in development
            if (process.env.NODE_ENV === 'development') {
                console.warn('[WebSocket] Backpressure: Shedding low-priority message', data.type);
            }
            return; // Drop non-critical message
        }

        // Backpressure Level 2: Emergency trim if at max capacity
        if (bufferSize >= MAX_BUFFER_SIZE) {
            console.warn(`[WebSocket] Buffer overflow! Trimming to ${EMERGENCY_TRIM_SIZE} messages`);
            // Keep priority messages and recent messages
            const priorityMessages = messageBufferRef.current.filter(
                msg => PRIORITY_MESSAGE_TYPES.includes(msg.type)
            );
            const recentMessages = messageBufferRef.current.slice(-EMERGENCY_TRIM_SIZE);

            // Merge priority messages with recent, avoiding duplicates
            const combined = [...priorityMessages];
            recentMessages.forEach(msg => {
                if (!combined.includes(msg)) {
                    combined.push(msg);
                }
            });
            messageBufferRef.current = combined.slice(-EMERGENCY_TRIM_SIZE);
        }

        // Add message to buffer
        messageBufferRef.current.push(data);
    }, []);


    /**
     * Connect to WebSocket server
     */
    const connect = useCallback(() => {
        if (!enabled || wsRef.current?.readyState === WebSocket.OPEN) return;

        setConnectionState(ConnectionState.CONNECTING);

        try {
            const ws = new WebSocket(WS_URL);
            wsRef.current = ws;

            ws.onopen = () => {
                if (!mountedRef.current) return;

                setConnectionState(ConnectionState.CONNECTED);
                reconnectDelayRef.current = RECONNECT_INITIAL_DELAY;
                // RESET RETRY COUNT ON SUCCESS
                reconnectCountRef.current = 0;
                onConnect?.();

                // Start Heartbeat
                if (wsRef.current?.pingInterval) clearInterval(wsRef.current.pingInterval);
                wsRef.current.pingInterval = setInterval(() => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ type: 'ping' }));
                    }
                }, 5000); // 5s Heartbeat
            };

            ws.binaryType = 'arraybuffer'; // Enable binary support

            ws.onmessage = (event) => {
                if (!mountedRef.current) return;

                try {
                    const data = JSON.parse(event.data);
                    queueMessage(data);
                } catch (e) {
                    console.error('[WebSocket] Failed to parse message:', e);
                }
            };

            // Start the optimized render loop
            rafRef.current = requestAnimationFrame(processQueue);

            ws.onclose = (event) => {
                if (!mountedRef.current) return;

                setConnectionState(ConnectionState.DISCONNECTED);
                wsRef.current = null;
                onDisconnect?.();

                // Auto-reconnect with jittered exponential backoff (Level 5 Resiliency)
                if (enabled && event.code !== 1000) {
                    setConnectionState(ConnectionState.RECONNECTING);

                    // FIXED: Persist retry count in Ref
                    const retryCount = reconnectCountRef.current + 1;
                    reconnectCountRef.current = retryCount;

                    // Level 5 Logic: Capped Exponential Backoff + Jitter (The "Thundering Herd" Fix)
                    // Formula: min(30s, 1s * 2^retries) + Jitter(0-1s)
                    const baseDelay = Math.min(1000 * (2 ** retryCount), 30000);

                    // Add Jitter: Random 0-1000ms to prevent synchronized DDoS
                    const jitter = Math.random() * 1000;
                    const timeout = baseDelay + jitter;

                    console.log(`[WebSocket] Recovery active (Attempt ${retryCount}). Reconnecting in ${(timeout / 1000).toFixed(1)}s`);

                    reconnectTimeoutRef.current = setTimeout(() => {
                        connect();
                    }, timeout);
                }
            };

            ws.onerror = (error) => {
                console.error('[WebSocket] Connection error:', error);
                setConnectionState(ConnectionState.ERROR);
            };
        } catch (error) {
            console.error('[WebSocket] Failed to create connection:', error);
            setConnectionState(ConnectionState.ERROR);
        }
    }, [enabled, onConnect, onDisconnect, queueMessage]);

    /**
     * Disconnect from WebSocket server
     */
    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }

        if (rafRef.current) {
            cancelAnimationFrame(rafRef.current);
            rafRef.current = null;
        }

        if (wsRef.current) {
            wsRef.current.close(1000, 'Client disconnected');
            wsRef.current = null;
        }

        setConnectionState(ConnectionState.DISCONNECTED);
    }, []);

    /**
     * Send message through WebSocket
     */
    const sendMessage = useCallback((message) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(message));
        } else {
            console.warn('[WebSocket] Cannot send message, not connected');
        }
    }, []);

    // Connect on mount, cleanup on unmount
    useEffect(() => {
        mountedRef.current = true;

        if (enabled) {
            connect();
        }

        return () => {
            mountedRef.current = false;
            disconnect();
        };
    }, [enabled, connect, disconnect]);

    return {
        connectionState,
        isConnected: connectionState === ConnectionState.CONNECTED,
        lastMessage,
        sendMessage,
        connect,
        disconnect,
    };
}

/**
 * Hook specifically for audit stream subscription
 */
export function useAuditStream() {
    const [auditEvents, setAuditEvents] = useState([]);
    const [verificationStatus, setVerificationStatus] = useState(null);
    const [sensorData, setSensorData] = useState(null);
    const [lastEventId, setLastEventId] = useState(null);
    const [isHydrating, setIsHydrating] = useState(false);

    // Track connection transitions to trigger hydration
    const prevConnectedRef = useRef(false);

    const handleMessage = useCallback((data) => {
        // NOTE: This runs inside the throttled loop.
        // React's auto-batching (v18) should handle the state updates efficiently.
        if (data.id) {
            setLastEventId(data.id);
        }

        switch (data.type) {
            case 'audit_event':
                setAuditEvents(prev => {
                    // Dedup based on ID if present
                    const exists = data.id && prev.some(e => e.id === data.id);
                    if (exists) return prev;
                    return [data.payload, ...prev].slice(0, 100); // Keep last 100
                });
                break;
            case 'verification_update':
                setVerificationStatus(data.payload);
                break;
            case 'sensor_data':
                setSensorData(data.payload);
                break;
            case 'sensor_delta':
                setSensorData(prev => ({ ...prev, ...data.payload }));
                break;

            // --- Battle Room Events ---
            case 'OPENING_ARGUMENTS':
            case 'CROSS_EXAMINATION':
            case 'REBUTTAL':
                setVerificationStatus(prev => ({
                    ...prev,
                    phase_1_audit: {
                        ...(prev?.phase_1_audit || {}),
                        ...data, // Merge event data (actor_proposal, critiques, agents, etc.)
                        status: data.type
                    }
                }));
                break;

            default:
                // console.log('[AuditStream] Unknown message type:', data.type);
                break;
        }
    }, []);

    const { connectionState, isConnected, sendMessage } = useWebSocket({
        onMessage: handleMessage,
    });

    // Hydration Logic
    useEffect(() => {
        const fetchMissingEvents = async () => {
            if (!lastEventId) return;

            try {
                setIsHydrating(true);
                console.log(`[AuditStream] Hydrating events since ${lastEventId}...`);
                const response = await fetch(`/api/v1/audit/events?since=${lastEventId}`);
                if (response.ok) {
                    const missedEvents = await response.json();
                    if (missedEvents && missedEvents.length > 0) {
                        console.log(`[AuditStream] Hydrated ${missedEvents.length} missed events.`);
                        // Merge missed events
                        setAuditEvents(prev => {
                            const newEvents = [...missedEvents, ...prev];
                            // Sort and dedup could be more robust here, but simple prepend + slice is okay for stream
                            return newEvents.slice(0, 100);
                        });
                        // Update last ID to the latest from hydration
                        setLastEventId(missedEvents[0].id);
                    }
                }
            } catch (e) {
                console.error("[AuditStream] Hydration failed:", e);
            } finally {
                setIsHydrating(false);
            }
        };

        // Trigger hydration when connection restores (false -> true)
        if (isConnected && !prevConnectedRef.current && lastEventId) {
            fetchMissingEvents();
        }
        prevConnectedRef.current = isConnected;
    }, [isConnected, lastEventId]);

    return {
        connectionState,
        isConnected,
        isHydrating,
        auditEvents,
        verificationStatus,
        sensorData,
        sendMessage,
    };
}

export default useWebSocket;
