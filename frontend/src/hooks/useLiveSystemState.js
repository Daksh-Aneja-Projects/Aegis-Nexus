import { useReducer, useEffect, useRef, useCallback } from 'react';

/**
 * useLiveSystemState - Production-Grade Live Data Hook
 * 
 * GRACEFUL DEGRADATION PATTERN:
 * 1. Primary: WebSocket for real-time updates
 * 2. Fallback: HTTP Polling when WebSocket fails
 * 3. User never notices the switch (Google-level UX)
 * 
 * This addresses the audit finding about:
 * "If the WebSocket fails, the dashboard likely freezes"
 */

// Predictable state transitions for Level 5 systems
const stateReducer = (state, action) => {
    switch (action.type) {
        case 'CONNECTING':
            return { ...state, status: 'CONNECTING' };
        case 'UPDATE':
            // Shallow spread for UI performance on main thread
            return {
                ...state,
                data: { ...state.data, ...action.payload },
                status: 'LIVE',
                errors: 0,
                lastUpdated: new Date(),
                isOptimistic: false,
                connectionType: action.connectionType || state.connectionType
            };
        case 'OPTIMISTIC_UPDATE':
            return {
                ...state,
                data: { ...state.data, ...action.payload },
                isOptimistic: true
            };
        case 'WS_CONNECTED':
            return {
                ...state,
                connectionType: 'websocket',
                status: state.data ? 'LIVE' : 'CONNECTING'
            };
        case 'WS_DISCONNECTED':
            return {
                ...state,
                connectionType: 'polling',
                // Don't change status if we have data - seamless fallback
            };
        case 'ERROR':
            const newCount = state.errors + 1;
            return {
                ...state,
                status: newCount >= 3 ? 'ERROR' : 'STALE',
                errors: newCount
            };
        default:
            return state;
    }
};

export const useLiveSystemState = (endpoint, initialData = null) => {
    const [state, dispatch] = useReducer(stateReducer, {
        data: initialData,
        status: 'CONNECTING',
        errors: 0,
        lastUpdated: null,
        isOptimistic: false,
        connectionType: 'initializing' // 'websocket' | 'polling' | 'initializing'
    });

    const mountedRef = useRef(true);
    const retryTimeout = useRef(null);
    const pollInterval = useRef(null);
    const wsRef = useRef(null);
    const wsReconnectAttempts = useRef(0);
    const WS_MAX_RETRIES = 3;

    // HTTP Polling fetch (fallback)
    const fetchData = useCallback(async () => {
        try {
            const response = await fetch(endpoint);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const payload = await response.json();
            if (!mountedRef.current) return;

            dispatch({ type: 'UPDATE', payload, connectionType: 'polling' });

        } catch (err) {
            if (!mountedRef.current) return;
            console.error("[LiveState] HTTP Polling Failure:", err);
            dispatch({ type: 'ERROR' });

            // Exponential Backoff with Jitter (Refinement 3)
            clearTimeout(retryTimeout.current);
            // Cap at 30s. Jitter adds up to 1000ms randomness to prevent synchronized reconnections
            const backoffDelay = Math.min(2000 * Math.pow(1.5, state.errors), 30000) + (Math.random() * 1000);
            console.debug(`[LiveState] Reconnecting in ${Math.round(backoffDelay)}ms`);
            retryTimeout.current = setTimeout(fetchData, backoffDelay);
        }
    }, [endpoint, state.errors]);

    // Throttled Dispatch (Gap 5)
    // Prevents React Render Thrashing during high-frequency WebSocket events (Chaos Scenarios)
    const pendingUpdates = useRef([]);
    const throttleFrame = useRef(null);

    const throttledDispatch = useCallback((payload, type = 'UPDATE') => {
        // Queue the update
        pendingUpdates.current.push(payload);

        // If no frame requested, request one
        if (!throttleFrame.current) {
            throttleFrame.current = requestAnimationFrame(() => {
                if (!mountedRef.current) return;

                // Batch processing: Merge all pending updates
                // For this system, we take the latest full state (if complete replacement)
                // or merge specific fields. Assuming payload is partial:
                const fusedPayload = pendingUpdates.current.reduce((acc, curr) => ({ ...acc, ...curr }), {});

                dispatch({ type: type, payload: fusedPayload, connectionType: 'websocket' });

                // Clear queue
                pendingUpdates.current = [];
                throttleFrame.current = null;
            });
        }
    }, []);

    // WebSocket connection (primary)
    const connectWebSocket = useCallback(() => {
        if (!mountedRef.current) return;
        if (wsReconnectAttempts.current >= WS_MAX_RETRIES) {
            console.log('[LiveState] WebSocket max retries reached, using HTTP polling');
            return;
        }

        try {
            // Convert HTTP endpoint to WebSocket endpoint
            // e.g., http://localhost:8000/api/v1/status -> ws://localhost:8000/ws/audit-stream
            const wsUrl = endpoint
                .replace(/^http/, 'ws')
                .replace(/\/api\/v\d+\/[^/]+$/, '/api/v1/ws/audit-stream');

            console.log('[LiveState] Connecting WebSocket:', wsUrl);
            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                if (!mountedRef.current) return;
                console.log('[LiveState] WebSocket connected');
                wsReconnectAttempts.current = 0;
                dispatch({ type: 'WS_CONNECTED' });

                // Slow down polling when WS is active
                if (pollInterval.current) {
                    clearInterval(pollInterval.current);
                    // Keep very slow polling as backup (every 30s)
                    pollInterval.current = setInterval(fetchData, 30000);
                }
            };

            ws.onmessage = (event) => {
                if (!mountedRef.current) return;
                try {
                    const data = JSON.parse(event.data);
                    // Handle different message types
                    if (data.type === 'pong') return; // Heartbeat response

                    const payload = data.payload || data;
                    // Use throttled dispatch instead of direct dispatch
                    throttledDispatch(payload);
                    // dispatch({ type: 'UPDATE', payload, connectionType: 'websocket' });
                } catch (e) {
                    console.error('[LiveState] Failed to parse WS message:', e);
                }
            };

            ws.onclose = (event) => {
                if (!mountedRef.current) return;
                console.log('[LiveState] WebSocket closed, code:', event.code);
                wsRef.current = null;
                dispatch({ type: 'WS_DISCONNECTED' });

                // Reconnect with exponential backoff and Jitter
                if (event.code !== 1000 && wsReconnectAttempts.current < WS_MAX_RETRIES) {
                    wsReconnectAttempts.current++;
                    const delay = Math.min(1000 * Math.pow(2, wsReconnectAttempts.current), 30000) + (Math.random() * 2000);
                    console.log(`[LiveState] Reconnecting in ${Math.round(delay)}ms (attempt ${wsReconnectAttempts.current})`);
                    setTimeout(connectWebSocket, delay);
                }

                // Resume faster polling when WS is down
                if (pollInterval.current) clearInterval(pollInterval.current);
                pollInterval.current = setInterval(fetchData, 3000);
            };

            ws.onerror = (error) => {
                console.error('[LiveState] WebSocket error:', error);
                // Error will trigger onclose
            };

        } catch (e) {
            console.log('[LiveState] WebSocket not available, using HTTP polling');
            wsReconnectAttempts.current = WS_MAX_RETRIES; // Skip future WS attempts
        }
    }, [endpoint, fetchData]);

    // Initialize connections
    useEffect(() => {
        mountedRef.current = true;

        // Start with HTTP fetch for immediate data
        fetchData();

        // Start HTTP polling as baseline
        pollInterval.current = setInterval(fetchData, 3000);

        // Try WebSocket for real-time upgrades
        connectWebSocket();

        return () => {
            mountedRef.current = false;
            clearInterval(pollInterval.current);
            clearTimeout(retryTimeout.current);
            if (wsRef.current) {
                wsRef.current.close(1000, 'Component unmounted');
            }
        };
    }, [endpoint, fetchData, connectWebSocket]);

    const optimisticUpdate = useCallback((payload) => {
        dispatch({ type: 'OPTIMISTIC_UPDATE', payload });
    }, []);

    const refresh = useCallback(() => {
        dispatch({ type: 'CONNECTING' });
        fetchData();
        // Also try to reconnect WebSocket if it was disconnected
        if (!wsRef.current && wsReconnectAttempts.current > 0) {
            wsReconnectAttempts.current = 0;
            connectWebSocket();
        }
    }, [fetchData, connectWebSocket]);

    return {
        ...state,
        refresh,
        optimisticUpdate,
        isLive: state.status === 'LIVE',
        isStale: state.status === 'STALE',
        isError: state.status === 'ERROR',
        isWebSocket: state.connectionType === 'websocket',
        isPolling: state.connectionType === 'polling',
    };
};
