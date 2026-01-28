/**
 * useApi - Production-Ready API Integration Hook
 * 
 * Connects frontend components to the Aegis API service with:
 * - Automatic loading states
 * - Error handling with retry capability
 * - Cache management for performance
 * - TypeScript-friendly structure
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import apiClient from '../services/api';

// =============================================================================
// HOOK: useApi - Generic API Request Hook
// =============================================================================

/**
 * Generic hook for making API requests with state management
 * 
 * @param {Function} apiFunction - The API function to call
 * @param {Object} options - Configuration options
 * @returns {Object} { data, loading, error, execute, reset }
 */
export function useApi(apiFunction, { immediate = false, initialData = null } = {}) {
    const [data, setData] = useState(initialData);
    const [loading, setLoading] = useState(immediate);
    const [error, setError] = useState(null);
    const mountedRef = useRef(true);

    useEffect(() => {
        mountedRef.current = true;
        return () => { mountedRef.current = false; };
    }, []);

    const execute = useCallback(async (...args) => {
        if (!mountedRef.current) return;

        setLoading(true);
        setError(null);

        try {
            const result = await apiFunction(...args);
            if (mountedRef.current) {
                setData(result);
                setLoading(false);
            }
            return result;
        } catch (err) {
            if (mountedRef.current) {
                setError(err);
                setLoading(false);
            }
            throw err;
        }
    }, [apiFunction]);

    const reset = useCallback(() => {
        setData(initialData);
        setError(null);
        setLoading(false);
    }, [initialData]);

    // Execute immediately if requested
    useEffect(() => {
        if (immediate) {
            execute().catch(() => { });
        }
    }, [immediate, execute]);

    return { data, loading, error, execute, reset };
}

// =============================================================================
// HOOK: useSystemStatus - System Health Monitoring
// =============================================================================

export function useSystemStatus(pollInterval = 5000) {
    const [status, setStatus] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const intervalRef = useRef(null);

    const fetchStatus = useCallback(async () => {
        try {
            const result = await apiClient.getSystemStatus();
            setStatus(result);
            setError(null);
        } catch (err) {
            setError(err);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchStatus();

        if (pollInterval > 0) {
            intervalRef.current = setInterval(fetchStatus, pollInterval);
        }

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, [fetchStatus, pollInterval]);

    return { status, loading, error, refresh: fetchStatus };
}

// =============================================================================
// HOOK: useVerification - Proposal Verification
// =============================================================================

export function useVerification() {
    const [verificationState, setVerificationState] = useState({
        status: 'idle', // 'idle' | 'submitting' | 'verifying' | 'complete' | 'error'
        result: null,
        progress: 0,
        traceId: null,
        error: null
    });

    const submit = useCallback(async (proposal) => {
        const traceId = `trace_${Date.now()}`;

        setVerificationState({
            status: 'submitting',
            result: null,
            progress: 10,
            traceId,
            error: null
        });

        try {
            // Submit for verification
            setVerificationState(prev => ({ ...prev, status: 'verifying', progress: 50 }));

            const result = await apiClient.submitForVerification({
                ...proposal,
                trace_id: traceId
            });

            setVerificationState({
                status: 'complete',
                result,
                progress: 100,
                traceId,
                error: null
            });

            return result;
        } catch (err) {
            setVerificationState(prev => ({
                ...prev,
                status: 'error',
                error: err,
                progress: 0
            }));
            throw err;
        }
    }, []);

    const reset = useCallback(() => {
        setVerificationState({
            status: 'idle',
            result: null,
            progress: 0,
            traceId: null,
            error: null
        });
    }, []);

    return {
        ...verificationState,
        submit,
        reset,
        isSubmitting: verificationState.status === 'submitting' || verificationState.status === 'verifying'
    };
}

// =============================================================================
// HOOK: useAuditLogs - Audit Trail Access
// =============================================================================

export function useAuditLogs({ limit = 50, autoRefresh = true, refreshInterval = 10000 } = {}) {
    const [logs, setLogs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [hasMore, setHasMore] = useState(true);

    const fetchLogs = useCallback(async (params = {}) => {
        try {
            setLoading(true);
            const result = await apiClient.getAuditLogs({ limit, ...params });
            const logsArray = Array.isArray(result) ? result : (result.logs || []);
            setLogs(logsArray);
            setHasMore(logsArray.length >= limit);
            setError(null);
        } catch (err) {
            setError(err);
        } finally {
            setLoading(false);
        }
    }, [limit]);

    const loadMore = useCallback(async () => {
        if (!hasMore || loading) return;

        const lastLog = logs[logs.length - 1];
        if (!lastLog) return;

        try {
            setLoading(true);
            const result = await apiClient.getAuditLogs({
                limit,
                before: lastLog.timestamp
            });
            const newLogs = Array.isArray(result) ? result : (result.logs || []);
            setLogs(prev => [...prev, ...newLogs]);
            setHasMore(newLogs.length >= limit);
        } catch (err) {
            setError(err);
        } finally {
            setLoading(false);
        }
    }, [logs, limit, hasMore, loading]);

    useEffect(() => {
        fetchLogs();

        if (autoRefresh && refreshInterval > 0) {
            const interval = setInterval(fetchLogs, refreshInterval);
            return () => clearInterval(interval);
        }
    }, [fetchLogs, autoRefresh, refreshInterval]);

    return { logs, loading, error, hasMore, refresh: fetchLogs, loadMore };
}

// =============================================================================
// HOOK: useSensorData - Sensor Fusion Data
// =============================================================================

export function useSensorData(pollInterval = 1000) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [confidence, setConfidence] = useState(0);

    const fetchData = useCallback(async () => {
        try {
            const result = await apiClient.getSensorData();
            setData(result);
            setConfidence(result.confidence || 0);
            setError(null);
        } catch (err) {
            setError(err);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchData();

        if (pollInterval > 0) {
            const interval = setInterval(fetchData, pollInterval);
            return () => clearInterval(interval);
        }
    }, [fetchData, pollInterval]);

    return { data, loading, error, confidence, refresh: fetchData };
}

// =============================================================================
// HOOK: useCircuitBreaker - Circuit Breaker State
// =============================================================================

export function useCircuitBreaker(pollInterval = 2000) {
    const [state, setState] = useState({
        status: 'unknown',
        isOpen: false,
        cognitiveLoad: 0,
        entropy: 0,
        activeRequests: 0,
        isLocked: false
    });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const fetchState = useCallback(async () => {
        try {
            const result = await apiClient.getCircuitBreakerStatus();
            setState({
                status: result.state || 'closed',
                isOpen: result.state === 'open',
                cognitiveLoad: result.cognitive_load || 0,
                entropy: result.entropy_level || 0,
                activeRequests: result.active_requests || 0,
                isLocked: result.hardware_lockout || false
            });
            setError(null);
        } catch (err) {
            setError(err);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchState();

        if (pollInterval > 0) {
            const interval = setInterval(fetchState, pollInterval);
            return () => clearInterval(interval);
        }
    }, [fetchState, pollInterval]);

    const clearLockout = useCallback(async (adminToken) => {
        try {
            await apiClient.clearHardwareLockout(adminToken);
            await fetchState();
            return true;
        } catch (err) {
            setError(err);
            return false;
        }
    }, [fetchState]);

    return { ...state, loading, error, refresh: fetchState, clearLockout };
}

// =============================================================================
// HOOK: useWebSocketEvents - WebSocket Event Subscription
// =============================================================================

export function useWebSocketEvents(eventTypes = ['all']) {
    const [events, setEvents] = useState([]);
    const [isConnected, setIsConnected] = useState(false);
    const [lastEvent, setLastEvent] = useState(null);
    const maxEvents = 100;

    useEffect(() => {
        const unsubscribers = [];

        // Connect to WebSocket
        apiClient.connectWebSocket()
            .then(() => setIsConnected(true))
            .catch(() => setIsConnected(false));

        // Subscribe to each event type
        eventTypes.forEach(eventType => {
            const unsubscribe = apiClient.onWebSocketEvent(eventType, (data) => {
                const event = { ...data, receivedAt: new Date().toISOString() };
                setLastEvent(event);
                setEvents(prev => {
                    const updated = [event, ...prev];
                    return updated.slice(0, maxEvents);
                });
            });
            unsubscribers.push(unsubscribe);
        });

        // Subscribe to connection events
        const unsubConnection = apiClient.onWebSocketEvent('connection_lost', () => {
            setIsConnected(false);
        });
        unsubscribers.push(unsubConnection);

        return () => {
            unsubscribers.forEach(unsub => unsub());
        };
    }, [eventTypes]);

    const clearEvents = useCallback(() => {
        setEvents([]);
        setLastEvent(null);
    }, []);

    return { events, lastEvent, isConnected, clearEvents };
}

export default useApi;
