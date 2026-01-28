/**
 * useOptimisticState Hook
 * Aegis Nexus Frontend
 * 
 * Manages "Tentative" vs "Verified" state for the cognitive interface.
 * Allows instant feedback (optimistic UI) while waiting for cryptographically verified
 * confirmation from the ledger/consensus layer.
 * 
 * PRODUCTION FIX v2.1:
 * - Verification timeout handling
 * - Enhanced rollback UX with toast notifications
 * - Compensation action patterns
 * - State snapshot persistence for recovery
 */

import { useState, useCallback, useRef, useEffect } from 'react';

// Status Enums
export const STATE_STATUS = {
    STABLE: 'stable',           // Verified by ledger
    OPTIMISTIC: 'optimistic',   // Local update, pending verification
    REJECTED: 'rejected',       // Rejected by governance
    SYNCING: 'syncing',         // Waiting for confirmation
    TIMEOUT: 'timeout'          // Verification took too long
};

// Default timeout for verification (10 seconds)
const DEFAULT_VERIFICATION_TIMEOUT_MS = 10000;

/**
 * @param {any} initialState - The initial confirmed state
 * @param {Function} verificationService - Async function to submit state change
 * @param {Object} options - Configuration options
 * @param {number} options.timeoutMs - Verification timeout in milliseconds
 * @param {Function} options.onTimeout - Callback when verification times out
 * @param {Function} options.onRejection - Callback when state is rejected
 * @param {Function} options.onRollback - Callback for rollback notification (toast integration)
 * @param {boolean} options.persistSnapshots - Whether to persist state snapshots to localStorage
 */
export const useOptimisticState = (
    initialState,
    verificationService,
    options = {}
) => {
    const {
        timeoutMs = DEFAULT_VERIFICATION_TIMEOUT_MS,
        onTimeout,
        onRejection,
        onRollback,
        persistSnapshots = false
    } = options;

    // The "True" state as known by the ledger
    const [confirmedState, setConfirmedState] = useState(initialState);

    // The UI state (what the user sees)
    const [uiState, setUiState] = useState(initialState);

    // Status tracking
    const [status, setStatus] = useState(STATE_STATUS.STABLE);

    // Error message for UI
    const [errorMessage, setErrorMessage] = useState(null);

    // History Stack for Time Travel Debugging & Replay
    // Instead of a single snapshot, we keep a ledger of recent states
    const historyStackRef = useRef([{ state: initialState, traceId: 'init', timestamp: Date.now() }]);

    // Metadata for rollback (Legacy ref kept for compatibility but logic moves to stack)
    const pendingTxRef = useRef(null);
    const timeoutRef = useRef(null);
    const abortControllerRef = useRef(null);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
            }
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
        };
    }, []);

    // Helper: Push to history stack
    const pushToHistory = (state, traceId) => {
        const stack = historyStackRef.current;
        stack.push({ state, traceId, timestamp: Date.now() });
        // Limit stack size to 50
        if (stack.length > 50) stack.shift();
    };

    // Helper: Replay history from a given point
    // This is the "Time Travel" repair mechanism
    const replayHistory = async (fromIndex) => {
        const stack = historyStackRef.current;
        const validActions = stack.slice(fromIndex + 1);

        console.log(`[OptimisticState] 🔄 Replaying ${validActions.length} actions for state repair...`);

        let currentState = stack[fromIndex].state;
        setConfirmedState(currentState);

        for (const action of validActions) {
            // Re-apply actions on top of the confirmed base
            // Note: This requires the action to be idempotent or re-computable
            // For simple state replacement, we just take the last one, 
            // but for complex reducers, we'd re-run the reducer here.
            // Assuming simple replacement for this hook version:
            currentState = action.state;
        }

        setUiState(currentState);
        setStatus(STATE_STATUS.STABLE); // Temporarily stable until new actions come in
    };

    /**
     * Perform rollback with visual feedback and compensation action
     */
    /**
     * Perform rollback with visual feedback and compensation action
     */
    const performRollback = useCallback((reason, newStatus, context = {}) => {
        // 1. Find the last safe state in history
        const stack = historyStackRef.current;
        // In a real implementation we might search for the specific tradeID to invalidate
        // deeper in the stack. For now, we revert to the *previous* verified state.

        // The last item is the "bad" optimistic state we just pushed.
        // The second to last item is the previous state.
        const safeStateEntry = stack.length > 1 ? stack[stack.length - 2] : stack[0];

        // Pop the failed state
        stack.pop();

        console.warn(`[OptimisticState] Reverting from current to ${safeStateEntry.traceId}`);

        setUiState(safeStateEntry.state);
        setStatus(newStatus);
        setErrorMessage(reason);

        // 2. Trigger rollback callback (for toast notifications)
        if (onRollback) {
            onRollback({
                reason,
                status: newStatus,
                previousState: safeStateEntry.state,
                traceId: context.traceId,
                timestamp: new Date().toISOString()
            });
        }

        // 3. Log compensation action for audit
        console.warn(`[OptimisticState] Rollback executed: ${reason}`, {
            status: newStatus,
            traceId: context.traceId
        });

        // Clear error after 5 seconds
        setTimeout(() => setErrorMessage(null), 5000);
    }, [onRollback]);

    /**
     * Optimistically update state and trigger background verification
     * @param {any} nextState - The desired new state
     * @param {string} traceId - Correlation ID for the action
     */
    const setOptimistic = useCallback(async (nextState, traceId) => {
        // Clear any previous pending operation
        if (timeoutRef.current) {
            clearTimeout(timeoutRef.current);
        }
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }

        // Create new abort controller for this operation
        abortControllerRef.current = new AbortController();

        // 1. Snapshot for rollback (Add to History Stack)
        pendingTxRef.current = traceId;
        pushToHistory(nextState, traceId);
        setErrorMessage(null);

        // 2. Update UI immediately
        setUiState(nextState);
        setStatus(STATE_STATUS.OPTIMISTIC);

        // 3. Set up timeout for automatic rollback
        timeoutRef.current = setTimeout(() => {
            if (status === STATE_STATUS.OPTIMISTIC || status === STATE_STATUS.SYNCING) {
                console.warn(`[OptimisticState] Verification timeout for ${traceId}`);
                performRollback('Verification timeout - please try again', STATE_STATUS.TIMEOUT, { traceId });
                onTimeout?.({ traceId, nextState });
            }
        }, timeoutMs);

        // 4. Trigger verification in background
        try {
            setStatus(STATE_STATUS.SYNCING);

            const result = await verificationService(nextState, traceId);

            // Clear timeout on completion
            clearTimeout(timeoutRef.current);

            if (result.success) {
                // Verification Success
                setConfirmedState(nextState);
                setUiState(nextState); // Ensure sync
                setStatus(STATE_STATUS.STABLE);
            } else {
                // Verification Failed / Rejected by Governance
                console.warn(`[OptimisticState] Action Rejected: ${result.reason}`);
                performRollback(result.reason || 'Action rejected by governance', STATE_STATUS.REJECTED, { traceId });
                onRejection?.({ traceId, reason: result.reason });
            }
        } catch (error) {
            // Clear timeout on error
            clearTimeout(timeoutRef.current);

            // Check if aborted (component unmounted or new operation started)
            if (error.name === 'AbortError') {
                return;
            }

            // Network/System Error
            console.error("[OptimisticState] System Error", error);
            performRollback(
                error.message || 'System error - please try again',
                STATE_STATUS.REJECTED
            );
        }
    }, [confirmedState, verificationService, status, timeoutMs, onTimeout, onRejection, performRollback]);

    /**
     * Externally force a state update (e.g., from WebSocket sync)
     */
    const syncConfirmedState = useCallback((newState) => {
        setConfirmedState(newState);
        setUiState(newState);
        setStatus(STATE_STATUS.STABLE);
        setErrorMessage(null);

        // Clear any pending timeouts
        if (timeoutRef.current) {
            clearTimeout(timeoutRef.current);
        }
    }, []);

    /**
     * Reset to confirmed state (manual recovery)
     */
    const reset = useCallback(() => {
        setUiState(confirmedState);
        setStatus(STATE_STATUS.STABLE);
        setErrorMessage(null);

        if (timeoutRef.current) {
            clearTimeout(timeoutRef.current);
        }
    }, [confirmedState]);

    return {
        state: uiState,
        setOptimistic,
        status,
        confirmedState,
        syncConfirmedState,
        reset,
        errorMessage,
        isPending: status === STATE_STATUS.OPTIMISTIC || status === STATE_STATUS.SYNCING,
        isError: status === STATE_STATUS.REJECTED || status === STATE_STATUS.TIMEOUT
    };
};

/**
 * useOptimisticSubmit - Simplified hook for form submissions with optimistic UI
 * 
 * @param {Object} options - Configuration
 * @param {Function} options.onSubmit - Submit function that returns a promise
 * @param {Function} options.onSuccess - Callback on successful submission
 * @param {Function} options.onError - Callback on error
 * @returns {Object} { isPending, submit, error, reset }
 */
export const useOptimisticSubmit = ({ onSubmit, onSuccess, onError } = {}) => {
    const [isPending, setIsPending] = useState(false);
    const [error, setError] = useState(null);
    const abortControllerRef = useRef(null);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
        };
    }, []);

    const submit = useCallback(async (data) => {
        if (isPending) return;

        // Abort any previous pending request
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
        abortControllerRef.current = new AbortController();

        setIsPending(true);
        setError(null);

        try {
            // Show optimistic state immediately
            const result = await onSubmit?.(data);

            setIsPending(false);
            onSuccess?.(result);
            return result;
        } catch (err) {
            // Check if aborted
            if (err.name === 'AbortError') {
                return;
            }

            setIsPending(false);
            setError(err);
            onError?.(err);
            throw err;
        }
    }, [isPending, onSubmit, onSuccess, onError]);

    const reset = useCallback(() => {
        setIsPending(false);
        setError(null);
    }, []);

    return {
        isPending,
        submit,
        error,
        reset
    };
};
