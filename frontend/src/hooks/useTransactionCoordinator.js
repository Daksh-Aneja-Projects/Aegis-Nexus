/**
 * useTransactionCoordinator Hook
 * Aegis Nexus Frontend
 * 
 * Coordinates optimistic state updates with WebSocket verification streams.
 * This hook bridges the gap between:
 * - useOptimisticState (manages tentative vs verified state)
 * - useWebSocket (receives real-time verification updates)
 * 
 * PRODUCTION FEATURES:
 * - Automatic rollback on WebSocket disconnection during pending transactions
 * - Transaction queue with timeout tracking
 * - Trace ID correlation for debugging
 * - Toast notification integration for user feedback
 */

import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { useWebSocket, ConnectionState } from './useWebSocket';
import { useOptimisticState, STATE_STATUS } from './useOptimisticState';

// Transaction states
export const TRANSACTION_STATUS = {
    PENDING: 'pending',
    SUBMITTED: 'submitted',
    VERIFIED: 'verified',
    REJECTED: 'rejected',
    TIMEOUT: 'timeout',
    CONNECTION_LOST: 'connection_lost'
};

// Default configuration
const DEFAULT_CONFIG = {
    transactionTimeoutMs: 30000,     // 30s overall transaction timeout
    verificationTimeoutMs: 15000,    // 15s for backend verification
    maxPendingTransactions: 10,      // Queue limit
    autoRollbackOnDisconnect: true,  // Rollback if WebSocket fails during tx
    reconnectGracePeriodMs: 5000     // Grace period before rollback on disconnect
};

/**
 * Transaction record for tracking pending operations
 */
class Transaction {
    constructor(traceId, payload, callbacks = {}) {
        this.traceId = traceId;
        this.payload = payload;
        this.status = TRANSACTION_STATUS.PENDING;
        this.createdAt = Date.now();
        this.submittedAt = null;
        this.completedAt = null;
        this.result = null;
        this.callbacks = callbacks; // { onVerified, onRejected, onTimeout }
    }

    markSubmitted() {
        this.status = TRANSACTION_STATUS.SUBMITTED;
        this.submittedAt = Date.now();
    }

    complete(status, result = null) {
        this.status = status;
        this.result = result;
        this.completedAt = Date.now();
    }

    get age() {
        return Date.now() - this.createdAt;
    }

    get isExpired() {
        return this.age > DEFAULT_CONFIG.transactionTimeoutMs;
    }
}

/**
 * useTransactionCoordinator
 * 
 * Unified hook for transactional state management with WebSocket verification.
 * 
 * @param {Object} options Configuration options
 * @param {string} options.wsUrl WebSocket URL (defaults to global)
 * @param {Function} options.submitAction Async function to submit action to backend
 * @param {Function} options.onNotification Callback for toast/notification display
 * @param {Object} options.config Override default configuration
 */
export function useTransactionCoordinator({
    submitAction,
    onNotification,
    config = {}
} = {}) {
    const mergedConfig = useMemo(() => ({ ...DEFAULT_CONFIG, ...config }), [config]);

    // Transaction queue
    const transactionsRef = useRef(new Map());
    const [activeTransactions, setActiveTransactions] = useState([]);

    // Reconnection grace period tracking
    const disconnectedAtRef = useRef(null);
    const gracePeriodTimerRef = useRef(null);

    // Notification helper
    const notify = useCallback((type, message, data = {}) => {
        if (onNotification) {
            onNotification({ type, message, ...data });
        } else {
            // Fallback to console
            console.log(`[TransactionCoordinator] ${type}: ${message}`, data);
        }
    }, [onNotification]);

    /**
     * Handle incoming WebSocket messages for verification updates
     */
    const handleVerificationMessage = useCallback((message) => {
        if (message.type !== 'verification_update') return;

        const { trace_id, status, proof, reason } = message.payload || {};
        if (!trace_id) return;

        const transaction = transactionsRef.current.get(trace_id);
        if (!transaction) {
            // Not our transaction, might be from another client
            return;
        }

        if (status === 'VERIFIED') {
            transaction.complete(TRANSACTION_STATUS.VERIFIED, { proof });

            // Invoke callback
            if (transaction.callbacks.onVerified) {
                transaction.callbacks.onVerified(transaction);
            }

            notify('success', 'Action verified by governance', { traceId: trace_id });

        } else if (status === 'REJECTED') {
            transaction.complete(TRANSACTION_STATUS.REJECTED, { reason });

            if (transaction.callbacks.onRejected) {
                transaction.callbacks.onRejected(transaction, reason);
            }

            notify('error', reason || 'Action rejected by governance', { traceId: trace_id });
        }

        // Update active transactions state
        transactionsRef.current.delete(trace_id);
        updateActiveTransactions();

    }, [notify]);

    /**
     * WebSocket connection with verification stream handling
     */
    const {
        connectionState,
        isConnected,
        sendMessage,
        lastMessage
    } = useWebSocket({
        onMessage: handleVerificationMessage,
        onConnect: () => {
            // Clear grace period timer on reconnect
            if (gracePeriodTimerRef.current) {
                clearTimeout(gracePeriodTimerRef.current);
                gracePeriodTimerRef.current = null;
            }
            disconnectedAtRef.current = null;
            notify('info', 'Connected to verification stream');
        },
        onDisconnect: () => {
            disconnectedAtRef.current = Date.now();

            // Start grace period timer for pending transactions
            if (mergedConfig.autoRollbackOnDisconnect && transactionsRef.current.size > 0) {
                gracePeriodTimerRef.current = setTimeout(() => {
                    rollbackPendingTransactions('WebSocket disconnected');
                }, mergedConfig.reconnectGracePeriodMs);
            }
        }
    });

    // Process last message
    useEffect(() => {
        if (lastMessage) {
            handleVerificationMessage(lastMessage);
        }
    }, [lastMessage, handleVerificationMessage]);

    /**
     * Update the active transactions state for UI consumption
     */
    const updateActiveTransactions = useCallback(() => {
        const active = Array.from(transactionsRef.current.values()).map(tx => ({
            traceId: tx.traceId,
            status: tx.status,
            age: tx.age,
            payload: tx.payload
        }));
        setActiveTransactions(active);
    }, []);

    /**
     * Rollback all pending transactions due to connection failure
     */
    const rollbackPendingTransactions = useCallback((reason) => {
        const pendingTxs = Array.from(transactionsRef.current.entries());

        pendingTxs.forEach(([traceId, transaction]) => {
            if (transaction.status === TRANSACTION_STATUS.SUBMITTED ||
                transaction.status === TRANSACTION_STATUS.PENDING) {

                transaction.complete(TRANSACTION_STATUS.CONNECTION_LOST, { reason });

                if (transaction.callbacks.onTimeout) {
                    transaction.callbacks.onTimeout(transaction, reason);
                }
            }
        });

        transactionsRef.current.clear();
        updateActiveTransactions();

        if (pendingTxs.length > 0) {
            notify('warning', `${pendingTxs.length} pending transaction(s) rolled back: ${reason}`, {
                transactions: pendingTxs.map(([id]) => id)
            });
        }
    }, [notify, updateActiveTransactions]);

    /**
     * Check for expired transactions
     */
    useEffect(() => {
        const checkInterval = setInterval(() => {
            let expired = false;

            transactionsRef.current.forEach((tx, traceId) => {
                if (tx.isExpired && tx.status === TRANSACTION_STATUS.SUBMITTED) {
                    tx.complete(TRANSACTION_STATUS.TIMEOUT, { reason: 'Verification timeout' });

                    if (tx.callbacks.onTimeout) {
                        tx.callbacks.onTimeout(tx, 'Verification timeout');
                    }

                    notify('warning', 'Transaction timed out', { traceId });
                    expired = true;
                }
            });

            if (expired) {
                // Clean up expired
                transactionsRef.current.forEach((tx, id) => {
                    if (tx.status === TRANSACTION_STATUS.TIMEOUT) {
                        transactionsRef.current.delete(id);
                    }
                });
                updateActiveTransactions();
            }
        }, 5000); // Check every 5 seconds

        return () => clearInterval(checkInterval);
    }, [notify, updateActiveTransactions]);

    /**
     * Submit a transaction for governance verification
     * 
     * @param {Object} payload - The action payload to submit
     * @param {Object} callbacks - Optional callbacks { onVerified, onRejected, onTimeout }
     * @returns {Promise<{ traceId, submitted }>}
     */
    const submitTransaction = useCallback(async (payload, callbacks = {}) => {
        // Generate trace ID
        const traceId = `tx_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

        // Check queue limit
        if (transactionsRef.current.size >= mergedConfig.maxPendingTransactions) {
            notify('error', 'Too many pending transactions. Please wait.');
            return { traceId: null, submitted: false, error: 'QUEUE_FULL' };
        }

        // Check connection
        if (!isConnected) {
            notify('warning', 'Not connected to verification stream. Attempting anyway...');
        }

        // Create transaction record
        const transaction = new Transaction(traceId, payload, callbacks);
        transactionsRef.current.set(traceId, transaction);
        updateActiveTransactions();

        try {
            // Submit to backend
            if (submitAction) {
                await submitAction(payload, traceId);
            } else {
                // Default: send via WebSocket
                sendMessage({
                    type: 'submit_action',
                    trace_id: traceId,
                    payload
                });
            }

            transaction.markSubmitted();
            updateActiveTransactions();

            notify('info', 'Action submitted for verification', { traceId });

            return { traceId, submitted: true };

        } catch (error) {
            transaction.complete(TRANSACTION_STATUS.REJECTED, { error: error.message });
            transactionsRef.current.delete(traceId);
            updateActiveTransactions();

            notify('error', `Failed to submit: ${error.message}`, { traceId });

            return { traceId, submitted: false, error: error.message };
        }
    }, [isConnected, mergedConfig, notify, sendMessage, submitAction, updateActiveTransactions]);

    /**
     * Cancel a pending transaction (client-side only)
     */
    const cancelTransaction = useCallback((traceId) => {
        const transaction = transactionsRef.current.get(traceId);
        if (transaction && transaction.status !== TRANSACTION_STATUS.VERIFIED) {
            transaction.complete(TRANSACTION_STATUS.REJECTED, { reason: 'Cancelled by user' });
            transactionsRef.current.delete(traceId);
            updateActiveTransactions();
            notify('info', 'Transaction cancelled', { traceId });
        }
    }, [notify, updateActiveTransactions]);

    /**
     * Get pending transaction count
     */
    const pendingCount = useMemo(() => {
        return Array.from(transactionsRef.current.values())
            .filter(tx => tx.status === TRANSACTION_STATUS.PENDING ||
                tx.status === TRANSACTION_STATUS.SUBMITTED)
            .length;
    }, [activeTransactions]); // eslint-disable-line react-hooks/exhaustive-deps

    return {
        // Connection state
        connectionState,
        isConnected,

        // Transaction management
        submitTransaction,
        cancelTransaction,
        activeTransactions,
        pendingCount,

        // For direct WebSocket use
        sendMessage,

        // Status flags
        hasPendingTransactions: pendingCount > 0,
        isConnectionStable: isConnected && disconnectedAtRef.current === null
    };
}

/**
 * Hook combining TransactionCoordinator with OptimisticState
 * 
 * This is the "batteries included" solution for most UI components.
 * 
 * @param {any} initialState Initial confirmed state
 * @param {Function} submitAction Backend submission function
 * @param {Object} options Configuration
 */
export function useOptimisticTransaction(
    initialState,
    submitAction,
    options = {}
) {
    const { onNotification, config, verificationService } = options;

    // Optimistic state management
    const {
        state,
        setOptimistic,
        status: optimisticStatus,
        confirmedState,
        syncConfirmedState,
        reset,
        errorMessage,
        isPending,
        isError
    } = useOptimisticState(initialState, verificationService || (async () => ({ success: true })), {
        timeoutMs: config?.transactionTimeoutMs || 30000,
        onRollback: (rollbackData) => {
            if (onNotification) {
                onNotification({
                    type: 'warning',
                    message: `Action rolled back: ${rollbackData.reason}`,
                    traceId: rollbackData.traceId
                });
            }
        }
    });

    // Transaction coordination
    const {
        connectionState,
        isConnected,
        submitTransaction,
        activeTransactions,
        pendingCount,
        hasPendingTransactions,
        isConnectionStable
    } = useTransactionCoordinator({
        submitAction,
        onNotification,
        config
    });

    /**
     * Submit an optimistic update with transaction tracking
     */
    const submitOptimistic = useCallback(async (nextState, callbacks = {}) => {
        const traceId = `opt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

        // Optimistically update UI
        await setOptimistic(nextState, traceId);

        // Submit for backend verification
        const result = await submitTransaction(nextState, {
            onVerified: (tx) => {
                // Sync confirmed state on verification
                syncConfirmedState(nextState);
                callbacks.onVerified?.(tx);
            },
            onRejected: (tx, reason) => {
                // Optimistic state will auto-rollback via timeout
                // But we can force it here for faster UX
                reset();
                callbacks.onRejected?.(tx, reason);
            },
            onTimeout: (tx, reason) => {
                reset();
                callbacks.onTimeout?.(tx, reason);
            }
        });

        return result;
    }, [setOptimistic, submitTransaction, syncConfirmedState, reset]);

    return {
        // State
        state,
        confirmedState,

        // Actions
        submitOptimistic,
        reset,

        // Status
        status: optimisticStatus,
        connectionState,
        isConnected,
        isPending,
        isError,
        errorMessage,

        // Transaction info
        activeTransactions,
        pendingCount,
        hasPendingTransactions,
        isConnectionStable
    };
}

export default useTransactionCoordinator;
