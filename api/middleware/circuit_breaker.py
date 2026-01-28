"""
Distributed Cognitive Circuit Breaker for Aegis Nexus
Implements cluster-wide cognitive load monitoring and graceful degradation.

This module provides a distributed circuit breaker that tracks global "Cognitive Load" 
and "Entropy" across all pods to prevent resource exhaustion attacks.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import redis.asyncio as redis
from fastapi import Request, HTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

# =============================================================================
# ATOMIC LUA SCRIPTS - Eliminate TOCTOU Race Conditions
# All state transitions happen inside Redis atomically
# =============================================================================

# Atomic request increment with threshold check
# Returns: 1 if allowed, 0 if threshold exceeded
ATOMIC_INCREMENT_SCRIPT = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
local threshold = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
if current < threshold then
    redis.call('INCR', KEYS[1])
    redis.call('EXPIRE', KEYS[1], ttl)
    return current + 1  -- Return new count
else
    return -1  -- Signal: threshold exceeded
end
"""

# Atomic decrement with floor at zero
ATOMIC_DECREMENT_SCRIPT = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
if current > 0 then
    return redis.call('DECR', KEYS[1])
else
    redis.call('SET', KEYS[1], 0)
    return 0
end
"""


# Welford's Online Algorithm for running mean/variance (entropy proxy)
# Computes variance incrementally in O(1) time per update
# Returns: current variance (0 if count < 2)
WELFORD_UPDATE_SCRIPT = """
local count = tonumber(redis.call('HGET', KEYS[1], 'count') or '0')
local mean = tonumber(redis.call('HGET', KEYS[1], 'mean') or '0')
local m2 = tonumber(redis.call('HGET', KEYS[1], 'm2') or '0')
local new_value = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])

count = count + 1
local delta = new_value - mean
mean = mean + delta / count
local delta2 = new_value - mean
m2 = m2 + delta * delta2

redis.call('HMSET', KEYS[1], 'count', tostring(count), 'mean', tostring(mean), 'm2', tostring(m2))
redis.call('EXPIRE', KEYS[1], ttl)

if count < 2 then 
    return '0'
end
return tostring(m2 / (count - 1))  -- Return variance
"""

# New Atomic Script: Update Cognitive Complexity (Z3 Metrics)
UPDATE_COGNITIVE_LOAD_SCRIPT = """
local current_load = tonumber(redis.call('GET', KEYS[1]) or '0')
local new_complexity = tonumber(ARGV[1]) -- e.g., solver cost
local decay_factor = tonumber(ARGV[2])   -- 0.9 for semantic backoff
local ttl = tonumber(ARGV[3])

-- Exponential Moving Average (EMA) for load
current_load = (current_load * decay_factor) + (new_complexity * (1 - decay_factor))

redis.call('SET', KEYS[1], current_load)
redis.call('EXPIRE', KEYS[1], ttl)

return tostring(current_load)
"""

# Atomic latency streak check with auto-trip
# Returns: current streak count, or -1 if circuit was tripped
ATOMIC_LATENCY_CHECK_SCRIPT = """
local current_streak = tonumber(redis.call('GET', KEYS[1]) or '0')
local is_slow = tonumber(ARGV[1])  -- 1 for slow, 0 for fast
local streak_threshold = tonumber(ARGV[2])
local state_key = KEYS[2]
local ttl = tonumber(ARGV[3])

if is_slow == 1 then
    current_streak = redis.call('INCR', KEYS[1])
    redis.call('EXPIRE', KEYS[1], ttl)
    
    if tonumber(current_streak) >= streak_threshold then
        redis.call('SET', state_key, 'open')
        redis.call('SET', KEYS[1], 0)
        return -1  -- Signal: circuit tripped
    end
    return current_streak
else
    redis.call('SET', KEYS[1], 0)
    return 0
end
"""

# Comprehensive atomic allow-request check
# Combines state check, increment, and threshold validation into ONE atomic op
ATOMIC_ALLOW_REQUEST_SCRIPT = """
local state_key = KEYS[1]
local active_key = KEYS[2]
local lockout_key = KEYS[3]
local entropy_key = KEYS[4]

local max_requests = tonumber(ARGV[1])
local entropy_threshold = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])

-- Check circuit state
local state = redis.call('GET', state_key)
if state == 'open' then
    return 'CIRCUIT_OPEN'
end

-- Check hardware lockout
local lockout = redis.call('GET', lockout_key)
if lockout then
    return 'HARDWARE_LOCKOUT'
end

-- Check and increment active requests atomically
local current_active = tonumber(redis.call('GET', active_key) or '0')
if current_active >= max_requests then
    return 'MAX_REQUESTS_EXCEEDED:' .. tostring(current_active)
end

-- Check entropy (if tracked)
local entropy_count = tonumber(redis.call('HGET', entropy_key, 'count') or '0')
if entropy_count >= 2 then
    local m2 = tonumber(redis.call('HGET', entropy_key, 'm2') or '0')
    local variance = m2 / (entropy_count - 1)
    -- Normalize variance to 0-1 range (assuming max variance of 10000)
    local normalized_entropy = math.min(variance / 10000, 1.0)
    if normalized_entropy > entropy_threshold then
        redis.call('SET', state_key, 'open')
        return 'HIGH_ENTROPY:' .. string.format('%.2f', normalized_entropy)
    end
end

-- All checks passed: increment and allow
redis.call('INCR', active_key)
redis.call('EXPIRE', active_key, ttl)
return 'ALLOWED:' .. tostring(current_active + 1)
"""

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Tripped, rejecting requests
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CachedCircuitState:
    """L1 Cache entry for circuit state."""
    state: CircuitState
    cached_at: float
    ttl_seconds: float = 1.0  # L1 cache TTL
    
    def is_valid(self) -> bool:
        return (time.time() - self.cached_at) < self.ttl_seconds


@dataclass
class CircuitMetrics:
    """Metrics tracked by the cognitive circuit breaker."""
    cognitive_load: float = 0.0
    entropy_level: float = 0.0
    hallucination_rate: float = 0.0
    active_requests: int = 0
    average_latency_ms: float = 0.0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()


class DistributedCognitiveCircuitBreaker:
    """
    Distributed cognitive circuit breaker using Redis for shared state.
    
    Tracks global cognitive load and entropy across all pods to prevent
    "Cognitive DDoS" attacks where adversaries flood the system with 
    paradoxical prompts designed to maximize computational cost.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        failure_threshold: float = 0.7,  # Trip when entropy > 70%
        recovery_timeout: int = 30,      # Seconds to wait before half-open
        window_size: int = 60,           # Time window for metrics (seconds)
        max_active_requests: int = 100   # Max concurrent requests
    ):
        """
        Initialize the distributed cognitive circuit breaker.
        
        Args:
            redis_url: Redis connection URL
            failure_threshold: Entropy threshold to trip circuit
            recovery_timeout: Time to wait before attempting recovery
            window_size: Time window for rolling metrics
            max_active_requests: Maximum concurrent requests allowed
        """
        self.redis_url = redis_url
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.window_size = window_size
        self.max_active_requests = max_active_requests
        
        self.redis_client: Optional[redis.Redis] = None
        self.metrics_key = "aegis_nexus:cognitive_metrics"
        self.state_key = "aegis_nexus:circuit_state"
        self.active_requests_key = "aegis_nexus:active_requests"
        self.latency_streak_key = "aegis_nexus:latency_streak"
        self.entropy_welford_key = "aegis_nexus:entropy_welford"
        self.lockout_key = "AEGIS_SYSTEM_LOCKOUT"
        
        # Script SHA caches (populated on initialize)
        self._script_shas: Dict[str, str] = {}
        
        # L1 Cache (Local Memory)
        self._l1_cache: Optional[CachedCircuitState] = None
        
    async def initialize(self) -> bool:
        """Initialize the Redis connection and register Lua scripts."""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=False
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Pre-register Lua scripts for EVALSHA (faster than EVAL)
            # Redis caches the compiled script and returns a SHA
            self._script_shas['increment'] = await self.redis_client.script_load(ATOMIC_INCREMENT_SCRIPT)
            self._script_shas['decrement'] = await self.redis_client.script_load(ATOMIC_DECREMENT_SCRIPT)
            self._script_shas['welford'] = await self.redis_client.script_load(WELFORD_UPDATE_SCRIPT)
            self._script_shas['latency'] = await self.redis_client.script_load(ATOMIC_LATENCY_CHECK_SCRIPT)
            self._script_shas['allow'] = await self.redis_client.script_load(ATOMIC_ALLOW_REQUEST_SCRIPT)
            self._script_shas['cognitive_load'] = await self.redis_client.script_load(UPDATE_COGNITIVE_LOAD_SCRIPT)
            
            logger.info("🧠 Distributed cognitive circuit breaker initialized with atomic Lua scripts")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize cognitive circuit breaker: {str(e)}")
            return False
    
    async def get_metrics(self) -> CircuitMetrics:
        """Get current circuit breaker metrics from Redis."""
        if not self.redis_client:
            return CircuitMetrics()
        
        try:
            # Get metrics from Redis hash
            metrics_data = await self.redis_client.hgetall(self.metrics_key)
            
            if not metrics_data:
                return CircuitMetrics()
            
            # Decode bytes to proper types
            decoded_metrics = {}
            for key, value in metrics_data.items():
                key_str = key.decode() if isinstance(key, bytes) else key
                value_str = value.decode() if isinstance(value, bytes) else value
                
                # Convert to appropriate type
                if key_str in ['cognitive_load', 'entropy_level', 'hallucination_rate', 'average_latency_ms']:
                    decoded_metrics[key_str] = float(value_str)
                elif key_str == 'active_requests':
                    decoded_metrics[key_str] = int(value_str)
                elif key_str == 'last_updated':
                    try:
                        decoded_metrics[key_str] = datetime.fromisoformat(value_str)
                    except ValueError:
                        decoded_metrics[key_str] = datetime.utcnow()
            
            return CircuitMetrics(**decoded_metrics)
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to get circuit metrics: {str(e)}")
            return CircuitMetrics()
    
    async def update_metrics(self, **kwargs) -> CircuitMetrics:
        """Update circuit breaker metrics in Redis."""
        if not self.redis_client:
            return CircuitMetrics()
        
        try:
            # Get current metrics
            current = await self.get_metrics()
            
            # Update with new values
            for key, value in kwargs.items():
                if hasattr(current, key):
                    setattr(current, key, value)
            
            current.last_updated = datetime.utcnow()
            
            # Store in Redis
            pipeline = self.redis_client.pipeline()
            
            # Set metrics as hash
            metric_dict = {
                'cognitive_load': str(current.cognitive_load),
                'entropy_level': str(current.entropy_level),
                'hallucination_rate': str(current.hallucination_rate),
                'active_requests': str(current.active_requests),
                'average_latency_ms': str(current.average_latency_ms),
                'last_updated': current.last_updated.isoformat()
            }
            
            pipeline.hset(self.metrics_key, mapping=metric_dict)
            
            # Set expiration
            pipeline.expire(self.metrics_key, self.window_size)
            
            await pipeline.execute()
            
            return current
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to update circuit metrics: {str(e)}")
            return CircuitMetrics()
    
    async def record_latency(self, latency_seconds: float):
        """
        Record request latency and check for Fail-Fast conditions.
        
        Fail-Fast Rule: If 5 consecutive requests take > 2 seconds, trip the circuit.
        """
        if not self.redis_client:
            return
            
        latency_ms = latency_seconds * 1000
        
        try:
            # We use a primitive Moving Average for simplicity here, or store it elsewhere.
            # For this specific task, we focus on the "Consecutive Streak" logic.
            
            if latency_seconds > 2.0:
                # Increment streak
                streak = await self.redis_client.incr(self.latency_streak_key)
                
                # Check threshold
                if streak >= 5:
                    await self.trip_circuit(f"Fail-Fast System Protection: {streak} consecutive slow requests (>2s)")
                    # Reset streak after tripping to avoid immediate re-trip on half-open
                    await self.redis_client.set(self.latency_streak_key, 0)
            else:
                # Reset streak on successful fast request
                # Only reset if it exists to avoid unnecessary writes?
                # A simple SET 0 is cheap.
                await self.redis_client.set(self.latency_streak_key, 0)
                
                # Optionally update moving average latency here
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to record latency metrics: {e}")

    async def record_solver_stats(self, conflicts: int, decisions: int):
        """
        Record Z3 Solver statistics to track Semantic Cognitive Load.
        """
        if not self.redis_client:
            return
            
        try:
            # Semantic Complexity Metric: (conflicts * 2) + decisions
            # This is a proxy for how "hard" the thinker is thinking
            complexity_score = (conflicts * 2) + (decisions * 0.1)
            
            # Normalize reasonably (0-100 range expected)
            normalized_score = min(complexity_score / 1000.0, 10.0)
            
            cognitive_load_key = "aegis_nexus:cognitive_load_ema"
            
            # SANITIZATION: Strict Float
            score_safe = float(normalized_score)
            
            result = await self.redis_client.evalsha(
                self._script_shas['cognitive_load'],
                1, # keys
                cognitive_load_key,
                score_safe,
                0.9, # decay
                int(self.window_size)
            )
            
            # If load is too high, update metrics for circuit breaker to see
            current_load = float(result)
            if current_load > self.failure_threshold * 10: # simple scaling
                await self.update_metrics(cognitive_load=current_load)
                
                if current_load > 8.0: # Hard Threshold
                    await self.trip_circuit(f"Cognitive Load Spike: {current_load:.2f} (Z3 Stress)")
                    
        except Exception as e:
            logger.warning(f"⚠️ Failed to record solver stats: {e}")

    # ... get_state/set_state ...

    async def increment_active_requests_atomic(self) -> tuple[bool, int]:
        """
        ATOMIC increment with threshold check.
        Eliminates TOCTOU race condition by combining check+increment in Redis.
        """
        if not self.redis_client:
            return True, 0  # Fail open if Redis unavailable
        
        try:
            # SANITIZATION
            max_reqs = int(self.max_active_requests)
            win_size = int(self.window_size)
            
            result = await self.redis_client.evalsha(
                self._script_shas['increment'],
                1,  # number of keys
                self.active_requests_key,
                max_reqs,
                win_size
            )
            count = int(result)
            if count == -1:
                return False, self.max_active_requests  # Threshold exceeded
            return True, count
        except Exception as e:
            logger.warning(f"⚠️  Atomic increment failed: {str(e)}")
            return True, 0  # Fail open
    
    async def increment_active_requests(self) -> int:
        """Legacy wrapper - use increment_active_requests_atomic for race-free ops."""
        allowed, count = await self.increment_active_requests_atomic()
        return count
    
    async def decrement_active_requests(self) -> int:
        """Decrement active request counter atomically with floor at zero."""
        if not self.redis_client:
            return 0
        
        try:
            result = await self.redis_client.evalsha(
                self._script_shas['decrement'],
                1,  # number of keys
                self.active_requests_key
            )
            return int(result)
        except Exception as e:
            logger.warning(f"⚠️  Atomic decrement failed: {str(e)}")
            return 0
    
    async def update_entropy_welford(self, value: float) -> float:
        """
        Update entropy using Welford's Online Algorithm.
        O(1) time and space complexity per update.
        
        Returns:
            Current variance (entropy proxy)
        """
        if not self.redis_client:
            return 0.0
        
        try:
            result = await self.redis_client.evalsha(
                self._script_shas['welford'],
                1,  # number of keys
                self.entropy_welford_key,
                value,
                self.window_size
            )
            return float(result.decode() if isinstance(result, bytes) else result)
        except Exception as e:
            logger.warning(f"⚠️  Welford entropy update failed: {str(e)}")
            return 0.0
    
    async def should_allow_request(self) -> tuple[bool, str]:
        """
        Request check with L1/L2 caching.
        
        L1 (Local Memory): 1 second TTL, avoids Redis round-trip
        L2 (Redis): Distributed state, canonical source
        """
        # L1 Cache Check (Fastest - no network)
        # If we know circuit is OPEN locally, reject immediately
        if self._l1_cache and self._l1_cache.is_valid():
            if self._l1_cache.state == CircuitState.OPEN:
                 return False, "Circuit breaker open (L1 cache)"
        
        # L2 Check (Redis - authoritative)
        result, reason = await self._should_allow_request_l2()
        
        # Update L1 cache
        # We fetch the definitive state to cache it
        # Optimization: We could infer state from result but get_state is safer
        # or we just cache OPEN if result is CIRCUIT_OPEN
        try:
            state = await self.get_state()
            self._l1_cache = CachedCircuitState(
                state=state,
                cached_at=time.time()
            )
        except Exception:
            pass # Don't fail request if cache update fails
            
        return result, reason

    async def _should_allow_request_l2(self) -> tuple[bool, str]:
        """
        ATOMIC request allowance check (L2 - Redis).
        
        Combines all checks (circuit state, lockout, active requests, entropy)
        into a SINGLE ATOMIC Redis operation to eliminate race conditions.
        """
        if not self.redis_client:
            return True, "Redis unavailable - failing open"
        
        try:
            # SANITIZATION (Gap 2): Ensure strict types for Lua ARGV
            # Prevent injection of weird strings or huge numbers
            max_reqs = int(self.max_active_requests)
            fail_thresh = float(self.failure_threshold)
            win_size = int(self.window_size)
            
            result = await self.redis_client.evalsha(
                self._script_shas['allow'],
                4,  # number of keys
                self.state_key,
                self.active_requests_key,
                self.lockout_key,
                self.entropy_welford_key,
                max_reqs,
                fail_thresh,
                win_size
            )
            
            result_str = result.decode() if isinstance(result, bytes) else str(result)
            
            if result_str.startswith('ALLOWED:'):
                count = result_str.split(':')[1]
                return True, f"Request allowed (active: {count})"
            elif result_str == 'CIRCUIT_OPEN':
                return False, "Circuit breaker open"
            elif result_str == 'HARDWARE_LOCKOUT':
                return False, "HARDWARE LOCKOUT ACTIVE: Proprioceptive Drift Detected"
            elif result_str.startswith('MAX_REQUESTS_EXCEEDED:'):
                count = result_str.split(':')[1]
                return False, f"Too many active requests: {count}/{self.max_active_requests}"
            elif result_str.startswith('HIGH_ENTROPY:'):
                entropy = result_str.split(':')[1]
                return False, f"High entropy detected: {entropy}"
            else:
                logger.warning(f"Unknown atomic check result: {result_str}")
                return True, "Unknown result - failing open"
                
        except Exception as e:
            logger.warning(f"⚠️  Atomic allow-request check failed: {str(e)}")
            # Fallback to legacy non-atomic check
            return await self._should_allow_request_legacy()
    
    async def _should_allow_request_legacy(self) -> tuple[bool, str]:
        """
        Legacy non-atomic request check.
        WARNING: Has TOCTOU race condition. Use only as fallback.
        """
        state = await self.get_state()
        metrics = await self.get_metrics()
        
        if state == CircuitState.OPEN:
            return False, f"Circuit breaker open. Entropy: {metrics.entropy_level:.2f}"
            
        try:
            lockout = await self.redis_client.get(self.lockout_key)
            if lockout:
                return False, "HARDWARE LOCKOUT ACTIVE: Proprioceptive Drift Detected"
        except Exception:
            pass
        
        if metrics.active_requests >= self.max_active_requests:
            return False, f"Too many active requests: {metrics.active_requests}/{self.max_active_requests}"
        
        if metrics.cognitive_load > self.failure_threshold:
            await self.trip_circuit(f"High cognitive load: {metrics.cognitive_load:.2f}")
            return False, f"High cognitive load: {metrics.cognitive_load:.2f}"
        
        if metrics.entropy_level > self.failure_threshold:
            await self.trip_circuit(f"High entropy: {metrics.entropy_level:.2f}")
            return False, f"High entropy: {metrics.entropy_level:.2f}"
        
        return True, "Request allowed (legacy)"
    
    async def trip_circuit(self, reason: str):
        """Trip the circuit breaker open."""
        logger.warning(f"🚨 Tripping cognitive circuit breaker: {reason}")
        await self.set_state(CircuitState.OPEN)
        
        # Invalidate L1 cache immediately
        self._l1_cache = None
    
    async def attempt_reset(self):
        """Attempt to reset the circuit breaker to closed state."""
        logger.info("🔄 Attempting to reset cognitive circuit breaker")
        await self.set_state(CircuitState.CLOSED)
    
    # =========================================================================
    # HARDWARE LOCKOUT INTEGRATION (Gap 4.3)
    # =========================================================================
    # Integration point for Reality Anchor drift detection to trigger emergency lockout
    
    async def trigger_hardware_lockout(self, reason: str, ttl: int = 3600) -> bool:
        """
        Trigger hardware lockout - called by drift monitor when 6σ deviation detected.
        
        This immediately blocks ALL cognitive processing until manually cleared.
        
        Args:
            reason: Why lockout was triggered (for audit log)
            ttl: Time-to-live in seconds (default 1 hour, must be manually cleared usually)
            
        Returns:
            bool: Whether lockout was successfully set
        """
        if not self.redis_client:
            logger.error("❌ Cannot trigger hardware lockout: Redis unavailable")
            return False
        
        try:
            # Set lockout flag with TTL
            await self.redis_client.setex(
                self.lockout_key,
                ttl,
                f"LOCKOUT:{datetime.utcnow().isoformat()}:{reason}"
            )
            
            # Also trip the circuit breaker
            await self.trip_circuit(f"HARDWARE LOCKOUT: {reason}")
            
            logger.critical(f"🚨 HARDWARE LOCKOUT TRIGGERED: {reason}")
            logger.critical(f"   Lockout will auto-expire in {ttl}s unless manually cleared")
            
            # Broadcast to all WebSocket clients
            try:
                from api.v1.endpoints.websockets import manager
                await manager.broadcast(
                    "system_lockout",
                    {
                        "severity": "CRITICAL",
                        "event": "HARDWARE_LOCKOUT",
                        "reason": reason,
                        "timestamp": datetime.utcnow().isoformat(),
                        "ttl_seconds": ttl
                    },
                    trace_id="system"
                )
            except Exception:
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to trigger hardware lockout: {e}")
            return False
    
    async def clear_hardware_lockout(self, admin_token: str) -> bool:
        """
        Clear hardware lockout - requires admin authorization.
        
        Args:
            admin_token: Admin authentication token (validated externally)
            
        Returns:
            bool: Whether lockout was cleared
        """
        if not self.redis_client:
            return False
        
        try:
            # Remove lockout flag
            result = await self.redis_client.delete(self.lockout_key)
            
            # Reset circuit breaker
            await self.set_state(CircuitState.CLOSED)
            
            logger.info(f"✅ Hardware lockout cleared by admin")
            return result > 0
            
        except Exception as e:
            logger.error(f"❌ Failed to clear hardware lockout: {e}")
            return False
    
    async def is_hardware_lockout_active(self) -> tuple[bool, Optional[str]]:
        """
        Check if hardware lockout is currently active.
        
        Returns:
            Tuple of (is_active, reason)
        """
        if not self.redis_client:
            return False, None
        
        try:
            lockout_data = await self.redis_client.get(self.lockout_key)
            if lockout_data:
                reason = lockout_data.decode() if isinstance(lockout_data, bytes) else str(lockout_data)
                return True, reason
            return False, None
            
        except Exception:
            return False, None
        
    async def save_recovery_checkpoint(self, recovery_data: Dict[str, Any], ttl: int = 300) -> str:
        """
        Save a checkpoint for stateful reentry if the system fails.
        """
        if not self.redis_client:
            return ""
            
        checkpoint_id = f"checkpoint:{int(time.time())}:{hash(str(recovery_data)) % 10000}"
        key = f"aegis_nexus:recovery:{checkpoint_id}"
        
        try:
            # Serialize
            data_str = str(recovery_data) # In prod use proper JSON serialization
            await self.redis_client.setex(key, ttl, data_str)
            return checkpoint_id
        except Exception as e:
            logger.warning(f"⚠️ Failed to save recovery checkpoint: {e}")
            return ""

    async def load_recovery_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load a saved checkpoint."""
        if not self.redis_client:
            return None
            
        key = f"aegis_nexus:recovery:{checkpoint_id}"
        try:
            data = await self.redis_client.get(key)
            if data:
                return {"restored": True, "raw": data}
            return None
        except Exception:
            return None


class CognitiveCircuitBreakerMiddleware:
    """
    Middleware that integrates the distributed cognitive circuit breaker.
    
    Monitors cognitive load and blocks requests when thresholds are exceeded.
    """
    
    def __init__(
        self, 
        app: ASGIApp, 
        circuit_breaker: DistributedCognitiveCircuitBreaker,
        protected_endpoints: Optional[list[str]] = None
    ):
        """
        Initialize the circuit breaker middleware.
        
        Args:
            app: ASGI application
            circuit_breaker: Distributed cognitive circuit breaker instance
            protected_endpoints: List of endpoints to protect (if None, all are protected)
        """
        self.app = app
        self.circuit_breaker = circuit_breaker
        self.protected_endpoints = protected_endpoints or ["/api/v1/submit"]
        
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope)
        endpoint_path = request.url.path
        
        # Only apply to protected endpoints
        if not any(endpoint_path.startswith(path) for path in self.protected_endpoints):
            await self.app(scope, receive, send)
            return
        
        # Check if request should be allowed
        should_allow, reason = await self.circuit_breaker.should_allow_request()
        
        if not should_allow:
            # Return safe mode response
            await self._send_safe_mode_response(send, reason)
            return
        
        # Increment active request counter
        await self.circuit_breaker.increment_active_requests()
        
        start_time = time.time()
        
        try:
            # Wrap the original send to decrement counter after response
            original_send = send
            
            async def wrapped_send(message):
                if message["type"] == "http.response.body":
                    # Decrement counter when response body is sent
                    await self.circuit_breaker.decrement_active_requests()
                    
                    # Record Latency
                    duration = time.time() - start_time
                    await self.circuit_breaker.record_latency(duration)
                
                await original_send(message)
            
            await self.app(scope, receive, wrapped_send)
            
        except Exception as e:
            # Make sure to decrement counter even if there's an error
            await self.circuit_breaker.decrement_active_requests()
            # Still record latency for failed requests?
            # Yes, a slow crash is still slow.
            duration = time.time() - start_time
            await self.circuit_breaker.record_latency(duration)
            raise

    async def _send_safe_mode_response(self, send: Send, reason: str):
        response_body = {
            "error": "Cognitive system under high load",
            "message": "System temporarily in safe mode to prevent resource exhaustion",
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
            "safe_mode": True
        }
        
        body = str(response_body).encode()
        
        response_headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"x-circuit-breaker", b"tripped")
        ]
        
        await send({
            "type": "http.response.start",
            "status": 503,
            "headers": response_headers
        })
        
        await send({
            "type": "http.response.body",
            "body": body,
            "more_body": False
        })


# Global instance
cognitive_circuit_breaker: Optional[DistributedCognitiveCircuitBreaker] = None


async def initialize_cognitive_circuit_breaker(
    redis_url: str = "redis://localhost:6379",
    failure_threshold: float = 0.7,
    recovery_timeout: int = 30,
    window_size: int = 60,
    max_active_requests: int = 100
) -> bool:
    """
    Initialize the global cognitive circuit breaker instance.
    """
    global cognitive_circuit_breaker
    
    try:
        cognitive_circuit_breaker = DistributedCognitiveCircuitBreaker(
            redis_url=redis_url,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            window_size=window_size,
            max_active_requests=max_active_requests
        )
        
        return await cognitive_circuit_breaker.initialize()
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize cognitive circuit breaker: {str(e)}")
        return False


def get_cognitive_circuit_breaker() -> DistributedCognitiveCircuitBreaker:
    """Get the global cognitive circuit breaker instance."""
    if cognitive_circuit_breaker is None:
        raise RuntimeError("Cognitive circuit breaker not initialized")
    return cognitive_circuit_breaker
