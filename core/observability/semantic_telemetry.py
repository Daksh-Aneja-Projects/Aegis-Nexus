"""
Semantic Telemetry for Aegis Nexus Governance Operations.

This module provides specialized telemetry for AI governance, including:
- Semantic drift visualization
- Constitutional verification tracing
- Reality anchor monitoring
- Governance decision path recording

INTEGRATES WITH:
- Glass Box Replay Engine (frontend)
- Prometheus metrics
- OpenTelemetry distributed tracing
"""

import logging
import time
import hashlib
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from datetime import datetime
from functools import wraps
from enum import Enum

from core.observability.tracing import (
    tracing_manager,
    SemanticAttributes,
    get_current_trace_id,
    OTEL_AVAILABLE
)

logger = logging.getLogger(__name__)


# =============================================================================
# GOVERNANCE DECISION RECORDING
# =============================================================================

class GovernancePhase(Enum):
    """Phases of the Aegis Nexus governance pipeline."""
    INGRESS = "ingress"
    LLM_PROCESSING = "llm_processing"
    ADVERSARIAL_AUDIT = "adversarial_audit"
    FORMAL_VERIFICATION = "formal_verification"
    BATTLE_ROOM = "battle_room"
    CONSENSUS = "consensus"
    EXECUTION = "execution"
    AUDIT_LOG = "audit_log"


@dataclass
class GovernanceEvent:
    """A single governance decision event for replay."""
    event_id: str
    trace_id: str
    phase: GovernancePhase
    timestamp: float
    duration_ms: float
    verdict: Optional[str] = None
    invariants: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernanceDecisionPath:
    """Complete decision path through the governance pipeline."""
    trace_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    final_verdict: Optional[str] = None
    events: List[GovernanceEvent] = field(default_factory=list)
    proof_hash: Optional[str] = None
    
    def add_event(self, event: GovernanceEvent):
        self.events.append(event)
        
    def finalize(self, verdict: str, proof: Optional[str] = None):
        self.completed_at = datetime.utcnow()
        self.final_verdict = verdict
        if proof:
            self.proof_hash = hashlib.sha256(proof.encode()).hexdigest()[:16]
            
    def to_replay_format(self) -> Dict[str, Any]:
        """Convert to Glass Box Replay format for frontend visualization."""
        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "final_verdict": self.final_verdict,
            "proof_hash": self.proof_hash,
            "decision_timeline": [
                {
                    "step": idx + 1,
                    "phase": event.phase.value,
                    "timestamp": event.timestamp,
                    "duration_ms": event.duration_ms,
                    "verdict": event.verdict,
                    "invariants_checked": event.invariants,
                    "confidence": event.confidence,
                    "details": event.metadata
                }
                for idx, event in enumerate(self.events)
            ],
            "total_duration_ms": sum(e.duration_ms for e in self.events)
        }


# Active decision paths (in-memory for now, could be Redis-backed)
_active_paths: Dict[str, GovernanceDecisionPath] = {}


def start_governance_path(trace_id: Optional[str] = None) -> GovernanceDecisionPath:
    """Start recording a new governance decision path."""
    trace_id = trace_id or get_current_trace_id()
    path = GovernanceDecisionPath(
        trace_id=trace_id,
        started_at=datetime.utcnow()
    )
    _active_paths[trace_id] = path
    return path


def get_governance_path(trace_id: str) -> Optional[GovernanceDecisionPath]:
    """Get an active governance decision path."""
    return _active_paths.get(trace_id)


def complete_governance_path(trace_id: str, verdict: str, proof: Optional[str] = None) -> Optional[GovernanceDecisionPath]:
    """Complete and finalize a governance decision path."""
    path = _active_paths.pop(trace_id, None)
    if path:
        path.finalize(verdict, proof)
    return path


# =============================================================================
# SEMANTIC DRIFT VISUALIZATION
# =============================================================================

@dataclass
class DriftMetric:
    """Tracks semantic or sensor drift over time."""
    metric_name: str
    timestamp: float
    current_value: float
    expected_value: float
    deviation: float
    severity: str  # 'normal', 'warning', 'critical'
    source: str


class DriftMonitor:
    """
    Monitors semantic drift in AI governance decisions.
    
    Semantic drift occurs when:
    - Model outputs gradually deviate from constitutional norms
    - Sensor fusion diverges from expected physical models
    - Governance decisions trend toward edge cases
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: List[DriftMetric] = []
        self.thresholds = {
            "normal": 0.1,
            "warning": 0.25,
            "critical": 0.5
        }
        
    def record_metric(
        self,
        metric_name: str,
        current_value: float,
        expected_value: float,
        source: str = "unknown"
    ) -> DriftMetric:
        """Record a drift metric and return analysis."""
        deviation = abs(current_value - expected_value) / max(abs(expected_value), 1e-10)
        
        # Determine severity
        if deviation >= self.thresholds["critical"]:
            severity = "critical"
        elif deviation >= self.thresholds["warning"]:
            severity = "warning"
        else:
            severity = "normal"
            
        metric = DriftMetric(
            metric_name=metric_name,
            timestamp=time.time(),
            current_value=current_value,
            expected_value=expected_value,
            deviation=deviation,
            severity=severity,
            source=source
        )
        
        # Maintain sliding window
        self.history.append(metric)
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        # Add to trace if available
        with tracing_manager.start_span("drift_metric") as span:
            span.set_attribute(f"aegis.drift.{metric_name}", deviation)
            span.set_attribute(f"aegis.drift.severity", severity)
            
        return metric
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of recent drift metrics for visualization."""
        if not self.history:
            return {"status": "no_data", "metrics": []}
            
        return {
            "status": "active",
            "window_size": self.window_size,
            "metrics_count": len(self.history),
            "critical_count": sum(1 for m in self.history if m.severity == "critical"),
            "warning_count": sum(1 for m in self.history if m.severity == "warning"),
            "average_deviation": sum(m.deviation for m in self.history) / len(self.history),
            "recent_metrics": [asdict(m) for m in self.history[-10:]]
        }


# Global drift monitor instance
drift_monitor = DriftMonitor()


# =============================================================================
# GOVERNANCE TRACING DECORATORS
# =============================================================================

def trace_governance_phase(phase: GovernancePhase, include_args: bool = False):
    """
    Decorator to trace a governance phase with semantic telemetry.
    
    Usage:
        @trace_governance_phase(GovernancePhase.FORMAL_VERIFICATION)
        async def verify_with_z3(action_code: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            trace_id = get_current_trace_id()
            start_time = time.time()
            
            with tracing_manager.start_span(f"governance.{phase.value}") as span:
                span.set_attribute(SemanticAttributes.GOVERNANCE_PHASE, phase.value)
                span.set_attribute(SemanticAttributes.TRACE_ID, trace_id)
                
                if include_args and args:
                    span.set_attribute("aegis.input_hash", 
                                       hashlib.sha256(str(args[0]).encode()).hexdigest()[:16])
                
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Record event
                    verdict = None
                    invariants = []
                    confidence = 0.0
                    
                    if hasattr(result, 'is_safe'):
                        verdict = "SAFE" if result.is_safe else "VIOLATION"
                        span.set_attribute(SemanticAttributes.GOVERNANCE_VERDICT, verdict)
                        
                    if hasattr(result, 'satisfied_invariants'):
                        invariants = result.satisfied_invariants
                        span.set_attribute(SemanticAttributes.Z3_INVARIANTS_SATISFIED, 
                                          len(invariants))
                                          
                    if hasattr(result, 'confidence'):
                        confidence = result.confidence
                        span.set_attribute(SemanticAttributes.GOVERNANCE_CONFIDENCE, confidence)
                    
                    # Add to decision path if active
                    path = get_governance_path(trace_id)
                    if path:
                        event = GovernanceEvent(
                            event_id=f"{trace_id}:{phase.value}",
                            trace_id=trace_id,
                            phase=phase,
                            timestamp=start_time,
                            duration_ms=duration_ms,
                            verdict=verdict,
                            invariants=invariants,
                            confidence=confidence
                        )
                        path.add_event(event)
                    
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    if OTEL_AVAILABLE:
                        from opentelemetry.trace import StatusCode
                        span.set_status(StatusCode.ERROR, str(e))
                    else:
                        span.set_status("ERROR", str(e))
                    raise
                    
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            trace_id = get_current_trace_id()
            start_time = time.time()
            
            with tracing_manager.start_span(f"governance.{phase.value}") as span:
                span.set_attribute(SemanticAttributes.GOVERNANCE_PHASE, phase.value)
                
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_attribute(SemanticAttributes.Z3_SOLVER_TIME_MS, duration_ms)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status("ERROR", str(e))
                    raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
        
    return decorator


def trace_reality_anchor(sensor_type: str = None):
    """
    Decorator for Reality Anchor (sensor fusion) tracing.
    
    Usage:
        @trace_reality_anchor(sensor_type="lidar")
        async def process_lidar_frame(frame_data):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tracing_manager.start_span(f"reality_anchor.{func.__name__}") as span:
                if sensor_type:
                    span.set_attribute(SemanticAttributes.SENSOR_SOURCE, sensor_type)
                    
                try:
                    result = await func(*args, **kwargs)
                    
                    # Extract sensor metrics if available
                    if isinstance(result, dict):
                        if 'confidence' in result:
                            span.set_attribute(SemanticAttributes.SENSOR_CONFIDENCE, 
                                              result['confidence'])
                        if 'anomaly_score' in result:
                            span.set_attribute(SemanticAttributes.SENSOR_ANOMALY_SCORE,
                                              result['anomaly_score'])
                        if 'divergence' in result:
                            span.set_attribute(SemanticAttributes.FUSION_DIVERGENCE,
                                              result['divergence'])
                    
                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise
                    
        return async_wrapper
    return decorator


# =============================================================================
# PROMETHEUS METRICS INTEGRATION
# =============================================================================

try:
    from prometheus_client import Histogram, Counter, Gauge
    
    GOVERNANCE_PHASE_DURATION = Histogram(
        'aegis_governance_phase_seconds',
        'Duration of each governance phase',
        ['phase', 'verdict'],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    )
    
    GOVERNANCE_DECISIONS = Counter(
        'aegis_governance_decisions_total',
        'Total governance decisions',
        ['verdict', 'phase']
    )
    
    SEMANTIC_DRIFT_GAUGE = Gauge(
        'aegis_semantic_drift',
        'Current semantic drift level',
        ['metric_name', 'source']
    )
    
    PROMETHEUS_AVAILABLE = True
    
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.debug("Prometheus client not available - semantic telemetry metrics disabled")


def record_governance_metrics(
    phase: GovernancePhase,
    duration_seconds: float,
    verdict: str
):
    """Record governance metrics to Prometheus."""
    if PROMETHEUS_AVAILABLE:
        GOVERNANCE_PHASE_DURATION.labels(
            phase=phase.value,
            verdict=verdict
        ).observe(duration_seconds)
        
        GOVERNANCE_DECISIONS.labels(
            verdict=verdict,
            phase=phase.value
        ).inc()


def record_drift_metric(metric_name: str, value: float, source: str = "unknown"):
    """Record drift metric to Prometheus gauge."""
    if PROMETHEUS_AVAILABLE:
        SEMANTIC_DRIFT_GAUGE.labels(
            metric_name=metric_name,
            source=source
        ).set(value)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'GovernancePhase',
    'GovernanceEvent',
    'GovernanceDecisionPath',
    'DriftMetric',
    'DriftMonitor',
    'drift_monitor',
    'start_governance_path',
    'get_governance_path',
    'complete_governance_path',
    'trace_governance_phase',
    'trace_reality_anchor',
    'record_governance_metrics',
    'record_drift_metric',
]
