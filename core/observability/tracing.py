"""
Observability and Distributed Tracing for Aegis Nexus.

This module provides a unified interface for tracing requests across the system.
Uses OpenTelemetry with OTLP export for production-grade distributed tracing.

PRODUCTION FEATURES:
- OTLP export to Jaeger/Tempo
- Semantic governance attributes
- Trace context propagation helpers
- Automatic instrumentation for FastAPI and Redis
"""

import logging
import os
import uuid
from typing import Optional, Dict, Any
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)

# Configuration from environment
OTEL_EXPORTER_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "aegis-nexus")
OTEL_ENABLED = os.getenv("OTEL_ENABLED", "true").lower() == "true"

# Try to import OpenTelemetry with full SDK
OTEL_AVAILABLE = False
tracer = None
meter = None

try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode, SpanKind
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.semconv.resource import ResourceAttributes
    
    # Try to import OTLP exporter
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        OTLP_AVAILABLE = True
    except ImportError:
        OTLP_AVAILABLE = False
        logger.warning("⚠️  OTLP exporter not available. Install opentelemetry-exporter-otlp.")
    
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("⚠️  OpenTelemetry SDK not installed. Using fallback tracing.")


# =============================================================================
# SEMANTIC ATTRIBUTE CONSTANTS - For searchable, structured traces
# =============================================================================

class SemanticAttributes:
    """Aegis Nexus semantic trace attributes for governance operations."""
    
    # Request/Transaction Attributes
    TRACE_ID = "aegis.trace_id"
    REQUEST_TYPE = "aegis.request_type"
    USER_ID = "aegis.user_id"
    
    # Governance Attributes
    GOVERNANCE_PHASE = "aegis.governance.phase"
    GOVERNANCE_VERDICT = "aegis.governance.verdict"
    GOVERNANCE_CONFIDENCE = "aegis.governance.confidence"
    
    # Z3 Verification Attributes
    Z3_RESULT = "aegis.z3.result"
    Z3_PROOF_HASH = "aegis.z3.proof_hash"
    Z3_INVARIANTS_SATISFIED = "aegis.z3.invariants_satisfied"
    Z3_INVARIANTS_VIOLATED = "aegis.z3.invariants_violated"
    Z3_SOLVER_TIME_MS = "aegis.z3.solver_time_ms"
    
    # Constitutional Attributes
    CONSTITUTION_VERSION = "aegis.constitution.version"
    INVARIANT_NAME = "aegis.invariant.name"
    INVARIANT_TYPE = "aegis.invariant.type"
    INVARIANT_RESULT = "aegis.invariant.result"
    
    # Reality Anchor Attributes
    SENSOR_SOURCE = "aegis.sensor.source"
    SENSOR_CONFIDENCE = "aegis.sensor.confidence"
    SENSOR_ANOMALY_SCORE = "aegis.sensor.anomaly_score"
    FUSION_DIVERGENCE = "aegis.fusion.divergence"
    
    # PQC Attributes
    PQC_ALGORITHM = "aegis.pqc.algorithm"
    PQC_MODE = "aegis.pqc.mode"  # 'pure_pqc', 'hybrid', 'fallback'
    PQC_SIGN_TIME_MS = "aegis.pqc.sign_time_ms"
    
    # Battle Room / Debate Attributes
    DEBATE_ID = "aegis.debate.id"
    DEBATE_STANCE = "aegis.debate.stance"
    DEBATE_ROUND = "aegis.debate.round"


def initialize_tracing() -> bool:
    """
    Initialize OpenTelemetry with OTLP export.
    
    Call this at application startup (e.g., in main.py or lifespan).
    """
    global tracer, meter, OTEL_AVAILABLE
    
    if not OTEL_ENABLED:
        logger.info("🔇 OpenTelemetry disabled via OTEL_ENABLED=false")
        return False
    
    if not OTEL_AVAILABLE:
        logger.warning("⚠️  OpenTelemetry not available - using fallback logging")
        return False
    
    try:
        # Create resource with service information
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: OTEL_SERVICE_NAME,
            ResourceAttributes.SERVICE_VERSION: os.getenv("AEGIS_VERSION", "2.1.0"),
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: os.getenv("AEGIS_ENV", "development"),
        })
        
        # Configure TracerProvider
        provider = TracerProvider(resource=resource)
        
        # Add OTLP exporter if available
        if OTLP_AVAILABLE:
            otlp_exporter = OTLPSpanExporter(
                endpoint=OTEL_EXPORTER_ENDPOINT,
                insecure=True  # Set to False for production with TLS
            )
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"📡 OTLP exporter configured: {OTEL_EXPORTER_ENDPOINT}")
        else:
            logger.info("📝 Using console span processor (no OTLP exporter)")
        
        # Set global tracer provider
        trace.set_tracer_provider(provider)
        
        # Get tracer instance
        tracer = trace.get_tracer(OTEL_SERVICE_NAME)
        
        # Configure MeterProvider for metrics (optional)
        try:
            meter_provider = MeterProvider(resource=resource)
            metrics.set_meter_provider(meter_provider)
            meter = metrics.get_meter(OTEL_SERVICE_NAME)
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize metrics: {e}")
        
        logger.info("✅ OpenTelemetry tracing initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize OpenTelemetry: {e}")
        OTEL_AVAILABLE = False
        return False


class TracingManager:
    """
    Manages distributed tracing across the Aegis Nexus system.
    Provides both context managers and decorators for span creation.
    """
    
    def __init__(self, service_name: str = "aegis-nexus"):
        self.service_name = service_name
        self._tracer = None
        
    @property
    def tracer(self):
        """Lazy-load tracer from global or get a new one."""
        if self._tracer is None:
            if OTEL_AVAILABLE and tracer:
                self._tracer = tracer
            elif OTEL_AVAILABLE:
                self._tracer = trace.get_tracer(self.service_name)
        return self._tracer

    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None, kind: str = "internal"):
        """
        Start a new trace span.
        Returns a context manager yielding the span.
        
        Args:
            name: Span name (e.g., 'z3_verification', 'pqc_sign')
            attributes: Initial span attributes
            kind: Span kind - 'internal', 'server', 'client', 'producer', 'consumer'
        """
        if OTEL_AVAILABLE and self.tracer:
            span_kind = {
                "internal": SpanKind.INTERNAL,
                "server": SpanKind.SERVER,
                "client": SpanKind.CLIENT,
                "producer": SpanKind.PRODUCER,
                "consumer": SpanKind.CONSUMER
            }.get(kind, SpanKind.INTERNAL)
            
            return self.tracer.start_as_current_span(
                name, 
                attributes=attributes or {},
                kind=span_kind
            )
        else:
            return self._fallback_span(name, attributes)

    @contextmanager
    def _fallback_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Fallback context manager for when OTel is missing."""
        span_id = str(uuid.uuid4())[:8]
        trace_id = str(uuid.uuid4())[:16]
        
        # Log start with structured data
        attrs_str = str(attributes) if attributes else ""
        logger.info(f"➡️  [TRACE START] {name} (trace={trace_id}, span={span_id}) {attrs_str}")
        
        class MockSpan:
            """Mock span for fallback logging."""
            def __init__(self):
                self.attributes = attributes or {}
                self._span_id = span_id
                self._trace_id = trace_id
                
            def set_attribute(self, key: str, value: Any):
                self.attributes[key] = value
                logger.debug(f"📝 [TRACE ATTR] {name} ({span_id}): {key}={value}")
                
            def set_status(self, status, description=None):
                if hasattr(status, 'name'):
                    status_name = status.name
                else:
                    status_name = str(status)
                    
                if status_name == "ERROR" or status == "ERROR":
                    logger.error(f"❌ [TRACE ERROR] {name} ({span_id}): {description}")
                else:
                    logger.debug(f"✓ [TRACE STATUS] {name} ({span_id}): {status_name}")
                    
            def record_exception(self, e):
                logger.error(f"❌ [TRACE EXCEPTION] {name} ({span_id}): {str(e)}")
                
            def add_event(self, name: str, attributes: Dict = None):
                logger.info(f"📌 [TRACE EVENT] {name} ({span_id}): {attributes}")
                
            def get_span_context(self):
                """Return mock span context for trace ID propagation."""
                class MockContext:
                    trace_id = int(trace_id, 16) if trace_id else 0
                    span_id = int(span_id, 16) if len(span_id) <= 8 else 0
                return MockContext()
        
        mock_span = MockSpan()
        try:
            yield mock_span
        finally:
            logger.info(f"⬅️  [TRACE END] {name} (trace={trace_id}, span={span_id})")


# =============================================================================
# DECORATORS - For easy instrumentation
# =============================================================================

def trace_governance(operation_name: str = None, phase: str = None):
    """
    Decorator to trace governance operations with semantic attributes.
    
    Usage:
        @trace_governance(operation_name="verify_safeguards", phase="formal_verification")
        async def verify_action(action_code: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            span_name = operation_name or func.__name__
            with tracing_manager.start_span(span_name) as span:
                if phase:
                    span.set_attribute(SemanticAttributes.GOVERNANCE_PHASE, phase)
                    
                try:
                    result = await func(*args, **kwargs)
                    
                    # Try to extract verdict from result
                    if hasattr(result, 'is_safe'):
                        span.set_attribute(SemanticAttributes.GOVERNANCE_VERDICT, 
                                          "SAFE" if result.is_safe else "VIOLATION")
                    elif isinstance(result, dict) and 'is_safe' in result:
                        span.set_attribute(SemanticAttributes.GOVERNANCE_VERDICT,
                                          "SAFE" if result['is_safe'] else "VIOLATION")
                    
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(StatusCode.ERROR if OTEL_AVAILABLE else "ERROR", str(e))
                    raise
                    
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            span_name = operation_name or func.__name__
            with tracing_manager.start_span(span_name) as span:
                if phase:
                    span.set_attribute(SemanticAttributes.GOVERNANCE_PHASE, phase)
                    
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(StatusCode.ERROR if OTEL_AVAILABLE else "ERROR", str(e))
                    raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
        
    return decorator


def get_current_trace_id() -> str:
    """
    Get the current trace ID for propagation (e.g., in WebSocket messages).
    Returns a hex string trace ID.
    """
    if OTEL_AVAILABLE:
        try:
            current_span = trace.get_current_span()
            if current_span:
                ctx = current_span.get_span_context()
                if ctx and ctx.trace_id:
                    return format(ctx.trace_id, '032x')
        except Exception:
            pass
    
    # Fallback: generate a new UUID-based trace ID
    return str(uuid.uuid4()).replace('-', '')[:32]


def inject_trace_context(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Inject trace context into HTTP headers for propagation.
    Uses W3C Trace Context format.
    """
    if OTEL_AVAILABLE:
        try:
            from opentelemetry.propagate import inject
            inject(headers)
        except Exception as e:
            logger.warning(f"Failed to inject trace context: {e}")
    else:
        # Fallback: Add a custom header
        headers["X-Aegis-Trace-Id"] = get_current_trace_id()
    
    return headers


def extract_trace_context(headers: Dict[str, str]):
    """
    Extract trace context from incoming HTTP headers.
    Returns a context object or None.
    """
    if OTEL_AVAILABLE:
        try:
            from opentelemetry.propagate import extract
            return extract(headers)
        except Exception as e:
            logger.warning(f"Failed to extract trace context: {e}")
    
    return None


# =============================================================================
# GLASS BOX REPLAY - Export trace for UI visualization
# =============================================================================

def format_trace_for_replay(
    trace_id: str,
    spans: list,
    governance_result: dict
) -> dict:
    """
    Format trace data for Glass Box Replay UI visualization.
    
    This creates a structured export format that the frontend's
    GlassBoxReplayEngine can consume for step-by-step visualization.
    """
    return {
        "trace_id": trace_id,
        "timestamp": spans[0].get("start_time") if spans else None,
        "governance_result": governance_result,
        "decision_path": [
            {
                "step": idx + 1,
                "name": span.get("name"),
                "phase": span.get("attributes", {}).get(SemanticAttributes.GOVERNANCE_PHASE),
                "duration_ms": span.get("duration_ms"),
                "verdict": span.get("attributes", {}).get(SemanticAttributes.GOVERNANCE_VERDICT),
                "invariants_checked": span.get("attributes", {}).get(SemanticAttributes.Z3_INVARIANTS_SATISFIED),
            }
            for idx, span in enumerate(spans)
        ],
        "replay_url": f"/api/v1/traces/{trace_id}/replay"
    }


# Global instances
tracing_manager = TracingManager()


def get_tracer():
    """Get the global tracing manager."""
    return tracing_manager


def get_otel_tracer():
    """Get the raw OpenTelemetry tracer (or None if unavailable)."""
    return tracer
