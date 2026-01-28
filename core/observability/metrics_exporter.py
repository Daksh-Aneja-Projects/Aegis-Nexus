\"\"\"
Metrics Exporter for Aegis Nexus
Exposes Prometheus-compatible metrics for LLM gateway, Z3 verifier, and system health.
\"\"\"

import logging
from typing import Dict, Any
from prometheus_client import Gauge, Counter, Histogram, generate_latest, REGISTRY

logger = logging.getLogger(__name__)

# LLM Provider Metrics
llm_hallucination_rate = Gauge(
    'aegis_llm_hallucination_rate',
    'Hallucination rate by provider (Z3 rejections / total)',
    ['provider']
)

llm_circuit_state = Gauge(
    'aegis_llm_circuit_state',
    'Circuit breaker state (0=closed, 1=half-open, 2=open)',
    ['provider']
)

llm_latency_ms = Histogram(
    'aegis_llm_latency_milliseconds',
    'LLM response latency in milliseconds',
    ['provider'],
    buckets=[100, 500, 1000, 2000, 5000, 10000, 30000]
)

llm_rate_limit_hits = Counter(
    'aegis_llm_rate_limit_hits_total',
    'Number of rate limit hits by provider',
    ['provider']
)

# Cognitive Load Metrics
cognitive_load = Gauge(
    'aegis_cognitive_load',
    'Current cognitive load (0.0-1.0)'
)

# Z3 Verification Metrics
z3_verification_duration = Histogram(
    'aegis_z3_verification_duration_seconds',
    'Z3 solver verification duration',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

z3_verification_outcome = Counter(
    'aegis_z3_verification_total',
    'Z3 verification outcomes',
    ['outcome']  # approved, rejected, error
)

# Phase Verification Metrics
phase_verification = Counter(
    'aegis_phase_verification_total',
    'Verification results by phase',
    ['phase', 'outcome']  # phase: 1,2,3; outcome: approved, rejected
)

# Sensor Drift Metrics
sensor_drift_magnitude = Gauge(
    'aegis_sensor_drift_magnitude',
    'Current proprioceptive drift magnitude (sigma)',
    ['sensor_type']
)

# Supreme Court Metrics
supreme_court_escalations = Counter(
    'aegis_supreme_court_escalations_total',
    'Total number of escalations to the Automated Supreme Court'
)

supreme_court_verdict = Counter(
    'aegis_supreme_court_verdict_total',
    'Outcomes of Supreme Court reviews',
    ['verdict']  # auto_veto, proceed_with_log, remedial_action_required
)

async def update_llm_metrics():
    \"\"\"Update LLM provider metrics from Lazarus gateway.\"\"\"
    try:
        from core.cognitive_architecture.llm_gateway import get_lazarus_gateway
        gateway = get_lazarus_gateway()
        metrics = gateway.get_metrics()
        
        #Update Prometheus metrics
        for key, value in metrics.items():
            if 'hallucination_rate' in key:
                provider = key.replace('_hallucination_rate', '')
                llm_hallucination_rate.labels(provider=provider).setvalue)
            elif 'circuit_state' in key:
                provider = key.replace('_circuit_state', '')
                state_map = {'closed': 0, 'half-open': 1, 'open': 2}
                llm_circuit_state.labels(provider=provider).set(state_map.get(value, 0))
            elif 'rate_limit_hits' in key:
                provider = key.replace('_rate_limit_hits', '')
                llm_rate_limit_hits.labels(provider=provider).inc(value)
            elif 'latency_p50_ms' in key:
                provider = key.replace('_latency_p50_ms', '')
                llm_latency_ms.labels(provider=provider).observe(value)
                
    except Exception as e:
        logger.warning(f\"Failed to update LLM metrics: {e}\")

async def update_cognitive_load_metric(load: float):
    \"\"\"Update cognitive load metric.\"\"\"
    cognitive_load.set(load)

async def update_z3_metrics(duration: float, outcome: str):
    \"\"\"Update Z3 verification metrics.\"\"\"
    z3_verification_duration.observe(duration)
    z3_verification_outcome.labels(outcome=outcome).inc()

async def update_phase_metrics(phase: int, outcome: str):
    \"\"\"Update phase verification metrics.\"\"\"
    phase_verification.labels(phase=str(phase), outcome=outcome).inc()

async def update_drift_metric(sensor_type: str, drift_sigma: float):
    \"\"\"Update sensor drift magnitude metric.\"\"\"
    sensor_drift_magnitude.labels(sensor_type=sensor_type).set(drift_sigma)

async def update_supreme_court_metrics(verdict: str, confidence: float):
    \"\"\"Update Supreme Court escalation and verdict metrics.\"\"\"
    supreme_court_escalations.inc()
    supreme_court_verdict.labels(verdict=verdict).inc()

def get_metrics_output() -> bytes:
    \"\"\"Get Prometheus-formatted metrics output.\"\"\"
    return generate_latest(REGISTRY)
