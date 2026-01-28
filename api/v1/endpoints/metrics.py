\"\"\"
Metrics endpoint for Aegis Nexus API
Exposes Prometheus-compatible metrics at /metrics
\"\"\"

import logging
from fastapi import APIRouter
from fastapi.responses import Response

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get(\"/metrics\")
async def get_prometheus_metrics():
    \"\"\"
    Prometheus metrics endpoint.
    Returns metrics in Prometheus text format.
    \"\"\"
    try:
        from core.observability.metrics_exporter import get_metrics_output, update_llm_metrics
        
        # Update LLM metrics before export
        await update_llm_metrics()
        
        # Generate Prometheus format
        metrics_output = get_metrics_output()
        
        return Response(
            content=metrics_output,
            media_type=\"text/plain; version=0.0.4\"
        )
        
    except Exception as e:
        logger.error(f\"Failed to export metrics: {e}\")
        return Response(
            content=f\"# Error exporting metrics: {str(e)}\",
            media_type=\"text/plain\"
        )

@router.get(\"/metrics/llm\")
async def get_llm_metrics():
    \"\"\"
    Get LLM provider metrics in JSON format for debugging.
    \"\"\"
    try:
        from core.cognitive_architecture.llm_gateway import get_lazarus_gateway
        gateway = get_lazarus_gateway()
        return gateway.get_metrics()
    except Exception as e:
        logger.error(f\"Failed to get LLM metrics: {e}\")
        return {\"error\": str(e)}
