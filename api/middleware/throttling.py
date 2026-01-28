import logging
import time
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from core.infrastructure.state_manager import RedisStateStore

logger = logging.getLogger(__name__)

class AdaptiveThrottlingMiddleware(BaseHTTPMiddleware):
    """
    Adaptive Throttling (Backpressure) Middleware.
    
    Checks the depth of the Z3 verification queue in Redis.
    If the queue depth exceeds a safety threshold, returns HTTP 503 
    to shed load before the system crashes.
    """
    
    def __init__(
        self, 
        app, 
        queue_name: str = "arq:queue", # Default arq queue name
        hard_limit: int = 100,
        soft_limit: int = 50
    ):
        super().__init__(app)
        self.queue_name = queue_name
        self.hard_limit = hard_limit
        self.soft_limit = soft_limit
        self.state_store = RedisStateStore()

    async def dispatch(self, request: Request, call_next):
        # Skip for health checks and static files
        if request.url.path in ["/health", "/ready", "/", "/favicon.ico"]:
            return await call_next(request)

        # Only throttle submission endpoints which trigger heavy Z3 math
        if "/api/v1/submission" in request.url.path:
            try:
                # Check Redis queue length
                # Note: arq uses a list in Redis
                queue_depth = await self.state_store._redis.llen(self.queue_name)
                
                if queue_depth >= self.hard_limit:
                    logger.warning(f"🚨 CRITICAL BACKPRESSURE: Queue depth {queue_depth} exceeds hard limit {self.hard_limit}")
                    return JSONResponse(
                        status_code=503,
                        content={
                            "error": "Service Unavailable",
                            "message": "System at capacity. Please retry with exponential backoff.",
                            "retry_after": 5
                        },
                        headers={"Retry-After": "5"}
                    )
                
                if queue_depth >= self.soft_limit:
                    logger.info(f"⚠️  SOFT BACKPRESSURE: Queue depth {queue_depth} approaching limit.")
                    # We could inject a header or log warning here
                    
            except Exception as e:
                # Fail open if Redis check fails to avoid blocking all traffic
                logger.error(f"❌ Throttling check failed: {e}")

        return await call_next(request)
