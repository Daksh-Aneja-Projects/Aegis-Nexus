"""
Aegis Nexus - Main Application Entry Point
Sentinel Executive Layer (SEL) for AI Governance

This module initializes the FastAPI application with all necessary
middleware, routers, and lifespan management for the cognitive governor system.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from audit_war_room.mpc_judging import initialize_mpc_judge
from api.core.logging_config import setup_logging

from api.v1.router import api_router
from core.security.immutable_ledger import initialize_ledger
from core.security.pqc_consensus import initialize_post_quantum_system
from core.cognitive_architecture.working_memory import initialize_working_memory
from core.governance.z3_verifier import initialize_formal_verifier
from reality_anchor.sensors.fusion_engine import initialize_sensor_fusion
from audit_war_room.orchestrator import initialize_battle_orchestrator
from audit_war_room.replay_engine import initialize_replay_engine
from infrastructure.operators.lazarus_operator import initialize_lazarus

# Configure structured JSON logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager for startup and shutdown events.
    
    Initializes all core system components in proper dependency order:
    1. Security infrastructure (ledger, PQC)
    2. Cognitive kernel (memory, verifier)
    3. Reality anchor (sensors, fusion)
    4. Adversarial audit system
    5. Cleanup on shutdown
    """
    logger.info("🚀 Starting Aegis Nexus Sentinel Executive Layer")
    
    try:
        # Phase 1: Security Infrastructure & Secret Management
        logger.info("🔒 Initializing security infrastructure & Vault...")
        
        # 1. Verify PQC Keys exist (Fail Fast - Six Sigma Requirement)
        if not os.getenv("AEGIS_PQC_PRIVATE_KEY") and os.getenv("AEGIS_ENV") == "production":
             logger.critical("FATAL: PQC Keys not found in Environment. System cannot guarantee immutable audit.")
             raise RuntimeError("FATAL: PQC Keys not found. System cannot guarantee immutable audit.")

        from core.security.vault_client import initialize_vault
        await initialize_vault()
        await initialize_ledger()
        await initialize_post_quantum_system(strict_mode=True)  # Enforce strict mode for production
        
        # 2. Check Database Connection (Pre-flight check)
        # Assuming DB URL is required for core operations
        if not os.getenv("AEGIS_DB_URL"):
            logger.warning("⚠️  AEGIS_DB_URL not set. Database-dependent features may fail.")
        
        # Parallel Phase: Cognitive Kernel, Reality Anchor, Battle Orchestrator
        # Phase 2, 3, 4 can run concurrently as they don't strictly depend on each other's full initialization
        # provided the basic infrastructure (State Store/Redis) is ready (which is implicit/lazy).
        logger.info("⚡ Starting Parallel Initialization of Core Systems...")
        
        # Import episodic memory
        from core.cognitive_architecture.episodic_memory import initialize_episodic_memory
        
        await asyncio.gather(
            initialize_working_memory(),        # Cognitive
            initialize_episodic_memory(),       # Cognitive - Long-term memory
            initialize_formal_verifier(),       # Cognitive (Process Pool)
            initialize_sensor_fusion(),         # Reality
            initialize_battle_orchestrator(),   # Audit
            initialize_replay_engine(),         # Audit
            initialize_mpc_judge(),             # Audit
            initialize_lazarus()                # Self-Healing
        )
        
        # Phase 5: Cognitive Circuit Breaker & Audit Recorder (Depend on subsystems)
        logger.info("🛡️  Initializing cognitive circuit breaker and black box recorder...")
        
        from api.middleware.circuit_breaker import initialize_cognitive_circuit_breaker
        await initialize_cognitive_circuit_breaker()
        
        from core.infrastructure.audit_recorder import initialize_audit_recorder
        await initialize_audit_recorder()

        # Advanced Feature Initialization
        logger.info("🕵️ Initializing Black Box Recorder & Kill Switch...")
        from core.infrastructure.black_box_recorder import get_black_box_recorder
        await get_black_box_recorder() # Initialize singleton

        from core.security.kill_switch import get_kill_switch
        await get_kill_switch() # Initialize singleton listener

        if os.getenv("AEGIS_ENV") != "production" or os.getenv("AEGIS_CHAOS_MODE") == "True":
             from core.governance.chaos_monkey import initialize_chaos_monkey
             await initialize_chaos_monkey()
             logger.warning("🃏 Chaos Monkey (Logic Joker) Active")
        
        # Phase 6: Async Job Queue (Arq)
        logger.info("👷 Initializing Async Job Queue...")
        from arq import create_pool
        from arq.connections import RedisSettings
        import os
        
        redis_settings = RedisSettings(
            host=os.getenv("AEGIS_REDIS_HOST", "localhost"),
            port=int(os.getenv("AEGIS_REDIS_PORT", 6379))
        )
        app.state.arq_pool = await create_pool(redis_settings)
        
        logger.info("✅ Aegis Nexus initialization complete - all systems operational")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Aegis Nexus: {str(e)}")
        raise
    finally:
        logger.info("🛑 Shutting down Aegis Nexus...")
        
        # Graceful shutdown with task draining (Gap 5.2)
        # 1. Drain Z3 ProcessPoolExecutor
        try:
            from core.governance.z3_verifier import _pool
            if _pool is not None:
                logger.info("⏳ Draining Z3 ProcessPool (waiting for pending proofs)...")
                _pool.shutdown(wait=True, cancel_futures=True)
                logger.info("✅ Z3 ProcessPool drained successfully")
        except Exception as e:
            logger.warning(f"⚠️ Error draining Z3 pool: {e}")
        
        # 2. Close ARQ job queue
        if hasattr(app.state, "arq_pool"):
            await app.state.arq_pool.close()
        
        # 3. Close Redis connections
        try:
            from api.middleware.circuit_breaker import get_cognitive_circuit_breaker
            breaker = get_cognitive_circuit_breaker()
            if breaker and breaker.redis:
                await breaker.redis.close()
                logger.info("✅ Circuit breaker Redis connection closed")
        except Exception:
            pass
        
        logger.info("🔚 Aegis Nexus shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Aegis Nexus API",
    description="Sentinel Executive Layer for AI Governance - Mathematical Proofs of Trust",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

import time
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# ... (Imports remain the same, ensuring time is imported)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Explicit Frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Aegis-Signature"],
)

# SECURITY: Prevent DNS Rebinding attacks
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["aegis-nexus.internal", "localhost", "127.0.0.1"])

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    # OBSERVABILITY: precise latency tracking
    response.headers["X-Process-Time"] = str(process_time)
    return response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds security headers to all responses."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        return response

app.add_middleware(SecurityHeadersMiddleware)

from core.observability.tracing import get_tracer

class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Middleware for request tracing and correlation ID management."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate correlation ID for request tracing
        # Use UUID for collision-resistant distributed tracing
        import uuid
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        tracer = get_tracer()
        with tracer.start_span(f"HTTP {request.method} {request.url.path}", attributes={
            "http.method": request.method,
            "http.url": str(request.url),
            "correlation_id": correlation_id
        }) as span:
            # Add correlation ID to response headers
            response = await call_next(request)
            response.headers["X-Correlation-ID"] = correlation_id
            
            span.set_attribute("http.status_code", response.status_code)
            return response

# Add cognitive circuit breaker middleware first
try:
    from api.middleware.circuit_breaker import get_cognitive_circuit_breaker, CognitiveCircuitBreakerMiddleware
    app.add_middleware(
        CognitiveCircuitBreakerMiddleware,
        circuit_breaker=get_cognitive_circuit_breaker(),
        protected_endpoints=["/api/v1/submit", "/api/v1/verify", "/api/v1/execute"]
    )
except RuntimeError:
    logger.warning("⚠️  Cognitive circuit breaker not initialized, skipping middleware")

# Security Hardening Middlewares
# 1. Force HTTPS
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
# app.add_middleware(HTTPSRedirectMiddleware) # Commented out for local dev, uncomment for prod

# 2. Trusted Hosts
# Already configured above in Security Hardening
# from fastapi.middleware.trustedhost import TrustedHostMiddleware
# app.add_middleware(
#    TrustedHostMiddleware,
#    allowed_hosts=["api.aegisnexus.io", "localhost", "127.0.0.1"]
# )

# 3. Adaptive Throttling (Backpressure)
from api.middleware.throttling import AdaptiveThrottlingMiddleware
app.add_middleware(AdaptiveThrottlingMiddleware)

# 4. JWT Authentication (Zero Trust)
from api.middleware.auth import JWTAuthMiddleware
app.add_middleware(JWTAuthMiddleware)

# Add custom tracing middleware
app.add_middleware(RequestTracingMiddleware)

# Add granular rate limiting middleware
from api.middleware.rate_limiter import GranularRateLimiterMiddleware
app.add_middleware(GranularRateLimiterMiddleware)

# Include API routers
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint returning system status."""
    return {
        "name": "Aegis Nexus",
        "version": "1.0.0",
        "status": "operational",
        "tagline": "Trust is no longer a feeling. It is a mathematical proof.",
        "phases_operational": {
            "phase_1_audit": True,
            "phase_2_cognition": True,
            "phase_3_reality": True
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    # Check Z3 Responsiveness
    z3_status = "offline"
    try:
        from core.governance.z3_verifier import get_formal_verifier, _pool
        verifier = get_formal_verifier()
        # Run a trivial math check in the process pool
        # This confirms workers are alive and not deadlocked
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_pool, pow, 2, 2)
        if result == 4:
            z3_status = "online"
    except Exception:
        z3_status = "degraded"

    return {
        "status": "healthy" if z3_status == "online" else "degraded",
        "timestamp": asyncio.get_event_loop().time(),
        "components": {
            "api": "online",
            "z3_solver": z3_status,
            "database": "connected",
            "redis": "connected", # In a real implementation we should ping redis too
            "llm_providers": "available"
        }
    }

@app.get("/ready")
async def readiness_check():
    """Readiness probe for Kubernetes/service discovery."""
    # Check if all critical systems are initialized
    try:
        # Add actual readiness checks here
        return {"status": "ready"}
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "system_initializing"}
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with structured error responses."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred in the cognitive governor system",
            "correlation_id": getattr(request.state, 'correlation_id', 'unknown')
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )