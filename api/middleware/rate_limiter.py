"""
Middleware for Granular Rate Limiting (Distributed)
Implements Sliding Window algorithm using Redis & Lua for atomic distributed counting.
"""

import time
import logging
import os
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Configuration
RATE_LIMIT_RULES = {
    "default": (100, 60),       # 100 req / 60s
    "/api/v1/verify": (50, 60), # 50 req / 60s
    "/api/v1/submit": (20, 60), # 20 req / 60s
    "/api/v1/audit": (10, 60),  # 10 req / 60s
}

# Lua Script for Atomic Sliding Window
# Keys: [rate_limit_key], Args: [window_size_seconds, limit]
LUA_SLIDING_WINDOW = """
-- Sliding Window Rate Limiter (Lua)
local key = KEYS[1]
local window = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])

-- Use Redis time for cluster-wide consistency
local redis_time = redis.call('TIME')
local now = tonumber(redis_time[1])

-- Remove old entries
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

-- Count current entries
local count = redis.call('ZCARD', key)

if count < limit then
    -- Add new request
    redis.call('ZADD', key, now, now)
    redis.call('EXPIRE', key, window)
    return {1, count + 1}
else
    return {0, count}
end
"""

class GranularRateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.redis_url = os.getenv("AEGIS_REDIS_URL", "redis://localhost:6379")
        self.redis = None
        self.script_sha = None

    async def dispatch(self, request: Request, call_next):
        # Initialize Redis lazily
        if not self.redis:
            self.redis = redis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
            # Register Lua script
            self.script_sha = await self.redis.script_load(LUA_SLIDING_WINDOW)

        path = request.url.path
        client_ip = request.client.host or "unknown"
        
        # 0. PRIORITY SHEDDING: Bypass for Governance Auditors (Refinement C)
        # Check Authorization header for critical roles
        auth_header = request.headers.get("Authorization")
        if auth_header and "Bearer" in auth_header:
            try:
                # In prod, verify JWT signature. Here we inspect claims for role.
                token = auth_header.split(" ")[1]
                # decode without verify for speed in rate limiter (signature verified in Auth middleware)
                import jwt
                payload = jwt.decode(token, options={"verify_signature": False})
                role = payload.get("role")
                
                if role == "SUPREME_COURT_AUDITOR" or role == "SYSTEM_ADMIN":
                     # BYPASS LIMITS for Critical Infrastructure Traffic
                     logger.info(f"🛡️ Priority Traffic: Bypassing Rate Limit for {role}")
                     return await call_next(request)
            except Exception:
                pass # Malformed token, fall back to limiting
        
        # 1. Identify Rule
        # Exact match or default
        limit, window = RATE_LIMIT_RULES.get(path, RATE_LIMIT_RULES["default"])
        
        # 2. Key Generation
        key = f"rl:{path}:{client_ip}"
        
        try:
            # 3. Lua Script Execution
            allowed, count = await self.redis.evalsha(
                self.script_sha, 
                1, 
                key, 
                window, 
                limit
            )
            
            if not allowed:
                logger.warning(f"⛔ Rate Limit Exceeded for {key}: {count}/{limit}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate Limit Exceeded",
                        "message": "Too many requests. Please try again later.",
                        "retry_after": str(window)
                    },
                    headers={"Retry-After": str(window)}
                )

            response = await call_next(request)
            
            # Add Headers
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(max(0, limit - count))
            
            return response
            
        except Exception as e:
            logger.error(f"Rate Limiter Redis Error: {e}")
            # Fail open
            return await call_next(request)
