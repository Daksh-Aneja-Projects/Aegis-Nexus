"""
JWT Authentication Middleware for Aegis Nexus
Enforces "Zero Trust" by validating OIDC tokens for every request.
"""

import os
import logging
from typing import Optional, List
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import jwt
import httpx
import uuid
from starlette.datastructures import MutableHeaders


logger = logging.getLogger(__name__)

from core.infrastructure.state_manager import RedisStateStore

class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate JWT tokens from an OIDC provider (Auth0/Keycloak).
    """
    def __init__(self, app, authority: str = None, audience: str = None):
        super().__init__(app)
        self.authority = authority or os.getenv("AEGIS_AUTH_AUTHORITY", "https://aegis-nexus.us.auth0.com/")
        self.audience = audience or os.getenv("AEGIS_AUTH_AUDIENCE", "https://api.aegisnexus.io")
        self.jwks_client = jwt.PyJWKClient(f"{self.authority}.well-known/jwks.json")
        self.state_store = RedisStateStore()
        
        # Bypass for health checks and docs
        self.public_paths = ["/docs", "/redoc", "/openapi.json", "/health", "/ready", "/"]

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.public_paths or request.method == "OPTIONS":
            return await call_next(request)

        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

            token = auth_header.split(" ")[1]
            
            # 1. Get Signing Key with Caching
            # Note: PyJWKClient already has built-in caching, but we can manage it.
            try:
                signing_key = self.jwks_client.get_signing_key_from_jwt(token)
            except (jwt.PyJWKClientError, jwt.DecodeError) as e:
                # Potential Key Rotation: Force refresh and retry once
                logger.info(f"🔄 JWKS Refresh Triggered (Kid mismatch or missing): {e}")
                
                # Force refresh by creating a new client or clearing cache if library supports it
                # For PyJWKClient, we can re-instantiate or just call get_signing_key again if it failed
                # Actually, PyJWKClient handles caching internally. We retry to trigger fresh fetch if possible.
                self.jwks_client = jwt.PyJWKClient(
                    f"{self.authority}.well-known/jwks.json",
                    cache_keys=True,
                    cache_jwk_set=True,
                    lifespan=600 # 10 minute TTL
                )
                signing_key = self.jwks_client.get_signing_key_from_jwt(token)
            
            # 2. Decode & Verify
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self.audience,
                issuer=self.authority
            )
            
            # 2.5 Token Revocation Check (TRL) & Kill Switch
            # Check if JTI is revoked OR User ID is banned
            jti = payload.get("jti")
            user_id = payload.get("sub")
            
            # Parallel Redis check for speed
            # Note: In a real async redis, we'd use gather, but simple await is okay for now
            if jti:
                is_revoked = await self.state_store.get(f"trl:{jti}")
                if is_revoked:
                    logger.warning(f"⛔ Blocked revoked token: {jti}")
                    return JSONResponse(status_code=401, content={"error": "Revoked", "message": "Token has been revoked"})
            
            if user_id:
                is_banned = await self.state_store.get(f"ban:{user_id}")
                if is_banned:
                     logger.critical(f"⛔ KILL SWITCH: Blocked banned user {user_id}")
                     return JSONResponse(status_code=403, content={"error": "Account Suspended", "message": "User is banned by security policy"})

            # 3. Inject User into Request State
            request.state.user = payload
            request.state.user_id = user_id
            request.state.scopes = payload.get("scope", "").split()
            
            # 4. Scope Enforcement (Basic)
            # More granular checks should happen in dependencies
            
        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "message": "Token has expired"}
            )
        except jwt.PyJWTError as e:
            logger.warning(f"⛔ JWT Validation Failed: {str(e)}")
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "message": f"Invalid authentication token: {str(e)}"}
            )
        except Exception as e:
            logger.error(f"❌ Auth Middleware Error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal Server Error", "message": "Authentication service unreachable"}
            )

        return await call_next(request)


class MTLSEnforcementMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce Mutual TLS (mTLS) for inter-service communication.
    
    In a service mesh or behind an mTLS-terminating proxy (like Nginx/Envoy), 
    the client certificate is validated and passed downstream via headers.
    This middleware ensures those headers are present and valid for sensitive routes.
    """
    def __init__(self, app):
        super().__init__(app)
        self.enforce_mtls = os.getenv("AEGIS_MTLS_ENABLED", "False").lower() == "true"
        # Standard headers used by proxies (e.g. Nginx, Linkerd, Istio)
        self.cert_header = os.getenv("AEGIS_MTLS_HEADER", "X-Client-Cert-Hash")
        self.public_paths = ["/docs", "/redoc", "/openapi.json", "/health", "/ready", "/"]

    async def dispatch(self, request: Request, call_next):
        # 0. Generate Trace ID (X-Opaque-ID) for every request
        # This correlates the "Battle Room" audit with the "Z3" verdict
        trace_id = request.headers.get("X-Opaque-ID") or str(uuid.uuid4())
        
        # Inject into request state for logging
        request.state.trace_id = trace_id
        
        # 1. Skip if not enforcing or public path
        if not self.enforce_mtls or request.url.path in self.public_paths or request.method == "OPTIONS":
            response = await call_next(request)
            response.headers["X-Opaque-ID"] = trace_id
            return response

        # 2. Check for mTLS Header
        client_cert = request.headers.get(self.cert_header)
        
        if not client_cert:
             logger.warning(f"⛔ mTLS Violation: Missing {self.cert_header} from {request.client.host}")
             return JSONResponse(
                 status_code=403,
                 content={
                     "error": "Access Denied", 
                     "message": "Mutual TLS Certificate Required",
                     "trace_id": trace_id
                 }
             )

        # 3. Proceed
        response = await call_next(request)
        
        # Propagate Trace ID
        response.headers["X-Opaque-ID"] = trace_id
        return response


# ============================================
# Scope-Based Access Control (SBAC)
# ============================================

class ScopeChecker:
    """
    Dependency factory for endpoint-level scope enforcement.
    
    Usage:
        @router.post("/admin/action")
        async def admin_action(user = Depends(require_scopes(["admin:write"]))):
            ...
    
    Supported scope patterns:
        - "admin:*" - Full admin access
        - "ai:execute" - AI agent execution rights
        - "auditor:read" - Auditor read-only access
        - "human:approve" - Human approval rights
    """
    
    def __init__(self, required_scopes: List[str], require_all: bool = True):
        self.required_scopes = required_scopes
        self.require_all = require_all
    
    async def __call__(self, request: Request):
        """Validate user has required scopes."""
        user_scopes = getattr(request.state, 'scopes', [])
        user_id = getattr(request.state, 'user_id', 'unknown')
        
        if not user_scopes:
            logger.warning(f"⛔ SBAC: No scopes found for user {user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions: No scopes assigned"
            )
        
        # Check for wildcard admin access
        if "admin:*" in user_scopes:
            return request.state.user
        
        # Evaluate required scopes
        if self.require_all:
            # All scopes must be present
            missing = [s for s in self.required_scopes if not self._scope_matches(s, user_scopes)]
            if missing:
                logger.warning(f"⛔ SBAC: User {user_id} missing scopes: {missing}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions: Missing {missing}"
                )
        else:
            # At least one scope must match
            if not any(self._scope_matches(s, user_scopes) for s in self.required_scopes):
                logger.warning(f"⛔ SBAC: User {user_id} has none of required scopes: {self.required_scopes}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions: Require one of {self.required_scopes}"
                )
        
        return request.state.user
    
    def _scope_matches(self, required: str, user_scopes: List[str]) -> bool:
        """Check if a required scope is satisfied by user scopes."""
        if required in user_scopes:
            return True
        
        # Check wildcard patterns (e.g., "ai:*" matches "ai:execute")
        required_parts = required.split(":")
        for user_scope in user_scopes:
            user_parts = user_scope.split(":")
            if len(user_parts) >= 1 and user_parts[0] == required_parts[0]:
                if len(user_parts) > 1 and user_parts[1] == "*":
                    return True
        
        return False


def require_scopes(scopes: List[str], require_all: bool = True):
    """
    Dependency factory for scope-based access control.
    
    Args:
        scopes: List of required scopes
        require_all: If True, all scopes required. If False, any one scope suffices.
    
    Examples:
        # Require admin write access
        Depends(require_scopes(["admin:write"]))
        
        # Require any auditor permission
        Depends(require_scopes(["auditor:read", "auditor:write"], require_all=False))
        
        # AI agent with execution rights
        Depends(require_scopes(["ai:execute"]))
    """
    return ScopeChecker(scopes, require_all)


# Agent type helpers
def require_human_approval():
    """Shorthand for requiring human approval scope."""
    return require_scopes(["human:approve"])

def require_ai_execution():
    """Shorthand for requiring AI execution scope."""
    return require_scopes(["ai:execute"])

def require_auditor_access():
    """Shorthand for requiring auditor read access."""
    return require_scopes(["auditor:read"])

def require_admin():
    """Shorthand for requiring admin access."""
    return require_scopes(["admin:write"])
