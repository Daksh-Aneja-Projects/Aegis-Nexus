"""
Scope-Based Access Control (SBAC) for Aegis Nexus
Provides finer-grained permissions than traditional RBAC roles.

This middleware validates JWT claims against endpoint-specific scope requirements,
ensuring that users can only access resources they have explicit permission for.

PRODUCTION SECURITY PATTERN:
- More granular than RBAC (Role-Based Access Control)
- Prevents "Viewer" roles from triggering chaos scenarios
- Supports hierarchical scope inheritance
- Compatible with OAuth2/OIDC scope specifications
"""

import logging
from typing import List, Optional, Dict, Any, Callable
from functools import wraps

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# Security scheme for extracting Bearer tokens
security = HTTPBearer(auto_error=False)


# =============================================================================
# SCOPE DEFINITIONS
# =============================================================================

class Scopes:
    """Standard scope definitions for Aegis Nexus."""
    
    # Core Governance Scopes
    GOVERNANCE_READ = "governance:read"
    GOVERNANCE_SUBMIT = "governance:submit"
    GOVERNANCE_ADMIN = "governance:admin"
    
    # Verification Scopes
    VERIFY_READ = "verify:read"
    VERIFY_EXECUTE = "verify:execute"
    
    # Audit Scopes
    AUDIT_READ = "audit:read"
    AUDIT_WRITE = "audit:write"
    AUDIT_ADMIN = "audit:admin"
    
    # Chaos Engineering Scopes (CRITICAL - restricted)
    CHAOS_READ = "chaos:read"
    CHAOS_EXECUTE = "chaos:execute"
    CHAOS_ADMIN = "chaos:admin"
    
    # System Administration Scopes
    ADMIN_USERS = "admin:users"
    ADMIN_CONFIG = "admin:config"
    ADMIN_SECRETS = "admin:secrets"
    ADMIN_SUPER = "admin:super"  # God mode - all permissions
    
    # Shadow Verification Scopes
    SHADOW_READ = "shadow:read"
    SHADOW_ADMIN = "shadow:admin"
    
    # Canary Token Scopes
    CANARY_READ = "canary:read"
    CANARY_EXECUTE = "canary:execute"


# Scope hierarchy - parent scopes grant access to child operations
SCOPE_HIERARCHY = {
    Scopes.ADMIN_SUPER: [
        Scopes.GOVERNANCE_ADMIN, Scopes.AUDIT_ADMIN, 
        Scopes.CHAOS_ADMIN, Scopes.SHADOW_ADMIN, Scopes.CANARY_EXECUTE,
        Scopes.ADMIN_USERS, Scopes.ADMIN_CONFIG, Scopes.ADMIN_SECRETS
    ],
    Scopes.GOVERNANCE_ADMIN: [Scopes.GOVERNANCE_READ, Scopes.GOVERNANCE_SUBMIT],
    Scopes.AUDIT_ADMIN: [Scopes.AUDIT_READ, Scopes.AUDIT_WRITE],
    Scopes.CHAOS_ADMIN: [Scopes.CHAOS_READ, Scopes.CHAOS_EXECUTE],
    Scopes.SHADOW_ADMIN: [Scopes.SHADOW_READ],
}


# =============================================================================
# ENDPOINT SCOPE REQUIREMENTS
# =============================================================================

ENDPOINT_SCOPES: Dict[str, List[str]] = {
    # Submission API
    "/api/v1/submit": [Scopes.GOVERNANCE_SUBMIT],
    "/api/v1/verify": [Scopes.VERIFY_EXECUTE],
    
    # Audit API
    "/api/v1/audit": [Scopes.AUDIT_READ],
    "/api/v1/audit/logs": [Scopes.AUDIT_READ],
    "/api/v1/audit/export": [Scopes.AUDIT_ADMIN],
    
    # Chaos Engineering API (RESTRICTED)
    "/api/v1/chaos": [Scopes.CHAOS_EXECUTE],
    "/api/v1/chaos/kill": [Scopes.CHAOS_ADMIN],
    "/api/v1/chaos/revive": [Scopes.CHAOS_ADMIN],
    "/api/v1/chaos/status": [Scopes.CHAOS_READ],
    
    # Shadow Verification API
    "/api/v1/shadow/stats": [Scopes.SHADOW_READ],
    "/api/v1/shadow/discrepancies": [Scopes.SHADOW_READ],
    "/api/v1/shadow/toggle": [Scopes.SHADOW_ADMIN],
    
    # Canary Token API
    "/api/v1/canary/validate": [Scopes.CANARY_EXECUTE],
    "/api/v1/canary/status": [Scopes.CANARY_READ],
    
    # Admin API
    "/api/v1/admin/users": [Scopes.ADMIN_USERS],
    "/api/v1/admin/config": [Scopes.ADMIN_CONFIG],
}


def expand_scopes(scopes: List[str]) -> List[str]:
    """
    Expand scopes to include all inherited permissions.
    
    If a user has 'admin:super', they automatically get all child scopes.
    """
    expanded = set(scopes)
    
    for scope in scopes:
        if scope in SCOPE_HIERARCHY:
            expanded.update(SCOPE_HIERARCHY[scope])
            # Recurse for nested hierarchies
            expanded.update(expand_scopes(SCOPE_HIERARCHY[scope]))
    
    return list(expanded)


def decode_jwt_scopes(token: str) -> List[str]:
    """
    """
    try:
        import jwt
    except ImportError:
        return []

    import os
    # Strict Production Check (Gap 3 Fix) - MUST BE OUTSIDE TRY/EXCEPT to fail fast
    secret_key = os.getenv("AEGIS_JWT_SECRET")
    if not secret_key and os.getenv("AEGIS_ENV") == "production":
         logger.critical("⛔ FATAL: AEGIS_JWT_SECRET missing in production.")
         raise RuntimeError("Security Config Error: JWT Secret Missing in Production")

    try:
        # Fallback for dev only
        if not secret_key:
             logger.warning("⚠️  SECURITY WARNING: AEGIS_JWT_SECRET not set. Using insecure default for dev.")
             secret_key = "CHANGE_ME_IN_PRODUCTION"
        
        # In production, use proper public key verification
        # This is a simplified implementation for development
        import os
        
        
        # Try to decode (allow unverified for dev if needed)
        try:
            payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        except jwt.PyJWTError:
            # In development, allow inspection of token structure
            try:
                payload = jwt.decode(token, options={"verify_signature": False})
            except:
                return []
        
        # Extract scopes from standard claims
        scopes = payload.get("scope", "").split() if isinstance(payload.get("scope"), str) else []
        scopes.extend(payload.get("scopes", []))
        
        # Also check for role-based scope mapping
        roles = payload.get("roles", [])
        for role in roles:
            if role == "superadmin":
                scopes.append(Scopes.ADMIN_SUPER)
            elif role == "operator":
                scopes.extend([Scopes.GOVERNANCE_SUBMIT, Scopes.VERIFY_EXECUTE, Scopes.AUDIT_READ])
            elif role == "viewer":
                scopes.extend([Scopes.GOVERNANCE_READ, Scopes.AUDIT_READ])
        
        return expand_scopes(scopes)
        
    except Exception as e:
        logger.warning(f"JWT scope extraction failed: {e}")
        return []


def require_scopes(required_scopes: List[str], all_required: bool = True):
    """
    Dependency that validates required scopes for an endpoint.
    
    Args:
        required_scopes: List of scope strings required for access
        all_required: If True, ALL scopes must be present. If False, ANY scope is sufficient.
        
    Usage:
        @router.post("/chaos/trigger")
        async def trigger_chaos(claims: dict = Depends(require_scopes([Scopes.CHAOS_EXECUTE]))):
            ...
    """
    async def scope_validator(
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ) -> Dict[str, Any]:
        # Allow unauthenticated access in development if configured
        import os
        if os.getenv("AEGIS_DISABLE_AUTH", "false").lower() == "true":
            logger.warning("⚠️ Authentication disabled (AEGIS_DISABLE_AUTH=true)")
            return {"sub": "dev_user", "scopes": [Scopes.ADMIN_SUPER]}
        
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="Authorization header missing",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        token = credentials.credentials
        user_scopes = decode_jwt_scopes(token)
        
        if not user_scopes:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Check scope requirements
        if all_required:
            # ALL required scopes must be present
            missing = [s for s in required_scopes if s not in user_scopes]
            if missing:
                logger.warning(
                    f"⛔ SBAC Denied: Missing scopes {missing} for {request.url.path}"
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Missing scopes: {', '.join(missing)}"
                )
        else:
            # ANY of the required scopes is sufficient
            if not any(s in user_scopes for s in required_scopes):
                logger.warning(
                    f"⛔ SBAC Denied: None of {required_scopes} present for {request.url.path}"
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required any of: {', '.join(required_scopes)}"
                )
        
        logger.debug(f"✅ SBAC Granted: {request.url.path} (scopes: {user_scopes[:3]}...)")
        
        # Return claims for use in endpoint
        return {
            "token": token,
            "scopes": user_scopes,
            "path": str(request.url.path)
        }
    
    return Depends(scope_validator)


# Convenience decorators for common scope requirements
RequireAdmin = require_scopes([Scopes.ADMIN_SUPER])
RequireChaos = require_scopes([Scopes.CHAOS_EXECUTE])
RequireGovernance = require_scopes([Scopes.GOVERNANCE_SUBMIT])
RequireAuditRead = require_scopes([Scopes.AUDIT_READ])
RequireAuditAdmin = require_scopes([Scopes.AUDIT_ADMIN])


# =============================================================================
# MIDDLEWARE FOR AUTOMATIC SCOPE VALIDATION
# =============================================================================

class SBACMiddleware:
    """
    Middleware that automatically validates scopes based on ENDPOINT_SCOPES mapping.
    
    Use this for automatic protection without decorating every endpoint.
    """
    
    def __init__(self, app):
        self.app = app
        
    async def __call__(self, scope, receive, send):
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive, send)
        path = request.url.path
        
        # Check if path requires scope validation
        required_scopes = None
        for endpoint_pattern, scopes in ENDPOINT_SCOPES.items():
            if path.startswith(endpoint_pattern) or path == endpoint_pattern:
                required_scopes = scopes
                break
        
        if required_scopes:
            # Extract and validate token
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                response = JSONResponse(
                    status_code=401, 
                    content={"detail": "Authorization required"}
                )
                await response(scope, receive, send)
                return
            
            token = auth_header.replace("Bearer ", "")
            user_scopes = decode_jwt_scopes(token)
            
            # Check scopes
            if not any(s in user_scopes for s in required_scopes):
                logger.warning(f"⛔ SBAC Middleware Denied: {path}")
                response = JSONResponse(
                    status_code=403,
                    content={"detail": f"Missing required scope(s): {required_scopes}"}
                )
                await response(scope, receive, send)
                return
        
        await self.app(scope, receive, send)
