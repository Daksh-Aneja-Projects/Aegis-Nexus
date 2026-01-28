"""
Aegis Nexus API v1 Router
Aggregates all API endpoints for the Sentinel Executive Layer
"""

from fastapi import APIRouter

from api.v1.endpoints import (
    submission, 
    verification, 
    audit_trail, 
    websockets, 
    shadow, 
    system,
    evolution
)

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    submission.router,
    prefix="/submit",
    tags=["prompt-submission"]
)

api_router.include_router(
    verification.router,
    prefix="/verify",
    tags=["verification-status"]
)

api_router.include_router(
    audit_trail.router,
    prefix="/audit",
    tags=["audit-trail"]
)

api_router.include_router(
    websockets.router,
    tags=["websockets"]
)

api_router.include_router(
    shadow.router,
    tags=["shadow-mode"]
)

api_router.include_router(
    system.router,
    prefix="/system",
    tags=["system-configuration"]
)

api_router.include_router(
    evolution.router,
    prefix="/evolution",
    tags=["evolutionary-governance"]
)