"""
Shadow Verifier API Endpoints
Provides monitoring and control for the Shadow Mode verification system.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import logging

from core.governance.shadow_verifier import get_shadow_verifier

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/shadow", tags=["Shadow Verifier"])


@router.get("/stats")
async def get_shadow_stats() -> Dict[str, Any]:
    """
    Get shadow verifier statistics.
    
    Returns metrics on verification activity, discrepancy rates, and performance.
    """
    try:
        shadow = get_shadow_verifier()
        stats = shadow.get_stats()
        
        return {
            "status": "active" if shadow._enabled else "disabled",
            "stats": stats,
            "traffic_percentage": shadow.shadow_traffic_percentage * 100
        }
    except Exception as e:
        logger.error(f"Failed to get shadow stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/discrepancies")
async def get_discrepancies(limit: int = 50) -> Dict[str, Any]:
    """
    Get recent discrepancies between primary and shadow verifier.
    
    Args:
        limit: Maximum number of discrepancies to return (default 50)
    """
    try:
        shadow = get_shadow_verifier()
        discrepancies = shadow.get_recent_discrepancies(limit)
        
        return {
            "count": len(discrepancies),
            "discrepancies": discrepancies
        }
    except Exception as e:
        logger.error(f"Failed to get discrepancies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enable")
async def enable_shadow_verifier() -> Dict[str, str]:
    """Enable the shadow verifier."""
    shadow = get_shadow_verifier()
    shadow.enable()
    return {"status": "enabled", "message": "Shadow verifier activated"}


@router.post("/disable")
async def disable_shadow_verifier() -> Dict[str, str]:
    """Disable the shadow verifier."""
    shadow = get_shadow_verifier()
    shadow.disable()
    return {"status": "disabled", "message": "Shadow verifier deactivated"}


@router.post("/traffic-percentage")
async def set_traffic_percentage(percentage: float) -> Dict[str, Any]:
    """
    Set the percentage of traffic to shadow verify.
    
    Args:
        percentage: Value between 0.0 and 100.0
    """
    if not 0.0 <= percentage <= 100.0:
        raise HTTPException(
            status_code=400, 
            detail="Percentage must be between 0 and 100"
        )
    
    shadow = get_shadow_verifier()
    shadow.set_traffic_percentage(percentage / 100.0)
    
    return {
        "status": "updated",
        "traffic_percentage": percentage
    }
