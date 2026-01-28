"""
Verification Status Endpoint for Aegis Nexus
Provides real-time status updates for ongoing verification processes.

This module implements the /verification_status endpoint for polling
verification progress and results.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.v1.schemas.request_models import (
    VerificationStatusRequest, 
    VerificationStatusResponse,
    VerificationPhase
)
from api.v1.schemas.response_models import APIResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Reference to global verification storage (would be imported in real implementation)
# from api.v1.endpoints.submission import verification_requests, verification_results

@router.get("/status/{request_id}", response_model=APIResponse)
async def get_verification_status(request_id: str):
    """
    Get the current status of a verification request.
    
    Args:
        request_id: The ID of the verification request to check
        
    Returns:
        APIResponse with current verification status and progress
    """
    try:
        # Check if this is implemented in the same process
        # In a real implementation, this would query a shared database/cache
        global verification_requests, verification_results
        
        # Check completed results first
        if request_id in verification_results:
            result = verification_results[request_id]
            return await _format_completed_result(result)
        
        # Check ongoing requests
        if request_id in verification_requests:
            request_data = verification_requests[request_id]
            return await _format_ongoing_request(request_id, request_data)
        
        # Request not found
        raise HTTPException(
            status_code=404,
            detail=f"Verification request {request_id} not found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving verification status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error retrieving verification status"
        )

@router.post("/status", response_model=APIResponse)
async def post_verification_status(request: VerificationStatusRequest):
    """
    Alternative POST endpoint for getting verification status.
    
    Args:
        request: Verification status request with request_id
        
    Returns:
        APIResponse with verification status
    """
    return await get_verification_status(request.request_id)

async def _format_completed_result(result: Dict[str, Any]) -> APIResponse:
    """Format a completed verification result for response."""
    try:
        # Build comprehensive status response
        response_data = VerificationStatusResponse(
            request_id=result["request_id"],
            current_phase=VerificationPhase.COMPLETED,
            phase_1_audit=_build_phase1_result(result["detailed_results"].get("phase_1")),
            phase_2_cognition=_build_phase2_result(result["detailed_results"].get("phase_2")),
            phase_3_reality=_build_phase3_result(result["detailed_results"].get("phase_3")),
            overall_status=result["final_status"],
            confidence_score=result["overall_confidence"],
            pqc_signature="signature_placeholder",  # Would be actual PQC signature
            audit_trail_hash="hash_placeholder",   # Would be actual hash
            completion_timestamp=datetime.fromisoformat(result["completion_timestamp"]),
            error_message=result.get("error_message")
        )
        
        return APIResponse(
            success=True,
            data=response_data.dict(),
            message=f"Verification {result['final_status'].upper()}",
            request_id=result["request_id"]
        )
        
    except Exception as e:
        logger.error(f"Error formatting completed result: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error formatting verification results"
        )

async def _format_ongoing_request(request_id: str, request_data: Dict[str, Any]) -> APIResponse:
    """Format an ongoing verification request for response."""
    try:
        current_phase = request_data.get("status", VerificationPhase.PENDING)
        results = request_data.get("results", {})
        
        # Build partial results for ongoing verification
        response_data = VerificationStatusResponse(
            request_id=request_id,
            current_phase=current_phase,
            phase_1_audit=_build_phase1_result(results.get("phase_1")) if "phase_1" in results else None,
            phase_2_cognition=_build_phase2_result(results.get("phase_2")) if "phase_2" in results else None,
            phase_3_reality=_build_phase3_result(results.get("phase_3")) if "phase_3" in results else None,
            overall_status="processing",
            confidence_score=0.0,  # Not available until completion
            completion_timestamp=None,
            error_message=None
        )
        
        # Add processing time estimate
        if request_data.get("processing_started_at"):
            elapsed = (datetime.utcnow() - request_data["processing_started_at"]).total_seconds()
            message = f"Processing - Elapsed: {elapsed:.1f}s"
        else:
            message = "Queued for processing"
        
        return APIResponse(
            success=True,
            data=response_data.dict(),
            message=message,
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Error formatting ongoing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error formatting verification status"
        )

def _build_phase1_result(phase_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build Phase 1 audit result structure."""
    if not phase_data:
        return None
    
    return {
        "actor_proposal": phase_data.get("actor_proposal", ""),
        "adversary_critique": phase_data.get("adversary_critique", ""),
        "judge_score": phase_data.get("confidence_score", 0.0),
        "debate_transcript": phase_data.get("debate_transcript", []),
        "flags_raised": phase_data.get("flags_raised", []),
        "phase_status": phase_data.get("status", "pending")
    }

def _build_phase2_result(phase_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build Phase 2 cognition result structure."""
    if not phase_data:
        return None
    
    return {
        "z3_proof_result": phase_data.get("z3_proof_result", False),
        "constitutional_compliance": phase_data.get("z3_proof_result", False),
        "cognitive_load": 0.5,  # Placeholder - would come from working memory
        "memory_conflicts": [],  # Placeholder
        "proof_trace": phase_data.get("proof_trace"),
        "phase_status": phase_data.get("status", "pending")
    }

def _build_phase3_result(phase_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build Phase 3 reality result structure."""
    if not phase_data:
        return None
    
    return {
        "sensor_readings": {},  # Placeholder - would contain actual sensor data
        "simulation_results": {
            "confidence": phase_data.get("simulation_confidence", 0.0),
            "risk_assessment": phase_data.get("risk_assessment", {}),
            "safety_violations": phase_data.get("safety_violations", [])
        },
        "health_score": phase_data.get("simulation_confidence", 0.0),
        "anomalies_detected": phase_data.get("anomalies_detected", []),
        "grounding_status": "available" if phase_data.get("ground_truth_available", False) else "unavailable",
        "phase_status": phase_data.get("status", "pending")
    }

@router.get("/queue/status", response_model=APIResponse)
async def get_verification_queue_status():
    """
    Get status of the verification request queue.
    
    Returns:
        APIResponse with queue statistics
    """
    try:
        global verification_requests, verification_results
        
        # Count different status types
        pending_count = sum(1 for req in verification_requests.values() 
                          if req["status"] == VerificationPhase.PENDING)
        processing_count = sum(1 for req in verification_requests.values() 
                             if req["status"] in [VerificationPhase.PHASE_1_AUDIT, 
                                                VerificationPhase.PHASE_2_COGNITION, 
                                                VerificationPhase.PHASE_3_REALITY])
        completed_count = len(verification_results)
        failed_count = sum(1 for result in verification_results.values() 
                          if result["final_status"] == "rejected")
        
        # Calculate average processing time
        completed_times = []
        for result in verification_results.values():
            if "completion_timestamp" in result:
                try:
                    completion_time = datetime.fromisoformat(result["completion_timestamp"])
                    request_data = verification_requests.get(result["request_id"], {})
                    if "processing_started_at" in request_data:
                        processing_time = (completion_time - request_data["processing_started_at"]).total_seconds()
                        completed_times.append(processing_time)
                except Exception:
                    pass
        
        avg_processing_time = sum(completed_times) / len(completed_times) if completed_times else 0
        
        # Find oldest pending request
        oldest_pending = None
        oldest_time = None
        for req_id, req_data in verification_requests.items():
            if (req_data["status"] == VerificationPhase.PENDING and 
                req_id not in verification_results):
                submitted_time = req_data["submitted_at"]
                if oldest_time is None or submitted_time < oldest_time:
                    oldest_time = submitted_time
                    oldest_pending = req_id
        
        response_data = {
            "total_requests": len(verification_requests),
            "pending_count": pending_count,
            "processing_count": processing_count,
            "completed_count": completed_count,
            "failed_count": failed_count,
            "average_processing_time_seconds": round(avg_processing_time, 2),
            "longest_pending_request": oldest_pending
        }
        
        return APIResponse(
            success=True,
            data=response_data,
            message="Verification queue status retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"❌ Error retrieving queue status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error retrieving queue status"
        )

@router.get("/statistics", response_model=APIResponse)
async def get_verification_statistics(
    hours: int = Query(24, ge=1, le=168, description="Hours of history to analyze")
):
    """
    Get verification statistics and performance metrics.
    
    Args:
        hours: Number of hours of history to analyze (1-168)
        
    Returns:
        APIResponse with statistical data
    """
    try:
        global verification_results
        
        # Filter results by time window
        cutoff_time = datetime.utcnow().replace(microsecond=0) - timedelta(hours=hours)
        
        recent_results = {
            req_id: result for req_id, result in verification_results.items()
            if datetime.fromisoformat(result["completion_timestamp"]) >= cutoff_time
        }
        
        if not recent_results:
            stats_data = {
                "time_period_hours": hours,
                "total_verifications": 0,
                "success_rate": 0.0,
                "average_confidence": 0.0,
                "verification_distribution": {}
            }
        else:
            total_count = len(recent_results)
            success_count = sum(1 for result in recent_results.values() 
                              if result["final_status"] == "approved")
            confidence_scores = [result["overall_confidence"] for result in recent_results.values()]
            
            # Count by phase failure
            phase_failures = {"phase_1": 0, "phase_2": 0, "phase_3": 0}
            for result in recent_results.values():
                if result["final_status"] == "rejected":
                    detailed_results = result.get("detailed_results", {})
                    for phase in ["phase_1", "phase_2", "phase_3"]:
                        if detailed_results.get(phase, {}).get("status") == "failed":
                            phase_failures[phase] += 1
            
            stats_data = {
                "time_period_hours": hours,
                "total_verifications": total_count,
                "success_rate": round(success_count / total_count * 100, 2),
                "average_confidence": round(sum(confidence_scores) / len(confidence_scores), 2),
                "verification_distribution": {
                    "approved": success_count,
                    "rejected": total_count - success_count
                },
                "phase_failure_rates": {
                    phase: round(count / total_count * 100, 2) 
                    for phase, count in phase_failures.items()
                }
            }
        
        return APIResponse(
            success=True,
            data=stats_data,
            message=f"Verification statistics for last {hours} hours"
        )
        
    except Exception as e:
        logger.error(f"❌ Error retrieving verification statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error retrieving statistics"
        )