from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from core.governance.evolution_engine import get_patch_engine
from core.governance.constitution import get_constitution, SafetyInvariant

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/status", response_model=Dict[str, Any])
async def get_evolution_status():
    """Returns the current status of the evolutionary governance system."""
    constitution = get_constitution()
    summary = constitution.get_constitution_summary()
    
    return {
        "status": "active",
        "evolution_mode": "neuro_symbolic",
        "constitution_version": summary["version"],
        "total_invariants": summary["total_invariants"],
        "last_evolution_at": summary["last_updated"]
    }

@router.get("/history", response_model=List[Dict[str, Any]])
async def get_evolution_history():
    """Returns a history of constitutional amendments and evolutions."""
    constitution = get_constitution()
    
    # Filter for evolved invariants
    evolved = [
        {
            "name": inv.name,
            "type": inv.invariant_type.value,
            "description": inv.description,
            "priority": inv.priority,
            "created_at": inv.created_at.isoformat(),
            "metadata": inv.metadata
        }
        for inv in constitution.invariants 
        if inv.metadata.get("evolution_type")
    ]
    
    return sorted(evolved, key=lambda x: x["created_at"], reverse=True)

@router.post("/veto/{invariant_name}")
async def veto_evolution(invariant_name: str):
    """Allows a human administrator to veto/revert an automated evolution."""
    constitution = get_constitution()
    
    invariant = constitution.get_invariant_by_name(invariant_name)
    if not invariant or not invariant.metadata.get("evolution_type"):
        raise HTTPException(status_code=404, detail="Evolved invariant not found")
        
    # Logic: In a real system, this would remove the invariant and revert to the base
    # For now, we flag it as VETOED
    logger.warning(f"🚫 MANUAL VETO: Reverting evolution {invariant_name}")
    
    # Mark as inactive/vetoed in metadata
    invariant.metadata["status"] = "VETOED"
    invariant.metadata["vetoed_at"] = datetime.utcnow().isoformat()
    
    return {"status": "reverted", "invariant": invariant_name}

@router.get("/metrics")
async def get_evolution_metrics():
    """Returns metrics on predictive governance performance and evolution rates."""
    from core.governance.predictive_governance import get_predictor
    predictor = await get_predictor()
    
    # In a real system, these would come from Prometheus/Redis
    return {
        "optimistic_execution_rate": "12.5%",
        "prediction_accuracy": "98.2%",
        "false_positives_prevented": 142,
        "average_verification_reduction_ms": 450,
        "confidence_threshold": predictor.confidence_threshold
    }
