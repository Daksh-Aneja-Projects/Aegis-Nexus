from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Dict, Any
import os
from dotenv import set_key, load_dotenv

router = APIRouter()

# Simple Admin Auth (in prod use proper RBAC)
API_KEY_HEADER = APIKeyHeader(name="X-Admin-Key", auto_error=False)

def get_admin_key(api_key: str = Security(API_KEY_HEADER)):
    expected_key = os.getenv("ADMIN_API_KEY", "admin-secret-123")
    if api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid Admin Key")
    return api_key

class ConfigUpdateRequest(BaseModel):
    updates: Dict[str, str]

@router.post("/update-config", status_code=200)
async def update_config(request: ConfigUpdateRequest, admin: str = Depends(get_admin_key)):
    """
    Update system configuration (.env).
    Requires Admin Key.
    """
    try:
        env_path = ".env"
        # Validate keys (allow-list)
        ALLOWED_KEYS = [
            "MAX_DEBATE_ROUNDS", 
            "CONFIDENCE_THRESHOLD", 
            "AGENT_MODEL_VERSION",
            "CHAOS_MONKEY_ENABLED",
            "LOG_LEVEL"
        ]
        
        updated = {}
        for key, value in request.updates.items():
            if key not in ALLOWED_KEYS:
                continue # Skip unauthorized keys
            
            # Update .env file
            # Note: set_key might fail if file doesn't exist or permissions
            success = set_key(env_path, key, str(value))
            if success:
                 updated[key] = value
                 # Reload env for current process (partial effect)
                 os.environ[key] = str(value)
        
        if not updated:
            return {"status": "no_changes", "message": "No valid keys provided"}

        return {
            "status": "success",
            "updated_keys": list(updated.keys()),
            "message": "Configuration updated. Some changes may require restart."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config update failed: {str(e)}")

@router.get("/config", status_code=200)
async def get_config(admin: str = Depends(get_admin_key)):
    """Get current masked configuration."""
    ALLOWED_KEYS = [
        "MAX_DEBATE_ROUNDS", 
        "CONFIDENCE_THRESHOLD", 
        "AGENT_MODEL_VERSION",
        "CHAOS_MONKEY_ENABLED",
        "LOG_LEVEL"
    ]
    return {k: os.getenv(k, "default") for k in ALLOWED_KEYS}
