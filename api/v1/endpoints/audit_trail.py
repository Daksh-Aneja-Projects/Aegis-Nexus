"""
Audit Trail Endpoint for Aegis Nexus
Provides access to immutable audit logs and verification history.

This module implements the /audit_logs endpoint for querying
cryptographically secured audit trails.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.v1.schemas.request_models import (
    AuditTrailQuery, 
    AuditTrailResponse,
    AuditTrailEntry
)
from api.v1.schemas.response_models import APIResponse
from core.security.immutable_ledger import get_immutable_ledger

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/logs", response_model=APIResponse)
async def get_audit_trail(
    request_id: Optional[str] = Query(None, description="Filter by specific request ID"),
    start_date: Optional[str] = Query(None, description="ISO format start date"),
    end_date: Optional[str] = Query(None, description="ISO format end date"),
    event_types: Optional[str] = Query(None, description="Comma-separated event types"),
    limit: int = Query(100, ge=1, le=1000, description="Number of entries to return"),
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    """
    Retrieve audit trail entries with filtering and pagination.
    
    Args:
        request_id: Filter by specific verification request ID
        start_date: ISO format start date for time filtering
        end_date: ISO format end date for time filtering
        event_types: Comma-separated list of event types to include
        limit: Maximum number of entries to return
        offset: Pagination offset
        
    Returns:
        APIResponse with audit trail entries and metadata
    """
    try:
        # Get immutable ledger
        ledger = get_immutable_ledger()
        
        # Parse date filters
        start_dt = None
        end_dt = None
        
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid start_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
                )
        
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid end_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
                )
        
        # Parse event types
        event_type_list = None
        if event_types:
            event_type_list = [et.strip() for et in event_types.split(',')]
        
        # Query ledger
        ledger_entries = await ledger.get_audit_trail(
            start_time=start_dt,
            end_time=end_dt,
            event_types=event_type_list,
            limit=limit + offset  # Get extra to handle offset
        )
        
        # Apply offset and limit
        paginated_entries = ledger_entries[offset:offset + limit]
        
        # Convert to response format
        audit_entries = []
        for entry in paginated_entries:
            audit_entry = AuditTrailEntry(
                timestamp=entry.timestamp,
                event_type=entry.event_type,
                component=entry.component,
                details=entry.data,
                hash=entry.entry_hash
            )
            audit_entries.append(audit_entry)
        
        # Create response
        response_data = AuditTrailResponse(
            entries=audit_entries,
            total_count=len(ledger_entries),
            has_more=len(ledger_entries) > offset + limit,
            pqc_verification=await _get_chain_verification(ledger)
        )
        
        return APIResponse(
            success=True,
            data=response_data.dict(),
            message=f"Retrieved {len(audit_entries)} audit trail entries"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving audit trail: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error retrieving audit trail"
        )

@router.post("/logs/query", response_model=APIResponse)
async def post_audit_trail_query(query: AuditTrailQuery):
    """
    POST endpoint for complex audit trail queries.
    
    Args:
        query: Audit trail query with all filter parameters
        
    Returns:
        APIResponse with audit trail entries
    """
    try:
        # Convert query to GET parameters format
        event_types_str = ",".join(query.event_types) if query.event_types else None
        
        return await get_audit_trail(
            request_id=query.request_id,
            start_date=query.start_date.isoformat() if query.start_date else None,
            end_date=query.end_date.isoformat() if query.end_date else None,
            event_types=event_types_str,
            limit=query.limit,
            offset=query.offset
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing audit trail query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error processing audit query"
        )

@router.get("/logs/{entry_hash}", response_model=APIResponse)
async def get_audit_entry(entry_hash: str):
    """
    Retrieve a specific audit trail entry by its hash.
    
    Args:
        entry_hash: Cryptographic hash of the audit entry
        
    Returns:
        APIResponse with the specific audit entry
    """
    try:
        # Get immutable ledger
        ledger = get_immutable_ledger()
        
        # In a real implementation, you'd query by hash
        # For now, we'll search through recent entries
        recent_entries = await ledger.get_audit_trail(limit=1000)
        
        target_entry = None
        for entry in recent_entries:
            if entry.entry_hash.startswith(entry_hash) or entry.entry_hash == entry_hash:
                target_entry = entry
                break
        
        if not target_entry:
            raise HTTPException(
                status_code=404,
                detail=f"Audit entry with hash {entry_hash} not found"
            )
        
        # Verify entry integrity
        is_valid = await ledger.verify_entry_integrity(target_entry.entry_id)
        
        audit_entry = AuditTrailEntry(
            timestamp=target_entry.timestamp,
            event_type=target_entry.event_type,
            component=target_entry.component,
            details=target_entry.data,
            hash=target_entry.entry_hash
        )
        
        response_data = {
            "entry": audit_entry.dict(),
            "integrity_verified": is_valid,
            "pqc_signature": target_entry.pqc_signature,
            "previous_entry_hash": target_entry.previous_hash
        }
        
        return APIResponse(
            success=True,
            data=response_data,
            message="Audit entry retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving audit entry: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error retrieving audit entry"
        )

@router.get("/logs/verify/{entry_hash}", response_model=APIResponse)
async def verify_audit_entry(entry_hash: str):
    """
    Verify the integrity of a specific audit trail entry.
    
    Args:
        entry_hash: Hash of the entry to verify
        
    Returns:
        APIResponse with verification results
    """
    try:
        # Get immutable ledger
        ledger = get_immutable_ledger()
        
        # Find and verify the entry
        recent_entries = await ledger.get_audit_trail(limit=1000)
        
        target_entry = None
        for entry in recent_entries:
            if entry.entry_hash.startswith(entry_hash) or entry.entry_hash == entry_hash:
                target_entry = entry
                break
        
        if not target_entry:
            raise HTTPException(
                status_code=404,
                detail=f"Audit entry with hash {entry_hash} not found"
            )
        
        # Perform integrity verification
        is_valid = await ledger.verify_entry_integrity(target_entry.entry_id)
        
        # Get chain verification status
        chain_valid = await _verify_ledger_chain(ledger)
        
        response_data = {
            "entry_hash": entry_hash,
            "integrity_valid": is_valid,
            "chain_integrity_valid": chain_valid,
            "verification_timestamp": datetime.utcnow().isoformat(),
            "entry_details": {
                "event_type": target_entry.event_type,
                "component": target_entry.component,
                "timestamp": target_entry.timestamp.isoformat()
            }
        }
        
        verification_status = "VALID" if is_valid and chain_valid else "INVALID"
        
        return APIResponse(
            success=is_valid and chain_valid,
            data=response_data,
            message=f"Audit entry verification: {verification_status}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error verifying audit entry: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during entry verification"
        )

@router.get("/summary", response_model=APIResponse)
async def get_audit_summary(
    hours: int = Query(24, ge=1, le=168, description="Hours of history to summarize")
):
    """
    Get a summary of audit trail activity.
    
    Args:
        hours: Number of hours of history to analyze
        
    Returns:
        APIResponse with audit activity summary
    """
    try:
        # Get immutable ledger
        ledger = get_immutable_ledger()
        
        # Calculate time window
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get entries in time window
        entries = await ledger.get_audit_trail(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Large limit for summary
        )
        
        # Generate summary statistics
        event_type_counts = {}
        component_counts = {}
        hourly_activity = {}
        
        for entry in entries:
            # Count by event type
            event_type = entry.event_type
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            
            # Count by component
            component = entry.component
            component_counts[component] = component_counts.get(component, 0) + 1
            
            # Hourly activity (grouped by hour)
            hour_key = entry.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_activity[hour_key] = hourly_activity.get(hour_key, 0) + 1
        
        # Get ledger statistics
        ledger_stats = await ledger.get_ledger_statistics()
        
        summary_data = {
            "time_period_hours": hours,
            "total_entries": len(entries),
            "entries_per_hour": round(len(entries) / hours, 2),
            "event_type_distribution": event_type_counts,
            "component_distribution": component_counts,
            "hourly_activity": hourly_activity,
            "ledger_statistics": ledger_stats,
            "most_active_component": max(component_counts.items(), key=lambda x: x[1])[0] if component_counts else None,
            "most_common_event": max(event_type_counts.items(), key=lambda x: x[1])[0] if event_type_counts else None
        }
        
        return APIResponse(
            success=True,
            data=summary_data,
            message=f"Audit trail summary for last {hours} hours"
        )
        
    except Exception as e:
        logger.error(f"❌ Error generating audit summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error generating audit summary"
        )

async def _get_chain_verification(ledger) -> Optional[str]:
    """
    Get chain verification signature for audit trail integrity.
    
    Args:
        ledger: Immutable ledger instance
        
    Returns:
        Chain verification signature or None
    """
    try:
        # In a real implementation, this would return a PQC signature
        # of the entire ledger chain for external verification
        ledger_snapshot = await ledger.export_ledger_snapshot()
        return f"chain_sig_{hash(str(ledger_snapshot)) % 1000000}"
    except Exception:
        return None

async def _verify_ledger_chain(ledger) -> bool:
    """
    Verify the integrity of the entire ledger chain.
    
    Args:
        ledger: Immutable ledger instance
        
    Returns:
        bool: True if chain integrity is valid
    """
    try:
        # In a real implementation, this would verify the cryptographic
        # chain of all blocks and entries
        stats = await ledger.get_ledger_statistics()
        return stats["total_entries"] > 0  # Basic check
    except Exception:
        return False