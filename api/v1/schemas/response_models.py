"""
Response models for API endpoints
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from api.v1.schemas.request_models import (
    VerificationPhase, 
    PromptSubmissionResponse, 
    VerificationStatusResponse,
    AuditTrailResponse
)


class APIResponse(BaseModel):
    """Base API response wrapper"""
    
    success: bool = Field(..., description="Whether the request succeeded")
    
    data: Optional[Any] = Field(None, description="Response data payload")
    
    message: Optional[str] = Field(None, description="Human-readable message")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    request_id: Optional[str] = Field(None, description="Associated request ID if applicable")


class ErrorResponse(APIResponse):
    """Standard error response format"""
    
    success: bool = False
    
    error_code: str = Field(..., description="Machine-readable error code")
    
    error_details: Optional[Dict[str, Any]] = Field(None, description="Detailed error information")
    
    correlation_id: Optional[str] = Field(None, description="For tracing and debugging")


class HealthCheckResponse(BaseModel):
    """System health check response"""
    
    status: str = Field(..., description="Overall system status")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    
    components: Dict[str, str] = Field(..., description="Individual component statuses")
    
    metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")


class SystemMetricsResponse(BaseModel):
    """System performance and operational metrics"""
    
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    
    request_count: int = Field(..., description="Total requests processed")
    
    average_response_time_ms: float = Field(..., description="Average response time")
    
    current_load: float = Field(..., description="Current system load factor")
    
    verification_queue_length: int = Field(..., description="Pending verification requests")
    
    last_verification_timestamp: Optional[datetime] = Field(None, description="Most recent verification")


class PromptQueueStatusResponse(BaseModel):
    """Status of the verification request queue"""
    
    total_requests: int = Field(..., description="Total requests in system")
    
    pending_count: int = Field(..., description="Requests awaiting processing")
    
    processing_count: int = Field(..., description="Currently processing")
    
    completed_count: int = Field(..., description="Successfully completed")
    
    failed_count: int = Field(..., description="Failed/rejected requests")
    
    average_wait_time_seconds: float = Field(..., description="Average queue wait time")
    
    longest_pending_request: Optional[str] = Field(None, description="ID of oldest pending request")


class VerificationStatisticsResponse(BaseModel):
    """Aggregate statistics about verification performance"""
    
    total_verifications: int = Field(..., description="Total verification attempts")
    
    success_rate: float = Field(..., description="Percentage of successful verifications")
    
    average_confidence_score: float = Field(..., description="Mean confidence across all verifications")
    
    phase_failure_rates: Dict[str, float] = Field(..., description="Failure rates by phase")
    
    common_rejection_reasons: List[Dict[str, Any]] = Field(..., description="Most frequent rejection causes")
    
    processing_time_distribution: Dict[str, float] = Field(..., description="Time distribution percentiles")