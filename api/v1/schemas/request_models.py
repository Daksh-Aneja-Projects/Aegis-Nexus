"""
Pydantic models for request validation and serialization
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator


class VerificationPhase(str, Enum):
    """Enumeration of verification phases"""
    PENDING = "pending"
    PHASE_1_AUDIT = "phase_1_audit"
    PHASE_2_COGNITION = "phase_2_cognition"
    PHASE_3_REALITY = "phase_3_reality"
    COMPLETED = "completed"
    REJECTED = "rejected"


class SeverityLevel(str, Enum):
    """Severity levels for audit findings"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PromptSubmissionRequest(BaseModel):
    """Request model for submitting prompts for verification"""
    
    prompt: str = Field(
        ..., 
        min_length=1, 
        max_length=10000,
        description="The prompt/text to be verified by the cognitive governor"
    )

    @validator('prompt')
    def sanitize_prompt(cls, v):
        """Sanitize prompt to prevent injection attacks."""
        import re
        # Block common shell escape sequences and injection patterns
        DANGEROUS_PATTERNS = [
            r";\s*rm\s+-rf",
            r"\|\s*bash",
            r">\s*/dev/null",
            r"&\s*curl",
            r"\$\(.*\)",
            r"`.*`",
            r"DROP TABLE",
            r"chmod 777"
        ]
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f"Dangerous pattern detected in prompt: {pattern}")
        return v
    
    context: Optional[Dict[str, Any]] = Field(
        default={},
        description="Additional context for the verification process"
    )
    
    llm_provider: str = Field(
        default="openai",
        description="LLM provider to use (openai, anthropic, gemini)"
    )
    
    priority: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Priority level (1-10) for processing queue"
    )

    start_time: Optional[datetime] = Field(None, description="Scheduled start time for the request")

    agent_count: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Number of AI agents to engage in the adversarial debate (1 Actor + N-1 Adversaries)"
    )


    complexity_score: Optional[float] = Field(
        default=None,
        description="Pre-computed complexity score for load balancing"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default={},
        description="Additional metadata for tracking and auditing"
    )


class PromptSubmissionResponse(BaseModel):
    """Response model for prompt submission"""
    
    request_id: str = Field(
        ...,
        description="Unique identifier for this verification request"
    )
    
    status: VerificationPhase = Field(
        default=VerificationPhase.PENDING,
        description="Current verification phase"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of submission"
    )
    
    estimated_completion_time: Optional[datetime] = Field(
        None,
        description="Estimated time for completion"
    )


class VerificationStatusRequest(BaseModel):
    """Request model for checking verification status"""
    
    request_id: str = Field(
        ...,
        description="The request ID to check status for"
    )


class Phase1AuditResult(BaseModel):
    """Results from Phase 1: Adversarial Audit"""
    
    actor_proposal: str = Field(..., description="Proposal from actor agent")
    adversary_critique: str = Field(..., description="Critique from adversary agent")
    judge_score: float = Field(..., ge=0, le=100, description="Judge's confidence score")
    debate_transcript: List[Dict[str, Any]] = Field(..., description="Full debate transcript")
    flags_raised: List[str] = Field(default=[], description="Security/logic flags raised")
    phase_status: str = Field(..., description="Pass/fail status")


class Phase2CognitionResult(BaseModel):
    """Results from Phase 2: Cognitive State Verification"""
    
    z3_proof_result: bool = Field(..., description="Formal verification result")
    constitutional_compliance: bool = Field(..., description="Constitution compliance")
    cognitive_load: float = Field(..., ge=0, le=1, description="Current cognitive load")
    memory_conflicts: List[str] = Field(default=[], description="Memory conflict detections")
    proof_trace: Optional[str] = Field(None, description="Z3 proof trace if available")
    phase_status: str = Field(..., description="Pass/fail status")


class Phase3RealityResult(BaseModel):
    """Results from Phase 3: Reality Grounding"""
    
    sensor_readings: Dict[str, Any] = Field(..., description="Fused sensor data")
    simulation_results: Dict[str, Any] = Field(..., description="Digital twin simulation")
    health_score: float = Field(..., ge=0, le=1, description="Data health score")
    anomalies_detected: List[str] = Field(default=[], description="Detected anomalies")
    grounding_status: str = Field(..., description="Reality grounding verification")
    phase_status: str = Field(..., description="Pass/fail status")


class VerificationStatusResponse(BaseModel):
    """Complete verification status response"""
    
    request_id: str = Field(..., description="Request identifier")
    
    current_phase: VerificationPhase = Field(..., description="Current processing phase")
    
    phase_1_audit: Optional[Phase1AuditResult] = Field(None, description="Phase 1 results")
    
    phase_2_cognition: Optional[Phase2CognitionResult] = Field(None, description="Phase 2 results")
    
    phase_3_reality: Optional[Phase3RealityResult] = Field(None, description="Phase 3 results")
    
    overall_status: str = Field(..., description="Final verdict: approved/rejected")
    
    confidence_score: float = Field(..., ge=0, le=100, description="Overall confidence score")
    
    pqc_signature: Optional[str] = Field(None, description="Post-quantum cryptographic signature")
    
    audit_trail_hash: Optional[str] = Field(None, description="Immutable audit trail hash")
    
    completion_timestamp: Optional[datetime] = Field(None, description="When processing completed")
    
    error_message: Optional[str] = Field(None, description="Error details if rejected")


class AuditTrailEntry(BaseModel):
    """Individual audit trail entry"""
    
    timestamp: datetime = Field(..., description="Entry timestamp")
    
    event_type: str = Field(..., description="Type of audit event")
    
    component: str = Field(..., description="Which system component")
    
    details: Dict[str, Any] = Field(..., description="Event details")
    
    hash: str = Field(..., description="Cryptographic hash of entry")


class AuditTrailQuery(BaseModel):
    """Query parameters for audit trail retrieval"""
    
    request_id: Optional[str] = Field(None, description="Filter by request ID")
    
    start_date: Optional[datetime] = Field(None, description="Start date for filtering")
    
    end_date: Optional[datetime] = Field(None, description="End date for filtering")
    
    event_types: Optional[List[str]] = Field(None, description="Filter by event types")
    
    limit: int = Field(default=100, ge=1, le=1000, description="Number of entries to return")
    
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class AuditTrailResponse(BaseModel):
    """Response for audit trail queries"""
    
    entries: List[AuditTrailEntry] = Field(..., description="Audit trail entries")
    
    total_count: int = Field(..., description="Total matching entries")
    
    has_more: bool = Field(..., description="Whether more entries exist")
    
    pqc_verification: Optional[str] = Field(None, description="Chain verification signature")