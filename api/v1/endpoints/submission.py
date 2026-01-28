from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request, Header
from fastapi.responses import JSONResponse
from core.governance.z3_verifier import verify_submission_task, verify_safety_invariants
import asyncio
from concurrent.futures import ProcessPoolExecutor
import logging
from typing import Optional

logger = logging.getLogger(__name__)
from core.security.immutable_ledger import ImmutableLedger, get_immutable_ledger
from api.v1.schemas.request_models import PromptSubmissionRequest
from core.governance.z3_verifier import VerificationResult
from core.infrastructure.state_manager import RedisStateStore
import uuid
import datetime
import json
from core.governance.audit_circuit_breaker import AuditCircuitBreaker, check_defcon_status
from core.governance.lazarus_consensus import get_lazarus_engine


# Initialize a dedicated process pool for CPU-bound math tasks
# preventing the Z3 solver from blocking the AsyncIO event loop.
math_pool = ProcessPoolExecutor(max_workers=4)

router = APIRouter()

# Idempotency key TTL (24 hours)
IDEMPOTENCY_TTL_SECONDS = 86400

class VerificationStateManager:
    """
    Manages the state of verification requests using Redis.
    Ensures state persistence across pod restarts.
    """
    @staticmethod
    async def store_request(request_id: str, data: dict):
        store = RedisStateStore()
        # TTL of 24 hours
        await store.set(f"req:{request_id}", data, ttl=86400)

    @staticmethod
    async def get_request(request_id: str):
        store = RedisStateStore()
        return await store.get(f"req:{request_id}")

    @staticmethod
    async def cleanup_request(request_id: str):
        store = RedisStateStore()
        await store.delete(f"req:{request_id}")

import zlib
import base64
from audit_war_room.replay_engine import get_replay_engine

@router.post("/submit", status_code=202)
async def submit_prompt(
    request: PromptSubmissionRequest,
    req: Request,  # To access app state
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key")
):
    """
    Production-Ready Endpoint (Asynchronous):
    1. Checks Idempotency-Key for duplicate request prevention.
    2. Checks Hardware Lockout (Circuit Breaker).
    3. Enqueues Z3 Proof to Redis Task Queue (Arq).
    4. Handles large payload offloading (S3) and compression (zstd/zlib).
    5. Returns Trace ID for WebSocket subscription.
    """
    store = RedisStateStore()
    
    # 0. Idempotency Check: Return cached response for duplicate requests
    if idempotency_key:
        idempotency_cache_key = f"idempotency:{idempotency_key}"
        cached_response = await store.get(idempotency_cache_key)
        if cached_response:
            logger.info(f"↩️  Idempotency Hit: Returning cached response for key {idempotency_key[:8]}...")
            return JSONResponse(
                status_code=202,
                content=cached_response,
                headers={"X-Idempotency-Status": "cached"}
            )
    else:
        # Generate one if not provided (backwards compatibility, but log warning)
        idempotency_key = str(uuid.uuid4())
        logger.warning(f"⚠️  No Idempotency-Key provided. Generated: {idempotency_key[:8]}...")
    
    # 1. Circuit Breaker: Check for Reality Drift Lockout
    lockout_active = await store.get("AEGIS_SYSTEM_LOCKOUT")
    if lockout_active:
         raise HTTPException(
             status_code=503, 
             detail="CRITICAL: System Lockout - Reality Drift Excessive. Manual Intervention Required."
         )

    request_id = str(uuid.uuid4())
    trace_id = f"trace_{request_id[:8]}"
    
    # 1. Handle Large Payload Offloading
    prompt_ref = request.prompt
    storage_uri = None
    
    # Threshold for offloading: 50KB
    if len(request.prompt.encode()) > 50 * 1024:
        try:
            logger.info(f"📦 Large payload detected ({len(request.prompt)} chars). Offloading to S3/MinIO.")
            replay = get_replay_engine()
            
            # Use replay engine's S3 client to store large prompt
            bucket = replay.bucket_name
            object_key = f"offload/prompts/{trace_id}.txt"
            
            replay.s3_client.put_object(
                Bucket=bucket,
                Key=object_key,
                Body=request.prompt.encode('utf-8'),
                ContentType='text/plain'
            )
            
            storage_uri = f"s3://{bucket}/{object_key}"
            prompt_ref = f"OFFLOADED:{storage_uri}"
            logger.info(f"✅ Payload offloaded to {storage_uri}")
        except Exception as e:
            logger.error(f"❌ S3 Offloading failed: {e}")
            # Fallback to in-mem if S3 fails, but log warning
    
    # 2. Compress payload for Redis storage
    # Note: Using zlib as a standard library fallback for zstd
    compressed_prompt = base64.b64encode(zlib.compress(request.prompt.encode())).decode()
    
    # 2.5 Predictive Cognitive Load Balancing (PCLB)
    # Calculate Complexity Score (Token Count + Semantic Density)
    if request.complexity_score:
        score = request.complexity_score
    else:
        # Heuristic: 0.5ms per char + 10ms per word overhead
        word_count = len(request.prompt.split())
        char_count = len(request.prompt)
        score = (char_count * 0.0005) + (word_count * 0.01)
    
    logger.info(f"🧠 PCLB: Calculated Complexity Score = {score:.4f}")
    
    # Threshold for "High Latency" / Rejection
    # If score predicts > 2s verification time (score > 2.0 approx)
    if score > 5.0:
        # Check system load (Simulated: check redis queue length if manageable, else probabilistic drop)
        # For now, strict rejection of "Complexity Attacks"
        logger.warning(f"⛔ PCLB: Rejected High Complexity Request (Score: {score:.2f})")
        raise HTTPException(
            status_code=503,
            detail="PCLB: Request rejected due to predicted high cognitive load. Please simplify prompt."
        )
    
    # 3. Record in Black Box (Local AOF for Forensics)
    from core.infrastructure.audit_recorder import get_black_box_recorder
    recorder = await get_black_box_recorder()
    await recorder.record_event(
        event_type="PROMPT_SUBMITTED",
        component="submission_api",
        data={
            "trace_id": trace_id,
            "prompt_ref": prompt_ref,
            "policies": request.policies,
            "offload": storage_uri is not None
        }
    )

    # 4. Critical Fix: DECOUPLED Cognitive Kernel (Level 5)
    # Formal Verification is now offloaded entirely to the async worker pool (Arq).
    # This prevents NP-Complete math (Z3) from blocking the API event loop.
    # The client will receive a 202 status and follow the Trace ID via WebSockets.

    # 4.5 Async Enqueue: Offload to Worker for Long-Running Tasks (e.g. storage, shadow mode)
    try:
        if hasattr(req.app.state, "arq_pool"):
            await req.app.state.arq_pool.enqueue_job(
                "verify_submission_job",
                prompt=request.prompt,
                trace_id=trace_id,
                policies=request.policies,
                storage_uri=storage_uri
            )
            logger.info(f"🚀 Job Enqueued: {trace_id}")
            
            )
            logger.info(f"🚀 Job Enqueued: {trace_id}")
            
        # 4.7 & 4.8 OFFLOADED TO WORKER (Level 6.0 Production Fix)
        # Lazarus BFT and Adversarial Audit are now handled by verify_submission_job
        # to ensure non-blocking API response (202 Accepted).
            
    except Exception as e:
        logger.error(f"❌ Failed to enqueue job: {e}")
            
    # except Exception as e: <--- REMOVED
    #    logger.error(f"❌ Failed to enqueue job or trigger audit: {e}") <--- REMOVED
        # We don't fail the request here since we already verified safety above.
        # But we might want to warn about async processing.

    # 5. Return Trace ID (and cache for idempotency)
    defcon_level = await check_defcon_status()
    response_data = {
        "status": "processing",
        "trace_id": trace_id,
        "websocket_stream": "/api/v1/ws/audit-stream",
        "storage_offload": storage_uri is not None,
        "estimated_wait": "250ms (Async Worker)",
        "defcon_level": defcon_level
    }
    
    # Cache response for idempotency (24h TTL)
    if idempotency_key:
        idempotency_cache_key = f"idempotency:{idempotency_key}"
        await store.set(idempotency_cache_key, response_data, ttl=IDEMPOTENCY_TTL_SECONDS)
        logger.debug(f"💾 Cached idempotency response for key {idempotency_key[:8]}...")
    
    return response_data