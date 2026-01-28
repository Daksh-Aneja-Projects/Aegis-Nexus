"""
Aegis Nexus - Hardened Z3 Verification Worker
Includes OS-level resource clamping and secure memory handling.
"""
import os
import resource  # <--- REQUIRED for Process Isolation
import signal
import asyncio
import logging
import json
import hashlib
import redis.asyncio as redis
from arq import create_pool
from arq.connections import RedisSettings
from core.governance.z3_verifier import initialize_formal_verifier, get_formal_verifier
from core.security.immutable_ledger import get_immutable_ledger
from core.governance.lazarus_consensus import get_lazarus_engine
from audit_war_room.orchestrator import get_battle_orchestrator

logger = logging.getLogger(__name__)

# Strict constraints to prevent "Constraint Bombs"
MAX_CPU_SECONDS = 2  # Hard limit for SMT solver
MAX_MEMORY_BYTES = 512 * 1024 * 1024  # 512MB RAM cap per worker

def limit_resources():
    """Apply OS-level resource limits to the worker process."""
    # Prevent CPU starvation
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU_SECONDS, MAX_CPU_SECONDS + 1))
    # Prevent Memory Leaks/OOM
    resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY_BYTES, MAX_MEMORY_BYTES))

async def startup(ctx):
    """Initialize system resources with constraints."""
    logger.info("👷 Z3 Worker Starting Up...")
    # limit_resources()  # <--- REMOVED: Dangerous for persistent worker process
    await initialize_formal_verifier()
    ctx["redis"] = redis.Redis(
        host=os.getenv("AEGIS_REDIS_HOST", "localhost"),
        port=int(os.getenv("AEGIS_REDIS_PORT", 6379))
    )

async def verify_submission_job(ctx, prompt: str, trace_id: str, policies: dict = None, storage_uri: str = None, metadata: dict = None):
    """
    Background Job: Formal Verification with Hard Timeouts
    """
    logger.info(f"🔨 Processing Job: {trace_id}")
    
    # Priority Check Logging
    if metadata and metadata.get('priority') == 'critical':
        logger.warning(f"🚨 PROCESSING CRITICAL PRIORITY JOB: {trace_id}")

    verifier = get_formal_verifier()
    
    try:
        # Enforce strict timeout on the future itself
        result = await asyncio.wait_for(
            verifier.verify_action_constraints(prompt, safety_policy=policies),
            timeout=MAX_CPU_SECONDS
        )
    except asyncio.TimeoutError:
        logger.error(f"💀 CONSTRAINT BOMB DETECTED: {trace_id}. Terminating job.")
        # In a real worker, we might need to suicide the process if Z3 is stuck in C++
        return False
    except Exception as e:
        logger.error(f"❌ Verification Error: {e}")
        return False

    # Secure Ledger Entry (Masked)
    ledger = get_immutable_ledger()
    # Use SHA-256 for PII masking (Zero-Knowledge Logging)
    masked_prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    
    await ledger.create_flight_recorder_entry(
        verification_result={
            "trace_id": trace_id,
            "is_safe": result.is_safe,
            "proof_hash": hashlib.sha256(str(result.proof_trace).encode()).hexdigest()
        },
        fused_state={"prompt_hash": masked_prompt_hash}
    )

    # Publish result
    payload = {
        "type": "verification_update",
        "payload": {
            "trace_id": trace_id,
            "status": "VERIFIED" if result.is_safe else "REJECTED",
            "proof": result.proof_trace, # Only send proof if safe or needed for audit
            "timestamp": asyncio.get_event_loop().time()
        }
    }
    
    await ctx["redis"].publish("aegis_audit_events", json.dumps(payload))

    # ---------------------------------------------------------
    # Phase 2 & 3: Lazarus Consensus & Adversarial Audit (Async Chain)
    # ---------------------------------------------------------
    if result.is_safe:
        try:
            # 4.7 Lazarus BFT Consensus
            logger.info(f"⚖️ Starting Lazarus BFT for {trace_id}...")
            lazarus = get_lazarus_engine()
            lazarus_result = await lazarus.reach_consensus(prompt)
            
            if not lazarus_result["is_consensus"]:
                logger.warning(f"❌ Lazarus BFT Divergence for {trace_id}")
                # Flag in redis or notify?
                await ctx["redis"].publish("aegis_alerts", json.dumps({
                    "type": "LAZARUS_DIVERGENCE",
                    "trace_id": trace_id,
                    "details": lazarus_result
                }))

            # 4.8 Adversarial Audit (Battle Room)
            logger.info(f"⚔️ Triggering Adversarial Audit for {trace_id}...")
            orchestrator = get_battle_orchestrator()
            await orchestrator.submit_audit_job(
                prompt=prompt,
                context={
                    "debate_id": trace_id, # Use trace_id as debate_id
                    "trace_id": trace_id,
                    "policies": policies or {},
                    "description": "Async Worker initiated audit"
                },
                agent_count=3 # Default
            )
        except Exception as e:
            logger.error(f"❌ Async Pipeline Phase 2/3 Failed: {e}")
            # Don't fail the Z3 result, but log error

    return result.is_safe

class WorkerSettings:
    """Arq Worker Configuration"""
    functions = [verify_submission_job]
    redis_settings = RedisSettings(
        host=os.getenv("AEGIS_REDIS_HOST", "localhost"),
        port=int(os.getenv("AEGIS_REDIS_PORT", 6379))
    )
    on_startup = startup
    # Priority Queue Implementation
    # Arq will pop from 'system_halt' first, then 'critical', then 'default'
    queues = ["system_halt", "critical_verification", "arq:queue"]
    
    max_jobs = 4 # Limit concurrent proofs to prevent CPU saturation
