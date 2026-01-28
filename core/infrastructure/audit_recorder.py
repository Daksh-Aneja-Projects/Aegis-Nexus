import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from core.infrastructure.state_manager import RedisStateStore
import hashlib
import zlib
import base64

logger = logging.getLogger(__name__)

class BlackBoxRecorder:
    """
    High-integrity "Black Box" Audit Recorder.
    
    Writes every cognitive decision and sensor state change to a local
    Append-Only File (AOF) with immediate fsync for maximum durability.
    Decoupled from the main request path via an internal queue and thread pool.
    """
    
    def __init__(self, log_dir: str = "data/audit"):
        self.log_dir = log_dir
        self.current_file = os.path.join(self.log_dir, f"blackbox_{datetime.utcnow().strftime('%Y%m%d')}.aof")
        self._queue: asyncio.Queue = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._stop_event = asyncio.Event()
        self._worker_task: Optional[asyncio.Task] = None
        self._lamport_clock = 0
        self._clock_lock = asyncio.Lock()
        self.last_hash = "GENESIS_HASH_00000000000000000000000000000000"

    async def initialize(self) -> bool:
        """Initialize core directories and start the recording worker."""
        try:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir, exist_ok=True)
            
            self._worker_task = asyncio.create_task(self._recording_loop())
            
            # Recover last hash from file if exists
            if os.path.exists(self.current_file):
                try:
                    with open(self.current_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        if lines:
                            last_line = json.loads(lines[-1])
                            self.last_hash = last_line.get("proof_hash", self.last_hash)
                            logger.info(f"🔗 Audit Chain Resumed from: {self.last_hash[:16]}...")
                except Exception as e:
                    logger.error(f"⚠️ Failed to recover audit chain: {e}")
            
            logger.info(f"📼 Flight Recorder Active: Writing to {self.current_file}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Black Box Recorder: {e}")
            return False

    async def record_event(self, event_type: str, component: str, data: Dict[str, Any], metadata: Optional[Dict] = None):
        """
        Public API to record an event.
        Adds to queue and returns immediately.
        """
        async with self._clock_lock:
            self._lamport_clock += 1
            current_time = self._lamport_clock

        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "lamport_clock": current_time,
            "event_type": event_type,
            "component": component,
            "data": data,
            "metadata": metadata or {},
            "version": "1.0.0"
        }
        
        # Level 5: Holographic Vectorization
        # Embed the state for precedent searching
        event["vector_embedding"] = self._generate_event_embedding(event)

        # Feature 2: Holographic State Replay - Compression (Gap)
        # If data is large (e.g. Z3 Search Tree), compress it
        if event_type == "flight_recorder_blackbox" or len(json.dumps(data)) > 1024:
             try:
                 json_bytes = json.dumps(data).encode('utf-8')
                 compressed = zlib.compress(json_bytes)
                 event["data_compressed"] = base64.b64encode(compressed).decode('utf-8')
                 event["compression"] = "zlib"
                 event["data"] = None # Remove raw data to save space
             except Exception as e:
                 logger.warning(f"Compression failed: {e}")
        
        await self._queue.put(event)

    def _generate_event_embedding(self, event: Dict) -> List[float]:
        """
        Generate a pseudo-embedding for the event state.
        In production, use a sentence-transformer model.
        """
        content = f"{event['event_type']}:{json.dumps(event['data'])}"
        h = hashlib.sha256(content.encode()).hexdigest()
        return [int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]

    async def _recording_loop(self):
        """Background worker that drains the queue and writes to disk."""
        while not self._stop_event.is_set():
            try:
                # Wait for an event
                event = await self._queue.get()
                
                # Perform blocking I/O in thread pool
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    self._executor, 
                    self._write_to_disk, 
                    event
                )
                
                self._queue.task_done()
            except Exception as e:
                logger.error(f"⚠️ Error in Black Box recording loop: {e}")
                await asyncio.sleep(1)

    def _write_to_disk(self, event: Dict[str, Any]):
        """Synchronous write and fsync with Hash Chaining."""
        try:
            # Hash Chaining (Immutable Ledger)
            # Calculate SHA256(prev_hash + current_data)
            # We must serialize data deterministically
            canonical_data = json.dumps(event, sort_keys=True)
            chain_content = f"{self.last_hash}{canonical_data}"
            event_hash = hashlib.sha256(chain_content.encode()).hexdigest()
            
            event["proof_hash"] = event_hash
            event["previous_hash"] = self.last_hash
            
            # Update state
            self.last_hash = event_hash

            line = json.dumps(event) + "\n"
            with open(self.current_file, "a", encoding="utf-8") as f:
                f.write(line)
                # Ensure data is actually flushed to the hardware
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.error(f"❌ DISK I/O ERROR: Failed to write to Black Box AOF: {e}")

    async def shutdown(self):
        """Gracefully drain the queue and stop the worker."""
        logger.info("🛑 Shutting down Black Box Recorder...")
        self._stop_event.set()
        
        # Wait for queue to drain with timeout
        try:
            await asyncio.wait_for(self._queue.join(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("⚠️ Black Box queue drain timed out. Some events may be lost.")
            
        if self._worker_task:
            self._worker_task.cancel()
        
        self._executor.shutdown(wait=True)
        logger.info("🔚 Black Box Recorder offline.")

# Global instance
_recorder = BlackBoxRecorder()

async def get_black_box_recorder() -> BlackBoxRecorder:
    return _recorder

async def initialize_audit_recorder() -> bool:
    return await _recorder.initialize()
