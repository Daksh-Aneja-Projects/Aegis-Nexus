"""
Black Box Recorder (Forensic Replay)
Records verification context (z3_seed, input_vector, memory_state) for post-incident analysis.
"""
import logging
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class BlackBoxRecorder:
    def __init__(self, storage_path: str = "./audit_logs/blackbox"):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
        
    async def record_decision(self, 
                            action_id: str, 
                            input_vector: str, 
                            z3_seed: int, 
                            verification_result: Dict[str, Any],
                            memory_snapshot: Optional[Dict] = None):
        """
        Record a decision event to the immutable append-only log.
        """
        try:
            # Create immutable record
            record_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()
            
            record = {
                "record_id": record_id,
                "timestamp": timestamp,
                "action_id": action_id,
                "input_vector": input_vector,
                "z3_seed": z3_seed,
                "verification_result": verification_result,
                "memory_snapshot": memory_snapshot or {}
            }
            
            # Serialize
            data = json.dumps(record, sort_keys=True)
            
            # Write to disk (Simulating S3/MinIO append)
            # using standard I/O in a blocking way for MVP or could offload to thread
            # Since this is "Black Box" forensic, durability > speed, but let's not block too long.
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            filename = f"{self.storage_path}/log_{date_str}.jsonl"
            
            with open(filename, mode='a') as f:
                f.write(data + "\n")
                
            logger.info(f"📼 Black Box Recorded: {action_id} (Seed: {z3_seed})")
            return record_id
            
        except Exception as e:
            logger.error(f"❌ Failed to record black box event: {e}")
            return None

    async def retrieve_record(self, record_id: str) -> Optional[Dict]:
        """Technically difficult in append-only without index, simple linear scan for MVP"""
        # Linear scan implementation for gap analysis proof
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith(".jsonl"):
                    with open(f"{self.storage_path}/{filename}", mode='r') as f:
                        for line in f:
                            record = json.loads(line)
                            if record['record_id'] == record_id:
                                return record
            return None
        except Exception:
            return None

# Global Singleton
_recorder = None

async def get_black_box_recorder():
    global _recorder
    if _recorder is None:
        _recorder = BlackBoxRecorder()
    return _recorder
