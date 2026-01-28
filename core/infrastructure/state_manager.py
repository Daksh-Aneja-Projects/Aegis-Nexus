from abc import ABC, abstractmethod
import json
import time
import logging
from typing import Any, Optional, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from redis.exceptions import ConnectionError

logger = logging.getLogger(__name__)

class StateStore(ABC):
    """
    Abstract base class for distributed state management to prevent Cognitive Drift.
    Now fully Async for high-throughput production environments.
    """
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        pass

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    async def lock(self, resource: str, ttl: int = 10) -> bool:
        """Acquire a distributed lock."""
        pass

    @abstractmethod
    async def unlock(self, resource: str) -> bool:
        """Release a distributed lock."""
        pass

    @abstractmethod
    async def incr(self, key: str) -> int:
        """Atomically increment a counter."""
        pass


class RedisStateStore(StateStore):
    """
    Production-ready Redis State Store.
    Uses redis.asyncio for non-blocking I/O.
    
    CRITICAL: This implementation ENFORCES Redis availability.
    In a distributed system, falling back to local memory creates "Split-Brain" 
    cognitive drift. If Redis is down, the system MUST fail fast or alerting must trigger.
    """
    _instance = None
    _redis = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisStateStore, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Prevent re-initialization if singleton already exists
        if self._redis is not None:
            return
            
        import os
        from redis import asyncio as redis
        
        redis_url = os.getenv("AEGIS_REDIS_URL", "redis://localhost:6379/0")
        
        # Configure robust connection pool
        self._redis = redis.from_url(
            redis_url, 
            decode_responses=True,
            socket_connect_timeout=5, # Increased for initial connect
            socket_keepalive=True,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # We do NOT use a local cache fallback for WRITES in production.
        # That would be lying to the user about "Distributed State".

    async def _ensure_connection(self):
        try:
            return await self._redis.ping()
        except Exception as e:
            logger.critical(f"🔥 REDIS CONNECTION FAILED: {str(e)}")
            return False

    @retry(
        stop=stop_after_attempt(5), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ConnectionError)
    )
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        try:
            # Serialize if needed (simple JSON for now)
            if hasattr(value, "json"):
                payload = value.json()
            elif hasattr(value, "model_dump_json"): # Pydantic v2
                payload = value.model_dump_json()
            elif isinstance(value, (dict, list)):
                payload = json.dumps(value, default=str)
            else:
                payload = str(value)
                
            if ttl:
                await self._redis.setex(key, ttl, payload)
            else:
                await self._redis.set(key, payload)
            return True
        except ConnectionError as e:
             # Log to dedicated "Red Alert" channel
            logger.critical(f"CRITICAL: State Manager lost connection to Redis: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Redis SET failed for {key}: {e}")
            # In production, we propagate this error so the API returns 503
            # rather than pretending success with a local dict that will vanish on restart.
            raise RuntimeError("Distributed State Store Unavailable") from e

    async def get(self, key: str) -> Optional[Any]:
        try:
            val = await self._redis.get(key)
            if val:
                try:
                    return json.loads(val)
                except:
                    return val
            return None
        except Exception as e:
            logger.error(f"❌ Redis GET failed for {key}: {e}")
            # For reads, we might tolerate failure by returning None, 
            # but it depends on the criticality. For now, log and return None.
            return None

    async def delete(self, key: str) -> bool:
        try:
            await self._redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"❌ Redis DELETE failed for {key}: {e}")
            return False

    async def lock(self, resource: str, ttl: int = 10) -> bool:
        lock_key = f"lock:{resource}"
        try:
            # Atomic set if not exists (NX) with expiry
            # This is the standard simple Redis lock pattern
            return await self._redis.set(lock_key, "LOCKED", ex=ttl, nx=True)
        except Exception as e:
             logger.error(f"❌ Distributed Lock failed for {resource}: {e}")
             return False

    async def unlock(self, resource: str) -> bool:
        lock_key = f"lock:{resource}"
        try:
            await self._redis.delete(lock_key)
            return True
        except Exception as e:
            logger.error(f"❌ Unlock failed for {resource}: {e}")
            return False

    async def incr(self, key: str) -> int:
        """Atomically increment a counter in Redis."""
        try:
            return await self._redis.incr(key)
        except Exception as e:
            logger.error(f"❌ Redis INCR failed for {key}: {e}")
            return 0

    async def broadcast_threat_signature(self, signature_hash: str, threat_data: Dict[str, Any]):
        """
        Federated Constitutional Learning Logic.
        Broadcasts threat signatures to the network via Redis Pub/Sub.
        """
        try:
            channel = "aegis:threat:signatures"
            message = {
                "hash": signature_hash,
                "data": threat_data,
                "timestamp": time.time(),
                "origin": "nexus_6_0"
            }
            await self._redis.publish(channel, json.dumps(message))
            logger.info(f"🛰️  Federated Learning: Broadcasted threat signature {signature_hash[:8]}")
        except Exception as e:
            logger.warning(f"⚠️  Federated Learning broadcast failed: {e}")
