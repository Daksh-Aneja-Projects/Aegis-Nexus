"""
WebSocket Manager for Real-Time War Room Observability.
Uses Redis Pub/Sub to scale across multiple API pods and receive events from Workers.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any, Optional
import logging
import json
import asyncio
import redis.asyncio as redis
import os

logger = logging.getLogger(__name__)

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.redis_url = os.getenv("AEGIS_REDIS_URL", "redis://localhost:6379")
        self.pubsub_task = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_connections)}")
        
        # Start Redis listener if not running
        if self.pubsub_task is None or self.pubsub_task.done():
            self.pubsub_task = asyncio.create_task(self._redis_listener())

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, event_type: str, payload: Dict[str, Any], trace_id: Optional[str] = None):
        """
        Standardized Level 5 Broadcast: Ensures cluster-wide consistency.
        Every event is published to Redis to ensure all pods receive it.
        """
        message = {
            "type": event_type,
            "timestamp": asyncio.get_event_loop().time(),
            "payload": {
                **payload,
                "trace_id": trace_id
            }
        }
        
        # 1. Publish to Redis for pod-to-pod synchronization
        try:
            r = redis.from_url(self.redis_url)
            await r.publish("aegis_audit_events", json.dumps(message))
        except Exception as e:
            logger.error(f"Redis Broadcast failed: {e}")
            # Fallback: immediate local broadcast if Redis is down
            await self._local_broadcast(message)

    async def _local_broadcast(self, message: Dict):
        """Send message only to clients connected to THIS pod."""
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Local WebSocket send failed: {e}")
                self.disconnect(connection)

    async def _redis_listener(self):
        """Background task to listen for Redis PubSub events."""
        try:
            r = redis.from_url(self.redis_url)
            pubsub = r.pubsub()
            await pubsub.subscribe("aegis_audit_events")
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        # Send to local clients only - recursion prevention
                        await self._local_broadcast(data)
                    except Exception as e:
                        logger.error(f"Failed to process Redis message: {e}")
        except Exception as e:
            logger.error(f"Redis Listener Error: {e}")
            # exponential backoff could go here

manager = ConnectionManager()

@router.websocket("/ws/audit-stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Keep alive logic
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except: 
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)

