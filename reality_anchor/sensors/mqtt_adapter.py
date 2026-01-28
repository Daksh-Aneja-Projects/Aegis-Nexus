"""
MQTT Adapter for Aegis Nexus Digital Twin
Bridges industrial IoT protocols (MQTT) into the Reality Anchor sensor fusion engine.
Allows Aegis to govern physical robotic and industrial systems.
"""

import logging
import json
import asyncio
import os
from typing import Dict, Any, Optional, Callable
from datetime import datetime

# Attempt to import paho-mqtt, fallback to mock for dev/test
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logging.warning("⚠️  paho-mqtt not installed. Using MOCK MQTT Adapter.")

logger = logging.getLogger(__name__)

class MQTTRealityAdapter:
    """
    Adapter to bridge MQTT sensor telemetry into the Aegis Sensor Fusion loop.
    Supports industrial 'Reality Anchors' (e.g., robotic arm position, valve status).
    """
    
    def __init__(self, broker_url: str = None, client_id: str = "aegis_nexus_twin"):
        self.broker_url = broker_url or os.getenv("AEGIS_MQTT_BROKER", "localhost")
        self.port = int(os.getenv("AEGIS_MQTT_PORT", 1883))
        self.client_id = client_id
        self.client: Optional[mqtt.Client] = None
        self.topics = ["aegis/sensors/#", "factory/robot/status"]
        self.on_telemetry_received: Optional[Callable] = None
        
    async def start(self, callback: Callable):
        """Start the MQTT subscriber loop."""
        self.on_telemetry_received = callback
        
        if not MQTT_AVAILABLE:
            logger.info("🧪 Starting MOCK MQTT telemetery loop")
            asyncio.create_task(self._mock_loop())
            return

        try:
            self.client = mqtt.Client(self.client_id)
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            
            logger.info(f"🔌 Connecting to MQTT Broker: {self.broker_url}:{self.port}")
            self.client.connect(self.broker_url, self.port, 60)
            self.client.loop_start()
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MQTT Broker: {e}")

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("✅ Connected to MQTT Broker")
            for topic in self.topics:
                client.subscribe(topic)
                logger.info(f"📡 Subscribed to: {topic}")
        else:
            logger.error(f"❌ MQTT Connection failed with result code {rc}")

    def _on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages and route to Sensor Fusion."""
        try:
            payload = json.loads(msg.payload.decode())
            topic = msg.topic
            
            logger.debug(f"📥 MQTT Received [{topic}]: {payload}")
            
            # Format as SensorReading (Internal Protocol)
            from reality_anchor.common.types import SensorReading
            
            reading = SensorReading(
                sensor_id=payload.get("sensor_id", topic.split('/')[-1]),
                sensor_type=payload.get("type", "industrial_iot"),
                value=payload.get("value", 0.0),
                quality=payload.get("quality", 1.0),
                timestamp=datetime.utcnow(),
                metadata={"topic": topic, "raw": payload}
            )
            
            if self.on_telemetry_received:
                # We need to bridge from sync MQTT thread to async loop
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.on_telemetry_received(reading))
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to process MQTT message: {e}")

    async def _mock_loop(self):
        """Simulate realistic IoT telemetry for development/testing."""
        while True:
            await asyncio.sleep(1.0)
            if self.on_telemetry_received:
                from reality_anchor.common.types import SensorReading
                # Simulate a robotic arm position
                mock_reading = SensorReading(
                    sensor_id="robotic_arm_01",
                    sensor_type="position",
                    value=25.5 + (0.1 * datetime.utcnow().second),
                    quality=0.99,
                    timestamp=datetime.utcnow(),
                    metadata={"mock": True}
                )
                await self.on_telemetry_received(mock_reading)

    def stop(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("🔌 MQTT Connection closed")

# Global Registry
_adapter = None

async def get_mqtt_adapter() -> MQTTRealityAdapter:
    global _adapter
    if _adapter is None:
        _adapter = MQTTRealityAdapter()
    return _adapter
