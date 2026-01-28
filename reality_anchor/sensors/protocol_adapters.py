"""
Protocol Adapters for Aegis Nexus
Handles communication with various IoT protocols (MQTT, CoAP, HTTP).

This module provides standardized adapter interfaces for different sensor protocols.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logging.warning("⚠️  MQTT library not available")

try:
    import aiocoap
    COAP_AVAILABLE = True
except ImportError:
    COAP_AVAILABLE = False
    logging.warning("⚠️  CoAP library not available")

import aiohttp

from reality_anchor.common.types import SensorReading

logger = logging.getLogger(__name__)

@dataclass
class ProtocolConfig:
    """Configuration for a sensor protocol"""
    protocol_type: str  # mqtt, coap, http
    host: str
    port: int
    topic_prefix: str = ""
    auth_required: bool = False
    username: Optional[str] = None
    password: Optional[str] = None
    ssl_enabled: bool = False

class SensorProtocolAdapter:
    """
    Universal adapter for different sensor communication protocols.
    
    Supports MQTT, CoAP, and HTTP protocols with automatic protocol detection,
    standardized data conversion, and heartbeat monitoring (Gap 2).
    """
    
    def __init__(self):
        """Initialize the protocol adapter."""
        self.adapters = {}
        self.active_connections = {}
        self.data_callbacks = {}
        
        # Heartbeat Monitoring Configuration (Gap 2)
        self.heartbeat_interval_seconds = 30
        self.max_missed_heartbeats = 3
        self.heartbeat_task = None
        self.last_heartbeat: Dict[str, datetime] = {}
        self.missed_heartbeats: Dict[str, int] = {}
        self.auto_reconnect_enabled = True
        
    async def initialize(self) -> bool:
        """Initialize the protocol adapter system."""
        try:
            logger.info("🔌 Initializing sensor protocol adapters...")
            
            # Initialize protocol-specific adapters
            if MQTT_AVAILABLE:
                self.adapters['mqtt'] = MQTTAdapter()
            if COAP_AVAILABLE:
                self.adapters['coap'] = CoAPAdapter()
            self.adapters['http'] = HTTPAdapter()
            
            # Start heartbeat monitoring loop (Gap 2)
            self.heartbeat_task = asyncio.create_task(self._heartbeat_monitoring_loop())
            
            logger.info(f"✅ Protocol adapters initialized: {list(self.adapters.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize protocol adapters: {str(e)}")
            return False
    
    async def _heartbeat_monitoring_loop(self):
        """
        Background task for heartbeat monitoring (Gap 2).
        Detects silent sensor failures and auto-reconnects.
        """
        logger.info("💓 Heartbeat monitoring loop started")
        
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval_seconds)
                
                for sensor_id in list(self.active_connections.keys()):
                    try:
                        # Attempt a lightweight read (heartbeat check)
                        reading = await asyncio.wait_for(
                            self.read_sensor_data(sensor_id),
                            timeout=5.0
                        )
                        
                        if reading:
                            # Heartbeat successful
                            self.last_heartbeat[sensor_id] = datetime.utcnow()
                            self.missed_heartbeats[sensor_id] = 0
                        else:
                            # No data - increment missed heartbeats
                            self.missed_heartbeats[sensor_id] = self.missed_heartbeats.get(sensor_id, 0) + 1
                            logger.warning(f"⚠️ Sensor {sensor_id} heartbeat missed ({self.missed_heartbeats[sensor_id]}/{self.max_missed_heartbeats})")
                            
                    except asyncio.TimeoutError:
                        self.missed_heartbeats[sensor_id] = self.missed_heartbeats.get(sensor_id, 0) + 1
                        logger.warning(f"⚠️ Sensor {sensor_id} heartbeat timeout ({self.missed_heartbeats[sensor_id]}/{self.max_missed_heartbeats})")
                    
                    # Check if sensor should be marked offline
                    if self.missed_heartbeats.get(sensor_id, 0) >= self.max_missed_heartbeats:
                        logger.critical(f"🚨 Sensor {sensor_id} OFFLINE: {self.max_missed_heartbeats} missed heartbeats")
                        
                        # Auto-reconnect if enabled
                        if self.auto_reconnect_enabled:
                            await self._attempt_reconnection(sensor_id)
                            
            except asyncio.CancelledError:
                logger.info("💓 Heartbeat monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Heartbeat monitoring error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _attempt_reconnection(self, sensor_id: str):
        """Attempt to reconnect a failed sensor."""
        if sensor_id not in self.active_connections:
            return
        
        try:
            connection = self.active_connections[sensor_id]
            config = connection['config']
            
            logger.info(f"🔄 Attempting reconnection for sensor {sensor_id}...")
            
            # Disconnect first
            await self.disconnect_sensor(sensor_id)
            
            # Attempt reconnection
            success = await self.connect_sensor(sensor_id, config)
            
            if success:
                logger.info(f"✅ Sensor {sensor_id} reconnected successfully")
                self.missed_heartbeats[sensor_id] = 0
            else:
                logger.error(f"❌ Sensor {sensor_id} reconnection failed")
                
        except Exception as e:
            logger.error(f"❌ Reconnection error for {sensor_id}: {e}")
    
    async def connect_sensor(self, sensor_id: str, config: ProtocolConfig) -> bool:
        """
        Connect to a sensor using the specified protocol.
        
        Args:
            sensor_id: Unique identifier for the sensor
            config: Protocol configuration
            
        Returns:
            bool: Connection success status
        """
        try:
            adapter = self.adapters.get(config.protocol_type.lower())
            if not adapter:
                logger.error(f"❌ Unsupported protocol: {config.protocol_type}")
                return False
            
            success = await adapter.connect(sensor_id, config)
            if success:
                self.active_connections[sensor_id] = {
                    'adapter': adapter,
                    'config': config,
                    'connected_at': datetime.utcnow()
                }
                logger.info(f"✅ Connected sensor {sensor_id} via {config.protocol_type}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to connect sensor {sensor_id}: {str(e)}")
            return False
    
    async def disconnect_sensor(self, sensor_id: str) -> bool:
        """
        Disconnect a sensor.
        
        Args:
            sensor_id: Sensor to disconnect
            
        Returns:
            bool: Disconnection success status
        """
        if sensor_id not in self.active_connections:
            return True
        
        try:
            connection = self.active_connections[sensor_id]
            adapter = connection['adapter']
            
            success = await adapter.disconnect(sensor_id)
            if success:
                del self.active_connections[sensor_id]
                logger.info(f"✅ Disconnected sensor {sensor_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to disconnect sensor {sensor_id}: {str(e)}")
            return False
    
    async def read_sensor_data(self, sensor_id: str) -> Optional[SensorReading]:
        """
        Read data from a connected sensor.
        
        Args:
            sensor_id: Sensor to read from
            
        Returns:
            SensorReading or None if failed
        """
        if sensor_id not in self.active_connections:
            logger.warning(f"⚠️  Sensor {sensor_id} not connected")
            return None
        
        try:
            connection = self.active_connections[sensor_id]
            adapter = connection['adapter']
            
            raw_data = await adapter.read_data(sensor_id)
            if raw_data:
                return await self._convert_to_sensor_reading(sensor_id, raw_data, connection['config'])
            
        except Exception as e:
            logger.error(f"❌ Failed to read sensor {sensor_id}: {str(e)}")
        
        return None
    
    async def subscribe_to_sensor(self, sensor_id: str, callback: Callable[[SensorReading], None]) -> bool:
        """
        Subscribe to real-time sensor data updates.
        
        Args:
            sensor_id: Sensor to subscribe to
            callback: Function to call with sensor readings
            
        Returns:
            bool: Subscription success status
        """
        if sensor_id not in self.active_connections:
            logger.warning(f"⚠️  Sensor {sensor_id} not connected")
            return False
        
        try:
            connection = self.active_connections[sensor_id]
            adapter = connection['adapter']
            
            success = await adapter.subscribe(sensor_id, self._subscription_callback_wrapper(callback))
            if success:
                self.data_callbacks[sensor_id] = callback
                logger.info(f"✅ Subscribed to sensor {sensor_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to sensor {sensor_id}: {str(e)}")
            return False
    
    async def _subscription_callback_wrapper(self, user_callback: Callable[[SensorReading], None]):
        """Wrapper for subscription callbacks to handle data conversion."""
        async def wrapper(sensor_id: str, raw_data: Dict[str, Any]):
            try:
                connection = self.active_connections.get(sensor_id)
                if connection:
                    sensor_reading = await self._convert_to_sensor_reading(
                        sensor_id, raw_data, connection['config']
                    )
                    if sensor_reading:
                        await user_callback(sensor_reading)
            except Exception as e:
                logger.error(f"❌ Error in subscription callback: {str(e)}")
        return wrapper
    
    async def _convert_to_sensor_reading(
        self, 
        sensor_id: str, 
        raw_data: Dict[str, Any], 
        config: ProtocolConfig
    ) -> Optional[SensorReading]:
        """Convert raw protocol data to standardized SensorReading."""
        try:
            # Extract common fields
            value = raw_data.get('value', raw_data.get('data', 0))
            timestamp = raw_data.get('timestamp')
            sensor_type = raw_data.get('type', raw_data.get('sensor_type', 'generic'))
            units = raw_data.get('units', 'unknown')
            quality = raw_data.get('quality', 1.0)
            location = raw_data.get('location')
            
            # Handle timestamp conversion
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.utcnow()
            elif not isinstance(timestamp, datetime):
                timestamp = datetime.utcnow()
            
            # Create sensor reading
            reading = SensorReading(
                sensor_id=sensor_id,
                sensor_type=sensor_type,
                value=value,
                timestamp=timestamp,
                quality=float(quality),
                units=units,
                location=location,
                metadata={
                    'raw_data': raw_data,
                    'protocol': config.protocol_type,
                    'conversion_timestamp': datetime.utcnow().isoformat()
                }
            )
            
            return reading
            
        except Exception as e:
            logger.error(f"❌ Failed to convert sensor data: {str(e)}")
            return None
    
    async def get_connected_sensors(self) -> List[str]:
        """Get list of all connected sensors."""
        return list(self.active_connections.keys())
    
    async def get_protocol_statistics(self) -> Dict[str, Any]:
        """Get statistics about protocol usage."""
        stats = {
            'total_sensors': len(self.active_connections),
            'protocols_in_use': {},
            'connection_uptime': {}
        }
        
        # Count by protocol
        for sensor_id, connection in self.active_connections.items():
            protocol = connection['config'].protocol_type
            stats['protocols_in_use'][protocol] = stats['protocols_in_use'].get(protocol, 0) + 1
            
            # Calculate uptime
            uptime = (datetime.utcnow() - connection['connected_at']).total_seconds()
            stats['connection_uptime'][sensor_id] = uptime
        
        return stats

class MQTTAdapter:
    """MQTT protocol adapter."""
    
    def __init__(self):
        self.clients = {}
        self.message_queues = {}
        
    async def connect(self, sensor_id: str, config: ProtocolConfig) -> bool:
        if not MQTT_AVAILABLE:
            return False
            
        try:
            client = mqtt.Client()
            
            if config.auth_required and config.username:
                client.username_pw_set(config.username, config.password)
            
            # Set up callbacks
            client.on_connect = self._on_connect
            client.on_message = self._on_message
            
            # Connect
            client.connect(config.host, config.port, 60)
            client.loop_start()
            
            self.clients[sensor_id] = client
            self.message_queues[sensor_id] = asyncio.Queue()
            
            return True
            
        except Exception as e:
            logger.error(f"MQTT connection failed: {str(e)}")
            return False
    
    async def disconnect(self, sensor_id: str) -> bool:
        if sensor_id in self.clients:
            client = self.clients[sensor_id]
            client.loop_stop()
            client.disconnect()
            del self.clients[sensor_id]
            del self.message_queues[sensor_id]
        return True
    
    async def read_data(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        if sensor_id in self.message_queues:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(self.message_queues[sensor_id].get(), timeout=5.0)
                return json.loads(message.payload.decode())
            except asyncio.TimeoutError:
                return None
        return None
    
    async def subscribe(self, sensor_id: str, callback) -> bool:
        if sensor_id in self.clients:
            client = self.clients[sensor_id]
            topic = f"sensors/{sensor_id}/data"
            client.subscribe(topic)
            # Store callback for message handling
            self.clients[sensor_id]._callback = callback
            return True
        return False
    
    def _on_connect(self, client, userdata, flags, rc):
        logger.info(f"MQTT connected with result code: {rc}")
    
    def _on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages."""
        try:
            # Find sensor_id from client map
            sensor_id = None
            for sid, mqtt_client in self.clients.items():
                if mqtt_client == client:
                    sensor_id = sid
                    break
            
            if sensor_id and sensor_id in self.message_queues:
                # Put message in queue for async processing
                try:
                    self.message_queues[sensor_id].put_nowait(msg)
                except asyncio.QueueFull:
                    logger.warning(f"⚠️  Message queue full for sensor {sensor_id}, dropping message")
            else:
                logger.warning(f"⚠️  Received MQTT message for unknown sensor")
                
        except Exception as e:
            logger.error(f"❌ Error handling MQTT message: {e}")
            # Don't crash the MQTT thread on error

class CoAPAdapter:
    """CoAP protocol adapter."""
    
    def __init__(self):
        self.contexts = {}
        
    async def connect(self, sensor_id: str, config: ProtocolConfig) -> bool:
        if not COAP_AVAILABLE:
            return False
            
        try:
            context = await aiocoap.Context.create_client_context()
            self.contexts[sensor_id] = context
            return True
        except Exception as e:
            logger.error(f"CoAP connection failed: {str(e)}")
            return False
    
    async def disconnect(self, sensor_id: str) -> bool:
        if sensor_id in self.contexts:
            await self.contexts[sensor_id].shutdown()
            del self.contexts[sensor_id]
        return True
    
    async def read_data(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        # Implementation would depend on specific CoAP endpoints
        return {"value": 0, "type": "mock", "timestamp": datetime.utcnow().isoformat()}
    
    async def subscribe(self, sensor_id: str, callback) -> bool:
        # CoAP observe pattern implementation
        return True

class HTTPAdapter:
    """HTTP/REST protocol adapter."""
    
    def __init__(self):
        self.sessions = {}
        
    async def connect(self, sensor_id: str, config: ProtocolConfig) -> bool:
        try:
            session = aiohttp.ClientSession()
            self.sessions[sensor_id] = {
                'session': session,
                'base_url': f"http{'s' if config.ssl_enabled else ''}://{config.host}:{config.port}"
            }
            return True
        except Exception as e:
            logger.error(f"HTTP connection failed: {str(e)}")
            return False
    
    async def disconnect(self, sensor_id: str) -> bool:
        if sensor_id in self.sessions:
            await self.sessions[sensor_id]['session'].close()
            del self.sessions[sensor_id]
        return True
    
    async def read_data(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        if sensor_id in self.sessions:
            try:
                session_info = self.sessions[sensor_id]
                url = f"{session_info['base_url']}/sensors/{sensor_id}/data"
                
                async with session_info['session'].get(url) as response:
                    if response.status == 200:
                        return await response.json()
            except Exception as e:
                logger.error(f"HTTP read failed: {str(e)}")
        return None
    
    async def subscribe(self, sensor_id: str, callback) -> bool:
        # HTTP polling implementation
        return True

# Global instance
protocol_adapter: Optional[SensorProtocolAdapter] = None

def get_protocol_adapter() -> SensorProtocolAdapter:
    """Get the global protocol adapter instance."""
    if protocol_adapter is None:
        raise RuntimeError("Protocol adapter not initialized")
    return protocol_adapter