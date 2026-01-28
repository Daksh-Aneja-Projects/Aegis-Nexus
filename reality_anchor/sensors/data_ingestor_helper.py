"""
Data Ingestor Helper - Aegis Nexus Reality Anchor
Provides circuit breaker integration for sensor data handling.
"""

import logging
from typing import Callable, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """Represents a single sensor reading."""
    sensor_id: str
    value: float
    timestamp: float
    metadata: Dict[str, Any] = None


class CircuitBreaker:
    """Simple circuit breaker for sensor connections."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = "closed"  # closed, open, half-open
        self.last_failure_time = None
    
    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            import time
            self.last_failure_time = time.time()
    
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        if self.state == "open":
            import time
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "half-open"
                return False
            return True
        return False


class DataIngestorHelper:
    """
    Helper class for data ingestion with circuit breaker support.
    Provides safe sensor data handling with automatic recovery.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def get_or_create_breaker(self, sensor_id: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a sensor."""
        if sensor_id not in self.circuit_breakers:
            self.circuit_breakers[sensor_id] = CircuitBreaker()
        return self.circuit_breakers[sensor_id]
    
    async def _handle_sensor_data(self, reading: SensorReading, callback: Callable):
        """
        Internal method to process sensor data.
        
        Args:
            reading: The sensor reading to process
            callback: Callback function to invoke with processed data
        """
        # Process the reading
        processed_data = {
            "sensor_id": reading.sensor_id,
            "value": reading.value,
            "timestamp": reading.timestamp,
            "processed": True
        }
        
        # Invoke callback with processed data
        if callback:
            await callback(processed_data)
        
        return processed_data
    
    async def _handle_sensor_data_safe(self, reading: SensorReading, callback: Callable, sensor_id: str):
        """
        Wrapper for sensor data handling that updates circuit breaker status.
        
        Args:
            reading: The sensor reading to process
            callback: Callback function for processed data
            sensor_id: ID of the sensor for circuit breaker tracking
        """
        try:
            # Check circuit breaker
            breaker = self.get_or_create_breaker(sensor_id)
            if breaker.is_open():
                logger.warning(f"⚡ Circuit breaker OPEN for sensor {sensor_id}")
                return None
            
            # Process data
            result = await self._handle_sensor_data(reading, callback)
            
            # Record success
            breaker.record_success()
            
            return result
            
        except Exception as e:
            # Record failure
            breaker = self.get_or_create_breaker(sensor_id)
            breaker.record_failure()
            logger.error(f"❌ Sensor data handling failed for {sensor_id}: {e}")
            raise


# Global instance
_helper_instance = None


def get_data_ingestor_helper() -> DataIngestorHelper:
    """Get the global DataIngestorHelper instance."""
    global _helper_instance
    if _helper_instance is None:
        _helper_instance = DataIngestorHelper()
    return _helper_instance
