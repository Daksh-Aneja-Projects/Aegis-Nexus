"""
Data Ingestor for Aegis Nexus
Normalizes and processes diverse IoT sensor inputs.

This module handles data ingestion from various sources and formats,
providing standardized data streams for the fusion engine.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

from reality_anchor.sensors.protocol_adapters import SensorProtocolAdapter, ProtocolConfig
from reality_anchor.sensors.fusion_engine import SensorReading

logger = logging.getLogger(__name__)

@dataclass
class IngestionPipeline:
    """Configuration for a data ingestion pipeline"""
    pipeline_id: str
    source_type: str  # sensor, database, api, file
    source_config: Dict[str, Any]
    transformation_rules: List[Dict[str, Any]]
    validation_rules: List[Dict[str, Any]]
    target_format: str  # json, csv, parquet, etc.


class CircuitBreakerState:
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: int = 30  # seconds
    half_open_max_calls: int = 2

class CircuitBreaker:
    """
    Circuit Breaker pattern to prevent cascading failures from sensor timeouts.
    """
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failures = 0
        self.last_failure_time = 0
        self.half_open_calls = 0

    def allow_request(self) -> bool:
        if self.state == CircuitBreakerState.OPEN:
            if (datetime.utcnow().timestamp() - self.last_failure_time) > self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                logger.info(f"🔌 Circuit Breaker {self.name}: HALF_OPEN (Testing recovery)")
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return self.half_open_calls < self.config.half_open_max_calls
        return True

    def record_success(self):
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.failures = 0
            logger.info(f"🔌 Circuit Breaker {self.name}: CLOSED (Recovered)")
        elif self.state == CircuitBreakerState.CLOSED:
             self.failures = 0

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = datetime.utcnow().timestamp()
        
        if self.state == CircuitBreakerState.CLOSED and self.failures >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"🔌 Circuit Breaker {self.name}: OPEN (Threshold reached, failing fast)")
        elif self.state == CircuitBreakerState.HALF_OPEN:
             self.state = CircuitBreakerState.OPEN
             logger.warning(f"🔌 Circuit Breaker {self.name}: Re-OPENED (Failed during recovery)")

class DataIngestor:
    """
    Data ingestion system for processing diverse sensor inputs.
    
    Handles normalization, validation, and preprocessing of incoming data
    from various IoT sources before passing to fusion engine.
    """
    
    def __init__(self):
        """Initialize the data ingestor."""
        self.pipelines: Dict[str, IngestionPipeline] = {}
        self.active_ingestions: Dict[str, Any] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {} # Sensor ID -> CircuitBreaker
        
        # Backpressure configuration
        self.max_queue_size = 1000
        self.ingestion_queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.dropped_frames = 0
        self._processing_task: Optional[asyncio.Task] = None
        
        self.processed_data_stats = {
            'total_records': 0,
            'successful_ingestions': 0,
            'failed_ingestions': 0,
            'data_quality_issues': 0
        }
        
    async def initialize(self) -> bool:
        """Initialize the data ingestor."""
        try:
            logger.info("📥 Initializing data ingestor...")
            
            # Start background processing worker
            self._processing_task = asyncio.create_task(self._process_ingestion_queue())
            
            logger.info("✅ Data ingestor initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize data ingestor: {str(e)}")
            return False
    
    async def create_pipeline(
        self, 
        pipeline_id: str, 
        source_type: str, 
        source_config: Dict[str, Any],
        transformation_rules: Optional[List[Dict[str, Any]]] = None,
        validation_rules: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Create a new data ingestion pipeline.
        
        Args:
            pipeline_id: Unique identifier for the pipeline
            source_type: Type of data source
            source_config: Configuration for the source
            transformation_rules: Data transformation rules
            validation_rules: Data validation rules
            
        Returns:
            bool: Success status
        """
        try:
            pipeline = IngestionPipeline(
                pipeline_id=pipeline_id,
                source_type=source_type,
                source_config=source_config,
                transformation_rules=transformation_rules or [],
                validation_rules=validation_rules or [],
                target_format=source_config.get('target_format', 'json')
            )
            
            self.pipelines[pipeline_id] = pipeline
            logger.info(f"✅ Created ingestion pipeline: {pipeline_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create pipeline {pipeline_id}: {str(e)}")
            return False
    
    async def start_ingestion(self, pipeline_id: str, callback: Optional[Callable] = None) -> bool:
        """
        Start data ingestion for a pipeline.
        
        Args:
            pipeline_id: Pipeline to start
            callback: Optional callback for processed data
            
        Returns:
            bool: Success status
        """
        if pipeline_id not in self.pipelines:
            logger.error(f"❌ Pipeline {pipeline_id} not found")
            return False
        
        try:
            pipeline = self.pipelines[pipeline_id]
            
            # Start appropriate ingestion based on source type
            if pipeline.source_type == 'sensor':
                success = await self._start_sensor_ingestion(pipeline, callback)
            elif pipeline.source_type == 'database':
                success = await self._start_database_ingestion(pipeline, callback)
            elif pipeline.source_type == 'api':
                success = await self._start_api_ingestion(pipeline, callback)
            elif pipeline.source_type == 'file':
                success = await self._start_file_ingestion(pipeline, callback)
            else:
                logger.error(f"❌ Unsupported source type: {pipeline.source_type}")
                return False
            
            if success:
                self.active_ingestions[pipeline_id] = {
                    'pipeline': pipeline,
                    'started_at': datetime.utcnow(),
                    'records_processed': 0,
                    'callback': callback
                }
                logger.info(f"✅ Started ingestion for pipeline: {pipeline_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to start ingestion for {pipeline_id}: {str(e)}")
            return False

    async def _process_ingestion_queue(self):
        """
        Background worker to process data from the ingestion queue.
        Decouples ingestion from processing to handle bursts.
        """
        logger.info("🔄 Starting ingestion queue processing worker")
        while True:
            try:
                # Get item from queue
                item = await self.ingestion_queue.get()
                
                reading, callback = item
                
                try:
                    # Process the item
                    await self._process_sensor_data(reading, callback)
                except Exception as e:
                    logger.error(f"Error processing queued item: {str(e)}")
                finally:
                    self.ingestion_queue.task_done()
                    
            except asyncio.CancelledError:
                logger.info("🛑 Ingestion worker cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in ingestion worker: {str(e)}")
                await asyncio.sleep(1)  # Prevent tight loop on error
    
    async def _start_sensor_ingestion(self, pipeline: IngestionPipeline, callback: Optional[Callable]) -> bool:
        """Start ingestion from sensor sources."""
        try:
            # Connect to sensors using protocol adapter
            protocol_adapter = SensorProtocolAdapter()
            await protocol_adapter.initialize()
            
            sensor_configs = pipeline.source_config.get('sensors', [])
            
            for sensor_config in sensor_configs:
                sensor_id = sensor_config['sensor_id']
                protocol_config = ProtocolConfig(**sensor_config['protocol_config'])
                
                # Circuit Breaker Check
                if sensor_id not in self.circuit_breakers:
                    self.circuit_breakers[sensor_id] = CircuitBreaker(sensor_id)
                
                cb = self.circuit_breakers[sensor_id]
                
                if not cb.allow_request():
                    logger.warning(f"⛔ Circuit Breaker blocking connection to sensor {sensor_id}")
                    continue

                try:
                    # Connect sensor
                    await protocol_adapter.connect_sensor(sensor_id, protocol_config)
                    cb.record_success()
                except Exception as e:
                    cb.record_failure()
                    logger.error(f"❌ Failed to connect sensor {sensor_id}: {e}")
                    continue

                # Subscribe to data updates
                if callback:
                    await protocol_adapter.subscribe_to_sensor(
                        sensor_id, 
                        lambda reading: self._handle_sensor_data_safe(reading, callback, sensor_id)
                    )
            
            return True
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Sensor ingestion failed: {str(e)}")
            return False

    async def _handle_sensor_data_safe(self, reading: SensorReading, callback: Callable, sensor_id: str):
        """
        Wrapper to update circuit breaker on data flow.
        """
        # If we are receiving data, the connection is healthy
        if sensor_id in self.circuit_breakers:
            self.circuit_breakers[sensor_id].record_success()
            
        await self._handle_sensor_data(reading, callback)

    
    async def _handle_sensor_data(self, reading: SensorReading, callback: Callable):
        """
        Handle incoming sensor data with backpressure.
        If queue is full, drop the frame (load shedding).
        """
        try:
            # Try to add to queue without blocking
            self.ingestion_queue.put_nowait((reading, callback))
        except asyncio.QueueFull:
            # Load shedding: Drop the frame
            self.dropped_frames += 1
            if self.dropped_frames % 100 == 0:
                logger.warning(f"⚠️  Backpressure active: Dropped {self.dropped_frames} frames so far (Queue size: {self.max_queue_size})")
    
    async def _process_sensor_data(self, reading: SensorReading, callback: Callable):
        """Process sensor data (formerly _handle_sensor_data logic)."""
        try:
            # Apply transformations
            transformed_data = await self._apply_transformations(
                reading, 
                self.active_ingestions.get('current_pipeline', {}).get('pipeline', {}).transformation_rules or []
            )
            
            # Validate data
            is_valid = await self._validate_data(transformed_data)
            
            if is_valid:
                # Apply callback
                if callback:
                    await callback(transformed_data)
                
                # Update statistics
                self.processed_data_stats['successful_ingestions'] += 1
            else:
                self.processed_data_stats['data_quality_issues'] += 1
                
        except Exception as e:
            logger.error(f"❌ Error handling sensor data: {str(e)}")
            self.processed_data_stats['failed_ingestions'] += 1
    
    async def _start_database_ingestion(self, pipeline: IngestionPipeline, callback: Optional[Callable]) -> bool:
        """Start ingestion from database sources."""
        # Implementation for database ingestion
        logger.info(f"Starting database ingestion for {pipeline.pipeline_id}")
        return True
    
    async def _start_api_ingestion(self, pipeline: IngestionPipeline, callback: Optional[Callable]) -> bool:
        """Start ingestion from API sources."""
        # Implementation for API ingestion
        logger.info(f"Starting API ingestion for {pipeline.pipeline_id}")
        return True
    
    async def _start_file_ingestion(self, pipeline: IngestionPipeline, callback: Optional[Callable]) -> bool:
        """Start ingestion from file sources."""
        # Implementation for file ingestion
        logger.info(f"Starting file ingestion for {pipeline.pipeline_id}")
        return True
    
    async def _apply_transformations(self, data: Any, rules: List[Dict[str, Any]]) -> Any:
        """Apply transformation rules to data."""
        transformed = data
        
        for rule in rules:
            transform_type = rule.get('type')
            if transform_type == 'normalize':
                transformed = await self._normalize_data(transformed, rule)
            elif transform_type == 'filter':
                transformed = await self._filter_data(transformed, rule)
            elif transform_type == 'aggregate':
                transformed = await self._aggregate_data(transformed, rule)
            elif transform_type == 'enrich':
                transformed = await self._enrich_data(transformed, rule)
        
        return transformed
    
    async def _normalize_data(self, data: Any, rule: Dict[str, Any]) -> Any:
        """Normalize data according to rule."""
        # Simple normalization example
        if isinstance(data, dict) and 'value' in data:
            min_val = rule.get('min', 0)
            max_val = rule.get('max', 100)
            current_val = float(data['value'])
            
            # Normalize to 0-1 range
            normalized = (current_val - min_val) / (max_val - min_val) if max_val != min_val else 0
            data['normalized_value'] = max(0, min(1, normalized))
        
        return data
    
    async def _filter_data(self, data: Any, rule: Dict[str, Any]) -> Any:
        """Filter data according to rule."""
        # Simple filtering example
        if isinstance(data, dict):
            conditions = rule.get('conditions', [])
            for condition in conditions:
                field = condition.get('field')
                operator = condition.get('operator')
                value = condition.get('value')
                
                if field in data:
                    data_value = data[field]
                    if operator == 'gt' and data_value <= value:
                        return None  # Filter out
                    elif operator == 'lt' and data_value >= value:
                        return None  # Filter out
        
        return data
    
    async def _aggregate_data(self, data: Any, rule: Dict[str, Any]) -> Any:
        """Aggregate data according to rule."""
        # Simple aggregation example
        if isinstance(data, list) and len(data) > 0:
            agg_type = rule.get('aggregation_type', 'average')
            field = rule.get('field', 'value')
            
            values = [item.get(field, 0) for item in data if isinstance(item, dict) and field in item]
            
            if values:
                if agg_type == 'average':
                    data = {'aggregated_value': sum(values) / len(values)}
                elif agg_type == 'sum':
                    data = {'aggregated_value': sum(values)}
                elif agg_type == 'max':
                    data = {'aggregated_value': max(values)}
                elif agg_type == 'min':
                    data = {'aggregated_value': min(values)}
        
        return data
    
    async def _enrich_data(self, data: Any, rule: Dict[str, Any]) -> Any:
        """Enrich data with additional information."""
        # Simple enrichment example
        if isinstance(data, dict):
            enrichments = rule.get('enrichments', {})
            for key, value in enrichments.items():
                data[key] = value
        
        return data
    
    async def _validate_data(self, data: Any) -> bool:
        """Validate data quality."""
        try:
            if data is None:
                return False
            
            # Basic validation checks
            if isinstance(data, dict):
                # Check required fields
                required_fields = ['value', 'timestamp']
                for field in required_fields:
                    if field not in data:
                        return False
                
                # Check data types
                if not isinstance(data['value'], (int, float, str)):
                    return False
                
                # Check timestamp validity
                if 'timestamp' in data:
                    timestamp = data['timestamp']
                    if isinstance(timestamp, str):
                        try:
                            datetime.fromisoformat(timestamp)
                        except ValueError:
                            return False
                    elif not isinstance(timestamp, datetime):
                        return False
            
            return True
            
        except Exception:
            return False
    
    async def stop_ingestion(self, pipeline_id: str) -> bool:
        """
        Stop data ingestion for a pipeline.
        
        Args:
            pipeline_id: Pipeline to stop
            
        Returns:
            bool: Success status
        """
        if pipeline_id in self.active_ingestions:
            try:
                # Stop the ingestion process
                del self.active_ingestions[pipeline_id]
                logger.info(f"✅ Stopped ingestion for pipeline: {pipeline_id}")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to stop ingestion for {pipeline_id}: {str(e)}")
        
        return False
    
    async def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status information for a pipeline.
        
        Args:
            pipeline_id: Pipeline to check
            
        Returns:
            Dict with status information or None if not found
        """
        if pipeline_id in self.active_ingestions:
            ingestion = self.active_ingestions[pipeline_id]
            return {
                'pipeline_id': pipeline_id,
                'status': 'active',
                'started_at': ingestion['started_at'].isoformat(),
                'records_processed': ingestion['records_processed'],
                'uptime_seconds': (datetime.utcnow() - ingestion['started_at']).total_seconds()
            }
        elif pipeline_id in self.pipelines:
            return {
                'pipeline_id': pipeline_id,
                'status': 'inactive',
                'created_at': datetime.utcnow().isoformat()
            }
        
        return None
    
    def get_ingestion_statistics(self) -> Dict[str, Any]:
        """Get overall ingestion statistics."""
        return {
            **self.processed_data_stats,
            'dropped_frames': self.dropped_frames,
            'queue_size': self.ingestion_queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'active_pipelines': len(self.active_ingestions),
            'configured_pipelines': len(self.pipelines)
        }

# Global instance
data_ingestor: Optional[DataIngestor] = None

async def initialize_data_ingestor() -> bool:
    """
    Initialize the global data ingestor instance.
    
    Returns:
        bool: Success status
    """
    global data_ingestor
    
    try:
        data_ingestor = DataIngestor()
        return await data_ingestor.initialize()
    except Exception as e:
        logger.error(f"❌ Failed to initialize data ingestor: {str(e)}")
        return False

def get_data_ingestor() -> DataIngestor:
    """Get the global data ingestor instance."""
    if data_ingestor is None:
        raise RuntimeError("Data ingestor not initialized")
    return data_ingestor