"""
Activity Recognizer for Aegis Nexus
Detects human and system activity patterns from sensor data.

This module implements machine learning models for activity recognition
and behavioral pattern detection.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from reality_anchor.sensors.fusion_engine import SensorReading

logger = logging.getLogger(__name__)

@dataclass
class ActivityPattern:
    """Represents a detected activity pattern"""
    pattern_id: str
    activity_type: str
    confidence: float
    start_time: datetime
    end_time: Optional[datetime]
    sensor_evidence: List[str]
    features_used: List[str]
    metadata: Dict[str, Any]

@dataclass
class BehavioralAnomaly:
    """Represents a detected behavioral anomaly"""
    anomaly_id: str
    anomaly_type: str
    severity: str  # low, medium, high, critical
    timestamp: datetime
    description: str
    affected_activities: List[str]
    confidence: float

class ActivityRecognizer:
    """
    Activity recognition system using machine learning.
    
    Detects human activities, system behaviors, and anomalous patterns
    from multimodal sensor data streams.
    """
    
    def __init__(self):
        """Initialize the activity recognizer."""
        self.activity_models = {}
        self.scalers = {}
        self.pattern_history: List[ActivityPattern] = []
        self.anomaly_history: List[BehavioralAnomaly] = []
        self.feature_extractors = {
            'time_based': self._extract_time_features,
            'frequency_based': self._extract_frequency_features,
            'statistical': self._extract_statistical_features
        }
        self.max_history_size = 1000
        
    async def initialize(self) -> bool:
        """Initialize the activity recognition system."""
        try:
            logger.info("🏃 Initializing activity recognition system...")
            
            # Initialize default activity models
            await self._initialize_activity_models()
            
            logger.info("✅ Activity recognition system initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize activity recognition: {str(e)}")
            return False
    
    async def _initialize_activity_models(self):
        """Initialize machine learning models for activity recognition."""
        # Human activities model
        self.activity_models['human_activities'] = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        # System activities model
        self.activity_models['system_activities'] = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        # Anomaly detection model
        self.activity_models['anomaly_detection'] = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=15
        )
        
        # Initialize scalers
        for model_name in self.activity_models.keys():
            self.scalers[model_name] = StandardScaler()
    
    async def recognize_activity(
        self, 
        sensor_readings: List[SensorReading],
        activity_context: Optional[Dict[str, Any]] = None
    ) -> List[ActivityPattern]:
        """
        Recognize activities from sensor readings.
        
        Args:
            sensor_readings: List of sensor readings
            activity_context: Additional context information
            
        Returns:
            List of detected activity patterns
        """
        logger.debug(f"🏃 Recognizing activities from {len(sensor_readings)} sensor readings")
        
        try:
            # Extract features from sensor data
            features = await self._extract_features(sensor_readings)
            
            if not features:
                return []
            
            # Recognize human activities
            human_activities = await self._recognize_human_activities(features, sensor_readings)
            
            # Recognize system activities
            system_activities = await self._recognize_system_activities(features, sensor_readings)
            
            # Combine all activities
            all_activities = human_activities + system_activities
            
            # Add to history
            self.pattern_history.extend(all_activities)
            
            # Maintain history size
            if len(self.pattern_history) > self.max_history_size:
                self.pattern_history = self.pattern_history[-self.max_history_size:]
            
            logger.debug(f"🏃 Detected {len(all_activities)} activity patterns")
            return all_activities
            
        except Exception as e:
            logger.error(f"❌ Activity recognition failed: {str(e)}")
            return []
    
    async def detect_anomalies(
        self, 
        sensor_readings: List[SensorReading],
        baseline_activities: Optional[List[ActivityPattern]] = None
    ) -> List[BehavioralAnomaly]:
        """
        Detect behavioral anomalies in sensor data.
        
        Args:
            sensor_readings: Current sensor readings
            baseline_activities: Baseline activity patterns for comparison
            
        Returns:
            List of detected anomalies
        """
        logger.debug("🔍 Detecting behavioral anomalies")
        
        try:
            # Extract features for anomaly detection
            features = await self._extract_features(sensor_readings)
            
            if not features:
                return []
            
            # Detect statistical anomalies
            statistical_anomalies = await self._detect_statistical_anomalies(features)
            
            # Detect pattern-based anomalies
            pattern_anomalies = await self._detect_pattern_anomalies(
                sensor_readings, baseline_activities
            )
            
            # Detect contextual anomalies
            contextual_anomalies = await self._detect_contextual_anomalies(
                sensor_readings, activity_context={}
            )
            
            # Combine all anomalies
            all_anomalies = statistical_anomalies + pattern_anomalies + contextual_anomalies
            
            # Add to history
            self.anomaly_history.extend(all_anomalies)
            
            # Maintain history size
            if len(self.anomaly_history) > self.max_history_size:
                self.anomaly_history = self.anomaly_history[-self.max_history_size:]
            
            logger.debug(f"🔍 Detected {len(all_anomalies)} anomalies")
            return all_anomalies
            
        except Exception as e:
            logger.error(f"❌ Anomaly detection failed: {str(e)}")
            return []
    
    async def _extract_features(self, sensor_readings: List[SensorReading]) -> Optional[np.ndarray]:
        """Extract features from sensor readings for ML models."""
        if not sensor_readings:
            return None
        
        try:
            # Convert readings to feature matrix
            feature_vectors = []
            
            for reading in sensor_readings:
                # Extract different types of features
                time_features = await self._extract_time_features(reading)
                frequency_features = await self._extract_frequency_features(reading)
                statistical_features = await self._extract_statistical_features(reading)
                
                # Combine all features
                combined_features = time_features + frequency_features + statistical_features
                feature_vectors.append(combined_features)
            
            # Convert to numpy array
            if feature_vectors:
                feature_matrix = np.array(feature_vectors)
                
                # Normalize features
                scaler = self.scalers.get('human_activities')
                if scaler:
                    feature_matrix = scaler.fit_transform(feature_matrix)
                
                return feature_matrix
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {str(e)}")
        
        return None
    
    async def _extract_time_features(self, reading: SensorReading) -> List[float]:
        """Extract time-based features from sensor reading."""
        features = []
        
        try:
            # Time of day features
            hour = reading.timestamp.hour
            minute = reading.timestamp.minute
            second = reading.timestamp.second
            
            # Cyclical encoding of time
            features.append(np.sin(2 * np.pi * hour / 24))
            features.append(np.cos(2 * np.pi * hour / 24))
            features.append(np.sin(2 * np.pi * minute / 60))
            features.append(np.cos(2 * np.pi * minute / 60))
            
            # Time since epoch (normalized)
            epoch_time = reading.timestamp.timestamp()
            features.append(epoch_time / (24 * 3600 * 365))  # Years since epoch
            
        except Exception:
            # Return zeros if feature extraction fails
            features = [0.0] * 5
        
        return features
    
    async def _extract_frequency_features(self, reading: SensorReading) -> List[float]:
        """Extract frequency-based features from sensor reading."""
        features = []
        
        try:
            # Convert value to numeric for frequency analysis
            if isinstance(reading.value, (int, float)):
                value = float(reading.value)
            elif isinstance(reading.value, str):
                value = float(hash(reading.value) % 1000)
            else:
                value = 0.0
            
            # Simple frequency features
            features.append(value)  # Raw value
            features.append(abs(value))  # Absolute value
            features.append(value ** 2)  # Squared value
            features.append(np.sqrt(abs(value)) if value != 0 else 0)  # Square root
            
        except Exception:
            features = [0.0] * 4
        
        return features
    
    async def _extract_statistical_features(self, reading: SensorReading) -> List[float]:
        """Extract statistical features from sensor reading."""
        features = []
        
        try:
            # Quality-based features
            features.append(reading.quality)
            features.append(1.0 - reading.quality)  # Inverse quality
            
            # Sensor type encoding (simplified)
            type_hash = hash(reading.sensor_type) % 100
            features.append(type_hash / 100.0)
            
            # Location encoding (if available)
            if reading.location:
                loc_hash = hash(reading.location) % 50
                features.append(loc_hash / 50.0)
            else:
                features.append(0.0)
            
        except Exception:
            features = [0.0] * 4
        
        return features
    
    async def _recognize_human_activities(
        self, 
        features: np.ndarray, 
        sensor_readings: List[SensorReading]
    ) -> List[ActivityPattern]:
        """Recognize human activities from features."""
        activities = []
        
        try:
            # Simple rule-based recognition for demonstration
            # In practice, this would use trained ML models
            for i, reading in enumerate(sensor_readings):
                if i < len(features):
                    feature_vector = features[i]
                    
                    # Simple activity detection based on sensor type and values
                    activity_type = await self._classify_human_activity(reading, feature_vector)
                    
                    if activity_type:
                        activity = ActivityPattern(
                            pattern_id=f"human_{hash(str(reading)) % 1000000}",
                            activity_type=activity_type,
                            confidence=reading.quality,
                            start_time=reading.timestamp,
                            end_time=None,
                            sensor_evidence=[reading.sensor_id],
                            features_used=['time', 'frequency', 'statistical'],
                            metadata={
                                'sensor_type': reading.sensor_type,
                                'value': str(reading.value),
                                'recognition_method': 'rule_based'
                            }
                        )
                        activities.append(activity)
        
        except Exception as e:
            logger.error(f"❌ Human activity recognition failed: {str(e)}")
        
        return activities
    
    async def _classify_human_activity(self, reading: SensorReading, features: np.ndarray) -> Optional[str]:
        """Classify specific human activity type."""
        # Simple rule-based classification
        sensor_type = reading.sensor_type.lower()
        value = reading.value
        
        # Motion/acceleration sensors
        if 'motion' in sensor_type or 'accel' in sensor_type:
            if isinstance(value, (int, float)) and abs(value) > 0.5:
                return 'movement_detected'
            else:
                return 'stationary'
        
        # Audio sensors
        elif 'audio' in sensor_type or 'sound' in sensor_type:
            if isinstance(value, (int, float)) and value > 0.3:
                return 'speech_detected'
            elif isinstance(value, str) and len(value) > 10:
                return 'conversation'
        
        # Temperature sensors
        elif 'temp' in sensor_type or 'temperature' in sensor_type:
            if isinstance(value, (int, float)):
                if value > 25:  # Celsius
                    return 'warm_environment'
                elif value < 18:
                    return 'cool_environment'
        
        # Light sensors
        elif 'light' in sensor_type or 'lux' in sensor_type:
            if isinstance(value, (int, float)):
                if value > 500:
                    return 'well_lit'
                elif value < 100:
                    return 'dim_lighting'
        
        return None
    
    async def _recognize_system_activities(
        self, 
        features: np.ndarray, 
        sensor_readings: List[SensorReading]
    ) -> List[ActivityPattern]:
        """Recognize system activities from features."""
        activities = []
        
        try:
            for i, reading in enumerate(sensor_readings):
                if i < len(features):
                    # Simple system activity detection
                    activity_type = await self._classify_system_activity(reading)
                    
                    if activity_type:
                        activity = ActivityPattern(
                            pattern_id=f"system_{hash(str(reading)) % 1000000}",
                            activity_type=activity_type,
                            confidence=reading.quality,
                            start_time=reading.timestamp,
                            end_time=None,
                            sensor_evidence=[reading.sensor_id],
                            features_used=['statistical'],
                            metadata={
                                'sensor_type': reading.sensor_type,
                                'value': str(reading.value),
                                'system_component': reading.metadata.get('component', 'unknown')
                            }
                        )
                        activities.append(activity)
        
        except Exception as e:
            logger.error(f"❌ System activity recognition failed: {str(e)}")
        
        return activities
    
    async def _classify_system_activity(self, reading: SensorReading) -> Optional[str]:
        """Classify specific system activity type."""
        sensor_type = reading.sensor_type.lower()
        value = reading.value
        
        # CPU/Memory sensors
        if 'cpu' in sensor_type:
            if isinstance(value, (int, float)) and value > 80:
                return 'high_cpu_usage'
            elif isinstance(value, (int, float)) and value < 20:
                return 'low_cpu_usage'
        
        elif 'memory' in sensor_type:
            if isinstance(value, (int, float)) and value > 85:
                return 'high_memory_usage'
        
        # Network sensors
        elif 'network' in sensor_type:
            if isinstance(value, (int, float)) and value > 1000:  # KB/s
                return 'high_network_traffic'
        
        # Disk sensors
        elif 'disk' in sensor_type:
            if isinstance(value, (int, float)) and value > 90:
                return 'disk_near_full'
        
        return None
    
    async def _detect_statistical_anomalies(self, features: np.ndarray) -> List[BehavioralAnomaly]:
        """Detect anomalies based on statistical analysis."""
        anomalies = []
        
        try:
            if len(features) == 0:
                return anomalies
            
            # Calculate statistical measures
            feature_means = np.mean(features, axis=0)
            feature_stds = np.std(features, axis=0)
            
            # Detect outliers using z-score
            for i, feature_vector in enumerate(features):
                z_scores = np.abs((feature_vector - feature_means) / (feature_stds + 1e-8))
                
                # Anomaly if any feature has high z-score
                if np.any(z_scores > 3.0):  # 3 sigma threshold
                    anomaly = BehavioralAnomaly(
                        anomaly_id=f"stat_{i}_{int(datetime.utcnow().timestamp())}",
                        anomaly_type="statistical_outlier",
                        severity="medium",
                        timestamp=datetime.utcnow(),
                        description="Statistical outlier detected in sensor data",
                        affected_activities=[],
                        confidence=min(1.0, np.max(z_scores) / 5.0)
                    )
                    anomalies.append(anomaly)
        
        except Exception as e:
            logger.error(f"❌ Statistical anomaly detection failed: {str(e)}")
        
        return anomalies
    
    async def _detect_pattern_anomalies(
        self, 
        sensor_readings: List[SensorReading],
        baseline_activities: Optional[List[ActivityPattern]]
    ) -> List[BehavioralAnomaly]:
        """Detect anomalies based on pattern deviation."""
        anomalies = []
        
        # Simple pattern-based anomaly detection
        if len(sensor_readings) > 1:
            # Check for sudden changes in sensor behavior
            for i in range(1, len(sensor_readings)):
                current = sensor_readings[i]
                previous = sensor_readings[i-1]
                
                # Large value jumps
                if isinstance(current.value, (int, float)) and isinstance(previous.value, (int, float)):
                    change = abs(current.value - previous.value)
                    if change > 10.0:  # Threshold
                        anomaly = BehavioralAnomaly(
                            anomaly_id=f"pattern_jump_{i}",
                            anomaly_type="value_spike",
                            severity="low",
                            timestamp=current.timestamp,
                            description=f"Sudden value change: {previous.value} -> {current.value}",
                            affected_activities=[],
                            confidence=min(1.0, change / 20.0)
                        )
                        anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_contextual_anomalies(
        self, 
        sensor_readings: List[SensorReading],
        activity_context: Dict[str, Any]
    ) -> List[BehavioralAnomaly]:
        """Detect anomalies based on contextual information."""
        anomalies = []
        
        # Time-based anomalies
        for reading in sensor_readings:
            hour = reading.timestamp.hour
            
            # Night time activity detection
            if hour >= 22 or hour <= 6:  # Night hours
                # Certain activities shouldn't happen at night
                if reading.sensor_type.lower() in ['door', 'motion', 'camera']:
                    anomaly = BehavioralAnomaly(
                        anomaly_id=f"context_night_{hash(str(reading)) % 1000000}",
                        anomaly_type="night_activity",
                        severity="medium",
                        timestamp=reading.timestamp,
                        description="Unexpected activity during night hours",
                        affected_activities=[],
                        confidence=0.7
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    async def get_activity_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent activities."""
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        recent_patterns = [
            pattern for pattern in self.pattern_history 
            if pattern.start_time >= cutoff_time
        ]
        
        recent_anomalies = [
            anomaly for anomaly in self.anomaly_history 
            if anomaly.timestamp >= cutoff_time
        ]
        
        # Group by activity type
        activity_counts = {}
        for pattern in recent_patterns:
            activity_type = pattern.activity_type
            activity_counts[activity_type] = activity_counts.get(activity_type, 0) + 1
        
        # Group anomalies by severity
        anomaly_severity_counts = {}
        for anomaly in recent_anomalies:
            severity = anomaly.severity
            anomaly_severity_counts[severity] = anomaly_severity_counts.get(severity, 0) + 1
        
        return {
            'time_window_hours': time_window_hours,
            'total_activities': len(recent_patterns),
            'activity_distribution': activity_counts,
            'total_anomalies': len(recent_anomalies),
            'anomaly_severity_distribution': anomaly_severity_counts,
            'most_common_activity': max(activity_counts.items(), key=lambda x: x[1])[0] if activity_counts else None
        }

# Global instance
activity_recognizer: Optional[ActivityRecognizer] = None

async def initialize_activity_recognizer() -> bool:
    """
    Initialize the global activity recognizer instance.
    
    Returns:
        bool: Success status
    """
    global activity_recognizer
    
    try:
        activity_recognizer = ActivityRecognizer()
        return await activity_recognizer.initialize()
    except Exception as e:
        logger.error(f"❌ Failed to initialize activity recognizer: {str(e)}")
        return False

def get_activity_recognizer() -> ActivityRecognizer:
    """Get the global activity recognizer instance."""
    if activity_recognizer is None:
        raise RuntimeError("Activity recognizer not initialized")
    return activity_recognizer