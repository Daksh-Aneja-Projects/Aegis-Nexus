"""
Common Types for Reality Anchor System
Data structures and types used throughout the reality anchoring components.

This module defines the common data structures for sensors, simulations,
and reality verification used by the Aegis Nexus system.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np

class SensorType(Enum):
    """Types of sensors supported by the system"""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    VIBRATION = "vibration"
    ACCELERATION = "acceleration"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
    GPS = "gps"
    CAMERA = "camera"
    MICROPHONE = "microphone"
    PROXIMITY = "proximity"
    LIGHT = "light"
    MOTION = "motion"
    POWER = "power"
    CURRENT = "current"
    VOLTAGE = "voltage"
    FLOW_RATE = "flow_rate"
    LEVEL = "level"
    WEIGHT = "weight"
    FORCE = "force"
    TORQUE = "torque"
    DISTANCE = "distance"
    ANGLE = "angle"
    SPEED = "speed"
    RPM = "rpm"

class SensorProtocol(Enum):
    """Communication protocols for sensors"""
    MQTT = "mqtt"
    COAP = "coap"
    HTTP = "http"
    WEBSOCKET = "websocket"
    MODBUS = "modbus"
    CAN = "can"
    SERIAL = "serial"
    I2C = "i2c"
    SPI = "spi"
    BLE = "ble"
    ZIGBEE = "zigbee"
    LORA = "lora"

class DataQuality(Enum):
    """Data quality classifications"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"

class ActivityType(Enum):
    """Types of activities that can be recognized"""
    HUMAN_WALKING = "human_walking"
    HUMAN_RUNNING = "human_running"
    HUMAN_SITTING = "human_sitting"
    HUMAN_STANDING = "human_standing"
    MACHINE_OPERATION = "machine_operation"
    MACHINE_IDLE = "machine_idle"
    MACHINE_MAINTENANCE = "machine_maintenance"
    VEHICLE_MOVEMENT = "vehicle_movement"
    DOOR_OPENING = "door_opening"
    DOOR_CLOSING = "door_closing"
    WINDOW_OPENING = "window_opening"
    WINDOW_CLOSING = "window_closing"
    LIGHT_ON = "light_on"
    LIGHT_OFF = "light_off"
    TEMPERATURE_CHANGE = "temperature_change"
    PRESSURE_CHANGE = "pressure_change"
    VIBRATION_START = "vibration_start"
    VIBRATION_STOP = "vibration_stop"

@dataclass
class SensorReading:
    """A single sensor reading with metadata"""
    sensor_id: str
    sensor_type: SensorType
    value: Union[float, int, str, bool, List[float]]
    unit: str
    timestamp: datetime
    quality: DataQuality = DataQuality.GOOD
    confidence: float = 1.0  # 0.0 to 1.0
    location: Optional[Tuple[float, float, float]] = None  # (x, y, z) coordinates
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class SensorConfig:
    """Configuration for a sensor"""
    sensor_id: str
    sensor_type: SensorType
    protocol: SensorProtocol
    endpoint: str
    sampling_rate: float  # Hz
    timeout: float  # seconds
    retry_count: int = 3
    calibration_offset: float = 0.0
    calibration_scale: float = 1.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SensorStream:
    """A stream of sensor readings"""
    stream_id: str
    sensor_id: str
    readings: List[SensorReading] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    buffer_size: int = 1000

    def add_reading(self, reading: SensorReading):
        """Add a reading to the stream."""
        self.readings.append(reading)

        # Maintain buffer size
        if len(self.readings) > self.buffer_size:
            self.readings.pop(0)

        # Update time bounds
        if self.start_time is None or reading.timestamp < self.start_time:
            self.start_time = reading.timestamp
        if self.end_time is None or reading.timestamp > self.end_time:
            self.end_time = reading.timestamp

    def get_readings_in_range(self, start: datetime, end: datetime) -> List[SensorReading]:
        """Get readings within a time range."""
        return [
            reading for reading in self.readings
            if start <= reading.timestamp <= end
        ]

@dataclass
class FusedState:
    """Fused state from multiple sensors representing physical reality"""
    timestamp: datetime
    position: Optional[Tuple[float, float, float]] = None  # (x, y, z)
    orientation: Optional[Tuple[float, float, float]] = None  # (roll, pitch, yaw)
    velocity: Optional[Tuple[float, float, float]] = None  # (vx, vy, vz)
    acceleration: Optional[Tuple[float, float, float]] = None  # (ax, ay, az)
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    activity: Optional[ActivityType] = None
    confidence: float = 0.0  # Overall confidence in fused state
    sensor_contributions: Dict[str, float] = field(default_factory=dict)  # Sensor ID -> contribution weight
    anomalies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KalmanFilterState:
    """State of a Kalman filter for sensor fusion"""
    state_vector: np.ndarray
    covariance_matrix: np.ndarray
    process_noise: np.ndarray
    measurement_noise: np.ndarray
    state_transition_matrix: np.ndarray
    measurement_matrix: np.ndarray
    last_update: datetime

@dataclass
class SensorFusionConfig:
    """Configuration for sensor fusion engine"""
    fusion_method: str = "kalman"  # kalman, particle, complementary
    max_sensors: int = 50
    fusion_rate: float = 10.0  # Hz
    outlier_threshold: float = 3.0  # Standard deviations
    temporal_alignment_window: float = 0.1  # seconds
    quality_weighting: bool = True
    anomaly_detection: bool = True

@dataclass
class ActivityRecognitionResult:
    """Result of activity recognition"""
    activity_type: ActivityType
    confidence: float
    start_time: datetime
    end_time: Optional[datetime] = None
    sensor_contributions: Dict[str, float] = field(default_factory=dict)
    features_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PrecisionValidationResult:
    """Result of precision validation"""
    data_stream_id: str
    timestamp: datetime
    health_score: float  # 0.0 to 1.0
    residuals: List[float]
    outliers_detected: int
    calibration_recommended: bool
    quality_assessment: DataQuality
    recommendations: List[str] = field(default_factory=list)

@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection"""
    sensor_id: str
    timestamp: datetime
    is_anomaly: bool
    anomaly_score: float  # 0.0 to 1.0 (higher = more anomalous)
    anomaly_type: str  # point, contextual, collective
    contributing_features: List[str] = field(default_factory=list)
    explanation: str = ""
    recommended_actions: List[str] = field(default_factory=list)

@dataclass
class DigitalTwinState:
    """State of a digital twin simulation"""
    twin_id: str
    timestamp: datetime
    physical_state: Dict[str, Any]
    simulated_state: Dict[str, Any]
    control_inputs: Dict[str, Any] = field(default_factory=dict)
    disturbances: Dict[str, Any] = field(default_factory=dict)
    fidelity_score: float = 0.0  # How well simulation matches reality
    convergence_status: str = "initializing"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InterventionPlan:
    """A plan for physical intervention"""
    plan_id: str
    description: str
    target_system: str
    actions: List[Dict[str, Any]]  # Sequence of actions
    preconditions: List[str]
    postconditions: List[str]
    risk_assessment: Dict[str, float]
    estimated_duration: float  # seconds
    safety_checks: List[str]
    rollback_plan: List[Dict[str, Any]]
    created_at: datetime

@dataclass
class SimulationResult:
    """Result of a digital twin simulation"""
    simulation_id: str
    intervention_plan: InterventionPlan
    initial_state: DigitalTwinState
    final_state: DigitalTwinState
    state_trajectory: List[DigitalTwinState]
    outcome_prediction: Dict[str, Any]
    risk_probability_distribution: Dict[str, float]
    safety_violations: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    simulation_duration: float = 0.0
    fidelity_score: float = 0.0

@dataclass
class SwarmAgent:
    """An agent in a swarm simulation"""
    agent_id: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    state: Dict[str, Any] = field(default_factory=dict)
    role: str = "worker"
    communication_range: float = 10.0
    capabilities: List[str] = field(default_factory=list)
    energy_level: float = 1.0

@dataclass
class SwarmState:
    """State of a swarm simulation"""
    swarm_id: str
    timestamp: datetime
    agents: List[SwarmAgent] = field(default_factory=list)
    global_objectives: List[str] = field(default_factory=list)
    emergent_behaviors: List[str] = field(default_factory=list)
    communication_network: Dict[str, List[str]] = field(default_factory=dict)  # agent_id -> connected_agents
    consensus_level: float = 0.0
    task_completion: float = 0.0

@dataclass
class InterventionForecast:
    """Forecast of intervention outcomes"""
    forecast_id: str
    intervention_plan: InterventionPlan
    prediction_horizon: float  # seconds into future
    predicted_states: List[DigitalTwinState]
    confidence_intervals: Dict[str, Tuple[float, float]]
    risk_evolution: List[Dict[str, float]]
    intervention_points: List[Dict[str, Any]]
    generated_at: datetime
    alternative_scenarios: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class RealityVerificationResult:
    """Result of reality verification process"""
    verification_id: str
    timestamp: datetime
    proposed_action: Dict[str, Any]
    reality_check_passed: bool
    hallucination_detected: bool
    safety_violation_detected: bool
    sensor_validation_score: float  # 0.0 to 1.0
    simulation_fidelity_score: float  # 0.0 to 1.0
    intervention_forecast: Optional[InterventionForecast] = None
    blocking_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    verification_duration: float = 0.0

@dataclass
class SensorHealthStatus:
    """Health status of a sensor"""
    sensor_id: str
    last_reading_time: Optional[datetime]
    readings_per_minute: float
    data_quality_score: float  # 0.0 to 1.0
    calibration_drift: float
    failure_probability: float
    recommended_actions: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)

@dataclass
class RealityAnchorStatus:
    """Overall status of the reality anchor system"""
    timestamp: datetime
    system_health: float  # 0.0 to 1.0
    active_sensors: int
    sensor_health_scores: Dict[str, float]  # sensor_id -> health_score
    fusion_confidence: float
    simulation_fidelity: float
    active_alerts: List[str] = field(default_factory=list)
    system_warnings: List[str] = field(default_factory=list)
    last_reality_check: Optional[datetime] = None
    reality_checks_passed: int = 0
    reality_checks_failed: int = 0