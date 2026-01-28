"""
Kalman Filter for Aegis Nexus Sensor Fusion
Provides state estimation, noise filtering, and data imputation for dirty IoT data.

This module implements:
1. Standard Kalman Filter for linear systems
2. Extended Kalman Filter (EKF) for nonlinear systems
3. Adaptive noise estimation
4. Outlier detection via innovation monitoring
5. Missing data imputation via prediction
6. Sensor fault detection

CRITICAL: Real IoT data is dirty, missing, and noisy. This filter provides
the "Kalman Filter or Data Imputation layer" mentioned in the audit.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger("KalmanFilter")


class SensorState(Enum):
    """State of a sensor based on filter diagnostics."""
    HEALTHY = "healthy"
    NOISY = "noisy"
    DRIFTING = "drifting"
    FAILED = "failed"
    MISSING = "missing"


@dataclass
class FilterState:
    """Kalman filter state for a single sensor or state variable."""
    x: np.ndarray  # State estimate
    P: np.ndarray  # Error covariance
    Q: np.ndarray  # Process noise covariance
    R: np.ndarray  # Measurement noise covariance
    last_update: datetime = field(default_factory=datetime.utcnow)
    innovation_history: List[float] = field(default_factory=list)
    measurement_count: int = 0


@dataclass 
class FilterDiagnostics:
    """Diagnostics from filter operation."""
    sensor_id: str
    state: SensorState
    innovation: float  # Difference between prediction and measurement
    innovation_normalized: float  # Innovation normalized by expected variance
    is_outlier: bool
    imputed: bool
    confidence: float  # 0.0-1.0 confidence in the estimate
    raw_value: Optional[float]
    filtered_value: float


class KalmanFilter:
    """
    Kalman Filter for sensor state estimation and noise filtering.
    
    Provides:
    - State prediction for missing data
    - Noise filtering for noisy measurements
    - Outlier rejection with configurable threshold
    - Adaptive noise estimation
    """
    
    def __init__(self,
                 state_dim: int = 1,
                 measurement_dim: int = 1,
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.1,
                 outlier_threshold: float = 3.0):
        """
        Initialize Kalman Filter.
        
        Args:
            state_dim: Dimension of state vector
            measurement_dim: Dimension of measurement vector
            process_noise: Initial process noise variance
            measurement_noise: Initial measurement noise variance
            outlier_threshold: Number of std deviations for outlier detection
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.outlier_threshold = outlier_threshold
        
        # State transition matrix (identity for simple case)
        self.F = np.eye(state_dim)
        
        # Measurement matrix (identity for direct observation)
        self.H = np.eye(measurement_dim, state_dim)
        
        # Initial covariances
        self.Q_init = np.eye(state_dim) * process_noise
        self.R_init = np.eye(measurement_dim) * measurement_noise
        
        # Filter states for each sensor
        self.filter_states: Dict[str, FilterState] = {}
        
        # Innovation history window for adaptive estimation
        self.innovation_window = 50
        
    def _get_or_create_state(self, sensor_id: str, initial_value: Optional[float] = None) -> FilterState:
        """Get existing filter state or create new one."""
        if sensor_id not in self.filter_states:
            x = np.zeros((self.state_dim, 1))
            if initial_value is not None:
                x[0, 0] = initial_value
            
            self.filter_states[sensor_id] = FilterState(
                x=x,
                P=np.eye(self.state_dim) * 1.0,  # Large initial uncertainty
                Q=self.Q_init.copy(),
                R=self.R_init.copy()
            )
        
        return self.filter_states[sensor_id]
    
    def predict(self, sensor_id: str) -> Tuple[float, float]:
        """
        Predict next state for a sensor.
        
        Returns:
            Tuple of (predicted_value, uncertainty)
        """
        state = self._get_or_create_state(sensor_id)
        
        # Predict state: x_pred = F * x
        x_pred = self.F @ state.x
        
        # Predict covariance: P_pred = F * P * F' + Q
        P_pred = self.F @ state.P @ self.F.T + state.Q
        
        return float(x_pred[0, 0]), float(np.sqrt(P_pred[0, 0]))
    
    def update(self, 
               sensor_id: str, 
               measurement: Optional[float],
               timestamp: Optional[datetime] = None) -> FilterDiagnostics:
        """
        Update filter with new measurement.
        
        Args:
            sensor_id: Unique sensor identifier
            measurement: New measurement (None if missing)
            timestamp: Measurement timestamp
            
        Returns:
            FilterDiagnostics with filter results
        """
        state = self._get_or_create_state(sensor_id, measurement)
        timestamp = timestamp or datetime.utcnow()
        
        # === PREDICT STEP ===
        # State prediction
        x_pred = self.F @ state.x
        P_pred = self.F @ state.P @ self.F.T + state.Q
        
        # Handle missing measurement
        if measurement is None:
            # Use prediction as best estimate (imputation)
            state.x = x_pred
            state.P = P_pred
            state.last_update = timestamp
            
            return FilterDiagnostics(
                sensor_id=sensor_id,
                state=SensorState.MISSING,
                innovation=0.0,
                innovation_normalized=0.0,
                is_outlier=False,
                imputed=True,
                confidence=max(0.0, 1.0 - float(np.sqrt(state.P[0, 0]))),
                raw_value=None,
                filtered_value=float(state.x[0, 0])
            )
        
        # === INNOVATION (RESIDUAL) ===
        z = np.array([[measurement]])
        y = z - self.H @ x_pred  # Innovation
        innovation = float(y[0, 0])
        
        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + state.R
        innovation_normalized = float(abs(innovation) / np.sqrt(S[0, 0]))
        
        # === OUTLIER DETECTION ===
        is_outlier = innovation_normalized > self.outlier_threshold
        
        if is_outlier:
            # Log outlier but still update with reduced weight
            logger.warning(f"⚠️ Outlier detected for {sensor_id}: {measurement:.4f} "
                          f"(expected: {float(x_pred[0,0]):.4f}, innovation: {innovation_normalized:.2f}σ)")
            
            # Inflate measurement noise for this update (soft rejection)
            R_adjusted = state.R * (innovation_normalized ** 2)
            S = self.H @ P_pred @ self.H.T + R_adjusted
        
        # === UPDATE STEP (unless hard outlier rejection) ===
        if innovation_normalized < self.outlier_threshold * 2:  # Hard rejection threshold
            # Kalman gain
            K = P_pred @ self.H.T @ np.linalg.inv(S)
            
            # State update
            state.x = x_pred + K @ y
            
            # Covariance update (Joseph form for numerical stability)
            I_KH = np.eye(self.state_dim) - K @ self.H
            state.P = I_KH @ P_pred @ I_KH.T + K @ state.R @ K.T
        else:
            # Hard rejection - use prediction only
            state.x = x_pred
            state.P = P_pred
        
        # Update diagnostics
        state.measurement_count += 1
        state.last_update = timestamp
        state.innovation_history.append(innovation)
        if len(state.innovation_history) > self.innovation_window:
            state.innovation_history.pop(0)
        
        # Adaptive noise estimation
        self._adapt_noise(state)
        
        # Determine sensor health state
        sensor_state = self._diagnose_sensor(state, is_outlier)
        
        # Calculate confidence
        confidence = max(0.0, min(1.0, 1.0 - innovation_normalized / self.outlier_threshold))
        
        return FilterDiagnostics(
            sensor_id=sensor_id,
            state=sensor_state,
            innovation=innovation,
            innovation_normalized=innovation_normalized,
            is_outlier=is_outlier,
            imputed=False,
            confidence=confidence,
            raw_value=measurement,
            filtered_value=float(state.x[0, 0])
        )
    
    def _adapt_noise(self, state: FilterState):
        """
        Adaptively estimate process and measurement noise from innovation statistics.
        
        The innovation sequence should be white noise with zero mean if the filter
        is properly tuned. Deviations indicate mistuning.
        """
        if len(state.innovation_history) < 10:
            return
        
        innovations = np.array(state.innovation_history)
        
        # Expected innovation variance: S = H*P*H' + R
        # If actual variance differs, adjust R
        actual_variance = np.var(innovations)
        expected_variance = float(state.R[0, 0])
        
        # Exponential smoothing of noise estimate
        alpha = 0.05
        new_R = (1 - alpha) * expected_variance + alpha * actual_variance
        state.R[0, 0] = max(0.001, new_R)  # Prevent collapse to zero
        
        # Check for bias (non-zero mean innovation indicates process drift)
        mean_innovation = np.mean(innovations[-10:])
        if abs(mean_innovation) > 0.1 * np.std(innovations):
            # Increase process noise to account for unmodeled dynamics
            state.Q[0, 0] *= 1.1
    
    def _diagnose_sensor(self, state: FilterState, is_outlier: bool) -> SensorState:
        """Diagnose sensor health based on filter statistics."""
        if len(state.innovation_history) < 5:
            return SensorState.HEALTHY
        
        recent_innovations = np.array(state.innovation_history[-10:])
        
        # Check for drift (systematic bias)
        mean_abs_innovation = np.mean(np.abs(recent_innovations))
        if mean_abs_innovation > 2 * np.std(recent_innovations):
            return SensorState.DRIFTING
        
        # Check for excessive noise
        if np.std(recent_innovations) > 3 * float(np.sqrt(state.R[0, 0])):
            return SensorState.NOISY
        
        # Frequent outliers indicate sensor failure
        recent_outliers = sum(1 for i in recent_innovations 
                             if abs(i) > self.outlier_threshold * float(np.sqrt(state.R[0, 0])))
        if recent_outliers > len(recent_innovations) * 0.3:
            return SensorState.FAILED
        
        return SensorState.HEALTHY
    
    def get_filtered_value(self, sensor_id: str) -> Optional[float]:
        """Get current filtered estimate for a sensor."""
        if sensor_id not in self.filter_states:
            return None
        return float(self.filter_states[sensor_id].x[0, 0])
    
    def get_sensor_diagnostics(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed diagnostics for a sensor."""
        if sensor_id not in self.filter_states:
            return None
        
        state = self.filter_states[sensor_id]
        
        return {
            "sensor_id": sensor_id,
            "estimate": float(state.x[0, 0]),
            "uncertainty": float(np.sqrt(state.P[0, 0])),
            "process_noise": float(state.Q[0, 0]),
            "measurement_noise": float(state.R[0, 0]),
            "measurement_count": state.measurement_count,
            "last_update": state.last_update.isoformat(),
            "recent_innovation_std": float(np.std(state.innovation_history)) if state.innovation_history else 0.0
        }
    
    def reset_sensor(self, sensor_id: str):
        """Reset filter state for a sensor."""
        if sensor_id in self.filter_states:
            del self.filter_states[sensor_id]


class MultiStateKalmanFilter(KalmanFilter):
    """
    Extended Kalman Filter for tracking multiple correlated state variables.
    
    Useful for:
    - Position/velocity tracking
    - Temperature/gradient tracking
    - Any system with physics-based relationships
    """
    
    def __init__(self,
                 dt: float = 0.1,
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.1):
        """
        Initialize multi-state filter for position/velocity tracking.
        
        Args:
            dt: Time step between measurements
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
        """
        super().__init__(
            state_dim=2,  # [position, velocity]
            measurement_dim=1,  # Only position measured
            process_noise=process_noise,
            measurement_noise=measurement_noise
        )
        
        self.dt = dt
        
        # State transition matrix for constant velocity model
        # [pos]     [1  dt] [pos]     [noise]
        # [vel]  =  [0   1] [vel]  +  [noise]
        self.F = np.array([
            [1, dt],
            [0, 1]
        ])
        
        # Measurement matrix - only observe position
        self.H = np.array([[1, 0]])
        
        # Process noise covariance (tuned for constant velocity model)
        self.Q_init = np.array([
            [dt**4/4, dt**3/2],
            [dt**3/2, dt**2]
        ]) * process_noise
    
    def update(self, 
               sensor_id: str, 
               measurement: Optional[float],
               timestamp: Optional[datetime] = None) -> FilterDiagnostics:
        """Update with position measurement, estimates velocity."""
        # Call parent update
        diagnostics = super().update(sensor_id, measurement, timestamp)
        
        # Extract velocity estimate for logging
        if sensor_id in self.filter_states:
            state = self.filter_states[sensor_id]
            velocity = float(state.x[1, 0])
            logger.debug(f"Sensor {sensor_id}: pos={diagnostics.filtered_value:.4f}, vel={velocity:.4f}")
        
        return diagnostics
    
    def get_velocity(self, sensor_id: str) -> Optional[float]:
        """Get estimated velocity for a sensor."""
        if sensor_id not in self.filter_states:
            return None
        return float(self.filter_states[sensor_id].x[1, 0])


class SensorFusionFilter:
    """
    Multi-sensor fusion using individual Kalman filters.
    
    Combines estimates from multiple sensors observing the same quantity,
    weighted by their respective uncertainties.
    """
    
    def __init__(self, outlier_threshold: float = 3.0):
        self.filters: Dict[str, KalmanFilter] = {}
        self.outlier_threshold = outlier_threshold
    
    def register_sensor(self, sensor_id: str, 
                       measurement_noise: float = 0.1,
                       process_noise: float = 0.01):
        """Register a sensor with its noise characteristics."""
        self.filters[sensor_id] = KalmanFilter(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            outlier_threshold=self.outlier_threshold
        )
    
    def update(self, sensor_id: str, measurement: Optional[float],
               timestamp: Optional[datetime] = None) -> Optional[FilterDiagnostics]:
        """Update a specific sensor."""
        if sensor_id not in self.filters:
            self.register_sensor(sensor_id)
        
        return self.filters[sensor_id].update(sensor_id, measurement, timestamp)
    
    def get_fused_estimate(self, quantity_id: str, 
                           sensor_ids: List[str]) -> Tuple[float, float]:
        """
        Get fused estimate from multiple sensors.
        
        Uses optimal fusion formula: weighted average by inverse variance.
        
        Returns:
            Tuple of (fused_estimate, fused_uncertainty)
        """
        estimates = []
        variances = []
        
        for sensor_id in sensor_ids:
            if sensor_id in self.filters:
                state = self.filters[sensor_id].filter_states.get(sensor_id)
                if state is not None:
                    estimates.append(float(state.x[0, 0]))
                    variances.append(float(state.P[0, 0]))
        
        if not estimates:
            return 0.0, float('inf')
        
        # Optimal fusion: inverse-variance weighted average
        weights = [1.0 / v for v in variances]
        total_weight = sum(weights)
        
        fused_estimate = sum(w * e for w, e in zip(weights, estimates)) / total_weight
        fused_variance = 1.0 / total_weight
        
        return fused_estimate, np.sqrt(fused_variance)


# Global instance
_sensor_filter: Optional[SensorFusionFilter] = None


def get_sensor_filter() -> SensorFusionFilter:
    """Get global sensor fusion filter instance."""
    global _sensor_filter
    if _sensor_filter is None:
        _sensor_filter = SensorFusionFilter()
    return _sensor_filter


def filter_sensor_reading(sensor_id: str, 
                         raw_value: Optional[float],
                         timestamp: Optional[datetime] = None) -> FilterDiagnostics:
    """
    Convenience function to filter a single sensor reading.
    
    Returns filtered value with diagnostics.
    """
    filter = get_sensor_filter()
    return filter.update(sensor_id, raw_value, timestamp)
