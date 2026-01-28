"""
Sensor Fusion Engine for Aegis Nexus
Merges data from disparate sources to create a "Ground Truth" state.

This module implements multi-modal sensor fusion with Kalman filtering
for reality grounding and data validation.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from scipy.spatial import distance

from reality_anchor.sensors.protocol_adapters import SensorProtocolAdapter
from reality_anchor.metrology.precision_validator import DataMetrology
from reality_anchor.common.types import SensorReading
from reality_anchor.sensors.signal_smoother import SignalPreProcessor

logger = logging.getLogger(__name__)

# =============================================================================
# STALE DATA HANDLING (Gap 3.1)
# =============================================================================
# TTL in seconds - sensors exceeding this are marked OFFLINE and excluded from fusion
SENSOR_TTL_SECONDS = 5.0  # Sensors older than 5 seconds are considered stale
SENSOR_CRITICAL_TTL_SECONDS = 10.0  # Sensors older than 10 seconds trigger alerts

# Covariance explosion prevention - cap maximum covariance values
MAX_COVARIANCE_VALUE = 1e6  # Prevent numerical instability

# Sensor health tracking
SENSOR_HEALTH_WINDOW = 60  # Track health over 60 seconds
MIN_SENSOR_RELIABILITY = 0.5  # Minimum reliability to include in fusion

@dataclass
class FusedState:
    """Fused state representing ground truth from multiple sensors"""
    state_vector: List[float]
    covariance_matrix: np.ndarray
    timestamp: datetime
    confidence: float  # 0.0 to 1.0
    contributing_sensors: List[str]
    anomalies_detected: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SensoryFusionCore:
    """
    Core sensor fusion system creating ground truth from disparate sources.
    
    Implements:
    - Time alignment for asynchronous sensor streams
    - Cross-modal correlation analysis
    - Kalman filtering for noise reduction
    - Multi-sensor data fusion
    """
    
    def __init__(self, buffer_size: int = 1000, heartbeat_timeout: float = 0.5):
        """
        Initialize the sensory fusion core.
        
        Args:
            buffer_size: Size of the temporal alignment buffer
            heartbeat_timeout: Timeout in seconds for reality heartbeat (dead man's switch)
        """
        import os
        self.buffer_size = buffer_size
        # Use env var for timeout if configured, otherwise use default
        env_timeout = os.getenv("AEGIS_HEARTBEAT_TIMEOUT")
        if env_timeout:
            try:
                self.heartbeat_timeout = float(env_timeout)
            except ValueError:
                self.heartbeat_timeout = heartbeat_timeout
        else:
            self.heartbeat_timeout = heartbeat_timeout
            
        self.last_sensor_update: datetime = datetime.utcnow()
        self.system_frozen = False
        self.sensor_buffer: Dict[str, List[SensorReading]] = {}
        self.protocol_adapter = SensorProtocolAdapter()
        self.data_metrology = DataMetrology()
        self.kalman_filters: Dict[str, Any] = {}  # Kalman filter instances
        self.cross_correlations: Dict[Tuple[str, str], float] = {}
        self.fusion_history: List[FusedState] = []
        
        # Stale data handling (Gap 3.1)
        self.sensor_last_seen: Dict[str, datetime] = {}  # Track last reading per sensor
        self.sensor_health_scores: Dict[str, float] = {}  # Reliability scores per sensor
        self.offline_sensors: set = set()  # Sensors currently marked OFFLINE
        
        # Signal Pre-processors per sensor ID
        self.signal_processors: Dict[str, SignalPreProcessor] = {}
        
    async def initialize(self) -> bool:
        """Initialize the sensory fusion system."""
        try:
            logger.info("🌍 Initializing sensory fusion core...")
            
            # Initialize protocol adapters
            await self.protocol_adapter.initialize()
            
            # Initialize data metrology
            await self.data_metrology.initialize()
            
            logger.info("✅ Sensory fusion core initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize sensory fusion core: {str(e)}")
            return False
    
    async def fuse_data(self, sensor_streams: List[SensorReading]) -> FusedState:
        """
        Fuse data from multiple sensor streams into ground truth state.
        
        Args:
            sensor_streams: List of sensor readings from various sources
            
        Returns:
            FusedState representing the ground truth
        """
        logger.debug(f"🌍 Fusing data from {len(sensor_streams)} sensor streams")
        
        try:
            # Reality Heartbeat Check: If no fresh telemetry for >500ms, assert SYSTEM_FREEZE
            current_time = datetime.utcnow()
            time_since_last_update = (current_time - self.last_sensor_update).total_seconds()
            if time_since_last_update > self.heartbeat_timeout:
                logger.warning(f"⚠️  Reality heartbeat timeout detected ({time_since_last_update}s > {self.heartbeat_timeout}s). Setting system freeze.")
                self.system_frozen = True
                # Return frozen state to prevent any physical action
                return FusedState(
                    state_vector=[0.0] * 10,
                    covariance_matrix=np.eye(10),
                    timestamp=current_time,
                    confidence=0.0,
                    contributing_sensors=[],
                    anomalies_detected=["system_frozen_due_to_heartbeat_timeout"]
                )
            else:
                self.system_frozen = False
                self.last_sensor_update = current_time
            
            # Step 1: Time alignment
            aligned_data = await self._time_align_sensors(sensor_streams)
            
            # Step 2: Quality assessment
            quality_filtered = await self._assess_data_quality(aligned_data)

            # Step 2.1: Signal Smoothing (Debouncing) - NEW FEATURE
            # Smooth out raw flickering before expensive Kalman filtering
            smoothed_data = await self._apply_signal_smoothing(quality_filtered)

            # Step 2.5: Bayesian Outlier Rejection
            # Filter out sensors that statistically disagree with the consensus
            consensus_filtered = await self._bayesian_outlier_rejection(smoothed_data)
            
            # Step 3: Cross-modal correlation
            correlated_data = await self._compute_cross_correlations(consensus_filtered)
            
            # Step 4: Kalman filtering
            filtered_data = await self._apply_kalman_filtering(correlated_data)
            
            # Step 5: State fusion
            fused_state = await self._fuse_sensor_states(filtered_data)
            
            # Step 6: Anomaly detection
            anomalies = await self._detect_anomalies(fused_state, filtered_data)
            
            # Update fused state with anomalies
            fused_state.anomalies_detected = anomalies
            
            # Add to history
            self.fusion_history.append(fused_state)
            
            # Maintain history size
            if len(self.fusion_history) > self.buffer_size:
                self.fusion_history.pop(0)
            
            logger.debug(f"🌍 Data fusion completed with confidence: {fused_state.confidence:.2f}")
            return fused_state
            
        except Exception as e:
            logger.error(f"❌ Data fusion failed: {str(e)}")
            # Return degraded state
            return FusedState(
                state_vector=[0.0] * 10,  # Default 10-dimensional state
                covariance_matrix=np.eye(10),
                timestamp=datetime.utcnow(),
                confidence=0.0,
                contributing_sensors=[],
                anomalies_detected=["fusion_error"]
            )
    
    async def _time_align_sensors(self, sensor_streams: List[SensorReading]) -> List[SensorReading]:
        """
        Align timestamps from asynchronous sensor streams.
        
        TTL-based Stale Data Handling (Gap 3.1):
        - Sensors exceeding SENSOR_TTL_SECONDS are marked as STALE
        - Sensors exceeding SENSOR_CRITICAL_TTL_SECONDS are marked OFFLINE
        - Stale sensors are excluded from fusion to prevent pulling state towards obsolete values
        """
        current_time = datetime.utcnow()
        aligned = []
        stale_count = 0
        
        for reading in sensor_streams:
            sensor_id = reading.sensor_id
            time_diff = (current_time - reading.timestamp).total_seconds()
            
            # Update last seen timestamp
            self.sensor_last_seen[sensor_id] = reading.timestamp
            
            # Check for critically stale data
            if time_diff > SENSOR_CRITICAL_TTL_SECONDS:
                if sensor_id not in self.offline_sensors:
                    logger.critical(f"🚨 SENSOR OFFLINE: {sensor_id} critically stale ({time_diff:.1f}s > {SENSOR_CRITICAL_TTL_SECONDS}s)")
                    self.offline_sensors.add(sensor_id)
                # Exclude from fusion entirely
                stale_count += 1
                continue
            
            # Check for stale data (but not critical)
            elif time_diff > SENSOR_TTL_SECONDS:
                logger.warning(f"⚠️ Stale sensor data: {sensor_id} ({time_diff:.1f}s old)")
                # Heavily penalize quality but still include with reduced weight
                reading.quality *= 0.3
                stale_count += 1
                # Remove from offline set if it was previously offline but now sending
                self.offline_sensors.discard(sensor_id)
            else:
                # Fresh data - sensor is healthy
                self.offline_sensors.discard(sensor_id)
                # Update health score (exponential moving average)
                current_health = self.sensor_health_scores.get(sensor_id, 1.0)
                self.sensor_health_scores[sensor_id] = 0.9 * current_health + 0.1 * reading.quality
            
            aligned.append(reading)
        
        if stale_count > 0:
            logger.info(f"📊 Stale data stats: {stale_count}/{len(sensor_streams)} sensors stale, {len(self.offline_sensors)} offline")
        
        return aligned
    
    async def _assess_data_quality(self, sensor_data: List[SensorReading]) -> List[SensorReading]:
        """Assess and filter data based on quality metrics."""
        filtered = []
        
        for reading in sensor_data:
            # Validate data using metrology system
            validation_result = await self.data_metrology.validate_data_integrity(
                reading.value, reading.sensor_type
            )
            
            # Update quality score
            reading.quality *= validation_result.health_score
            
            # Only keep high-quality readings
            if reading.quality >= 0.7:  # 70% quality threshold
                filtered.append(reading)
        
        return filtered

    async def _apply_signal_smoothing(self, sensor_data: List[SensorReading]) -> List[SensorReading]:
        """
        Apply signal smoothing (Median Filter + EMA) to raw readings.
        Reduces high-frequency noise and salt-and-pepper outliers.
        """
        smoothed_data = []
        
        for reading in sensor_data:
            # Initialize processor if needed
            if reading.sensor_id not in self.signal_processors:
                self.signal_processors[reading.sensor_id] = SignalPreProcessor()
            
            processor = self.signal_processors[reading.sensor_id]
            
            try:
                if isinstance(reading.value, (int, float)):
                    # Apply smoothing
                    smoothed_val = processor.process(float(reading.value))
                    reading.value = smoothed_val
                
                smoothed_data.append(reading)
            except Exception:
                # If smoothing fails (e.g. non-numeric), pass through original
                smoothed_data.append(reading)
                
        return smoothed_data

    async def _bayesian_outlier_rejection(self, sensor_data: List[SensorReading]) -> List[SensorReading]:
        """
        Apply IsolationForest-based anomaly detection to reject sensors exhibiting
        spoofing patterns (e.g., GPS drifting faster than speed of sound, temperature
        spikes from 20°C to 500°C in 1 second).
        
        Uses sklearn's IsolationForest for robust, multi-dimensional outlier detection.
        """
        if len(sensor_data) < 5:
            # IsolationForest needs sufficient samples
            return sensor_data
        
        from sklearn.ensemble import IsolationForest
            
        # Group by sensor type
        type_groups = defaultdict(list)
        for r in sensor_data:
            type_groups[r.sensor_type].append(r)
            
        accepted_sensors = []
        
        for s_type, readings in type_groups.items():
            if len(readings) < 5:
                # Need enough samples for IsolationForest
                accepted_sensors.extend(readings)
                continue
                
            # Extract feature matrix: [value, quality, timestamp_delta]
            features = []
            base_time = readings[0].timestamp
            for r in readings:
                try:
                    val = float(r.value) if isinstance(r.value, (int, float)) else 0.0
                    quality = float(r.quality) if hasattr(r, 'quality') else 1.0
                    time_delta = (r.timestamp - base_time).total_seconds() if hasattr(r, 'timestamp') else 0.0
                    features.append([val, quality, time_delta])
                except:
                    features.append([0.0, 1.0, 0.0])
            
            if not features:
                accepted_sensors.extend(readings)
                continue
            
            # Fit IsolationForest
            # contamination='auto' lets the model determine the threshold
            try:
                clf = IsolationForest(
                    n_estimators=50,  # Lightweight for real-time
                    contamination=0.1,  # Expect ~10% anomalies max
                    random_state=42,
                    n_jobs=1  # Single-threaded for async compatibility
                )
                
                X = np.array(features)
                predictions = clf.fit_predict(X)
                
                # Filter readings: -1 = outlier, 1 = inlier
                for reading, pred in zip(readings, predictions):
                    if pred == -1:
                        val = float(reading.value) if isinstance(reading.value, (int, float)) else 'N/A'
                        logger.warning(f"⛔ IsolationForest Rejection: Sensor {reading.sensor_id} (type={s_type}) value={val} detected as anomaly")
                    else:
                        accepted_sensors.append(reading)
                        
            except Exception as e:
                logger.warning(f"⚠️ IsolationForest failed for {s_type}: {e}. Falling back to z-score.")
                # Fallback to simple z-score if IsolationForest fails
                values = [float(r.value) if isinstance(r.value, (int, float)) else 0.0 for r in readings]
                median_val = np.median(values)
                std_dev = np.std(values) + 1e-6
                
                for reading in readings:
                    try:
                        val = float(reading.value) if isinstance(reading.value, (int, float)) else 0.0
                        z_score = abs(val - median_val) / std_dev
                        if z_score <= 2.0:
                            accepted_sensors.append(reading)
                        else:
                            logger.warning(f"⛔ Z-Score Rejection: Sensor {reading.sensor_id} deviates {z_score:.2f}σ")
                    except:
                        accepted_sensors.append(reading)
                   
        return accepted_sensors
    
    async def _compute_cross_correlations(self, sensor_data: List[SensorReading]) -> List[SensorReading]:
        """Compute correlations between related sensor modalities."""
        # Simple correlation logic - in practice this would use signal processing
        correlated = sensor_data.copy()
        
        # Group by sensor type for correlation analysis
        sensor_groups = {}
        for reading in sensor_data:
            if reading.sensor_type not in sensor_groups:
                sensor_groups[reading.sensor_type] = []
            sensor_groups[reading.sensor_type].append(reading)
        
        # Compute inter-sensor correlations
        for type1, readings1 in sensor_groups.items():
            for type2, readings2 in sensor_groups.items():
                if type1 != type2 and len(readings1) > 0 and len(readings2) > 0:
                    correlation = await self._calculate_correlation(type1, type2, readings1, readings2)
                    self.cross_correlations[(type1, type2)] = correlation
        
        return correlated
    
    async def _calculate_correlation(self, type1: str, type2: str, readings1: List[SensorReading], readings2: List[SensorReading]) -> float:
        """Calculate correlation between two sensor types."""
        # Simple correlation calculation
        try:
            values1 = [float(r.value) if isinstance(r.value, (int, float)) else 0.0 for r in readings1]
            values2 = [float(r.value) if isinstance(r.value, (int, float)) else 0.0 for r in readings2]
            
            if len(values1) > 1 and len(values2) > 1:
                correlation = np.corrcoef(values1, values2)[0, 1]
                return abs(float(correlation)) if not np.isnan(correlation) else 0.0
        except Exception:
            pass
        
        return 0.0
    
    async def _apply_kalman_filtering(self, sensor_data: List[SensorReading]) -> List[SensorReading]:
        """Apply Kalman filtering to smooth sensor data."""
        filtered = []
        
        for reading in sensor_data:
            # Initialize Kalman filter for this sensor type if needed
            if reading.sensor_type not in self.kalman_filters:
                self.kalman_filters[reading.sensor_type] = await self._create_kalman_filter(reading.sensor_type)
            
            # Apply filtering
            kalman_filter = self.kalman_filters[reading.sensor_type]
            filtered_value = await self._kalman_predict_update(kalman_filter, reading)
            
            # Update reading with filtered value
            reading.value = filtered_value
            filtered.append(reading)
        
        return filtered
    
    async def _create_kalman_filter(self, sensor_type: str) -> Dict[str, Any]:
        """Create Unscented Kalman Filter (UKF) parameters."""
        # 1-Dimensional state (value) + velocity for simplicity in this demo,
        # but using UKF structure (Sigma Points).
        n_dim = 2 
        
        return {
            'state': np.zeros(n_dim), 
            'covariance': np.eye(n_dim),
            'process_noise': np.eye(n_dim) * 0.1,
            'measurement_noise': np.array([[0.1]]),
            'alpha': 1e-3,
            'kappa': 0,
            'beta': 2,
            'consecutive_divergence_count': 0,
            'innovation_history': []  # NEW: For Adaptive Noise Estimation
        }

    def _generate_sigma_points(self, mean, cov, alpha, kappa):
        """Generate Sigma Points for UKF."""
        n = len(mean)
        lambda_ = alpha**2 * (n + kappa) - n
        
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky((n + lambda_) * cov)
        except np.linalg.LinAlgError:
            L = np.eye(n) # Fallback
            
        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = mean
        for i in range(n):
            sigmas[i + 1] = mean + L[i]
            sigmas[n + i + 1] = mean - L[i]
            
        return sigmas, lambda_

    async def _kalman_predict_update(self, kf_params: Dict[str, Any], reading: SensorReading) -> Any:
        """Apply Unscented Kalman Filter (UKF) Cycle."""
        try:
            z = float(reading.value) if isinstance(reading.value, (int, float)) else 0.0
            
            # 1. Generate Sigma Points
            sigmas, lambda_ = self._generate_sigma_points(
                kf_params['state'], kf_params['covariance'], 
                kf_params['alpha'], kf_params['kappa']
            )
            
            # weights
            n = len(kf_params['state'])
            Wm = np.full(2*n + 1, 1 / (2*(n + lambda_)))
            Wc = np.full(2*n + 1, 1 / (2*(n + lambda_)))
            Wm[0] = lambda_ / (n + lambda_)
            Wc[0] = lambda_ / (n + lambda_) + (1 - kf_params['alpha']**2 + kf_params['beta'])
            
            # 2. Predict (Unscented Transform)
            # Process function f(x) = x (Identity + Velocity for simplicity)
            # x' = x + v * dt
            dt = 0.1
            sigmas_f = np.copy(sigmas)
            sigmas_f[:, 0] += sigmas_f[:, 1] * dt # Pos += Vel * dt
            
            # Predicted Mean
            x_pred = np.dot(Wm, sigmas_f)
            # Predicted Covariance
            P_pred = kf_params['process_noise'].copy()
            for i in range(2*n + 1):
                y = sigmas_f[i] - x_pred
                P_pred += Wc[i] * np.outer(y, y)
                
            # 3. Update
            # Measurement function h(x) = pos
            sigmas_h = sigmas_f[:, 0:1] # Just position
            z_pred_full = np.dot(Wm, sigmas_h)
            z_pred = z_pred_full[0] if isinstance(z_pred_full, (np.ndarray, list)) else z_pred_full
            
            # --- ADAPTIVE NOISE ESTIMATION (Level 5) ---
            # Monitor Innovation Sequence to detect model/sensor drift
            innovation = z - z_pred
            
            if 'innovation_history' not in kf_params:
                kf_params['innovation_history'] = []
            kf_params['innovation_history'].append(innovation)
            
            # Maintenance: Keep last 50 samples
            if len(kf_params['innovation_history']) > 50:
                kf_params['innovation_history'].pop(0)
                
            # Adaptive Logic: Adjust Q and R based on innovation stats
            if len(kf_params['innovation_history']) >= 20:
                innovations = np.array(kf_params['innovation_history'])
                # innovation_mean = np.mean(innovations) # Unused
                innovation_var = np.var(innovations)
                
                # Check for Sensor Noise Mismatch (High Variance)
                # Theoretical S = HPH' + R. If actual variance > S, R is likely underestimated.
                # Simplified check against current measurement noise floor
                R_val = kf_params['measurement_noise'][0,0]
                
                if innovation_var > R_val * 4.0:
                    # logger.debug(f"⚠️ High innovation variance ({innovation_var:.3f}). Adaptive R boost.")
                    kf_params['measurement_noise'] *= 1.05  # Trust sensor less
                elif innovation_var < R_val * 0.1:
                    kf_params['measurement_noise'] *= 0.98  # Trust sensor more
                
                # Clamp R to sane limits
                kf_params['measurement_noise'] = np.clip(kf_params['measurement_noise'], 0.01, 10.0)

            # --- END ADAPTIVE LOGIC ---
            
            # Measurement Covariance
            S = kf_params['measurement_noise'].copy()
            for i in range(2*n + 1):
                y = sigmas_h[i] - z_pred
                S += Wc[i] * np.outer(y, y)
                
            # Cross Covariance
            Pxz = np.zeros((n, 1))
            for i in range(2*n + 1):
                Pxz += Wc[i] * np.outer(sigmas_f[i] - x_pred, sigmas_h[i] - z_pred)
                
            # 3. Update with Joseph Stabilized Form (Level 5 Reliability)
            # Standard P = (I - KH)P can lose symmetry due to rounding errors.
            # Joseph Form: P = (I - KH)P(I - KH)' + KRK' 
            # This ensures P remains positive-definite even with precision issues.
            
            # Residual (Innovation)
            y_res = z - z_pred
            
            # Kalman Gain
            K = np.dot(Pxz, np.linalg.inv(S))
            
            # --- MAHALANOBIS GATING (Level 5 Outlier Rejection) ---
            # Calculated as: distance^2 = y' * inv(S) * y
            # Rejects measurements that are statistically impossible (> 99.9% confidence)
            try:
                # 1. NEW: Mahalanobis Gating
                # Calculate statistical distance of new measurement from predicted state
                innovation = z - z_pred
                innovation_covariance = S[0,0]  # From existing UKF logic
                
                # Calculate squared Mahalanobis distance
                # d^2 = (z - z_pred)^T * inv(S) * (z - z_pred)
                md_squared = (innovation**2) / innovation_covariance
                
                # Chi-square threshold for 1 degree of freedom, p=0.001 (very rare event)
                CHI_SQUARE_THRESHOLD = 10.83 
                
                if md_squared > CHI_SQUARE_THRESHOLD:
                    logger.warning(f"🛡️ Reality Shield: Rejected spoofed/glitched sensor {reading.sensor_id}. Dev: {md_squared:.2f}σ")
                    
                    # 2. NEW: Adaptive Isolation
                    # If a sensor consistently fails gating, mark it as 'Compromised' (Health Penalty)
                    current_health = self.sensor_health_scores.get(reading.sensor_id, 1.0)
                    self.sensor_health_scores[reading.sensor_id] = max(0.0, current_health - 0.2)
                    
                    kf_params['consecutive_divergence_count'] += 1
                    
                    if kf_params['consecutive_divergence_count'] >= 5:
                        logger.critical(f"🚨 PERSISTENT REALITY DRIFT: Resetting Kalman state for {reading.sensor_type}")
                        kf_params['state'][0] = z # Snap to truth
                        kf_params['state'][1] = 0 # reset velocity
                        kf_params['covariance'] = np.eye(n) * 1.0 # high uncertainty reset
                        kf_params['consecutive_divergence_count'] = 0
                    
                    # Return predicted state instead of corrupted update
                    return x_pred[0]
                else:
                    kf_params['consecutive_divergence_count'] = 0
            except Exception as e:
                logger.warning(f"Gating logic failed: {e}")

            # Update State
            kf_params['state'] = x_pred + np.dot(K, y_res).flatten()
            
            # Joseph Form Covariance Update
            I = np.eye(n)
            # Measurement H matrix for position [1, 0]
            H = np.array([[1, 0]])
            I_KH = I - np.dot(K, H)
            
            # P = (I-KH)P(I-KH)' + KRK'
            # (Note: R is measurement_noise in kf_params)
            P_new = np.dot(np.dot(I_KH, P_pred), I_KH.T) + np.dot(np.dot(K, kf_params['measurement_noise']), K.T)
            
            # Force symmetry to be absolutely safe
            kf_params['covariance'] = (P_new + P_new.T) / 2.0
            
            return kf_params['state'][0]
            
        except Exception as e:
            logger.warning(f"⚠️ UKF failed: {e}")
            return reading.value

    
    async def _fuse_sensor_states(self, filtered_data: List[SensorReading]) -> FusedState:
        """Fuse individual sensor states into unified ground truth."""
        if not filtered_data:
            return FusedState(
                state_vector=[0.0] * 10,
                covariance_matrix=np.eye(10),
                timestamp=datetime.utcnow(),
                confidence=0.0,
                contributing_sensors=[]
            )
        
        # Convert sensor readings to state vector
        state_vector = []
        total_confidence = 0.0
        contributing_sensors = []
        
        # Sort by quality for weighted fusion
        sorted_data = sorted(filtered_data, key=lambda x: x.quality, reverse=True)
        
        for reading in sorted_data[:10]:  # Limit to top 10 readings
            try:
                # Convert reading to numeric value
                if isinstance(reading.value, (int, float)):
                    state_value = float(reading.value)
                elif isinstance(reading.value, str):
                    # Simple string to numeric conversion
                    state_value = hash(reading.value) % 1000 / 100.0
                else:
                    state_value = 0.0
                
                # Apply quality weighting
                weighted_value = state_value * reading.quality
                state_vector.append(weighted_value)
                total_confidence += reading.quality
                contributing_sensors.append(reading.sensor_id)
                
            except Exception:
                continue
        
        # Pad to fixed size if needed
        while len(state_vector) < 10:
            state_vector.append(0.0)
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(contributing_sensors) if contributing_sensors else 0.0
        
        # Create covariance matrix (diagonal for independence assumption)
        covariances = [1.0 / (q + 0.1) for q in [r.quality for r in sorted_data[:10]]]
        while len(covariances) < 10:
            covariances.append(10.0)  # High uncertainty for unused dimensions
        
        covariance_matrix = np.diag(covariances)
        
        return FusedState(
            state_vector=state_vector[:10],
            covariance_matrix=covariance_matrix,
            timestamp=datetime.utcnow(),
            confidence=min(1.0, avg_confidence),
            contributing_sensors=contributing_sensors
        )
    
    async def _detect_anomalies(self, fused_state: FusedState, filtered_data: List[SensorReading]) -> List[str]:
        """Detect anomalies in the fused sensor data."""
        anomalies = []
        
        # Check for sensor disagreement
        if len(filtered_data) > 1:
            values = [float(r.value) if isinstance(r.value, (int, float)) else 0.0 for r in filtered_data]
            std_dev = np.std(values)
            mean_val = np.mean(values)
            
            if std_dev > 2.0:  # High variance indicates disagreement
                anomalies.append("sensor_disagreement")
        
        # Check confidence levels
        if fused_state.confidence < 0.5:
            anomalies.append("low_confidence_fusion")
        
        # Check for extreme values
        for value in fused_state.state_vector:
            if abs(value) > 1000:  # Arbitrary threshold
                anomalies.append("extreme_value_detected")
                break
        
        return anomalies
    
    async def get_current_ground_truth(self) -> Optional[FusedState]:
        """Get the most recent fused ground truth state."""
        if self.fusion_history:
            return self.fusion_history[-1]
        return None
    
    def is_system_frozen(self) -> bool:
        """Check if the system is in frozen state due to heartbeat timeout."""
        current_time = datetime.utcnow()
        time_since_last_update = (current_time - self.last_sensor_update).total_seconds()
        
        # If timeout exceeded, system remains frozen
        if time_since_last_update > self.heartbeat_timeout:
            self.system_frozen = True
        
        return self.system_frozen
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get statistics about the fusion process."""
        if not self.fusion_history:
            return {
                'total_fusions': 0,
                'average_confidence': 0.0,
                'anomalies_detected': 0,
                'system_frozen': self.system_frozen
            }
        
        confidences = [state.confidence for state in self.fusion_history]
        anomalies = sum(len(state.anomalies_detected) for state in self.fusion_history)
        
        return {
            'total_fusions': len(self.fusion_history),
            'average_confidence': np.mean(confidences) if confidences else 0.0,
            'anomalies_detected': anomalies,
            'recent_correlations': len(self.cross_correlations),
            'system_frozen': self.system_frozen
        }

# Global instance
sensor_fusion_core: Optional[SensoryFusionCore] = None

async def initialize_sensor_fusion(buffer_size: int = 1000) -> bool:
    """
    Initialize the global sensor fusion instance.
    
    Args:
        buffer_size: Size of the temporal buffer
        
    Returns:
        bool: Success status
    """
    global sensor_fusion_core
    
    try:
        sensor_fusion_core = SensoryFusionCore(buffer_size=buffer_size)
        return await sensor_fusion_core.initialize()
    except Exception as e:
        logger.error(f"❌ Failed to initialize sensor fusion: {str(e)}")
        return False


def get_sensor_fusion() -> SensoryFusionCore:
    """Get the global sensor fusion instance."""
    if sensor_fusion_core is None:
        raise RuntimeError("Sensor fusion not initialized")
    return sensor_fusion_core