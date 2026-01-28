import asyncio
import logging
import random
from typing import Dict, Any, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [PROPRIOCEPTION] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ProprioceptiveMonitor")

# =============================================================================
# SIX SIGMA THRESHOLD ENFORCEMENT (Gap 3.2)
# =============================================================================
# Six Sigma quality standard: 6σ = 3.4 defects per million opportunities
# Progressive severity levels with corresponding actions

SIGMA_LEVELS = {
    3.0: "WARNING",       # 3σ = 99.73% confidence - Log only
    4.0: "ALERT",         # 4σ = 99.994% - Alert operators
    5.0: "CRITICAL",      # 5σ = 99.99994% - Prepare lockout
    6.0: "LOCKOUT"        # 6σ = 99.9999998% - Hardware lockout (Six Sigma standard)
}

# OLS Trend Detection settings
OLS_WINDOW_SIZE = 50  # Observations for trend line
OLS_SLOPE_THRESHOLD = 0.5  # Drift rate threshold per second

# Hardware lockout integration
LOCKOUT_REDIS_KEY = "AEGIS_HARDWARE_LOCKOUT"
LOCKOUT_TTL_SECONDS = 3600  # 1 hour default

class RealityAnchor:
    """Mock interface to physical sensors."""
    def get_sensor_data(self) -> Dict[str, List[float]]:
        # Simulate 3 redundant sensors for each metric to enable consensus
        return {
            "temperature_core": [
                45.0 + random.uniform(-2, 2),
                45.0 + random.uniform(-2, 2),
                95.0 if random.random() < 0.05 else 45.0 + random.uniform(-2, 2) # Occasional spoofed outlier
            ],
            "pressure_hydraulic": [
                1200.0 + random.uniform(-50, 50),
                1200.0 + random.uniform(-50, 50),
                1200.0 + random.uniform(-50, 50)
            ]
        }

import numpy as np
from collections import deque

class KalmanFilter1D:
    """
    Simple 1D Kalman Filter for predictive sensor tracking.
    State: [position, velocity]
    """
    def __init__(self, dt=0.1, u=0.0, std_acc=0.1, std_meas=0.1):
        self.dt = dt
        self.u = u
        self.std_acc = std_acc
        self.A = np.array([[1, dt],
                           [0, 1]])
        self.B = np.array([[dt**2/2], 
                           [dt]])
        self.H = np.array([[1, 0]])
        self.Q = np.array([[dt**4/4, dt**3/2],
                           [dt**3/2, dt**2]]) * std_acc**2
        self.R = std_meas**2
        self.P = np.eye(2)
        self.x = np.array([[0], 
                           [0]]) # Initial state

    def predict(self):
        # x = A*x + B*u
        self.x = np.dot(self.A, self.x) + self.B * self.u
        # P = A*P*A' + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0][0] # Return predicted position

    def update(self, z):
        # S = H*P*H' + R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # K = P*H'*inv(S)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # x = x + K*(z - H*x)
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        # P = P - K*H*P
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        return self.x[0][0] # Return updated position

class AdaptiveDriftMonitor:
    """
    Adaptive drift detection using Kalman Filtering, Z-score, and OLS Trend.
    Implements Six Sigma severity levels (Gap 3.2).
    """
    def __init__(self, window_size=50, sigma_threshold=3.0):
        self.history = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)  # For OLS trend
        self.sigma_threshold = sigma_threshold
        self.kalman = KalmanFilter1D(dt=0.1, std_meas=2.0)
        self.predicted_next = None
        self.start_time = None

    def is_drift_detected(self, current_reading: float) -> Tuple[bool, float, str, float]:
        """
        Check if the current reading is a statistically significant drift.
        
        Returns: 
            (is_drift, z_score, severity_level, ols_slope)
            - is_drift: Boolean if drift exceeds threshold
            - z_score: The sigma deviation
            - severity_level: One of WARNING/ALERT/CRITICAL/LOCKOUT
            - ols_slope: Trend rate (drift velocity)
        """
        import time
        
        if self.start_time is None:
            self.start_time = time.time()
        
        current_time = time.time() - self.start_time
        
        # 1. Update Kalman Filter
        if self.predicted_next is not None:
            prediction_error = abs(current_reading - self.predicted_next)
        else:
            prediction_error = 0.0
             
        smoothed_val = self.kalman.update(current_reading)
        self.predicted_next = self.kalman.predict()
        
        if len(self.history) < 10:
            self.history.append(current_reading)
            self.timestamps.append(current_time)
            return False, 0.0, "NOMINAL", 0.0

        # Calculate rolling statistics
        data = np.array(self.history)
        mean = np.mean(data)
        std = np.std(data)
        
        # Avoid division by zero
        if std < 1e-6: 
            std = 1e-6

        # Z-Score check
        z_score = abs(current_reading - mean) / std
        
        # OLS Trend Detection (Gap 3.2)
        ols_slope = 0.0
        if len(self.history) >= OLS_WINDOW_SIZE:
            try:
                times = np.array(self.timestamps)
                values = np.array(self.history)
                # Simple linear regression: y = mx + b
                n = len(times)
                sum_x = np.sum(times)
                sum_y = np.sum(values)
                sum_xy = np.sum(times * values)
                sum_x2 = np.sum(times ** 2)
                
                denominator = n * sum_x2 - sum_x ** 2
                if abs(denominator) > 1e-6:
                    ols_slope = (n * sum_xy - sum_x * sum_y) / denominator
            except Exception:
                ols_slope = 0.0
        
        self.history.append(current_reading)
        self.timestamps.append(current_time)
        
        # Determine Six Sigma severity level
        severity = "NOMINAL"
        for sigma_level in sorted(SIGMA_LEVELS.keys()):
            if z_score >= sigma_level:
                severity = SIGMA_LEVELS[sigma_level]
        
        # Is drift detected based on configured threshold?
        is_drift = z_score > self.sigma_threshold
        
        # Also trigger on rapid trend (OLS slope)
        if abs(ols_slope) > OLS_SLOPE_THRESHOLD and len(self.history) >= OLS_WINDOW_SIZE:
            logger.warning(f"⚠️ OLS Trend Alert: Slope={ols_slope:.4f}/s exceeds threshold")
            is_drift = True
            if severity == "NOMINAL":
                severity = "WARNING"
            
        return is_drift, z_score, severity, ols_slope

class DriftMonitor:
    """
    Monitors 'Proprioceptive Drift' - the divergence between the AI's internal model 
    (what it thinks it is doing) and the physical reality (sensor data).
    """
    def __init__(self):
        self.anchor = RealityAnchor()
        # Legacy static threshold for backwards compatibility
        self.drift_threshold = 0.01  
        self.lockout_mode = False
        # Level 5: Adaptive monitoring
        self.adaptive_monitors = {
            "temperature_core": AdaptiveDriftMonitor(window_size=100, sigma_threshold=4.5),
            "pressure_hydraulic": AdaptiveDriftMonitor(window_size=100, sigma_threshold=4.5)
        }

    async def monitor_loop(self):
        logger.info("👁️  Proprioceptive Drift Monitor ACTIVE (Level 5: Adaptive Mode)")
        from core.infrastructure.state_manager import RedisStateStore
        import json
        
        self.store = RedisStateStore()
        # Start background flusher
        asyncio.create_task(self._db_flusher_loop())
        
        while True:
            try:
                # 1. Get Ground Truth (Redundant Sensors)
                raw_sensors = self.anchor.get_sensor_data()
                
                # 1.1 Byzantine Sensor Consensus (Outlier Trimming)
                consensus_sensors = {}
                for key, readings in raw_sensors.items():
                    if len(readings) >= 3:
                        readings.sort()
                        consensus_sensors[key] = readings[len(readings)//2] 
                    else:
                        consensus_sensors[key] = sum(readings) / len(readings) if readings else 0
                
                # 1.5 Telemetry Ingestion (Redis Buffering)
                if self.store._redis:
                    await self.store._redis.rpush("telemetry_stream", json.dumps(consensus_sensors))
                
                # 2. Get LLM 'Intent' (Simulated)
                # In production, this comes from the Cognitive Engine's belief state
                llm_belief = {
                    "temperature_core": consensus_sensors["temperature_core"] * random.uniform(0.98, 1.02)
                }
                
                # 3. Calculate and Check Drift with Six Sigma Severity (Gap 3.2)
                for key, real_val in consensus_sensors.items():
                    if key in self.adaptive_monitors:
                        # Use Adaptive Kalman Monitor with Six Sigma severity
                        is_drift, z_score, severity, ols_slope = self.adaptive_monitors[key].is_drift_detected(real_val)
                        
                        # Log based on severity level
                        if severity == "WARNING":
                            logger.warning(f"⚠️ [{severity}] Drift in {key}: σ={z_score:.2f}, slope={ols_slope:.4f}/s")
                        elif severity == "ALERT":
                            logger.error(f"🚨 [{severity}] Drift in {key}: σ={z_score:.2f}, slope={ols_slope:.4f}/s")
                        elif severity == "CRITICAL":
                            logger.critical(f"⛔ [{severity}] Drift in {key}: σ={z_score:.2f}, slope={ols_slope:.4f}/s")
                        elif severity == "LOCKOUT":
                            logger.critical(f"💀 [{severity}] SIX SIGMA VIOLATION in {key}: σ={z_score:.2f}")
                        
                        if is_drift:
                            # NEURO-SYMBOLIC BRIDGE CHECK (Refinement B)
                            # Check if the "Mind" is also hallucinating (high uncertainty)
                            hallucination_score = 0.0
                            try:
                                h_score = self.store._redis.get("cognitive:hallucination_index") if self.store._redis else None
                                if h_score: hallucination_score = float(h_score)
                            except Exception: pass
                            
                            # Six Sigma Severity Matrix
                            if severity == "LOCKOUT":
                                # 6σ deviation - immediate hardware lockout
                                logger.critical(f"💀 SIX SIGMA LOCKOUT: {key} deviated {z_score:.2f}σ from baseline")
                                await self._trigger_lockout()
                                break
                            elif severity == "CRITICAL" and hallucination_score > 0.7:
                                # 5σ with hallucination - neuro-symbolic divergence
                                logger.critical("💀 CRITICAL: NEURO-SYMBOLIC DIVERGENCE. AI is hallucinating sensor data.")
                                logger.critical(f"   Sensor Divergence: {z_score:.2f}σ | Hallucination Index: {hallucination_score:.2f}")
                                await self._trigger_lockout()
                                break
                            elif severity in ["CRITICAL", "ALERT"]:
                                # Try evolutionary calibration first
                                try:
                                    from core.governance.evolution_engine import get_patch_engine
                                    patch_engine = get_patch_engine()
                                    await patch_engine.propose_calibration_update(key, {
                                        "drift_value": real_val, 
                                        "z_score": z_score,
                                        "severity": severity,
                                        "ols_slope": ols_slope
                                    })
                                except Exception as e:
                                    logger.error(f"Failed to trigger evolution: {e}")
                    
                    # Fallback/Parallel check for Mean Percentage Error drift
                    if key in llm_belief:
                        belief_val = llm_belief[key]
                        mpe_drift = abs(belief_val - real_val) / real_val if real_val != 0 else 0
                        if mpe_drift > self.drift_threshold:
                            logger.critical(f"⛔ STATIC DRIFT DETECTED in {key}: {mpe_drift:.2%} > {self.drift_threshold:.2%}")
                            await self._trigger_lockout()
                            break
                
                if not self.lockout_mode:
                    if random.random() < 0.05: # Reduced logging frequency for production
                        logger.info("✅ Proprioception Stable. Consensus sensors within adaptive bounds.")
                        
                await asyncio.sleep(0.1) # 10Hz Monitor Tick Rate
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {str(e)}")
                await asyncio.sleep(5)

    async def _db_flusher_loop(self):
        """
        Periodic async dump to Postgres (Mocked).
        Prevents write-heavy thrashing by batching updates.
        """
        logger.info("💾 Database Flusher Loop Started")
        while True:
            try:
                await asyncio.sleep(10) # 10s batch window
                
                if not self.store._redis:
                    continue
                    
                # Atomic batch retrieval
                # LPOP 100 items or LRANGE based on need. 
                # Using Pipeline for atomicity in real prod, here simple fetch.
                batch_size = 100
                items = await self.store._redis.lrange("telemetry_stream", 0, batch_size - 1)
                
                if items:
                    # Mock DB Write
                    logger.info(f"🧱 Bulk Inserting {len(items)} sensor records to Postgres TimescaleDB...")
                    
                    # Trim processed items
                    await self.store._redis.ltrim("telemetry_stream", len(items), -1)
                    
            except Exception as e:
                logger.error(f"❌ DB Flush Error: {e}")
                await asyncio.sleep(5)

    def _calculate_drift(self, reality: Dict[str, float], belief: Dict[str, float]) -> float:
        """Calculate mean percentage error between belief and reality."""
        total_drift = 0
        count = 0
        
        for key, perceived_val in belief.items():
            if key in reality:
                real_val = reality[key]
                if real_val == 0: continue
                # Absolute percentage error
                drift = abs(perceived_val - real_val) / real_val
                total_drift += drift
                count += 1
        
        return total_drift / count if count > 0 else 0

    async def _trigger_lockout(self):
        if not self.lockout_mode:
            logger.critical("🔒 INITIATING HARDWARE LOCKOUT. AI DISCONNECTED FROM ACTUATORS.")
            self.lockout_mode = True
            
            try:
                # Use Redis to propagate lockout to all pods
                from core.infrastructure.state_manager import RedisStateStore
                store = RedisStateStore()
                # Set with 1 hour TTL, must be manually cleared
                await store.set("AEGIS_SYSTEM_LOCKOUT", "TRUE", ttl=3600)
                logger.critical("✅ GLOBAL LOCKOUT SIGNAL SENT TO REDIS")
            except Exception as e:
                logger.error(f"❌ Failed to propagate lockout signal: {e}")

async def main():
    monitor = DriftMonitor()
    await monitor.monitor_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Monitor stopped.")
