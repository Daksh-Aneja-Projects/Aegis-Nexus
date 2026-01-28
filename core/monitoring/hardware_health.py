"""
Aegis Nexus - Hardware Health Monitor (Proprioceptive Drift)
Level 5 Safety Feature

This sidecar service continuously monitors the "Mahalanobis Distance" between
the Digital Twin's expected state and the Raw Sensor Data.

If the drift exceeds the safety threshold, it triggers a HARDWARE_LOCKOUT
in Redis, physically preventing the AI from executing dangerous actions.
"""
import os
import time
import json
import logging
import asyncio
import numpy as np
import redis
from typing import List, Dict

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HardwareHealthMonitor")

# Configuration
REDIS_HOST = os.getenv("AEGIS_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("AEGIS_REDIS_PORT", 6379))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", 2.5))  # Sigma deviation
CHECK_INTERVAL_MS = 100  # 10Hz monitoring

# Keys
LOCKOUT_KEY = "AEGIS_SYSTEM_LOCKOUT"
SENSOR_STREAM_KEY = "aegis:sensor_stream"
DIGITAL_TWIN_KEY = "aegis:digital_twin:state"

class ProprioceptiveDriftMonitor:
    def __init__(self):
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        # Initialize historical data for covariance matrix
        # In a real system, this would load from a database
        self.history: List[np.ndarray] = []
        self.max_history = 1000
        
        # Mock initial healthy state covariance (Identity for start)
        self.inv_covariance = np.eye(3) 
        self.mean_vector = np.zeros(3)

    def calculate_mahalanobis_distance(self, x: np.ndarray) -> float:
        """
        Calculate Mahalanobis Distance: D = sqrt((x - u)^T * S^-1 * (x - u))
        """
        delta = x - self.mean_vector
        term = np.dot(delta.T, self.inv_covariance)
        distance = np.sqrt(np.dot(term, delta))
        return distance

    def update_model(self, new_data: np.ndarray):
        """Update the statistical model (Online Learning)."""
        self.history.append(new_data)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Recompute Stats (Simplified for demo, usually use incremental updates)
        if len(self.history) > 10:
            data_matrix = np.array(self.history)
            self.mean_vector = np.mean(data_matrix, axis=0)
            try:
                covariance = np.cov(data_matrix, rowvar=False)
                # Add regularization to prevent singular matrix
                covariance += np.eye(covariance.shape[0]) * 1e-6
                self.inv_covariance = np.linalg.inv(covariance)
            except np.linalg.LinAlgError as e:
                logger.warning(f"⚠️  DEGRADED MODE: Could not compute covariance inverse: {e}")
                logger.info("📉 System operating with reduced statistical confidence")
                # Keep using previous inverse covariance in degraded mode
                pass

    def run(self):
        logger.info(f"👁️  Hardware Health Monitor Active. Threshold: {DRIFT_THRESHOLD}")
        
        # Clear any stale lockouts on startup
        self.redis.delete(LOCKOUT_KEY)
        
        while True:
            try:
                # 1. Fetch latest sensor data (Simulated read from Redis Stream or Key)
                # For demo, we read a mocked key or generate synthetic data
                
                # Mock: Generate "Real" vs "Expected"
                # In prod, this comes from the Fusion Engine
                raw_sensor_data = np.random.normal(0, 1, 3) # Normal operation
                
                # Introduce Drift/Fault randomly (Simulating sensor wear)
                # if np.random.rand() > 0.99:
                #     raw_sensor_data = np.random.normal(5, 2, 3) # ERROR STATE
                
                # 2. Calculate Drift
                distance = self.calculate_mahalanobis_distance(raw_sensor_data)
                
                # 3. Update Model
                self.update_model(raw_sensor_data)
                
                # 4. Safety Check
                if distance > DRIFT_THRESHOLD:
                    logger.critical(f"🚨 PROPRIOCEPTIVE DRIFT DETECTED! Distance: {distance:.2f} > {DRIFT_THRESHOLD}")
                    
                    # TRIGGER HARDWARE LOCKOUT
                    self.redis.set(LOCKOUT_KEY, "TRUE", ex=60) # Lock for 60s or until reset
                    
                    # Publish Alert
                    alert = {
                        "type": "lockdown_alert",
                        "reason": "Proprioceptive Drift - Hardware/Digital Twin Mismatch",
                        "drift_metric": distance,
                        "timestamp": time.time()
                    }
                    self.redis.publish("aegis_audit_events", json.dumps(alert))
                    
                else:
                    # Clean bill of health - only log every 100 iterations
                    # logger.debug(f"✅ System Healthy. Drift: {distance:.2f}")
                    pass
                
                time.sleep(CHECK_INTERVAL_MS / 1000.0)
                
            except Exception as e:
                logger.error(f"❌ Error in monitor loop: {e}")
                logger.warning("⚠️  DEGRADED MODE: Hardware health monitoring degraded, continuing with reduced accuracy")
                time.sleep(1)

if __name__ == "__main__":
    monitor = ProprioceptiveDriftMonitor()
    monitor.run()
