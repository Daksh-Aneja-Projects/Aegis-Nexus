"""
Production Kalman Filter with Joseph Stabilized Form
Prevents numerical divergence in 24/7 continuous operation.

PRODUCTION HARDENING:
- Joseph Stabilized Form for covariance updates (guarantees positive-definiteness)
- Mahalanobis gating for Chi-squared outlier rejection
- Covariance symmetry enforcement after each update
- Eigenvalue monitoring to detect early signs of divergence

This addresses the numerical instability vulnerability where cumulative
floating-point errors cause the filter to diverge in long-running systems.
"""

import numpy as np
from scipy import stats
from scipy.linalg import cholesky, LinAlgError
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FilterState:
    """State vector and covariance for Kalman filter."""
    state: np.ndarray           # State estimate vector
    covariance: np.ndarray      # Covariance matrix (positive-definite)
    timestamp: datetime
    
    def __post_init__(self):
        # Ensure proper shapes
        self.state = np.atleast_1d(self.state).astype(np.float64)
        self.covariance = np.atleast_2d(self.covariance).astype(np.float64)


@dataclass
class MeasurementGateResult:
    """Result of Mahalanobis gating check."""
    accepted: bool
    mahalanobis_distance: float
    chi_squared_threshold: float
    confidence_level: float
    reason: str


class StabilizedKalmanFilter:
    """
    Production-grade Kalman filter with numerical stability guarantees.
    
    Key Features:
    1. Joseph Stabilized Form: P = (I-KH)P(I-KH)' + KRK'
       - Guarantees positive-definiteness even with precision errors
    2. Mahalanobis Gating: χ² distance-based outlier rejection
       - Prevents sensor glitches from corrupting state estimate
    3. Symmetry Enforcement: P = (P + P') / 2 after updates
       - Corrects accumulated asymmetry from floating-point ops
    """
    
    # Default thresholds for χ² gating (degrees of freedom -> 99% confidence)
    CHI_SQUARED_THRESHOLDS = {
        1: 6.635,
        2: 9.210,
        3: 11.345,  # Default for 3D state (position, velocity, acceleration)
        4: 13.277,
        5: 15.086,
        6: 16.812,
    }
    
    def __init__(
        self,
        state_dim: int,
        measurement_dim: int,
        process_noise: Optional[np.ndarray] = None,
        measurement_noise: Optional[np.ndarray] = None,
        confidence_level: float = 0.99
    ):
        """
        Initialize the stabilized Kalman filter.
        
        Args:
            state_dim: Dimension of state vector
            measurement_dim: Dimension of measurement vector
            process_noise: Process noise covariance (Q)
            measurement_noise: Measurement noise covariance (R)
            confidence_level: Confidence level for χ² gating (default 99%)
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.confidence_level = confidence_level
        
        # Initialize covariance matrices
        self.Q = process_noise if process_noise is not None else np.eye(state_dim) * 0.01
        self.R = measurement_noise if measurement_noise is not None else np.eye(measurement_dim) * 0.1
        
        # Default transition and measurement matrices (identity for simplicity)
        self.F = np.eye(state_dim)  # State transition
        self.H = np.eye(measurement_dim, state_dim)  # Measurement
        
        # Calculate χ² threshold for gating
        dof = measurement_dim
        self.chi_squared_threshold = self._get_chi_squared_threshold(dof, confidence_level)
        
        # State initialization
        self.current_state: Optional[FilterState] = None
        
        # Statistics
        self.update_count = 0
        self.rejected_measurements = 0
        self.last_eigenvalues: Optional[np.ndarray] = None
        
        # Level 5: Eigenvalue Monitoring Configuration
        self.eigenvalue_check_interval = 100  # Check every N updates
        self.condition_number_threshold = 1e12  # Maximum acceptable condition number
        self.min_eigenvalue_threshold = 1e-15  # Minimum eigenvalue (detect collapse)
        self.max_eigenvalue_threshold = 1e15  # Maximum eigenvalue (detect explosion)
        self.eigenvalue_history: List[Dict] = []  # History for trend analysis
        
        logger.info(f"🔬 Stabilized Kalman Filter initialized: state_dim={state_dim}, "
                   f"measurement_dim={measurement_dim}, χ² threshold={self.chi_squared_threshold:.3f}")
    
    def _get_chi_squared_threshold(self, dof: int, confidence: float) -> float:
        """Get χ² threshold for given DOF and confidence level."""
        if dof in self.CHI_SQUARED_THRESHOLDS and confidence == 0.99:
            return self.CHI_SQUARED_THRESHOLDS[dof]
        return stats.chi2.ppf(confidence, dof)
    
    def initialize(self, initial_state: np.ndarray, initial_covariance: Optional[np.ndarray] = None):
        """
        Initialize the filter with an initial state estimate.
        
        Args:
            initial_state: Initial state vector
            initial_covariance: Initial covariance (default: identity * 1.0)
        """
        state = np.atleast_1d(initial_state).astype(np.float64)
        
        if initial_covariance is not None:
            cov = np.atleast_2d(initial_covariance).astype(np.float64)
        else:
            cov = np.eye(self.state_dim)
        
        self.current_state = FilterState(
            state=state,
            covariance=cov,
            timestamp=datetime.utcnow()
        )
        
        logger.debug(f"Filter initialized with state: {state}")
    
    def predict(self, control_input: Optional[np.ndarray] = None) -> FilterState:
        """
        Prediction step: project state and covariance forward.
        
        Args:
            control_input: Optional control vector (B*u term)
            
        Returns:
            Predicted state
        """
        if self.current_state is None:
            raise ValueError("Filter not initialized. Call initialize() first.")
        
        x = self.current_state.state
        P = self.current_state.covariance
        
        # State prediction: x_pred = F*x + B*u
        x_pred = self.F @ x
        if control_input is not None:
            x_pred += control_input
        
        # Covariance prediction: P_pred = F*P*F' + Q
        P_pred = self.F @ P @ self.F.T + self.Q
        
        # Enforce symmetry
        P_pred = self._enforce_symmetry(P_pred)
        
        self.current_state = FilterState(
            state=x_pred,
            covariance=P_pred,
            timestamp=datetime.utcnow()
        )
        
        return self.current_state
    
    def mahalanobis_gate(
        self,
        measurement: np.ndarray,
        predicted_measurement: np.ndarray,
        innovation_covariance: np.ndarray
    ) -> MeasurementGateResult:
        """
        Perform Mahalanobis distance-based gating for outlier rejection.
        
        The Mahalanobis distance measures how many standard deviations
        the measurement is from the predicted value, accounting for
        correlations between dimensions.
        
        d² = (z - ẑ)' * S⁻¹ * (z - ẑ)
        
        Args:
            measurement: Actual measurement vector (z)
            predicted_measurement: Predicted measurement (ẑ = H*x)
            innovation_covariance: Innovation covariance (S = H*P*H' + R)
            
        Returns:
            MeasurementGateResult with accept/reject decision
        """
        innovation = measurement - predicted_measurement
        
        try:
            # Compute Mahalanobis distance squared
            S_inv = np.linalg.inv(innovation_covariance)
            d_squared = float(innovation.T @ S_inv @ innovation)
            d = np.sqrt(d_squared)
            
            # Chi-squared test
            accepted = d_squared < self.chi_squared_threshold
            
            if accepted:
                reason = f"Within gate: d²={d_squared:.2f} < χ²={self.chi_squared_threshold:.2f}"
            else:
                reason = f"REJECTED: d²={d_squared:.2f} >= χ²={self.chi_squared_threshold:.2f}"
                logger.warning(f"📊 Measurement rejected by Mahalanobis gate: {reason}")
            
            return MeasurementGateResult(
                accepted=accepted,
                mahalanobis_distance=d,
                chi_squared_threshold=self.chi_squared_threshold,
                confidence_level=self.confidence_level,
                reason=reason
            )
            
        except np.linalg.LinAlgError:
            logger.error("❌ Singular innovation covariance - rejecting measurement")
            return MeasurementGateResult(
                accepted=False,
                mahalanobis_distance=float('inf'),
                chi_squared_threshold=self.chi_squared_threshold,
                confidence_level=self.confidence_level,
                reason="REJECTED: Singular innovation covariance"
            )
    
    def update_covariance_joseph(
        self,
        K: np.ndarray,
        H: np.ndarray,
        P: np.ndarray,
        R: np.ndarray
    ) -> np.ndarray:
        """
        Joseph Stabilized Form for covariance update.
        
        P_new = (I - K*H) * P * (I - K*H)' + K * R * K'
        
        This formulation guarantees that P_new remains positive-definite
        even in the presence of numerical precision errors, because it's
        expressed as the sum of two symmetric positive semi-definite matrices.
        
        Args:
            K: Kalman gain
            H: Measurement matrix
            P: Prior covariance
            R: Measurement noise covariance
            
        Returns:
            Updated covariance (guaranteed positive-definite)
        """
        I = np.eye(P.shape[0])
        I_KH = I - K @ H
        
        # Joseph form: sum of two quadratic forms
        P_new = I_KH @ P @ I_KH.T + K @ R @ K.T
        
        # Enforce symmetry to eliminate accumulated asymmetry
        P_new = self._enforce_symmetry(P_new)
        
        # Validate positive-definiteness
        if not self._is_positive_definite(P_new):
            logger.warning("⚠️ Covariance not positive-definite after Joseph update - applying correction")
            P_new = self._make_positive_definite(P_new)
        
        return P_new
    
    def _enforce_symmetry(self, P: np.ndarray) -> np.ndarray:
        """Enforce matrix symmetry: P = (P + P') / 2"""
        return (P + P.T) / 2
    
    def _is_positive_definite(self, P: np.ndarray) -> bool:
        """Check if matrix is positive-definite using Cholesky decomposition."""
        try:
            cholesky(P)
            return True
        except LinAlgError:
            return False
    
    def _make_positive_definite(self, P: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
        """
        Force matrix to be positive-definite by adjusting eigenvalues.
        
        This is a recovery mechanism for edge cases where numerical
        errors cause slight negative eigenvalues.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(P)
        
        # Set minimum eigenvalue floor
        eigenvalues = np.maximum(eigenvalues, epsilon)
        
        # Reconstruct matrix
        P_corrected = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        self.last_eigenvalues = eigenvalues
        return P_corrected
    
    def update(
        self,
        measurement: np.ndarray,
        apply_gating: bool = True
    ) -> Tuple[FilterState, MeasurementGateResult]:
        """
        Update step with Joseph stabilization and Mahalanobis gating.
        
        Args:
            measurement: Measurement vector
            apply_gating: If True, apply Mahalanobis gate (recommended)
            
        Returns:
            Tuple of (updated state, gating result)
        """
        if self.current_state is None:
            raise ValueError("Filter not initialized. Call initialize() first.")
        
        z = np.atleast_1d(measurement).astype(np.float64)
        x = self.current_state.state
        P = self.current_state.covariance
        
        # Predicted measurement
        z_pred = self.H @ x
        
        # Innovation covariance: S = H*P*H' + R
        S = self.H @ P @ self.H.T + self.R
        
        # Mahalanobis gating
        gate_result = self.mahalanobis_gate(z, z_pred, S)
        
        if apply_gating and not gate_result.accepted:
            # Reject measurement - state unchanged
            self.rejected_measurements += 1
            return self.current_state, gate_result
        
        # Kalman gain: K = P*H' * S^(-1)
        try:
            K = P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.error("❌ Singular innovation covariance - skipping update")
            self.rejected_measurements += 1
            return self.current_state, gate_result
        
        # State update: x_new = x + K*(z - z_pred)
        innovation = z - z_pred
        x_new = x + K @ innovation
        
        # Covariance update using Joseph Stabilized Form
        P_new = self.update_covariance_joseph(K, self.H, P, self.R)
        
        self.current_state = FilterState(
            state=x_new,
            covariance=P_new,
            timestamp=datetime.utcnow()
        )
        
        self.update_count += 1
        
        if self.update_count % 1000 == 0:
            logger.info(f"📊 Kalman filter: {self.update_count} updates, "
                       f"{self.rejected_measurements} rejected, "
                       f"trace(P)={np.trace(P_new):.4f}")
        
        # Eigenvalue Monitoring (Level 5 - Automated Divergence Detection)
        if self.update_count % self.eigenvalue_check_interval == 0:
            divergence_status = self.check_filter_divergence()
            if divergence_status['is_diverging']:
                logger.warning(f"⚠️ FILTER DIVERGENCE DETECTED: {divergence_status['reason']}")
        
        return self.current_state, gate_result
    
    def check_filter_divergence(self) -> Dict[str, Any]:
        """
        Check for early signs of filter divergence.
        
        Level 5 Production Hardening:
        - Monitors condition number (ratio of max/min eigenvalue)
        - Detects covariance explosion or collapse
        - Provides actionable diagnostics for operators
        
        Returns:
            Dict with divergence status and diagnostics
        """
        if self.current_state is None:
            return {"is_diverging": False, "reason": "Filter not initialized"}
        
        P = self.current_state.covariance
        eigenvalues = np.linalg.eigvalsh(P)
        self.last_eigenvalues = eigenvalues
        
        min_eig = float(np.min(eigenvalues))
        max_eig = float(np.max(eigenvalues))
        condition_number = max_eig / max(min_eig, 1e-15)
        trace_val = float(np.trace(P))
        
        result = {
            "is_diverging": False,
            "reason": None,
            "min_eigenvalue": min_eig,
            "max_eigenvalue": max_eig,
            "condition_number": condition_number,
            "trace": trace_val,
            "update_count": self.update_count
        }
        
        # Divergence Criteria (Level 5 thresholds)
        if condition_number > self.condition_number_threshold:
            result["is_diverging"] = True
            result["reason"] = f"Condition number {condition_number:.2e} exceeds threshold {self.condition_number_threshold:.2e}"
        
        if min_eig < self.min_eigenvalue_threshold:
            result["is_diverging"] = True  
            result["reason"] = f"Min eigenvalue {min_eig:.2e} below threshold {self.min_eigenvalue_threshold:.2e} (covariance collapse)"
        
        if max_eig > self.max_eigenvalue_threshold:
            result["is_diverging"] = True
            result["reason"] = f"Max eigenvalue {max_eig:.2e} exceeds threshold {self.max_eigenvalue_threshold:.2e} (covariance explosion)"
        
        # Store history for trend analysis
        self.eigenvalue_history.append({
            "update": self.update_count,
            "condition_number": condition_number,
            "min_eig": min_eig,
            "max_eig": max_eig
        })
        
        # Keep only last 100 entries
        if len(self.eigenvalue_history) > 100:
            self.eigenvalue_history.pop(0)
        
        return result
    
    def get_stability_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive stability metrics for dashboard integration.
        
        Returns:
            Dict with stability metrics including trend analysis
        """
        divergence_status = self.check_filter_divergence()
        
        # Calculate trend if we have enough history
        trend = "stable"
        if len(self.eigenvalue_history) >= 10:
            recent = self.eigenvalue_history[-10:]
            condition_trend = [h["condition_number"] for h in recent]
            if condition_trend[-1] > condition_trend[0] * 1.5:
                trend = "degrading"
            elif condition_trend[-1] < condition_trend[0] * 0.5:
                trend = "improving"
        
        return {
            **divergence_status,
            "trend": trend,
            "history_length": len(self.eigenvalue_history),
            "filter_healthy": not divergence_status["is_diverging"],
            "recommended_action": self._recommend_action(divergence_status)
        }
    
    def _recommend_action(self, divergence_status: Dict) -> str:
        """Recommend action based on divergence status."""
        if not divergence_status["is_diverging"]:
            return "none"
        
        if "collapse" in str(divergence_status.get("reason", "")):
            return "increase_process_noise"
        elif "explosion" in str(divergence_status.get("reason", "")):
            return "reinitialize_filter"
        else:
            return "manual_review"
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get filter diagnostics for monitoring."""
        diagnostics = {
            "update_count": self.update_count,
            "rejected_measurements": self.rejected_measurements,
            "rejection_rate": self.rejected_measurements / max(1, self.update_count + self.rejected_measurements),
            "chi_squared_threshold": self.chi_squared_threshold,
        }
        
        if self.current_state is not None:
            P = self.current_state.covariance
            eigenvalues = np.linalg.eigvalsh(P)
            diagnostics.update({
                "covariance_trace": float(np.trace(P)),
                "covariance_det": float(np.linalg.det(P)),
                "min_eigenvalue": float(np.min(eigenvalues)),
                "max_eigenvalue": float(np.max(eigenvalues)),
                "condition_number": float(np.max(eigenvalues) / max(np.min(eigenvalues), 1e-15)),
                "is_positive_definite": self._is_positive_definite(P),
            })
            
            # Add stability metrics (Level 5)
            stability = self.get_stability_metrics()
            diagnostics["filter_healthy"] = stability["filter_healthy"]
            diagnostics["stability_trend"] = stability["trend"]
        
        return diagnostics


# Global instance
fusion_filter: Optional[StabilizedKalmanFilter] = None


async def initialize_fusion_engine(
    state_dim: int = 3,
    measurement_dim: int = 3,
    process_noise_scale: float = 0.01,
    measurement_noise_scale: float = 0.1
) -> bool:
    """
    Initialize the global stabilized Kalman filter.
    
    Returns:
        bool: Success status
    """
    global fusion_filter
    
    try:
        fusion_filter = StabilizedKalmanFilter(
            state_dim=state_dim,
            measurement_dim=measurement_dim,
            process_noise=np.eye(state_dim) * process_noise_scale,
            measurement_noise=np.eye(measurement_dim) * measurement_noise_scale
        )
        
        # Initialize with zero state
        fusion_filter.initialize(np.zeros(state_dim))
        
        logger.info("✅ Fusion engine initialized with Joseph-stabilized Kalman filter")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize fusion engine: {e}")
        return False


def get_fusion_filter() -> StabilizedKalmanFilter:
    """Get the global fusion filter instance."""
    if fusion_filter is None:
        raise RuntimeError("Fusion engine not initialized. Call initialize_fusion_engine() first.")
    return fusion_filter
