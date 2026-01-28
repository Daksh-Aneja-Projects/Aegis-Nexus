"""
Anomaly Detector for Aegis Nexus
Statistical outlier detection using Ordinary Least Squares modeling.

This module implements advanced anomaly detection using OLS regression
models and statistical process control methods.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection analysis"""
    detection_id: str
    data_series: str
    anomalies_found: List[Dict[str, Any]]
    anomaly_count: int
    confidence_level: float
    detection_method: str
    statistical_summary: Dict[str, float]
    recommendations: List[str]
    processing_time_ms: float
    timestamp: datetime

@dataclass
class AnomalyPoint:
    """Individual anomaly point with details"""
    index: int
    value: float
    expected_value: float
    residual: float
    z_score: float
    severity: str  # low, medium, high, critical
    anomaly_type: str  # statistical, contextual, seasonal
    description: str

class AnomalyDetector:
    """
    Advanced anomaly detection system using statistical methods.
    
    Implements OLS modeling, residual analysis, and multiple detection
    algorithms for identifying outliers and anomalous patterns.
    """
    
    def __init__(self, sensitivity: float = 0.95):
        """
        Initialize the anomaly detector.
        
        Args:
            sensitivity: Detection sensitivity threshold (0.0-1.0)
        """
        self.sensitivity = sensitivity
        self.models: Dict[str, LinearRegression] = {}
        self.thresholds: Dict[str, float] = {}
        self.detection_history: List[AnomalyDetectionResult] = []
        self.max_history_size = 500
        
    async def initialize(self) -> bool:
        """Initialize the anomaly detection system."""
        try:
            logger.info("🔍 Initializing anomaly detection system...")
            
            logger.info("✅ Anomaly detection system initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize anomaly detection: {str(e)}")
            return False
    
    async def detect_anomalies(
        self, 
        data: Union[List[float], np.ndarray, pd.Series], 
        series_name: str = "default_series",
        method: str = "auto"
    ) -> AnomalyDetectionResult:
        """
        Detect anomalies in a data series.
        
        Args:
            data: Time series data to analyze
            series_name: Name identifier for the data series
            method: Detection method ('auto', 'statistical', 'ols', 'combined')
            
        Returns:
            AnomalyDetectionResult with findings
        """
        start_time = datetime.utcnow()
        logger.debug(f"🔍 Detecting anomalies in series: {series_name}")
        
        try:
            # Generate detection ID
            detection_id = f"detect_{series_name}_{int(start_time.timestamp())}"
            
            # Convert data to numpy array
            if isinstance(data, pd.Series):
                np_data = data.values
            elif isinstance(data, list):
                np_data = np.array(data, dtype=float)
            else:
                np_data = np.array(data, dtype=float)
            
            # Remove NaN values
            clean_data = np_data[~np.isnan(np_data)]
            
            if len(clean_data) < 10:  # Minimum data requirement
                return AnomalyDetectionResult(
                    detection_id=detection_id,
                    data_series=series_name,
                    anomalies_found=[],
                    anomaly_count=0,
                    confidence_level=0.0,
                    detection_method=method,
                    statistical_summary={'error': 'insufficient_data'},
                    recommendations=['Insufficient data for reliable anomaly detection'],
                    processing_time_ms=0.0,
                    timestamp=start_time
                )
            
            # Choose detection method
            if method == "auto":
                method = await self._select_optimal_method(clean_data)
            
            # Perform anomaly detection
            anomalies = await self._perform_detection(clean_data, method, series_name)
            
            # Calculate statistical summary
            statistical_summary = await self._calculate_statistical_summary(clean_data, anomalies)
            
            # Generate recommendations
            recommendations = await self._generate_detection_recommendations(anomalies, statistical_summary)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Calculate confidence level
            confidence_level = await self._calculate_detection_confidence(anomalies, clean_data)
            
            # Create result
            result = AnomalyDetectionResult(
                detection_id=detection_id,
                data_series=series_name,
                anomalies_found=[vars(anomaly) for anomaly in anomalies],
                anomaly_count=len(anomalies),
                confidence_level=confidence_level,
                detection_method=method,
                statistical_summary=statistical_summary,
                recommendations=recommendations,
                processing_time_ms=processing_time,
                timestamp=datetime.utcnow()
            )
            
            # Add to history
            self.detection_history.append(result)
            
            # Maintain history size
            if len(self.detection_history) > self.max_history_size:
                self.detection_history = self.detection_history[-self.max_history_size:]
            
            logger.debug(f"🔍 Anomaly detection completed: {len(anomalies)} anomalies found "
                        f"in {processing_time:.2f}ms")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(f"❌ Anomaly detection failed: {str(e)}")
            
            return AnomalyDetectionResult(
                detection_id=f"error_{int(datetime.utcnow().timestamp())}",
                data_series=series_name,
                anomalies_found=[],
                anomaly_count=0,
                confidence_level=0.0,
                detection_method=method,
                statistical_summary={'error': str(e)},
                recommendations=['Anomaly detection failed - manual review required'],
                processing_time_ms=processing_time,
                timestamp=datetime.utcnow()
            )
    
    async def _select_optimal_method(self, data: np.ndarray) -> str:
        """Automatically select the optimal detection method."""
        # Simple heuristic - in practice this would be more sophisticated
        data_length = len(data)
        data_variance = np.var(data)
        
        if data_length > 100 and data_variance > 0:
            return "combined"  # Use combination of methods for robust detection
        elif data_length > 50:
            return "ols"  # OLS method for moderate-sized datasets
        else:
            return "statistical"  # Statistical method for small datasets
    
    async def _perform_detection(
        self, 
        data: np.ndarray, 
        method: str, 
        series_name: str
    ) -> List[AnomalyPoint]:
        """Perform anomaly detection using specified method."""
        anomalies = []
        
        if method == "statistical":
            anomalies = await self._statistical_detection(data)
        elif method == "ols":
            anomalies = await self._ols_detection(data, series_name)
        elif method == "combined":
            # Combine multiple detection methods
            stat_anomalies = await self._statistical_detection(data)
            ols_anomalies = await self._ols_detection(data, series_name)
            
            # Merge and deduplicate anomalies
            anomalies = await self._merge_anomalies(stat_anomalies, ols_anomalies)
        
        return anomalies
    
    async def _statistical_detection(self, data: np.ndarray) -> List[AnomalyPoint]:
        """Perform statistical anomaly detection."""
        anomalies = []
        
        # Calculate basic statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # Z-score based detection
        z_scores = np.abs((data - mean_val) / std_val) if std_val > 0 else np.zeros_like(data)
        
        # Adaptive threshold based on sensitivity
        threshold = stats.norm.ppf(1 - (1 - self.sensitivity) / 2)
        
        # Identify anomalies
        for i, (value, z_score) in enumerate(zip(data, z_scores)):
            if z_score > threshold and std_val > 0:
                # Determine severity
                if z_score > threshold * 2:
                    severity = "critical"
                elif z_score > threshold * 1.5:
                    severity = "high"
                elif z_score > threshold * 1.2:
                    severity = "medium"
                else:
                    severity = "low"
                
                anomaly = AnomalyPoint(
                    index=i,
                    value=float(value),
                    expected_value=float(mean_val),
                    residual=float(value - mean_val),
                    z_score=float(z_score),
                    severity=severity,
                    anomaly_type="statistical",
                    description=f"Statistical outlier (z-score: {z_score:.2f})"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _ols_detection(self, data: np.ndarray, series_name: str) -> List[AnomalyPoint]:
        """Perform OLS regression-based anomaly detection."""
        anomalies = []
        
        try:
            # Create time indices
            indices = np.arange(len(data)).reshape(-1, 1)
            
            # Fit OLS model
            model = LinearRegression()
            model.fit(indices, data)
            
            # Store model for future use
            self.models[series_name] = model
            
            # Calculate predictions and residuals
            predictions = model.predict(indices)
            residuals = data - predictions
            
            # Calculate residual statistics
            residual_mean = np.mean(residuals)
            residual_std = np.std(residuals)
            
            # Store threshold
            threshold_percentile = self.sensitivity * 100
            residual_threshold = np.percentile(np.abs(residuals), threshold_percentile)
            self.thresholds[series_name] = residual_threshold
            
            # Detect anomalies based on residuals
            for i, (value, pred, residual) in enumerate(zip(data, predictions, residuals)):
                abs_residual = abs(residual)
                
                if abs_residual > residual_threshold and residual_std > 0:
                    # Calculate standardized residual
                    std_residual = abs_residual / residual_std
                    
                    # Determine severity
                    if std_residual > 3.0:
                        severity = "critical"
                    elif std_residual > 2.5:
                        severity = "high"
                    elif std_residual > 2.0:
                        severity = "medium"
                    else:
                        severity = "low"
                    
                    anomaly = AnomalyPoint(
                        index=i,
                        value=float(value),
                        expected_value=float(pred),
                        residual=float(residual),
                        z_score=float(std_residual),
                        severity=severity,
                        anomaly_type="ols_regression",
                        description=f"OLS regression outlier (residual: {residual:.2f})"
                    )
                    anomalies.append(anomaly)
        
        except Exception as e:
            logger.warning(f"OLS detection failed: {str(e)}")
        
        return anomalies
    
    async def _merge_anomalies(
        self, 
        anomalies1: List[AnomalyPoint], 
        anomalies2: List[AnomalyPoint]
    ) -> List[AnomalyPoint]:
        """Merge and deduplicate anomalies from different methods."""
        # Combine anomalies
        all_anomalies = anomalies1 + anomalies2
        
        # Deduplicate by index
        unique_anomalies = {}
        for anomaly in all_anomalies:
            if anomaly.index not in unique_anomalies:
                unique_anomalies[anomaly.index] = anomaly
            else:
                # Keep the more severe anomaly
                existing = unique_anomalies[anomaly.index]
                severity_order = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
                if severity_order[anomaly.severity] > severity_order[existing.severity]:
                    unique_anomalies[anomaly.index] = anomaly
        
        return list(unique_anomalies.values())
    
    async def _calculate_statistical_summary(
        self, 
        data: np.ndarray, 
        anomalies: List[AnomalyPoint]
    ) -> Dict[str, float]:
        """Calculate statistical summary of the analysis."""
        if len(data) == 0:
            return {}
        
        # Basic statistics
        summary = {
            'data_points': len(data),
            'anomalies_detected': len(anomalies),
            'anomaly_rate': len(anomalies) / len(data) if len(data) > 0 else 0,
            'data_mean': float(np.mean(data)),
            'data_std': float(np.std(data)),
            'data_min': float(np.min(data)),
            'data_max': float(np.max(data)),
            'data_median': float(np.median(data))
        }
        
        # Anomaly statistics
        if anomalies:
            anomaly_values = [a.value for a in anomalies]
            anomaly_indices = [a.index for a in anomalies]
            
            summary.update({
                'anomaly_mean': float(np.mean(anomaly_values)),
                'anomaly_std': float(np.std(anomaly_values)),
                'first_anomaly_index': min(anomaly_indices),
                'last_anomaly_index': max(anomaly_indices),
                'severity_distribution': self._calculate_severity_distribution(anomalies)
            })
        
        return summary
    
    def _calculate_severity_distribution(self, anomalies: List[AnomalyPoint]) -> Dict[str, int]:
        """Calculate distribution of anomaly severities."""
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for anomaly in anomalies:
            severity_counts[anomaly.severity] += 1
        return severity_counts
    
    async def _generate_detection_recommendations(
        self, 
        anomalies: List[AnomalyPoint], 
        statistical_summary: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on detection results."""
        recommendations = []
        
        anomaly_count = len(anomalies)
        anomaly_rate = statistical_summary.get('anomaly_rate', 0)
        
        # General recommendations based on anomaly count
        if anomaly_count == 0:
            recommendations.append("No anomalies detected - data appears normal")
        elif anomaly_count <= 3:
            recommendations.append("Few anomalies detected - investigate individually")
        elif anomaly_count <= 10:
            recommendations.append("Moderate number of anomalies - consider pattern analysis")
        else:
            recommendations.append("High number of anomalies - systemic issue likely present")
        
        # Recommendations based on anomaly rate
        if anomaly_rate > 0.1:  # 10% anomaly rate
            recommendations.append("High anomaly rate suggests underlying process issues")
            recommendations.append("Recommend root cause analysis and process improvement")
        elif anomaly_rate > 0.05:  # 5% anomaly rate
            recommendations.append("Monitor trends - may indicate emerging issues")
        
        # Severity-based recommendations
        severity_dist = statistical_summary.get('severity_distribution', {})
        if severity_dist.get('critical', 0) > 0:
            recommendations.append("Critical anomalies require immediate investigation")
        if severity_dist.get('high', 0) > 2:
            recommendations.append("Multiple high-severity anomalies warrant detailed analysis")
        
        # Statistical recommendations
        data_std = statistical_summary.get('data_std', 0)
        if data_std > statistical_summary.get('data_mean', 1) * 0.5:
            recommendations.append("High data variability - consider data stabilization techniques")
        
        if not recommendations:
            recommendations.append("Continue monitoring with current parameters")
        
        return recommendations
    
    async def _calculate_detection_confidence(
        self, 
        anomalies: List[AnomalyPoint], 
        data: np.ndarray
    ) -> float:
        """Calculate confidence level in the detection results."""
        if len(data) == 0:
            return 0.0
        
        # Base confidence on data quality and method appropriateness
        confidence_factors = []
        
        # Data size factor
        data_size_factor = min(1.0, len(data) / 100.0)
        confidence_factors.append(data_size_factor)
        
        # Anomaly rate factor (too many or too few anomalies reduce confidence)
        anomaly_rate = len(anomalies) / len(data)
        if 0.01 <= anomaly_rate <= 0.2:  # Reasonable anomaly rate
            rate_factor = 1.0
        elif anomaly_rate < 0.01:
            rate_factor = 0.8  # Very few anomalies - might miss real issues
        else:
            rate_factor = 0.6  # Too many anomalies - might be false positives
        confidence_factors.append(rate_factor)
        
        # Method appropriateness factor
        if len(data) >= 50:
            method_factor = 1.0  # OLS method works well
        else:
            method_factor = 0.8  # Statistical method more appropriate for small data
        confidence_factors.append(method_factor)
        
        # Return geometric mean of confidence factors
        if confidence_factors:
            confidence = np.prod(confidence_factors) ** (1/len(confidence_factors))
            return max(0.1, min(1.0, confidence))
        
        return 0.5  # Default confidence
    
    async def get_detection_history(self, limit: int = 50) -> List[AnomalyDetectionResult]:
        """Get recent detection history."""
        return self.detection_history[-limit:] if self.detection_history else []
    
    def get_model_info(self, series_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a fitted model."""
        if series_name not in self.models:
            return None
        
        model = self.models[series_name]
        threshold = self.thresholds.get(series_name, 0.0)
        
        return {
            'series_name': series_name,
            'model_coefficients': model.coef_.tolist() if hasattr(model, 'coef_') else [],
            'model_intercept': float(model.intercept_) if hasattr(model, 'intercept_') else 0.0,
            'residual_threshold': float(threshold),
            'model_fitted': True
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        total_detections = len(self.detection_history)
        
        if total_detections == 0:
            return {
                'total_detections': 0,
                'average_anomalies_per_detection': 0.0,
                'models_trained': len(self.models)
            }
        
        total_anomalies = sum(result.anomaly_count for result in self.detection_history)
        avg_anomalies = total_anomalies / total_detections
        avg_confidence = np.mean([result.confidence_level for result in self.detection_history])
        
        return {
            'total_detections': total_detections,
            'total_anomalies_found': total_anomalies,
            'average_anomalies_per_detection': avg_anomalies,
            'average_detection_confidence': avg_confidence,
            'models_trained': len(self.models),
            'series_monitored': list(self.models.keys())
        }

# Global instance
anomaly_detector: Optional[AnomalyDetector] = None

async def initialize_anomaly_detector(sensitivity: float = 0.95) -> bool:
    """
    Initialize the global anomaly detector instance.
    
    Args:
        sensitivity: Detection sensitivity threshold
        
    Returns:
        bool: Success status
    """
    global anomaly_detector
    
    try:
        anomaly_detector = AnomalyDetector(sensitivity=sensitivity)
        return await anomaly_detector.initialize()
    except Exception as e:
        logger.error(f"❌ Failed to initialize anomaly detector: {str(e)}")
        return False

def get_anomaly_detector() -> AnomalyDetector:
    """Get the global anomaly detector instance."""
    if anomaly_detector is None:
        raise RuntimeError("Anomaly detector not initialized")
    return anomaly_detector