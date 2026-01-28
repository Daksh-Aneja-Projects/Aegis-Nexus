"""
Precision Validator for Aegis Nexus
Applies Six Sigma validation standards to data integrity checking.

This module implements industrial-grade data validation using statistical
process control and outlier detection methods.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation"""
    data_id: str
    is_valid: bool
    health_score: float  # 0.0 to 1.0
    anomalies_detected: List[str]
    statistical_measures: Dict[str, float]
    six_sigma_status: str  # pass, warning, fail
    recommendations: List[str]
    validation_timestamp: datetime

@dataclass
class HistoricalBaseline:
    """Historical data baseline for comparison"""
    baseline_id: str
    metric_name: str
    mean: float
    std_dev: float
    sample_size: int
    confidence_interval: Tuple[float, float]
    last_updated: datetime
    data_points: List[float]

class DataMetrology:
    """
    Industrial-grade data validation using Six Sigma standards.
    
    Implements statistical process control, outlier detection, and
    precision validation for information quality assurance.
    """
    
    def __init__(self, sigma_level: float = 6.0):
        """
        Initialize the data metrology system.
        
        Args:
            sigma_level: Sigma level for validation (default 6 for Six Sigma)
        """
        self.sigma_level = sigma_level
        self.baselines: Dict[str, HistoricalBaseline] = {}
        self.validation_history: List[ValidationResult] = []
        self.ols_models: Dict[str, LinearRegression] = {}
        self.residual_thresholds: Dict[str, float] = {}
        self.max_history_size = 1000
        
    async def initialize(self) -> bool:
        """Initialize the data metrology system."""
        try:
            logger.info("📏 Initializing data metrology system...")
            
            # Initialize baseline metrics
            await self._initialize_baselines()
            
            # Load OLS models for different data types
            await self._initialize_ols_models()
            
            logger.info("✅ Data metrology system initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize data metrology: {str(e)}")
            return False
    
    async def _initialize_baselines(self):
        """Initialize baseline metrics for common data types."""
        # CPU usage baseline
        self.baselines['cpu_usage'] = HistoricalBaseline(
            baseline_id='cpu_baseline_001',
            metric_name='cpu_usage',
            mean=0.3,  # 30% average CPU usage
            std_dev=0.15,
            sample_size=1000,
            confidence_interval=(0.0, 0.8),
            last_updated=datetime.utcnow(),
            data_points=[]
        )
        
        # Memory usage baseline
        self.baselines['memory_usage'] = HistoricalBaseline(
            baseline_id='memory_baseline_001',
            metric_name='memory_usage',
            mean=0.4,  # 40% average memory usage
            std_dev=0.12,
            sample_size=1000,
            confidence_interval=(0.0, 0.85),
            last_updated=datetime.utcnow(),
            data_points=[]
        )
        
        # Response time baseline
        self.baselines['response_time'] = HistoricalBaseline(
            baseline_id='response_baseline_001',
            metric_name='response_time_ms',
            mean=150.0,  # 150ms average response time
            std_dev=50.0,
            sample_size=1000,
            confidence_interval=(0.0, 1000.0),
            last_updated=datetime.utcnow(),
            data_points=[]
        )
        
        # Error rate baseline
        self.baselines['error_rate'] = HistoricalBaseline(
            baseline_id='error_baseline_001',
            metric_name='error_rate',
            mean=0.01,  # 1% average error rate
            std_dev=0.005,
            sample_size=1000,
            confidence_interval=(0.0, 0.05),
            last_updated=datetime.utcnow(),
            data_points=[]
        )
    
    async def _initialize_ols_models(self):
        """Initialize OLS regression models for different metrics."""
        # Simple linear models for demonstration
        # In practice, these would be trained on actual historical data
        
        for metric_name in ['cpu_usage', 'memory_usage', 'response_time', 'error_rate']:
            model = LinearRegression()
            # Dummy training data - in practice this would be real historical data
            X_dummy = np.array([[i] for i in range(100)])
            y_dummy = np.random.normal(0, 1, 100)
            model.fit(X_dummy, y_dummy)
            self.ols_models[metric_name] = model
    
    async def validate_data_integrity(
        self, 
        data: Union[float, int, List[float], Dict[str, Any]], 
        data_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate data integrity using Six Sigma standards.
        
        Args:
            data: Data to validate
            data_type: Type of data being validated
            context: Additional context for validation
            
        Returns:
            ValidationResult with validation outcomes
        """
        logger.debug(f"📏 Validating {data_type} data integrity")
        
        try:
            # Generate unique data ID
            data_id = f"validation_{hash(str(data)) % 1000000}_{int(datetime.utcnow().timestamp())}"
            
            # Convert data to appropriate format
            numeric_data = await self._convert_to_numeric(data)
            
            if numeric_data is None or len(numeric_data) == 0:
                return ValidationResult(
                    data_id=data_id,
                    is_valid=False,
                    health_score=0.0,
                    anomalies_detected=['invalid_data_format'],
                    statistical_measures={},
                    six_sigma_status='fail',
                    recommendations=['Invalid data format provided'],
                    validation_timestamp=datetime.utcnow()
                )
            
            # Perform statistical analysis
            statistical_measures = await self._calculate_statistical_measures(numeric_data)
            
            # Check against baseline
            baseline_comparison = await self._compare_against_baseline(
                numeric_data, data_type, statistical_measures
            )
            
            # Perform Six Sigma validation
            six_sigma_result = await self._perform_six_sigma_validation(
                numeric_data, data_type, statistical_measures, baseline_comparison
            )
            
            # Calculate health score
            health_score = await self._calculate_health_score(
                statistical_measures, baseline_comparison, six_sigma_result
            )
            
            # Identify anomalies
            anomalies = await self._detect_anomalies(
                numeric_data, statistical_measures, baseline_comparison
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                health_score, anomalies, six_sigma_result
            )
            
            # Determine overall validity
            is_valid = health_score >= 0.95 and len(anomalies) == 0
            
            # Create validation result
            result = ValidationResult(
                data_id=data_id,
                is_valid=is_valid,
                health_score=health_score,
                anomalies_detected=anomalies,
                statistical_measures=statistical_measures,
                six_sigma_status=six_sigma_result['status'],
                recommendations=recommendations,
                validation_timestamp=datetime.utcnow()
            )
            
            # Add to validation history
            self.validation_history.append(result)
            
            # Maintain history size
            if len(self.validation_history) > self.max_history_size:
                self.validation_history = self.validation_history[-self.max_history_size:]
            
            # Update baseline with new data
            await self._update_baseline(data_type, numeric_data)
            
            logger.debug(f"📏 Validation completed: {data_id} "
                        f"(health: {health_score:.2f}, status: {six_sigma_result['status']})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Data validation failed: {str(e)}")
            return ValidationResult(
                data_id=f"error_{int(datetime.utcnow().timestamp())}",
                is_valid=False,
                health_score=0.0,
                anomalies_detected=['validation_system_error'],
                statistical_measures={'error': str(e)},
                six_sigma_status='fail',
                recommendations=['Validation system error - manual review required'],
                validation_timestamp=datetime.utcnow()
            )
    
    async def _convert_to_numeric(self, data: Any) -> Optional[List[float]]:
        """Convert input data to numeric format for analysis."""
        try:
            if isinstance(data, (int, float)):
                return [float(data)]
            elif isinstance(data, list):
                return [float(x) for x in data if isinstance(x, (int, float))]
            elif isinstance(data, dict):
                # Extract numeric values from dictionary
                numeric_values = []
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        numeric_values.append(float(value))
                    elif isinstance(value, list):
                        numeric_values.extend([float(x) for x in value if isinstance(x, (int, float))])
                return numeric_values
            elif isinstance(data, str):
                # Try to parse as number or list of numbers
                try:
                    return [float(data)]
                except ValueError:
                    # Try parsing as comma-separated values
                    try:
                        return [float(x.strip()) for x in data.split(',') if x.strip()]
                    except ValueError:
                        return None
            else:
                return None
                
        except Exception:
            return None
    
    async def _calculate_statistical_measures(self, data: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistical measures."""
        if not data:
            return {}
        
        # Basic statistics
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        variance = np.var(data)
        
        # Percentiles
        percentiles = np.percentile(data, [25, 50, 75, 95, 99])
        
        # Skewness and kurtosis
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        # Range and interquartile range
        data_range = np.max(data) - np.min(data)
        iqr = percentiles[2] - percentiles[0]  # Q3 - Q1
        
        # Coefficient of variation
        cv = (std_val / mean_val) if mean_val != 0 else 0
        
        return {
            'count': len(data),
            'mean': float(mean_val),
            'median': float(median_val),
            'std_dev': float(std_val),
            'variance': float(variance),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'range': float(data_range),
            'q25': float(percentiles[0]),
            'q50': float(percentiles[1]),  # median
            'q75': float(percentiles[2]),
            'q95': float(percentiles[3]),
            'q99': float(percentiles[4]),
            'iqr': float(iqr),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'coefficient_of_variation': float(cv)
        }
    
    async def _compare_against_baseline(
        self, 
        data: List[float], 
        data_type: str, 
        statistical_measures: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compare data against established baselines."""
        baseline = self.baselines.get(data_type)
        if not baseline:
            return {'baseline_available': False}
        
        current_mean = statistical_measures.get('mean', 0)
        current_std = statistical_measures.get('std_dev', 0)
        
        # Calculate z-score against baseline
        if baseline.std_dev > 0:
            z_score = abs(current_mean - baseline.mean) / baseline.std_dev
        else:
            z_score = float('inf')
        
        # Check if within confidence interval
        within_bounds = (
            baseline.confidence_interval[0] <= current_mean <= baseline.confidence_interval[1]
        )
        
        # Calculate drift from baseline
        mean_drift = abs(current_mean - baseline.mean) / baseline.mean if baseline.mean != 0 else 0
        std_drift = abs(current_std - baseline.std_dev) / baseline.std_dev if baseline.std_dev != 0 else 0
        
        return {
            'baseline_available': True,
            'z_score': z_score,
            'within_confidence_bounds': within_bounds,
            'mean_drift': mean_drift,
            'std_drift': std_drift,
            'baseline_mean': baseline.mean,
            'baseline_std': baseline.std_dev
        }
    
    async def _perform_six_sigma_validation(
        self, 
        data: List[float], 
        data_type: str,
        statistical_measures: Dict[str, float],
        baseline_comparison: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform Six Sigma validation using OLS residuals."""
        try:
            # Get OLS model for this data type
            model = self.ols_models.get(data_type)
            if not model:
                return {'status': 'warning', 'reason': 'no_model_available'}
            
            # Calculate expected values using OLS model
            indices = np.array(range(len(data))).reshape(-1, 1)
            expected_values = model.predict(indices)
            actual_values = np.array(data)
            
            # Calculate residuals
            residuals = actual_values - expected_values
            
            # Calculate residual statistics
            residual_mean = np.mean(residuals)
            residual_std = np.std(residuals)
            
            # Calculate 95th percentile threshold for residuals
            residual_threshold = np.percentile(np.abs(residuals), 95)
            
            # Store threshold for future use
            self.residual_thresholds[data_type] = residual_threshold
            
            # Check Six Sigma compliance (±6σ tolerance)
            six_sigma_limit = 6 * residual_std
            max_residual = np.max(np.abs(residuals))
            
            # Determine status
            if max_residual <= six_sigma_limit:
                status = 'pass'
                reason = 'Data within Six Sigma tolerance limits'
            elif max_residual <= 8 * residual_std:  # ±8σ warning zone
                status = 'warning'
                reason = 'Data approaching Six Sigma limits'
            else:
                status = 'fail'
                reason = 'Data exceeds Six Sigma tolerance limits'
            
            return {
                'status': status,
                'reason': reason,
                'residual_mean': float(residual_mean),
                'residual_std': float(residual_std),
                'max_residual': float(max_residual),
                'six_sigma_limit': float(six_sigma_limit),
                'residual_threshold': float(residual_threshold),
                'within_tolerance': max_residual <= six_sigma_limit
            }
            
        except Exception as e:
            logger.error(f"Six Sigma validation error: {str(e)}")
            return {
                'status': 'fail',
                'reason': f'Validation error: {str(e)}'
            }
    
    async def _calculate_health_score(
        self, 
        statistical_measures: Dict[str, float],
        baseline_comparison: Dict[str, Any],
        six_sigma_result: Dict[str, Any]
    ) -> float:
        """Calculate overall data health score."""
        scores = []
        
        # Statistical quality score (based on coefficient of variation)
        cv = statistical_measures.get('coefficient_of_variation', 1.0)
        stat_score = max(0.0, 1.0 - min(1.0, cv))  # Lower CV = higher score
        scores.append(stat_score)
        
        # Baseline compliance score
        if baseline_comparison.get('baseline_available', False):
            z_score = baseline_comparison.get('z_score', float('inf'))
            baseline_score = max(0.0, 1.0 - min(1.0, z_score / 3.0))  # 3σ tolerance
            scores.append(baseline_score)
        else:
            scores.append(0.8)  # Default score when no baseline
        
        # Six Sigma compliance score
        six_sigma_status = six_sigma_result.get('status', 'fail')
        if six_sigma_status == 'pass':
            six_sigma_score = 1.0
        elif six_sigma_status == 'warning':
            six_sigma_score = 0.7
        else:
            six_sigma_score = 0.3
        scores.append(six_sigma_score)
        
        # Return geometric mean of all scores
        if scores:
            health_score = np.prod(scores) ** (1/len(scores))
            return max(0.0, min(1.0, health_score))
        
        return 0.5  # Default score
    
    async def _detect_anomalies(
        self, 
        data: List[float], 
        statistical_measures: Dict[str, float],
        baseline_comparison: Dict[str, Any]
    ) -> List[str]:
        """Detect anomalies in the data."""
        anomalies = []
        
        if not data:
            return ['no_data']
        
        # Statistical anomalies using z-scores
        mean_val = statistical_measures.get('mean', 0)
        std_val = statistical_measures.get('std_dev', 1)
        
        for value in data:
            if std_val > 0:
                z_score = abs(value - mean_val) / std_val
                if z_score > 3.0:  # 3σ threshold
                    anomalies.append(f'outlier_detected_z{z_score:.1f}')
        
        # Range anomalies
        min_val = statistical_measures.get('min', 0)
        max_val = statistical_measures.get('max', 0)
        data_range = max_val - min_val
        
        if data_range == 0:
            anomalies.append('zero_variance')
        
        # Baseline drift anomalies
        if baseline_comparison.get('baseline_available', False):
            mean_drift = baseline_comparison.get('mean_drift', 0)
            if mean_drift > 0.2:  # 20% drift threshold
                anomalies.append(f'significant_mean_drift_{mean_drift:.2f}')
        
        # Distribution shape anomalies
        skewness = statistical_measures.get('skewness', 0)
        kurtosis = statistical_measures.get('kurtosis', 0)
        
        if abs(skewness) > 2.0:
            anomalies.append(f'high_skewness_{skewness:.2f}')
        if abs(kurtosis) > 7.0:  # Excess kurtosis threshold
            anomalies.append(f'high_kurtosis_{kurtosis:.2f}')
        
        return list(set(anomalies))  # Remove duplicates
    
    async def _generate_recommendations(
        self, 
        health_score: float, 
        anomalies: List[str], 
        six_sigma_result: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Health score based recommendations
        if health_score < 0.7:
            recommendations.append('Data quality critically low - immediate investigation required')
        elif health_score < 0.9:
            recommendations.append('Data quality suboptimal - consider data cleaning')
        elif health_score < 0.95:
            recommendations.append('Data quality acceptable but could be improved')
        else:
            recommendations.append('Data quality excellent - meets Six Sigma standards')
        
        # Anomaly specific recommendations
        for anomaly in anomalies:
            if 'outlier' in anomaly:
                recommendations.append('Investigate and remove statistical outliers')
            elif 'drift' in anomaly:
                recommendations.append('Recalibrate baseline measurements')
            elif 'skewness' in anomaly:
                recommendations.append('Address data distribution asymmetry')
            elif 'kurtosis' in anomaly:
                recommendations.append('Investigate heavy-tailed distributions')
        
        # Six Sigma recommendations
        six_sigma_status = six_sigma_result.get('status', 'fail')
        if six_sigma_status == 'fail':
            recommendations.append('Implement Six Sigma process improvements')
            recommendations.append('Reduce process variation through root cause analysis')
        elif six_sigma_status == 'warning':
            recommendations.append('Monitor process closely for Six Sigma compliance')
        
        if not recommendations:
            recommendations.append('Continue monitoring with current processes')
        
        return recommendations
    
    async def _update_baseline(self, data_type: str, new_data: List[float]):
        """Update baseline statistics with new data."""
        if data_type not in self.baselines:
            return
        
        baseline = self.baselines[data_type]
        
        # Add new data points
        baseline.data_points.extend(new_data)
        
        # Keep only recent data points (sliding window)
        max_points = 10000
        if len(baseline.data_points) > max_points:
            baseline.data_points = baseline.data_points[-max_points:]
        
        # Recalculate statistics
        if baseline.data_points:
            baseline.mean = float(np.mean(baseline.data_points))
            baseline.std_dev = float(np.std(baseline.data_points))
            baseline.sample_size = len(baseline.data_points)
            
            # Update confidence interval (95%)
            margin_of_error = 1.96 * (baseline.std_dev / np.sqrt(baseline.sample_size))
            baseline.confidence_interval = (
                max(0, baseline.mean - margin_of_error),
                baseline.mean + margin_of_error
            )
        
        baseline.last_updated = datetime.utcnow()
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get statistics about validation performance."""
        if not self.validation_history:
            return {
                'total_validations': 0,
                'pass_rate': 0.0,
                'average_health_score': 0.0
            }
        
        total_validations = len(self.validation_history)
        passed_validations = sum(1 for result in self.validation_history if result.is_valid)
        health_scores = [result.health_score for result in self.validation_history]
        
        six_sigma_status_counts = {}
        for result in self.validation_history:
            status = result.six_sigma_status
            six_sigma_status_counts[status] = six_sigma_status_counts.get(status, 0) + 1
        
        return {
            'total_validations': total_validations,
            'pass_rate': passed_validations / total_validations if total_validations > 0 else 0.0,
            'average_health_score': np.mean(health_scores) if health_scores else 0.0,
            'six_sigma_distribution': six_sigma_status_counts,
            'baselines_tracked': len(self.baselines)
        }
    
    def get_baseline_info(self, data_type: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific baseline."""
        baseline = self.baselines.get(data_type)
        if not baseline:
            return None
        
        return {
            'baseline_id': baseline.baseline_id,
            'metric_name': baseline.metric_name,
            'current_mean': baseline.mean,
            'current_std': baseline.std_dev,
            'sample_size': baseline.sample_size,
            'confidence_interval': baseline.confidence_interval,
            'last_updated': baseline.last_updated.isoformat(),
            'data_points_count': len(baseline.data_points)
        }

# Global instance
data_metrology: Optional[DataMetrology] = None

async def initialize_data_metrology(sigma_level: float = 6.0) -> bool:
    """
    Initialize the global data metrology instance.
    
    Args:
        sigma_level: Sigma level for validation
        
    Returns:
        bool: Success status
    """
    global data_metrology
    
    try:
        data_metrology = DataMetrology(sigma_level=sigma_level)
        return await data_metrology.initialize()
    except Exception as e:
        logger.error(f"❌ Failed to initialize data metrology: {str(e)}")
        return False

def get_data_metrology() -> DataMetrology:
    """Get the global data metrology instance."""
    if data_metrology is None:
        raise RuntimeError("Data metrology not initialized")
    return data_metrology