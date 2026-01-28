"""
Signal Smoothing Utilities for Sensor Debouncing.

Provides real-time filters to reduce high-frequency noise before
expensive Kalman filtering steps.
"""

import numpy as np
from collections import deque
from typing import Optional, List, Any

class ExponentialMovingAverage:
    """
    Fast Exponential Moving Average (EMA) filter.
    Recursive implementation: O(1) time complexity.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: Smoothing factor (0 < alpha <= 1). 
                   Smaller = more smoothing, more lag.
        """
        self.alpha = alpha
        self._value: Optional[float] = None
    
    def update(self, new_value: float) -> float:
        """Update filter with new measurement."""
        if self._value is None:
            self._value = new_value
        else:
            self._value = self.alpha * new_value + (1 - self.alpha) * self._value
        return self._value
    
    def reset(self):
        """Reset filter state."""
        self._value = None


class MedianFilter:
    """
    Sliding window Median Filter.
    Excellent for removing "salt and pepper" impulse noise (outliers).
    """
    
    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: Number of samples in sliding window.
        """
        self.window = deque(maxlen=window_size)
    
    def update(self, new_value: float) -> float:
        """Update window and return median."""
        self.window.append(new_value)
        return float(np.median(list(self.window)))
    
    def reset(self):
        """Reset filter state."""
        self.window.clear()


class SignalPreProcessor:
    """
    Combined signal pre-processor for sensor streams.
    Applies Median Filter (Outlier Rejection) -> EMA (Smoothing).
    """
    
    def __init__(self, ema_alpha: float = 0.3, median_window: int = 5):
        self.median_filter = MedianFilter(window_size=median_window)
        self.ema_filter = ExponentialMovingAverage(alpha=ema_alpha)
        
    def process(self, value: float) -> float:
        """Apply full pre-processing chain."""
        denoised = self.median_filter.update(value)
        smoothed = self.ema_filter.update(denoised)
        return smoothed
