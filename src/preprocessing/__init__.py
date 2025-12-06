"""
GPS Signal Preprocessing Module

This module provides functions for preprocessing GPS I/Q signals including:
- Signal loading from various formats
- Normalization and filtering
- Windowing and segmentation
- C/N0 estimation
"""

from .signal_io import load_signal, read_iq_binary
from .normalization import normalize_signal, remove_dc
from .filtering import bandpass_filter, remove_outliers, smooth_signal
from .resampling import resample_signal
from .windowing import window_segment, align_channels
from .cn0_estimation import estimate_cn0_from_correlation, estimate_cn0_from_signal

__all__ = [
    'load_signal',
    'read_iq_binary',
    'normalize_signal',
    'remove_dc',
    'bandpass_filter',
    'remove_outliers',
    'smooth_signal',
    'resample_signal',
    'window_segment',
    'align_channels',
    'estimate_cn0_from_correlation',
    'estimate_cn0_from_signal',
]
