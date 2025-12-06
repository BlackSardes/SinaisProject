<<<<<<< HEAD
"""
GPS Signal Feature Extraction Module

This module provides functions for extracting features from GPS signals
focused on correlation profiles and signal quality metrics.
"""

from .correlation import (
    compute_cross_correlation,
    compute_autocorrelation,
    generate_local_code
)
from .correlation_features import (
    extract_correlation_features,
    compute_peak_height,
    compute_fwhm,
    compute_peak_ratio,
    compute_peak_offset
)
from .temporal_features import (
    extract_temporal_features,
    compute_cn0_variation_features
)
from .feature_pipeline import (
    build_feature_vector,
    preprocess_features
=======
"""Feature extraction module for GPS spoofing detection"""
from .correlation import (
    compute_cross_correlation,
    compute_autocorrelation,
    extract_correlation_features,
)
from .temporal import (
    extract_temporal_features,
)
from .pipeline import (
    build_feature_vector,
    preprocess_features,
>>>>>>> main
)

__all__ = [
    'compute_cross_correlation',
    'compute_autocorrelation',
<<<<<<< HEAD
    'generate_local_code',
    'extract_correlation_features',
    'compute_peak_height',
    'compute_fwhm',
    'compute_peak_ratio',
    'compute_peak_offset',
    'extract_temporal_features',
    'compute_cn0_variation_features',
=======
    'extract_correlation_features',
    'extract_temporal_features',
>>>>>>> main
    'build_feature_vector',
    'preprocess_features',
]
