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
)

__all__ = [
    'compute_cross_correlation',
    'compute_autocorrelation',
    'extract_correlation_features',
    'extract_temporal_features',
    'build_feature_vector',
    'preprocess_features',
]
