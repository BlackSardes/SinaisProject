"""Feature extraction module for GPS spoofing detection."""

from .correlation import compute_cross_correlation, compute_autocorrelation
from .metrics import fwhm
from .feature_extraction import extract_correlation_features, build_feature_vector

__all__ = [
    'compute_cross_correlation',
    'compute_autocorrelation',
    'fwhm',
    'extract_correlation_features',
    'build_feature_vector',
]
