"""
Feature extraction from correlation profiles.

These features capture the morphology of the correlation peak,
which is distorted by spoofing attacks.
"""
import numpy as np
from scipy import stats
from typing import Dict, Optional


def compute_peak_height(corr_magnitude: np.ndarray) -> float:
    """
    Compute height of correlation peak.
    
    Args:
        corr_magnitude: Magnitude of correlation profile
    
    Returns:
        Peak height value
    """
    return float(np.max(corr_magnitude))


def compute_fwhm(
    corr_magnitude: np.ndarray,
    samples_per_chip: Optional[int] = None,
    fraction: float = 0.5
) -> float:
    """
    Compute Full Width at Half Maximum (FWHM) of correlation peak.
    
    FWHM measures the width of the correlation peak, which increases
    when multiple signals (authentic + spoofing) are present.
    
    Args:
        corr_magnitude: Magnitude of correlation profile
        samples_per_chip: Samples per chip (for normalization)
        fraction: Fraction of peak height to measure width (default: 0.5 = half max)
    
    Returns:
        FWHM in samples (or normalized by samples_per_chip if provided)
        
    Example:
        >>> fwhm = compute_fwhm(corr_mag, samples_per_chip=5)
        >>> print(f"FWHM: {fwhm:.2f} chips")
    """
    peak_idx = np.argmax(corr_magnitude)
    peak_val = corr_magnitude[peak_idx]
    threshold = fraction * peak_val
    
    # Find points above threshold
    above_threshold = corr_magnitude > threshold
    
    if not np.any(above_threshold):
        return 0.0
    
    # Find left and right edges
    indices = np.where(above_threshold)[0]
    left_idx = indices[0]
    right_idx = indices[-1]
    
    width = right_idx - left_idx + 1
    
    # Normalize by samples per chip if provided
    if samples_per_chip is not None and samples_per_chip > 0:
        width = width / samples_per_chip
    
    return float(width)


def compute_peak_ratio(
    corr_magnitude: np.ndarray,
    samples_per_chip: int,
    exclude_window_chips: float = 2.0
) -> float:
    """
    Compute ratio of primary to secondary peak.
    
    A high ratio indicates clean signal, while low ratio suggests
    multipath or spoofing interference.
    
    Args:
        corr_magnitude: Magnitude of correlation profile
        samples_per_chip: Samples per chip
        exclude_window_chips: Window around main peak to exclude when finding secondary
    
    Returns:
        Peak-to-secondary ratio
    """
    peak_idx = np.argmax(corr_magnitude)
    peak_val = corr_magnitude[peak_idx]
    
    # Exclude region around main peak
    exclude_window = int(exclude_window_chips * samples_per_chip)
    mask = np.ones_like(corr_magnitude, dtype=bool)
    start = max(0, peak_idx - exclude_window)
    end = min(len(mask), peak_idx + exclude_window)
    mask[start:end] = False
    
    # Find secondary peak
    if np.any(mask):
        secondary_val = np.max(corr_magnitude[mask])
    else:
        secondary_val = 0.0
    
    # Compute ratio
    if secondary_val > 0:
        ratio = peak_val / secondary_val
    else:
        ratio = 999.0  # Very high ratio for clean signal
    
    return float(ratio)


def compute_peak_offset(
    corr_magnitude: np.ndarray,
    expected_idx: Optional[int] = None
) -> float:
    """
    Compute offset of peak from expected position.
    
    Args:
        corr_magnitude: Magnitude of correlation profile
        expected_idx: Expected peak index (None = center of array)
    
    Returns:
        Peak offset in samples
    """
    peak_idx = np.argmax(corr_magnitude)
    
    if expected_idx is None:
        expected_idx = len(corr_magnitude) // 2
    
    offset = peak_idx - expected_idx
    return float(offset)


def compute_asymmetry(
    corr_magnitude: np.ndarray,
    samples_per_chip: int
) -> float:
    """
    Compute asymmetry of correlation peak.
    
    Measures left-right asymmetry around the peak, which can indicate
    synchronized spoofing attacks.
    
    Args:
        corr_magnitude: Magnitude of correlation profile
        samples_per_chip: Samples per chip
    
    Returns:
        Asymmetry metric [-1, 1] where 0 is symmetric
    """
    peak_idx = np.argmax(corr_magnitude)
    
    # Define windows around peak
    window = samples_per_chip
    left_start = max(0, peak_idx - window)
    right_end = min(len(corr_magnitude), peak_idx + window + 1)
    
    # Compute areas
    left_area = np.sum(corr_magnitude[left_start:peak_idx])
    right_area = np.sum(corr_magnitude[peak_idx+1:right_end])
    
    # Compute asymmetry
    total = left_area + right_area
    if total > 0:
        asymmetry = (right_area - left_area) / total
    else:
        asymmetry = 0.0
    
    return float(asymmetry)


def compute_energy_window(
    corr_magnitude: np.ndarray,
    samples_per_chip: int,
    window_chips: float = 3.0
) -> float:
    """
    Compute integrated energy in window around peak.
    
    Args:
        corr_magnitude: Magnitude of correlation profile
        samples_per_chip: Samples per chip
        window_chips: Window size in chips
    
    Returns:
        Integrated energy value
    """
    peak_idx = np.argmax(corr_magnitude)
    window = int(window_chips * samples_per_chip)
    
    start = max(0, peak_idx - window // 2)
    end = min(len(corr_magnitude), peak_idx + window // 2 + 1)
    
    energy = np.sum(corr_magnitude[start:end])
    return float(energy)


def extract_correlation_features(
    corr_magnitude: np.ndarray,
    samples_per_chip: int,
    feature_names: Optional[list] = None
) -> Dict[str, float]:
    """
    Extract all correlation-based features.
    
    Args:
        corr_magnitude: Magnitude of correlation profile
        samples_per_chip: Samples per chip
        feature_names: List of features to extract (None = all)
    
    Returns:
        Dictionary of feature name -> value
        
    Available features:
        - peak_height: Maximum correlation value
        - fwhm: Full width at half maximum
        - peak_ratio: Primary to secondary peak ratio
        - peak_offset: Offset from expected position
        - asymmetry: Left-right asymmetry
        - energy_window: Integrated energy around peak
        - skewness: Statistical skewness
        - kurtosis: Statistical kurtosis
        
    Example:
        >>> features = extract_correlation_features(corr_mag, samples_per_chip=5)
        >>> print(features['peak_ratio'])
        12.5
    """
    all_features = {
        'peak_height': lambda: compute_peak_height(corr_magnitude),
        'fwhm': lambda: compute_fwhm(corr_magnitude, samples_per_chip),
        'peak_ratio': lambda: compute_peak_ratio(corr_magnitude, samples_per_chip),
        'peak_offset': lambda: compute_peak_offset(corr_magnitude),
        'asymmetry': lambda: compute_asymmetry(corr_magnitude, samples_per_chip),
        'energy_window': lambda: compute_energy_window(corr_magnitude, samples_per_chip),
        'skewness': lambda: float(stats.skew(corr_magnitude)),
        'kurtosis': lambda: float(stats.kurtosis(corr_magnitude)),
    }
    
    if feature_names is None:
        feature_names = list(all_features.keys())
    
    features = {}
    for name in feature_names:
        if name in all_features:
            try:
                features[name] = all_features[name]()
            except Exception as e:
                print(f"Warning: Failed to compute {name}: {e}")
                features[name] = np.nan
    
    return features
