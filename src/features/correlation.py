"""
Correlation-based feature extraction for GPS spoofing detection.
Adapted from extract_features.py with enhancements.
"""
import numpy as np
from typing import Dict, Optional
from scipy.stats import skew, kurtosis


def compute_cross_correlation(signal: np.ndarray, prn_code: np.ndarray) -> np.ndarray:
    """
    Compute cross-correlation between signal and PRN code using FFT.
    
    Args:
        signal: Complex IQ signal
        prn_code: PRN code (real-valued, +1/-1)
    
    Returns:
        Correlation magnitude profile
    """
    # Ensure same length
    if len(prn_code) < len(signal):
        prn_code = np.tile(prn_code, int(np.ceil(len(signal) / len(prn_code))))
    prn_code = prn_code[:len(signal)]
    
    # FFT-based correlation
    fft_signal = np.fft.fft(signal)
    fft_code = np.fft.fft(prn_code)
    corr = np.fft.ifft(fft_signal * np.conj(fft_code))
    
    return np.abs(corr)


def compute_autocorrelation(signal: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """
    Compute autocorrelation of signal.
    
    Args:
        signal: Input signal
        max_lag: Maximum lag to compute (None = full)
    
    Returns:
        Autocorrelation values
    """
    if max_lag is None:
        max_lag = len(signal) - 1
    
    autocorr = np.correlate(signal, signal, mode='full')
    center = len(autocorr) // 2
    return autocorr[center:center + max_lag + 1]


def extract_correlation_features(corr_profile: np.ndarray, fs: float, 
                                  ca_chip_rate: float = 1.023e6) -> Dict[str, float]:
    """
    Extract comprehensive features from correlation profile (SQMs).
    
    Args:
        corr_profile: Correlation magnitude profile
        fs: Sampling frequency (Hz)
        ca_chip_rate: C/A code chip rate (Hz)
    
    Returns:
        Dictionary of correlation-based features
    
    Features extracted:
        - peak_height: Maximum correlation value
        - peak_index: Index of peak
        - peak_to_mean: Ratio of peak to mean
        - peak_to_secondary: Ratio of primary to secondary peak
        - fwhm: Full Width at Half Maximum
        - skewness: Distribution skewness around peak
        - kurtosis: Distribution kurtosis around peak
        - energy_window: Energy in window around peak
        - ratio_first_second_peak: Ratio of first to second highest peak
        - peak_offset: Offset of peak from expected position
    """
    features = {}
    
    # Basic peak features
    peak_idx = np.argmax(corr_profile)
    peak_value = corr_profile[peak_idx]
    features['peak_height'] = float(peak_value)
    features['peak_index'] = int(peak_idx)
    
    # Peak to mean ratio
    mean_value = np.mean(corr_profile)
    features['peak_to_mean'] = float(peak_value / mean_value) if mean_value > 0 else 0.0
    
    # Secondary peak detection (adapted from extract_features.py)
    samples_per_chip = int(fs / ca_chip_rate)
    peak_window_samples = int(2 * samples_per_chip)
    
    temp_corr = corr_profile.copy()
    temp_corr = np.roll(temp_corr, -peak_idx)
    temp_corr[:peak_window_samples] = 0
    temp_corr = np.roll(temp_corr, peak_idx)
    
    secondary_peak = np.max(temp_corr)
    features['secondary_peak_value'] = float(secondary_peak)
    features['peak_to_secondary'] = float(peak_value / secondary_peak) if secondary_peak > 0 else 999.0
    
    # FWHM (Full Width at Half Maximum)
    half_max = peak_value / 2.0
    above_half = np.where(corr_profile >= half_max)[0]
    if len(above_half) > 0:
        fwhm = above_half[-1] - above_half[0]
    else:
        fwhm = 0
    features['fwhm'] = int(fwhm)
    
    # Fractional Peak Width at 80% (from extract_features.py)
    FRACTIONAL_PEAK_THRESHOLD = 0.8
    frac_level = FRACTIONAL_PEAK_THRESHOLD * peak_value
    above_frac = np.where(corr_profile > frac_level)[0]
    fpw = above_frac[-1] - above_frac[0] if above_frac.size > 0 else 0
    features['fpw'] = int(fpw)
    
    # Asymmetry (from extract_features.py)
    left_area = np.sum(corr_profile[max(0, peak_idx - samples_per_chip):peak_idx])
    right_area = np.sum(corr_profile[peak_idx + 1:min(len(corr_profile), peak_idx + samples_per_chip + 1)])
    if (right_area + left_area) != 0:
        asymmetry = (right_area - left_area) / (right_area + left_area)
    else:
        asymmetry = 0.0
    features['asymmetry'] = float(asymmetry)
    
    # Statistical features around peak
    window_size = min(samples_per_chip * 4, len(corr_profile) // 4)
    start = max(0, peak_idx - window_size // 2)
    end = min(len(corr_profile), peak_idx + window_size // 2)
    window = corr_profile[start:end]
    
    if len(window) > 3:
        features['skewness'] = float(skew(window))
        features['kurtosis'] = float(kurtosis(window))
    else:
        features['skewness'] = 0.0
        features['kurtosis'] = 0.0
    
    # Energy in window around peak
    energy_window = np.sum(window**2)
    features['energy_window'] = float(energy_window)
    
    # Peak offset from expected center
    expected_center = len(corr_profile) // 2
    features['peak_offset'] = int(peak_idx - expected_center)
    
    # Gradient features (rate of change around peak)
    if peak_idx > 0 and peak_idx < len(corr_profile) - 1:
        left_gradient = corr_profile[peak_idx] - corr_profile[peak_idx - 1]
        right_gradient = corr_profile[peak_idx + 1] - corr_profile[peak_idx]
        features['left_gradient'] = float(left_gradient)
        features['right_gradient'] = float(right_gradient)
    else:
        features['left_gradient'] = 0.0
        features['right_gradient'] = 0.0
    
    return features
