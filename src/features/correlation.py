"""
Correlation Functions for GPS Signal Analysis

This module provides correlation-based feature extraction for GPS signals,
which is fundamental to detecting spoofing attacks through peak distortions.
"""

import numpy as np
from scipy.stats import skew, kurtosis
from typing import Dict, Tuple, Optional


def compute_correlation_fft(
    signal: np.ndarray,
    local_code: np.ndarray
) -> np.ndarray:
    """
    Compute correlation using FFT method (faster for long signals).
    
    Parameters
    ----------
    signal : np.ndarray
        Received signal (complex or real)
    local_code : np.ndarray
        Local PRN code (real, +1/-1)
        
    Returns
    -------
    np.ndarray
        Correlation magnitude
        
    Notes
    -----
    Uses the convolution theorem: correlation in time domain equals
    multiplication in frequency domain.
    
    corr(x, y) = IFFT(FFT(x) * conj(FFT(y)))
    
    This is much faster than direct correlation for long signals:
    O(N log N) vs O(N^2)
    """
    # Ensure same length
    if len(signal) != len(local_code):
        raise ValueError(f"Signal and code must have same length: {len(signal)} vs {len(local_code)}")
    
    # FFT-based correlation
    fft_signal = np.fft.fft(signal)
    fft_code = np.fft.fft(local_code)
    corr_fft = fft_signal * np.conj(fft_code)
    corr = np.fft.ifft(corr_fft)
    
    # Return magnitude
    return np.abs(corr)


def compute_autocorrelation(signal: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """
    Compute autocorrelation of signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    max_lag : int, optional
        Maximum lag to compute (default: len(signal)-1)
        
    Returns
    -------
    np.ndarray
        Autocorrelation values
        
    Notes
    -----
    Autocorrelation measures similarity of signal with delayed version of itself.
    Useful for detecting periodic patterns and self-similarity.
    """
    if max_lag is None:
        max_lag = len(signal) - 1
    
    autocorr = np.correlate(signal, signal, mode='full')
    center = len(autocorr) // 2
    return autocorr[center:center+max_lag+1]


def compute_crosscorrelation(
    signal1: np.ndarray,
    signal2: np.ndarray
) -> np.ndarray:
    """
    Compute cross-correlation between two signals.
    
    Parameters
    ----------
    signal1 : np.ndarray
        First signal
    signal2 : np.ndarray
        Second signal
        
    Returns
    -------
    np.ndarray
        Cross-correlation values
        
    Notes
    -----
    Cross-correlation measures similarity between two signals at different lags.
    In GPS, used to correlate received signal with local PRN replica.
    """
    return np.correlate(signal1, signal2, mode='full')


def extract_peak_metrics(
    correlation: np.ndarray,
    samples_per_chip: int,
    fs: Optional[float] = None
) -> Dict[str, float]:
    """
    Extract comprehensive metrics from correlation peak.
    
    Parameters
    ----------
    correlation : np.ndarray
        Correlation magnitude array
    samples_per_chip : int
        Number of samples per C/A code chip
    fs : float, optional
        Sampling frequency (for time-based metrics)
        
    Returns
    -------
    dict
        Dictionary with peak metrics:
        - peak_value: Maximum correlation value
        - peak_index: Index of peak
        - peak_to_secondary: Ratio of main peak to second highest peak
        - peak_to_mean: Ratio of peak to mean correlation
        - fwhm: Full Width at Half Maximum (samples)
        - fwhm_chips: FWHM in chips (if fs provided)
        - fpw: Fractional Peak Width at 80% (samples)
        - asymmetry: Asymmetry of peak (left-right imbalance)
        - skewness: Statistical skewness around peak
        - kurtosis: Statistical kurtosis around peak
        - energy_around_peak: Integral in window around peak
        - peak_sharpness: Second derivative at peak
        - sidelobe_level: Maximum sidelobe level
        
    Notes
    -----
    These metrics capture the "shape" of the correlation peak, which
    is distorted by spoofing attacks. Key indicators:
    
    - Peak-to-secondary ratio: Decreases with spoofing (multiple peaks)
    - Asymmetry: Increases when genuine and spoofed signals overlap
    - FWHM: Increases with multipath or multiple signal sources
    - Skewness/Kurtosis: Deviation from ideal triangular shape
    """
    metrics = {}
    
    # Basic peak detection
    peak_index = np.argmax(correlation)
    peak_value = correlation[peak_index]
    metrics['peak_value'] = float(peak_value)
    metrics['peak_index'] = int(peak_index)
    
    # Peak-to-mean ratio
    mean_corr = np.mean(correlation)
    metrics['peak_to_mean'] = float(peak_value / mean_corr) if mean_corr > 0 else 0.0
    
    # Find secondary peak (exclude main peak region)
    peak_window_samples = int(2 * samples_per_chip)
    temp_corr = correlation.copy()
    
    # Zero out main peak region
    start_exclude = max(0, peak_index - peak_window_samples)
    end_exclude = min(len(correlation), peak_index + peak_window_samples)
    temp_corr[start_exclude:end_exclude] = 0
    
    secondary_peak_value = np.max(temp_corr)
    metrics['secondary_peak_value'] = float(secondary_peak_value)
    
    # Peak-to-secondary ratio (critical for spoofing detection)
    if secondary_peak_value > 0:
        metrics['peak_to_secondary'] = float(peak_value / secondary_peak_value)
    else:
        metrics['peak_to_secondary'] = 999.0  # Very high ratio (good)
    
    # Full Width at Half Maximum (FWHM)
    half_max = peak_value / 2.0
    above_half = np.where(correlation > half_max)[0]
    if len(above_half) > 0:
        fwhm = above_half[-1] - above_half[0]
        metrics['fwhm'] = int(fwhm)
        if fs is not None:
            chip_rate = 1.023e6
            metrics['fwhm_chips'] = float(fwhm / samples_per_chip)
    else:
        metrics['fwhm'] = 0
        metrics['fwhm_chips'] = 0.0
    
    # Fractional Peak Width (80% level)
    frac_level = 0.8 * peak_value
    above_frac = np.where(correlation > frac_level)[0]
    if len(above_frac) > 0:
        fpw = above_frac[-1] - above_frac[0]
        metrics['fpw'] = int(fpw)
    else:
        metrics['fpw'] = 0
    
    # Asymmetry (left-right balance around peak)
    left_start = max(0, peak_index - samples_per_chip)
    right_end = min(len(correlation), peak_index + samples_per_chip + 1)
    
    left_area = np.sum(correlation[left_start:peak_index])
    right_area = np.sum(correlation[peak_index+1:right_end])
    
    total_area = left_area + right_area
    if total_area > 0:
        asymmetry = (right_area - left_area) / total_area
        metrics['asymmetry'] = float(asymmetry)
    else:
        metrics['asymmetry'] = 0.0
    
    # Extract window around peak for statistical analysis
    window_size = 4 * samples_per_chip
    window_start = max(0, peak_index - window_size // 2)
    window_end = min(len(correlation), peak_index + window_size // 2)
    peak_window = correlation[window_start:window_end]
    
    # Skewness (asymmetry measure)
    if len(peak_window) > 2:
        metrics['skewness'] = float(skew(peak_window))
    else:
        metrics['skewness'] = 0.0
    
    # Kurtosis (tail weight / peakedness)
    if len(peak_window) > 3:
        metrics['kurtosis'] = float(kurtosis(peak_window))
    else:
        metrics['kurtosis'] = 0.0
    
    # Energy around peak
    metrics['energy_around_peak'] = float(np.sum(peak_window))
    
    # Peak sharpness (second derivative at peak)
    if peak_index > 0 and peak_index < len(correlation) - 1:
        second_deriv = correlation[peak_index-1] - 2*correlation[peak_index] + correlation[peak_index+1]
        metrics['peak_sharpness'] = float(abs(second_deriv))
    else:
        metrics['peak_sharpness'] = 0.0
    
    # Sidelobe level (maximum value outside main peak)
    sidelobe_mask = np.ones(len(correlation), dtype=bool)
    sidelobe_mask[start_exclude:end_exclude] = False
    if np.any(sidelobe_mask):
        metrics['sidelobe_level'] = float(np.max(correlation[sidelobe_mask]))
    else:
        metrics['sidelobe_level'] = 0.0
    
    return metrics


def extract_temporal_gradient(
    correlation_history: list,
    metric_name: str = 'peak_value'
) -> Dict[str, float]:
    """
    Extract temporal gradient features from correlation history.
    
    Parameters
    ----------
    correlation_history : list
        List of correlation metrics from consecutive windows
    metric_name : str, optional
        Name of metric to analyze (default: 'peak_value')
        
    Returns
    -------
    dict
        Temporal gradient features:
        - mean_gradient: Average rate of change
        - std_gradient: Variability of change
        - max_gradient: Maximum change between windows
        - trend: Overall trend (positive/negative)
        
    Notes
    -----
    Spoofing attacks often cause sudden changes in correlation metrics.
    Temporal gradients can detect these transitions.
    """
    if len(correlation_history) < 2:
        return {
            'mean_gradient': 0.0,
            'std_gradient': 0.0,
            'max_gradient': 0.0,
            'trend': 0.0
        }
    
    # Extract metric values
    values = [metrics[metric_name] for metrics in correlation_history]
    
    # Compute gradients (differences)
    gradients = np.diff(values)
    
    return {
        'mean_gradient': float(np.mean(gradients)),
        'std_gradient': float(np.std(gradients)),
        'max_gradient': float(np.max(np.abs(gradients))),
        'trend': float(values[-1] - values[0]) if len(values) > 1 else 0.0
    }


def compute_correlation_profile(
    signal: np.ndarray,
    local_code: np.ndarray,
    normalize: bool = True
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute full correlation profile with metrics.
    
    Parameters
    ----------
    signal : np.ndarray
        Received complex signal
    local_code : np.ndarray
        Local PRN code
    normalize : bool, optional
        Normalize correlation output (default: True)
        
    Returns
    -------
    correlation : np.ndarray
        Correlation magnitude
    metrics : dict
        Correlation metrics
        
    Notes
    -----
    This is a convenience function that combines correlation computation
    and metric extraction in one call.
    """
    # Compute correlation
    correlation = compute_correlation_fft(signal, local_code)
    
    # Normalize if requested
    if normalize:
        max_val = np.max(correlation)
        if max_val > 0:
            correlation = correlation / max_val
    
    # Extract metrics (assume standard GPS sampling)
    # This is a simplified version; real implementation should get samples_per_chip from config
    samples_per_chip = len(signal) // 1023  # Approximate
    if samples_per_chip < 1:
        samples_per_chip = 1
    
    metrics = extract_peak_metrics(correlation, samples_per_chip)
    
    return correlation, metrics
