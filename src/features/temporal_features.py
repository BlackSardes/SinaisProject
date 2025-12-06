"""
Temporal feature extraction from GPS signals.
"""
import numpy as np
from typing import Dict, List, Optional
from ..preprocessing.cn0_estimation import estimate_cn0_from_signal, estimate_cn0_variation


def extract_temporal_features(
    signal: np.ndarray,
    fs: float,
    feature_names: Optional[list] = None
) -> Dict[str, float]:
    """
    Extract temporal/statistical features from signal.
    
    Args:
        signal: Complex I/Q signal
        fs: Sampling frequency in Hz
        feature_names: List of features to extract (None = all)
    
    Returns:
        Dictionary of feature name -> value
        
    Available features:
        - mean_i: Mean of I component
        - mean_q: Mean of Q component
        - std_i: Standard deviation of I
        - std_q: Standard deviation of Q
        - mean_amplitude: Mean amplitude
        - std_amplitude: Standard deviation of amplitude
        - mean_power: Mean power
        - snr_estimate: Simple SNR estimate
        - cn0_estimate: C/N0 estimate
        
    Example:
        >>> features = extract_temporal_features(signal, fs=5e6)
        >>> print(features['snr_estimate'])
        15.3
    """
    I = np.real(signal)
    Q = np.imag(signal)
    amplitude = np.abs(signal)
    power = amplitude ** 2
    
    all_features = {
        'mean_i': lambda: float(np.mean(I)),
        'mean_q': lambda: float(np.mean(Q)),
        'std_i': lambda: float(np.std(I)),
        'std_q': lambda: float(np.std(Q)),
        'mean_amplitude': lambda: float(np.mean(amplitude)),
        'std_amplitude': lambda: float(np.std(amplitude)),
        'mean_power': lambda: float(np.mean(power)),
        'snr_estimate': lambda: _compute_snr_estimate(signal),
        'cn0_estimate': lambda: estimate_cn0_from_signal(signal, fs, method='snv'),
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


def _compute_snr_estimate(signal: np.ndarray) -> float:
    """
    Simple SNR estimate from signal statistics.
    
    Args:
        signal: Complex I/Q signal
    
    Returns:
        SNR estimate in dB
    """
    power = np.mean(np.abs(signal) ** 2)
    variance = np.var(np.abs(signal))
    
    if variance > 0:
        snr_linear = power / variance
        snr_db = 10 * np.log10(snr_linear)
        return float(snr_db)
    else:
        return 0.0


def compute_cn0_variation_features(
    signal: np.ndarray,
    fs: float,
    window_s: float = 0.1,
    hop_s: float = 0.05
) -> Dict[str, float]:
    """
    Compute C/N0 variation features over time.
    
    Temporal variations in C/N0 can indicate spoofing attacks
    or signal degradation events.
    
    Args:
        signal: Complex I/Q signal
        fs: Sampling frequency in Hz
        window_s: Window duration for C/N0 estimation
        hop_s: Hop size between windows
    
    Returns:
        Dictionary with variation statistics:
        - cn0_mean: Mean C/N0 over time
        - cn0_std: Standard deviation of C/N0
        - cn0_min: Minimum C/N0
        - cn0_max: Maximum C/N0
        - cn0_range: Range of C/N0 values
        - cn0_trend: Linear trend coefficient
        
    Example:
        >>> features = compute_cn0_variation_features(signal, fs=5e6)
        >>> print(f"C/N0 variation: {features['cn0_std']:.2f} dB-Hz")
    """
    cn0_values, time_points, variation_std = estimate_cn0_variation(
        signal, fs, window_s, hop_s
    )
    
    if len(cn0_values) < 2:
        return {
            'cn0_mean': float(cn0_values[0]) if len(cn0_values) > 0 else 0.0,
            'cn0_std': 0.0,
            'cn0_min': float(cn0_values[0]) if len(cn0_values) > 0 else 0.0,
            'cn0_max': float(cn0_values[0]) if len(cn0_values) > 0 else 0.0,
            'cn0_range': 0.0,
            'cn0_trend': 0.0,
        }
    
    # Compute trend using linear regression
    coeffs = np.polyfit(time_points, cn0_values, deg=1)
    trend = coeffs[0]  # Slope
    
    return {
        'cn0_mean': float(np.mean(cn0_values)),
        'cn0_std': float(variation_std),
        'cn0_min': float(np.min(cn0_values)),
        'cn0_max': float(np.max(cn0_values)),
        'cn0_range': float(np.max(cn0_values) - np.min(cn0_values)),
        'cn0_trend': float(trend),
    }


def compute_peak_gradient_over_time(
    correlation_profiles: List[np.ndarray],
    time_points: np.ndarray
) -> Dict[str, float]:
    """
    Compute temporal gradient of correlation peak characteristics.
    
    Analyzes how the correlation peak changes over time, useful for
    detecting gradual spoofing attacks.
    
    Args:
        correlation_profiles: List of correlation magnitude profiles
        time_points: Time stamps for each profile
    
    Returns:
        Dictionary with gradient features:
        - peak_height_gradient: Change rate of peak height
        - peak_width_gradient: Change rate of peak width
        - peak_position_gradient: Change rate of peak position
        
    Example:
        >>> features = compute_peak_gradient_over_time(corr_profiles, times)
    """
    if len(correlation_profiles) < 2:
        return {
            'peak_height_gradient': 0.0,
            'peak_width_gradient': 0.0,
            'peak_position_gradient': 0.0,
        }
    
    # Extract peak metrics over time
    peak_heights = []
    peak_positions = []
    peak_widths = []
    
    for corr in correlation_profiles:
        peak_idx = np.argmax(corr)
        peak_val = corr[peak_idx]
        
        peak_heights.append(peak_val)
        peak_positions.append(peak_idx)
        
        # Compute width at half max
        threshold = 0.5 * peak_val
        above = corr > threshold
        if np.any(above):
            indices = np.where(above)[0]
            width = indices[-1] - indices[0] + 1
        else:
            width = 0
        peak_widths.append(width)
    
    # Compute gradients
    peak_heights = np.array(peak_heights)
    peak_positions = np.array(peak_positions)
    peak_widths = np.array(peak_widths)
    
    # Use linear fit to get gradient
    if len(time_points) > 1:
        height_grad = np.polyfit(time_points, peak_heights, deg=1)[0]
        position_grad = np.polyfit(time_points, peak_positions, deg=1)[0]
        width_grad = np.polyfit(time_points, peak_widths, deg=1)[0]
    else:
        height_grad = 0.0
        position_grad = 0.0
        width_grad = 0.0
    
    return {
        'peak_height_gradient': float(height_grad),
        'peak_width_gradient': float(width_grad),
        'peak_position_gradient': float(position_grad),
    }
