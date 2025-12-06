"""
Temporal and power-based feature extraction.
"""
import numpy as np
from typing import Dict


def extract_temporal_features(signal: np.ndarray, fs: float, 
                               correlation_peak: float = 0.0,
                               correlation_secondary: float = 0.0) -> Dict[str, float]:
    """
    Extract temporal and power-based features from signal.
    
    Args:
        signal: Complex IQ signal
        fs: Sampling frequency (Hz)
        correlation_peak: Peak correlation value (for C/N0 estimation)
        correlation_secondary: Secondary peak value
    
    Returns:
        Dictionary of temporal features
    
    Features extracted:
        - mean_real: Mean of real component
        - mean_imag: Mean of imaginary component
        - var_real: Variance of real component
        - var_imag: Variance of imaginary component
        - std_amplitude: Standard deviation of amplitude
        - total_power: Mean signal power
        - cn0_estimate: Estimated C/N0 (dB-Hz)
        - noise_floor: Estimated noise floor
        - snr_estimate: Estimated SNR
    """
    features = {}
    
    # Basic statistics
    features['mean_real'] = float(np.mean(np.real(signal)))
    features['mean_imag'] = float(np.mean(np.imag(signal)))
    features['var_real'] = float(np.var(np.real(signal)))
    features['var_imag'] = float(np.var(np.imag(signal)))
    features['std_amplitude'] = float(np.std(np.abs(signal)))
    
    # Power metrics
    total_power = np.mean(np.abs(signal)**2)
    features['total_power'] = float(total_power)
    
    # C/N0 estimation (adapted from extract_features.py)
    if correlation_peak > 0:
        carrier_power_proxy = correlation_peak**2 / len(signal)
        noise_power_est = total_power - carrier_power_proxy
        if noise_power_est <= 0:
            noise_power_est = 1e-12
        
        # C/N0 = 10*log10(C / (N0 * BW))
        cn0_estimate = 10 * np.log10(carrier_power_proxy / (noise_power_est / fs))
        features['cn0_estimate'] = float(cn0_estimate)
        features['noise_floor'] = float(noise_power_est)
        
        # SNR estimate
        if noise_power_est > 0:
            snr = carrier_power_proxy / noise_power_est
            features['snr_estimate'] = float(10 * np.log10(snr)) if snr > 0 else -100.0
        else:
            features['snr_estimate'] = 0.0
    else:
        features['cn0_estimate'] = 0.0
        features['noise_floor'] = float(total_power)
        features['snr_estimate'] = 0.0
    
    # Amplitude statistics
    amplitude = np.abs(signal)
    features['max_amplitude'] = float(np.max(amplitude))
    features['min_amplitude'] = float(np.min(amplitude))
    features['median_amplitude'] = float(np.median(amplitude))
    
    # Phase statistics
    phase = np.angle(signal)
    features['mean_phase'] = float(np.mean(phase))
    features['std_phase'] = float(np.std(phase))
    
    # Frequency domain features (simple)
    fft_mag = np.abs(np.fft.fft(signal))
    features['spectral_mean'] = float(np.mean(fft_mag))
    features['spectral_std'] = float(np.std(fft_mag))
    features['spectral_max'] = float(np.max(fft_mag))
    
    return features


def extract_cn0_variation_features(signals: list, fs: float, ca_chip_rate: float = 1.023e6) -> Dict[str, float]:
    """
    Extract C/N0 variation features across multiple signal windows.
    
    Args:
        signals: List of signal windows
        fs: Sampling frequency (Hz)
        ca_chip_rate: C/A code chip rate (Hz)
    
    Returns:
        Dictionary with C/N0 variation features
    """
    features = {}
    
    if len(signals) < 2:
        features['cn0_mean'] = 0.0
        features['cn0_std'] = 0.0
        features['cn0_trend'] = 0.0
        return features
    
    # For each window, estimate C/N0 (simplified)
    cn0_values = []
    for signal in signals:
        total_power = np.mean(np.abs(signal)**2)
        # Very simplified - would need proper correlation
        cn0_est = 10 * np.log10(total_power * fs / 1000) if total_power > 0 else 0.0
        cn0_values.append(cn0_est)
    
    cn0_values = np.array(cn0_values)
    features['cn0_mean'] = float(np.mean(cn0_values))
    features['cn0_std'] = float(np.std(cn0_values))
    
    # Trend (linear fit)
    if len(cn0_values) > 1:
        x = np.arange(len(cn0_values))
        trend = np.polyfit(x, cn0_values, 1)[0]
        features['cn0_trend'] = float(trend)
    else:
        features['cn0_trend'] = 0.0
    
    return features
