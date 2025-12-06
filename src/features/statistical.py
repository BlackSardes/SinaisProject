"""
Statistical Feature Extraction for GPS Signals

This module provides statistical and temporal features that complement
correlation-based features for spoofing detection.
"""

import numpy as np
from scipy.stats import skew, kurtosis, entropy
from typing import Dict, Optional


def extract_power_features(
    signal: np.ndarray,
    fs: float,
    peak_value: Optional[float] = None,
    secondary_peak: Optional[float] = None
) -> Dict[str, float]:
    """
    Extract power-related features from signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Preprocessed complex signal
    fs : float
        Sampling frequency in Hz
    peak_value : float, optional
        Correlation peak value (if available)
    secondary_peak : float, optional
        Secondary correlation peak value (if available)
        
    Returns
    -------
    dict
        Power features:
        - total_power: Total signal power
        - carrier_power: Estimated carrier power
        - noise_power: Estimated noise power
        - cn0_estimate: C/N0 in dB-Hz
        - snr_estimate: Signal-to-noise ratio in dB
        - mean_real: Mean of real component
        - mean_imag: Mean of imaginary component
        - std_real: Std of real component
        - std_imag: Std of imaginary component
        - std_amplitude: Std of signal magnitude
        - rms_amplitude: RMS amplitude
        
    Notes
    -----
    Power metrics are critical for spoofing detection as most attacks
    involve higher power signals to overpower the authentic signal.
    """
    features = {}
    
    # Total power
    total_power = np.mean(np.abs(signal) ** 2)
    features['total_power'] = float(total_power)
    
    # Estimate carrier power from correlation peak if available
    if peak_value is not None:
        carrier_power = (peak_value ** 2) / len(signal)
        features['carrier_power'] = float(carrier_power)
    else:
        # Rough estimate
        carrier_power = total_power * 0.1
        features['carrier_power'] = float(carrier_power)
    
    # Noise power estimate
    noise_power = total_power - carrier_power
    if noise_power <= 0:
        noise_power = 1e-12
    features['noise_power'] = float(noise_power)
    
    # C/N0 estimate
    cn0_linear = carrier_power / (noise_power / fs)
    features['cn0_estimate'] = float(10 * np.log10(cn0_linear)) if cn0_linear > 0 else -np.inf
    
    # SNR estimate
    snr_linear = carrier_power / noise_power
    features['snr_estimate'] = float(10 * np.log10(snr_linear)) if snr_linear > 0 else -np.inf
    
    # Component statistics
    features['mean_real'] = float(np.mean(np.real(signal)))
    features['mean_imag'] = float(np.mean(np.imag(signal)))
    features['std_real'] = float(np.std(np.real(signal)))
    features['std_imag'] = float(np.std(np.imag(signal)))
    
    # Amplitude statistics
    amplitude = np.abs(signal)
    features['std_amplitude'] = float(np.std(amplitude))
    features['rms_amplitude'] = float(np.sqrt(np.mean(amplitude ** 2)))
    
    return features


def extract_statistical_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Extract general statistical features from signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Complex signal
        
    Returns
    -------
    dict
        Statistical features:
        - mean_magnitude: Mean of signal magnitude
        - std_magnitude: Standard deviation of magnitude
        - max_magnitude: Maximum magnitude
        - min_magnitude: Minimum magnitude
        - median_magnitude: Median magnitude
        - skewness_magnitude: Skewness of magnitude distribution
        - kurtosis_magnitude: Kurtosis of magnitude distribution
        - entropy_magnitude: Entropy of magnitude histogram
        - mean_phase: Mean phase angle
        - std_phase: Standard deviation of phase
        - phase_discontinuities: Count of large phase jumps
        
    Notes
    -----
    Statistical features capture distributional properties that may
    change under spoofing conditions.
    """
    features = {}
    
    # Magnitude statistics
    magnitude = np.abs(signal)
    features['mean_magnitude'] = float(np.mean(magnitude))
    features['std_magnitude'] = float(np.std(magnitude))
    features['max_magnitude'] = float(np.max(magnitude))
    features['min_magnitude'] = float(np.min(magnitude))
    features['median_magnitude'] = float(np.median(magnitude))
    
    # Higher-order statistics
    if len(magnitude) > 2:
        features['skewness_magnitude'] = float(skew(magnitude))
    else:
        features['skewness_magnitude'] = 0.0
    
    if len(magnitude) > 3:
        features['kurtosis_magnitude'] = float(kurtosis(magnitude))
    else:
        features['kurtosis_magnitude'] = 0.0
    
    # Entropy (measure of randomness)
    hist, _ = np.histogram(magnitude, bins=50, density=True)
    hist = hist + 1e-12  # Avoid log(0)
    features['entropy_magnitude'] = float(entropy(hist))
    
    # Phase statistics
    phase = np.angle(signal)
    features['mean_phase'] = float(np.mean(phase))
    features['std_phase'] = float(np.std(phase))
    
    # Phase discontinuities (large jumps)
    phase_diff = np.diff(phase)
    phase_diff = np.abs(phase_diff)
    discontinuities = np.sum(phase_diff > np.pi / 2)  # Jumps larger than 90 degrees
    features['phase_discontinuities'] = int(discontinuities)
    
    return features


def extract_spectral_features(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extract frequency-domain features from signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Complex signal
    fs : float
        Sampling frequency in Hz
        
    Returns
    -------
    dict
        Spectral features:
        - spectral_centroid: Center frequency of spectrum
        - spectral_spread: Spread of spectrum
        - spectral_flatness: Flatness measure (tone vs noise)
        - spectral_rolloff: Frequency below which 85% of energy
        - peak_frequency: Frequency of maximum spectral component
        - bandwidth_90: Bandwidth containing 90% of energy
        
    Notes
    -----
    Spectral features can detect frequency-domain anomalies caused
    by spoofing signals with different Doppler shifts or modulation.
    """
    features = {}
    
    # Compute spectrum
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    magnitude_spectrum = np.abs(spectrum)
    power_spectrum = magnitude_spectrum ** 2
    
    # Frequency bins
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/fs))
    
    # Normalize power spectrum
    total_power = np.sum(power_spectrum)
    if total_power > 0:
        power_spectrum_norm = power_spectrum / total_power
    else:
        power_spectrum_norm = power_spectrum
    
    # Spectral centroid (center of mass in frequency)
    centroid = np.sum(freqs * power_spectrum_norm)
    features['spectral_centroid'] = float(centroid)
    
    # Spectral spread (standard deviation in frequency)
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * power_spectrum_norm))
    features['spectral_spread'] = float(spread)
    
    # Spectral flatness (geometric mean / arithmetic mean)
    # Measure of how "tone-like" vs "noise-like" the spectrum is
    geometric_mean = np.exp(np.mean(np.log(magnitude_spectrum + 1e-12)))
    arithmetic_mean = np.mean(magnitude_spectrum)
    flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0
    features['spectral_flatness'] = float(flatness)
    
    # Peak frequency
    peak_idx = np.argmax(magnitude_spectrum)
    features['peak_frequency'] = float(freqs[peak_idx])
    
    # Spectral rolloff (frequency below which 85% of energy)
    cumsum_power = np.cumsum(power_spectrum_norm)
    rolloff_idx = np.where(cumsum_power >= 0.85)[0]
    if len(rolloff_idx) > 0:
        features['spectral_rolloff'] = float(freqs[rolloff_idx[0]])
    else:
        features['spectral_rolloff'] = float(freqs[-1])
    
    # Bandwidth containing 90% of energy
    indices_90 = np.where((cumsum_power >= 0.05) & (cumsum_power <= 0.95))[0]
    if len(indices_90) > 0:
        bandwidth_90 = freqs[indices_90[-1]] - freqs[indices_90[0]]
        features['bandwidth_90'] = float(abs(bandwidth_90))
    else:
        features['bandwidth_90'] = 0.0
    
    return features


def extract_temporal_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Extract temporal features from signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Complex signal
        
    Returns
    -------
    dict
        Temporal features:
        - zero_crossing_rate: Rate of zero crossings
        - autocorr_lag1: Autocorrelation at lag 1
        - autocorr_lag10: Autocorrelation at lag 10
        - rms_derivative: RMS of signal derivative
        - energy: Total signal energy
        - peak_to_average: Peak-to-average power ratio
        
    Notes
    -----
    Temporal features capture time-domain characteristics that
    may be affected by spoofing or multipath.
    """
    features = {}
    
    # Work with magnitude
    magnitude = np.abs(signal)
    
    # Zero crossing rate
    real_part = np.real(signal)
    zero_crossings = np.sum(np.diff(np.sign(real_part)) != 0)
    features['zero_crossing_rate'] = float(zero_crossings / len(signal))
    
    # Autocorrelation at specific lags
    autocorr = np.correlate(magnitude - np.mean(magnitude), magnitude - np.mean(magnitude), mode='full')
    autocorr = autocorr / np.max(autocorr)  # Normalize
    center = len(autocorr) // 2
    
    if center + 1 < len(autocorr):
        features['autocorr_lag1'] = float(autocorr[center + 1])
    else:
        features['autocorr_lag1'] = 0.0
    
    if center + 10 < len(autocorr):
        features['autocorr_lag10'] = float(autocorr[center + 10])
    else:
        features['autocorr_lag10'] = 0.0
    
    # Derivative (rate of change)
    derivative = np.diff(magnitude)
    features['rms_derivative'] = float(np.sqrt(np.mean(derivative ** 2)))
    
    # Energy
    features['energy'] = float(np.sum(magnitude ** 2))
    
    # Peak-to-average ratio (crest factor)
    average_power = np.mean(magnitude ** 2)
    peak_power = np.max(magnitude) ** 2
    features['peak_to_average'] = float(peak_power / average_power) if average_power > 0 else 0.0
    
    return features


def extract_all_statistical_features(
    signal: np.ndarray,
    fs: float,
    peak_value: Optional[float] = None,
    secondary_peak: Optional[float] = None
) -> Dict[str, float]:
    """
    Extract all statistical, spectral, and temporal features.
    
    Parameters
    ----------
    signal : np.ndarray
        Complex signal
    fs : float
        Sampling frequency in Hz
    peak_value : float, optional
        Correlation peak value
    secondary_peak : float, optional
        Secondary correlation peak value
        
    Returns
    -------
    dict
        Combined feature dictionary
    """
    features = {}
    
    # Power features
    features.update(extract_power_features(signal, fs, peak_value, secondary_peak))
    
    # Statistical features
    features.update(extract_statistical_features(signal))
    
    # Spectral features
    features.update(extract_spectral_features(signal, fs))
    
    # Temporal features
    features.update(extract_temporal_features(signal))
    
    return features
