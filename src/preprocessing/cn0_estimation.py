"""
Carrier-to-Noise Density Ratio (C/N0) estimation functions.

C/N0 is a critical metric in GPS signal processing that indicates
signal quality and is often affected by spoofing attacks.
"""
import numpy as np
from typing import Optional


def estimate_cn0_from_correlation(
    corr_profile: np.ndarray,
    fs: float,
    coherent_integration_time: float = 0.001
) -> float:
    """
    Estimate C/N0 from correlation profile.
    
    Uses the relationship between correlation peak power and noise floor
    to estimate the carrier-to-noise density ratio.
    
    Args:
        corr_profile: Correlation magnitude profile
        fs: Sampling frequency in Hz
        coherent_integration_time: Integration time in seconds (default: 1 ms)
    
    Returns:
        C/N0 estimate in dB-Hz
        
    Example:
        >>> cn0 = estimate_cn0_from_correlation(corr_mag, fs=5e6)
        >>> print(f"C/N0: {cn0:.1f} dB-Hz")
        C/N0: 45.2 dB-Hz
    """
    # Find peak (carrier power proxy)
    peak_value = np.max(corr_profile)
    peak_index = np.argmax(corr_profile)
    
    # Estimate noise floor (exclude region around peak)
    noise_region = corr_profile.copy()
    peak_width = int(fs * coherent_integration_time)
    start = max(0, peak_index - peak_width)
    end = min(len(noise_region), peak_index + peak_width)
    noise_region[start:end] = np.nan
    
    noise_floor = np.nanmean(noise_region)
    
    # Protect against invalid values
    if noise_floor <= 0 or peak_value <= noise_floor:
        return 0.0
    
    # C/N0 calculation
    # peak_value^2 ~ carrier power * integration_time
    # noise_floor ~ noise power
    carrier_power = (peak_value ** 2) / len(corr_profile)
    noise_power = noise_floor / len(corr_profile)
    
    cn0 = 10 * np.log10(carrier_power / noise_power) + 10 * np.log10(fs)
    
    return float(cn0)


def estimate_cn0_from_signal(
    signal: np.ndarray,
    fs: float,
    method: str = 'snv'
) -> float:
    """
    Estimate C/N0 directly from signal without correlation.
    
    Supports multiple estimation methods suitable for different scenarios.
    
    Args:
        signal: Complex I/Q signal
        fs: Sampling frequency in Hz
        method: Estimation method
            - 'snv': Signal-to-Noise Variance ratio (Narrow-Wide correlator)
            - 'moment': Second and fourth moment method
            - 'beaulieu': Beaulieu's narrowband-to-wideband power ratio
    
    Returns:
        C/N0 estimate in dB-Hz
        
    Example:
        >>> cn0 = estimate_cn0_from_signal(signal, fs=5e6, method='snv')
    """
    if method == 'snv':
        # Signal-to-Noise Variance (simple estimator)
        power = np.mean(np.abs(signal) ** 2)
        variance = np.var(np.abs(signal))
        
        if variance <= 0:
            return 0.0
        
        snr = power / np.sqrt(variance)
        cn0 = 10 * np.log10(snr) + 10 * np.log10(fs)
        
        return float(max(0.0, cn0))
    
    elif method == 'moment':
        # Second and fourth moment method
        amp = np.abs(signal)
        M2 = np.mean(amp ** 2)
        M4 = np.mean(amp ** 4)
        
        if M2 <= 0:
            return 0.0
        
        # For Gaussian noise: M4/M2^2 = 2
        # For signal+noise, ratio is different
        ratio = M4 / (M2 ** 2)
        
        # Simplified estimate (exact formula depends on modulation)
        if ratio < 2.0:
            snr_linear = (2.0 - ratio) / ratio
            cn0 = 10 * np.log10(snr_linear) + 10 * np.log10(fs)
            return float(max(0.0, cn0))
        else:
            return 0.0
    
    elif method == 'beaulieu':
        # Narrowband-to-wideband power ratio method
        # Split signal into narrow and wide bands
        fft_sig = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/fs)
        
        # Narrow band: central 10% of bandwidth
        narrow_mask = np.abs(freqs) < (0.05 * fs)
        wide_mask = np.abs(freqs) < (0.5 * fs)
        
        P_narrow = np.mean(np.abs(fft_sig[narrow_mask]) ** 2)
        P_wide = np.mean(np.abs(fft_sig[wide_mask]) ** 2)
        
        if P_wide <= 0 or P_narrow <= P_wide:
            return 0.0
        
        # Estimate C/N0
        ratio = P_narrow / P_wide
        cn0 = 10 * np.log10(ratio - 1) + 10 * np.log10(fs)
        
        return float(max(0.0, cn0))
    
    else:
        raise ValueError(f"Unknown C/N0 estimation method: {method}")


def estimate_cn0_variation(
    signal: np.ndarray,
    fs: float,
    window_s: float = 0.1,
    hop_s: float = 0.05
) -> tuple:
    """
    Estimate C/N0 variation over time.
    
    Computes C/N0 in sliding windows to detect temporal variations,
    which can indicate spoofing attacks or signal degradation.
    
    Args:
        signal: Complex I/Q signal
        fs: Sampling frequency in Hz
        window_s: Window duration in seconds
        hop_s: Hop size in seconds
    
    Returns:
        Tuple of (cn0_values, time_points, variation_std)
        
    Example:
        >>> cn0_vals, times, std = estimate_cn0_variation(signal, fs=5e6)
        >>> print(f"C/N0 std: {std:.2f} dB-Hz")
    """
    window_samples = int(window_s * fs)
    hop_samples = int(hop_s * fs)
    
    cn0_values = []
    time_points = []
    
    start = 0
    while start + window_samples <= len(signal):
        segment = signal[start:start + window_samples]
        cn0 = estimate_cn0_from_signal(segment, fs, method='snv')
        
        cn0_values.append(cn0)
        time_points.append(start / fs)
        
        start += hop_samples
    
    cn0_values = np.array(cn0_values)
    time_points = np.array(time_points)
    
    variation_std = np.std(cn0_values) if len(cn0_values) > 1 else 0.0
    
    return cn0_values, time_points, variation_std
