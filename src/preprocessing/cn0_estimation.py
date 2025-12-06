"""C/N0 (Carrier-to-Noise Density Ratio) estimation utilities."""

import numpy as np
from typing import Union


def estimate_cn0_from_correlation(corr_profile: np.ndarray, fs: float) -> float:
    """
    Estimate C/N0 from correlation profile.
    
    Parameters
    ----------
    corr_profile : np.ndarray
        Correlation profile (magnitude)
    fs : float
        Sampling frequency in Hz
    
    Returns
    -------
    float
        Estimated C/N0 in dB-Hz
    
    Notes
    -----
    This method estimates C/N0 based on the ratio of peak power to noise floor
    in the correlation profile, following standard GPS signal processing techniques.
    """
    # Find peak
    peak_value = np.max(np.abs(corr_profile))
    peak_idx = np.argmax(np.abs(corr_profile))
    
    # Estimate noise floor (exclude region around peak)
    peak_window = int(len(corr_profile) * 0.05)  # 5% window around peak
    mask = np.ones(len(corr_profile), dtype=bool)
    start_idx = max(0, peak_idx - peak_window)
    end_idx = min(len(corr_profile), peak_idx + peak_window)
    mask[start_idx:end_idx] = False
    
    if np.any(mask):
        noise_floor = np.mean(np.abs(corr_profile[mask]) ** 2)
    else:
        # Fallback: use median
        noise_floor = np.median(np.abs(corr_profile) ** 2)
    
    # Ensure noise floor is not zero
    if noise_floor < 1e-12:
        noise_floor = 1e-12
    
    # Signal power (peak)
    signal_power = peak_value ** 2 / len(corr_profile)
    
    # C/N0 = 10 * log10(signal_power / (noise_floor / fs))
    cn0 = 10 * np.log10(signal_power / (noise_floor / fs))
    
    return float(cn0)


def estimate_cn0_from_signal(sig: np.ndarray, fs: float, 
                            method: str = 'snr') -> float:
    """
    Estimate C/N0 directly from signal.
    
    Parameters
    ----------
    sig : np.ndarray
        Input signal (complex)
    fs : float
        Sampling frequency in Hz
    method : str, default='snr'
        Estimation method:
        - 'snr': Simple SNR-based estimation
        - 'beaulieu': Beaulieu's method (narrowband approximation)
    
    Returns
    -------
    float
        Estimated C/N0 in dB-Hz
    
    Notes
    -----
    Direct C/N0 estimation from raw signal is challenging and typically less
    accurate than correlation-based methods. This provides a rough estimate.
    """
    if method == 'snr':
        # Simple method: total power vs noise estimate
        total_power = np.mean(np.abs(sig) ** 2)
        
        # Estimate signal power using variance of phase
        # Assumes constant envelope signal
        envelope = np.abs(sig)
        carrier_power = np.mean(envelope) ** 2
        noise_power = total_power - carrier_power
        
        if noise_power <= 0:
            noise_power = total_power * 0.01  # Assume 1% noise
        
        cn0 = 10 * np.log10(carrier_power / (noise_power / fs))
        return float(cn0)
    
    elif method == 'beaulieu':
        # Narrowband approximation
        # Estimate based on moment method
        M2 = np.mean(np.abs(sig) ** 2)
        M4 = np.mean(np.abs(sig) ** 4)
        
        # Estimate SNR from moments
        if M2 > 1e-12:
            k = M4 / (M2 ** 2)
            # For Rician fading: k ≈ 2 + SNR
            snr_linear = max(0, k - 2)
            cn0 = 10 * np.log10(snr_linear * fs)
            return float(cn0)
        else:
            return -np.inf
    
    else:
        raise ValueError(f"Unknown method: {method}")
