"""Feature extraction for GPS spoofing detection."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from scipy.stats import skew, kurtosis

from .correlation import compute_cross_correlation
from .metrics import fwhm


def extract_correlation_features(corr_profile: np.ndarray, fs: float, 
                                ca_chip_rate: float = 1.023e6) -> Dict[str, float]:
    """
    Extract features from correlation profile.
    
    Parameters
    ----------
    corr_profile : np.ndarray
        Correlation profile (magnitude)
    fs : float
        Sampling frequency in Hz
    ca_chip_rate : float, default=1.023e6
        C/A code chip rate in Hz
    
    Returns
    -------
    dict
        Dictionary of extracted features
    """
    # Find peak
    peak_idx = np.argmax(np.abs(corr_profile))
    peak_value = np.abs(corr_profile[peak_idx])
    
    # FWHM
    fwhm_samples = fwhm(np.abs(corr_profile), peak_idx)
    fwhm_seconds = fwhm_samples / fs
    
    # Find secondary peak (exclude main peak region)
    samples_per_chip = int(fs / ca_chip_rate)
    peak_window = 2 * samples_per_chip
    
    temp_corr = np.abs(corr_profile).copy()
    start_excl = max(0, peak_idx - peak_window)
    end_excl = min(len(temp_corr), peak_idx + peak_window)
    temp_corr[start_excl:end_excl] = 0
    
    secondary_peak_idx = np.argmax(temp_corr)
    secondary_peak_value = temp_corr[secondary_peak_idx]
    
    # Ratio of first to second peak
    ratio_first_second = peak_value / secondary_peak_value if secondary_peak_value > 1e-12 else 999.0
    
    # Peak offset from center
    center_idx = len(corr_profile) // 2
    peak_offset_samples = peak_idx - center_idx
    peak_offset_seconds = peak_offset_samples / fs
    
    # Statistical features
    corr_mag = np.abs(corr_profile)
    
    return {
        'peak_height': float(peak_value),
        'fwhm_s': float(fwhm_seconds),
        'ratio_first_second': float(ratio_first_second),
        'peak_offset_s': float(peak_offset_seconds),
        'skewness': float(skew(corr_mag)),
        'kurtosis': float(kurtosis(corr_mag)),
    }


def build_feature_vector(windows: List[np.ndarray], prn: int, fs: float,
                        reference_code: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Build feature vectors from signal windows.
    
    Parameters
    ----------
    windows : List[np.ndarray]
        List of signal windows
    prn : int
        PRN number of satellite
    fs : float
        Sampling frequency in Hz
    reference_code : np.ndarray, optional
        Reference C/A code for correlation. If None, features based on
        signal statistics only.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with features for each window:
        - peak_height: Peak correlation value
        - fwhm_s: Full width at half maximum (seconds)
        - skewness: Skewness of correlation profile
        - kurtosis: Kurtosis of correlation profile
        - energy_window: Signal energy in window
        - ratio_first_second: Ratio of primary to secondary peak
        - peak_offset_s: Peak offset from center (seconds)
        - mean: Mean signal magnitude
        - var: Signal variance
        - snr_est: Estimated SNR
    """
    features_list = []
    
    for idx, window in enumerate(windows):
        feature_dict = {'window_idx': idx, 'prn': prn}
        
        # Basic signal statistics
        if np.iscomplexobj(window):
            magnitude = np.abs(window)
        else:
            magnitude = window
        
        feature_dict['mean'] = float(np.mean(magnitude))
        feature_dict['var'] = float(np.var(magnitude))
        feature_dict['energy_window'] = float(np.sum(magnitude ** 2))
        
        # Simple SNR estimate
        signal_power = np.mean(magnitude ** 2)
        noise_power = np.var(magnitude)
        if noise_power > 1e-12:
            snr_linear = signal_power / noise_power
            feature_dict['snr_est'] = float(10 * np.log10(snr_linear))
        else:
            feature_dict['snr_est'] = 0.0
        
        # If reference code provided, compute correlation features
        if reference_code is not None:
            # Ensure same length
            if len(window) != len(reference_code):
                # Resample or trim
                ref_len = len(reference_code)
                if len(window) > ref_len:
                    window_use = window[:ref_len]
                else:
                    # Repeat reference code
                    repeats = int(np.ceil(len(window) / ref_len))
                    reference_code_use = np.tile(reference_code, repeats)[:len(window)]
                    window_use = window
            else:
                window_use = window
                reference_code_use = reference_code
            
            # Compute correlation
            corr = compute_cross_correlation(window_use, reference_code_use, mode='fft')
            corr_mag = np.abs(corr)
            
            # Extract correlation features
            corr_features = extract_correlation_features(corr_mag, fs)
            feature_dict.update(corr_features)
        else:
            # Add placeholder values
            feature_dict.update({
                'peak_height': 0.0,
                'fwhm_s': 0.0,
                'ratio_first_second': 1.0,
                'peak_offset_s': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
            })
        
        features_list.append(feature_dict)
    
    return pd.DataFrame(features_list)
