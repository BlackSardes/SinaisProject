"""
Preprocessing Pipeline for GPS Signals

This module provides a complete preprocessing pipeline that combines
all preprocessing steps in the correct order.
"""

import numpy as np
from typing import Optional, Dict, Any
from .signal_io import load_signal
from .signal_processing import (
    normalize_signal,
    remove_dc,
    apply_frequency_correction,
    apply_pulse_blanking,
    apply_frequency_domain_interference_mitigation,
    bandpass_filter,
    notch_filter
)


def preprocess_signal(
    signal: np.ndarray,
    fs: float,
    config: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Complete preprocessing pipeline for a GPS signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Raw complex signal (I+jQ)
    fs : float
        Sampling frequency in Hz
    config : dict, optional
        Configuration dictionary with preprocessing options:
        - 'remove_dc': bool (default: True)
        - 'freq_correction': float in Hz (default: 0.0)
        - 'apply_notch': bool (default: False)
        - 'notch_freq': float in Hz (default: 1e6)
        - 'notch_q': float (default: 30.0)
        - 'apply_bandpass': bool (default: False)
        - 'bandpass_low': float in Hz (default: -2e6)
        - 'bandpass_high': float in Hz (default: 2e6)
        - 'pulse_blanking': bool (default: False)
        - 'pb_threshold': float (default: 4.0)
        - 'freq_domain_mitigation': bool (default: False)
        - 'fdm_threshold': float (default: 3.5)
        - 'normalize': str (default: 'power')
        
    Returns
    -------
    np.ndarray
        Preprocessed signal
        
    Notes
    -----
    Processing order (optimized for GPS):
    1. DC removal
    2. Frequency correction (IF + Doppler)
    3. Interference mitigation (pulse blanking, frequency domain)
    4. Filtering (notch, bandpass)
    5. Normalization
    """
    # Default configuration
    if config is None:
        config = {}
    
    # Extract configuration with defaults
    remove_dc_flag = config.get('remove_dc', True)
    freq_correction = config.get('freq_correction', 0.0)
    apply_notch_flag = config.get('apply_notch', False)
    notch_freq = config.get('notch_freq', 1e6)
    notch_q = config.get('notch_q', 30.0)
    apply_bandpass_flag = config.get('apply_bandpass', False)
    bandpass_low = config.get('bandpass_low', -2e6)
    bandpass_high = config.get('bandpass_high', 2e6)
    pulse_blanking_flag = config.get('pulse_blanking', False)
    pb_threshold = config.get('pb_threshold', 4.0)
    freq_domain_mitigation_flag = config.get('freq_domain_mitigation', False)
    fdm_threshold = config.get('fdm_threshold', 3.5)
    normalize_method = config.get('normalize', 'power')
    
    # Start preprocessing
    processed = signal.copy()
    
    # Step 1: Remove DC offset
    if remove_dc_flag:
        processed = remove_dc(processed)
    
    # Step 2: Frequency correction
    if freq_correction != 0.0:
        processed = apply_frequency_correction(processed, fs, freq_correction)
    
    # Step 3: Interference mitigation
    if pulse_blanking_flag:
        processed = apply_pulse_blanking(processed, pb_threshold)
    
    if freq_domain_mitigation_flag:
        processed = apply_frequency_domain_interference_mitigation(processed, fdm_threshold)
    
    # Step 4: Filtering
    if apply_notch_flag:
        processed = notch_filter(processed, fs, notch_freq, notch_q)
    
    if apply_bandpass_flag:
        processed = bandpass_filter(processed, fs, bandpass_low, bandpass_high)
    
    # Step 5: Normalization (always do last)
    processed = normalize_signal(processed, method=normalize_method)
    
    return processed


def preprocess_file_segment(
    file_path: str,
    start_offset_samples: int,
    num_samples: int,
    fs: float,
    config: Optional[Dict[str, Any]] = None
) -> Optional[np.ndarray]:
    """
    Load and preprocess a segment from a file.
    
    Parameters
    ----------
    file_path : str
        Path to signal file
    start_offset_samples : int
        Starting sample position
    num_samples : int
        Number of samples to read
    fs : float
        Sampling frequency in Hz
    config : dict, optional
        Preprocessing configuration (see preprocess_signal)
        
    Returns
    -------
    np.ndarray or None
        Preprocessed signal segment or None on error
    """
    # Load signal segment
    signal = load_signal(file_path, start_offset_samples, num_samples)
    
    if signal is None:
        return None
    
    # Check if we got enough samples
    if len(signal) < num_samples:
        print(f"Warning: Only got {len(signal)} samples instead of {num_samples}")
        return None
    
    # Preprocess
    return preprocess_signal(signal, fs, config)


def create_preprocessing_config(
    preset: str = 'default'
) -> Dict[str, Any]:
    """
    Create preprocessing configuration from preset.
    
    Parameters
    ----------
    preset : str, optional
        Preset name: 'default', 'minimal', 'aggressive', 'texbat'
        
    Returns
    -------
    dict
        Configuration dictionary
        
    Notes
    -----
    Presets:
    - 'default': Balanced preprocessing for most cases
    - 'minimal': Only essential steps (DC removal, normalization)
    - 'aggressive': All mitigation techniques enabled
    - 'texbat': Optimized for TEXBAT dataset
    """
    if preset == 'minimal':
        return {
            'remove_dc': True,
            'freq_correction': 0.0,
            'apply_notch': False,
            'apply_bandpass': False,
            'pulse_blanking': False,
            'freq_domain_mitigation': False,
            'normalize': 'power'
        }
    
    elif preset == 'aggressive':
        return {
            'remove_dc': True,
            'freq_correction': 0.0,
            'apply_notch': True,
            'notch_freq': 1e6,
            'notch_q': 30.0,
            'apply_bandpass': True,
            'bandpass_low': 0.5e6,
            'bandpass_high': 2e6,
            'pulse_blanking': True,
            'pb_threshold': 3.5,
            'freq_domain_mitigation': True,
            'fdm_threshold': 3.0,
            'normalize': 'power'
        }
    
    elif preset == 'texbat':
        return {
            'remove_dc': True,
            'freq_correction': 0.0,
            'apply_notch': True,
            'notch_freq': 1e6,
            'notch_q': 30.0,
            'apply_bandpass': False,
            'pulse_blanking': False,
            'freq_domain_mitigation': False,
            'normalize': 'power'
        }
    
    else:  # 'default'
        return {
            'remove_dc': True,
            'freq_correction': 0.0,
            'apply_notch': False,
            'apply_bandpass': False,
            'pulse_blanking': False,
            'freq_domain_mitigation': False,
            'normalize': 'power'
        }
