"""Noise reduction and signal smoothing utilities."""

import numpy as np
from scipy.signal import savgol_filter, medfilt
from typing import Union


def remove_outliers(signal: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Remove outliers from signal using z-score method.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    threshold : float, default=3.0
        Z-score threshold for outlier detection
    
    Returns
    -------
    np.ndarray
        Signal with outliers removed (replaced with median)
    """
    if np.iscomplexobj(signal):
        # Handle magnitude for complex signals
        magnitude = np.abs(signal)
        mean = np.mean(magnitude)
        std = np.std(magnitude)
        
        if std < 1e-12:
            return signal
        
        z_scores = np.abs((magnitude - mean) / std)
        outliers = z_scores > threshold
        
        # Replace outliers with median
        result = signal.copy()
        median = np.median(signal[~outliers]) if np.any(~outliers) else np.median(signal)
        result[outliers] = median
        return result
    else:
        mean = np.mean(signal)
        std = np.std(signal)
        
        if std < 1e-12:
            return signal
        
        z_scores = np.abs((signal - mean) / std)
        outliers = z_scores > threshold
        
        result = signal.copy()
        median = np.median(signal[~outliers]) if np.any(~outliers) else np.median(signal)
        result[outliers] = median
        return result


def smooth_signal(signal: np.ndarray, method: str = 'savgol', 
                 window_length: int = 11, **kwargs) -> np.ndarray:
    """
    Smooth signal using specified method.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    method : str, default='savgol'
        Smoothing method: 'savgol' or 'median'
    window_length : int, default=11
        Length of smoothing window (must be odd)
    **kwargs : dict
        Additional arguments for smoothing method
        - polyorder : int for Savitzky-Golay filter (default=3)
    
    Returns
    -------
    np.ndarray
        Smoothed signal
    """
    if window_length % 2 == 0:
        window_length += 1  # Ensure odd window length
    
    if window_length > len(signal):
        window_length = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
    
    if window_length < 3:
        return signal
    
    if method == 'savgol':
        polyorder = kwargs.get('polyorder', 3)
        if polyorder >= window_length:
            polyorder = window_length - 1
        
        if np.iscomplexobj(signal):
            real_smoothed = savgol_filter(np.real(signal), window_length, polyorder)
            imag_smoothed = savgol_filter(np.imag(signal), window_length, polyorder)
            return real_smoothed + 1j * imag_smoothed
        else:
            return savgol_filter(signal, window_length, polyorder)
    
    elif method == 'median':
        if np.iscomplexobj(signal):
            real_smoothed = medfilt(np.real(signal), window_length)
            imag_smoothed = medfilt(np.imag(signal), window_length)
            return real_smoothed + 1j * imag_smoothed
        else:
            return medfilt(signal, window_length)
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
