"""Correlation computation utilities."""

import numpy as np
from typing import Optional


def compute_cross_correlation(signal1: np.ndarray, signal2: np.ndarray, 
                              mode: str = 'fft') -> np.ndarray:
    """
    Compute cross-correlation between two signals.
    
    Parameters
    ----------
    signal1 : np.ndarray
        First signal
    signal2 : np.ndarray
        Second signal (reference/template)
    mode : str, default='fft'
        Computation mode: 'fft' (fast) or 'direct' (for short signals)
    
    Returns
    -------
    np.ndarray
        Cross-correlation result
    """
    if mode == 'fft':
        # Ensure same length
        n = max(len(signal1), len(signal2))
        
        # Compute FFT-based correlation
        fft_sig1 = np.fft.fft(signal1, n=n)
        fft_sig2 = np.fft.fft(signal2, n=n)
        
        # Cross-correlation in frequency domain
        corr_fft = fft_sig1 * np.conj(fft_sig2)
        corr = np.fft.ifft(corr_fft)
        
        return corr
    
    elif mode == 'direct':
        # Direct correlation using numpy
        return np.correlate(signal1, signal2, mode='full')
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_autocorrelation(signal: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """
    Compute autocorrelation of signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    max_lag : int, optional
        Maximum lag to compute (default: len(signal) - 1)
    
    Returns
    -------
    np.ndarray
        Autocorrelation result
    """
    n = len(signal)
    
    if max_lag is None:
        max_lag = n - 1
    
    max_lag = min(max_lag, n - 1)
    
    # Normalize signal
    signal_norm = signal - np.mean(signal)
    variance = np.var(signal)
    
    if variance < 1e-12:
        return np.zeros(max_lag + 1)
    
    # Compute autocorrelation using FFT
    fft_sig = np.fft.fft(signal_norm, n=2*n)
    autocorr = np.fft.ifft(fft_sig * np.conj(fft_sig))
    autocorr = np.real(autocorr[:n]) / variance / n
    
    return autocorr[:max_lag + 1]
