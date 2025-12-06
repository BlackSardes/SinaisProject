"""Signal filtering utilities."""

import numpy as np
from scipy.signal import butter, filtfilt
from typing import Optional


def bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float, 
                   fs: float, order: int = 5) -> np.ndarray:
    """
    Apply bandpass filter to signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    lowcut : float
        Low cutoff frequency in Hz
    highcut : float
        High cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    order : int, default=5
        Filter order
    
    Returns
    -------
    np.ndarray
        Filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = butter(order, [low, high], btype='band')
    
    # Handle complex signals
    if np.iscomplexobj(signal):
        filtered_real = filtfilt(b, a, np.real(signal))
        filtered_imag = filtfilt(b, a, np.imag(signal))
        return filtered_real + 1j * filtered_imag
    else:
        return filtfilt(b, a, signal)


def remove_dc(signal: np.ndarray) -> np.ndarray:
    """
    Remove DC component from signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (real or complex)
    
    Returns
    -------
    np.ndarray
        Signal with DC component removed
    """
    if np.iscomplexobj(signal):
        return signal - np.mean(signal)
    else:
        return signal - np.mean(signal)
