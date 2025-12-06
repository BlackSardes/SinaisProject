"""Signal normalization utilities."""

import numpy as np
from typing import Union


def normalize_signal(signal: np.ndarray, method: str = 'power') -> np.ndarray:
    """
    Normalize signal using specified method.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (real or complex)
    method : str, default='power'
        Normalization method:
        - 'power': Normalize by root mean square power
        - 'max': Normalize by maximum absolute value
        - 'std': Normalize to zero mean and unit variance
    
    Returns
    -------
    np.ndarray
        Normalized signal
    """
    if method == 'power':
        power = np.mean(np.abs(signal) ** 2)
        if power > 1e-12:
            return signal / np.sqrt(power)
        return signal
    
    elif method == 'max':
        max_val = np.max(np.abs(signal))
        if max_val > 1e-12:
            return signal / max_val
        return signal
    
    elif method == 'std':
        signal = signal - np.mean(signal)
        std_val = np.std(signal)
        if std_val > 1e-12:
            return signal / std_val
        return signal
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
