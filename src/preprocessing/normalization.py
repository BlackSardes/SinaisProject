"""
Signal normalization functions.
"""
import numpy as np
from typing import Optional


def normalize_signal(
    signal: np.ndarray,
    method: str = 'power',
    power_target: float = 1.0,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Normalize signal to unit power or amplitude.
    
    Normalization ensures that signal amplitudes are consistent across
    different receivers and recording sessions, which is critical for
    robust feature extraction and classification.
    
    Args:
        signal: Complex or real signal array
        method: Normalization method
            - 'power': Normalize to target average power (default)
            - 'amplitude': Normalize to target peak amplitude
            - 'std': Normalize to unit standard deviation
        power_target: Target power level (for 'power' method)
        eps: Small constant to avoid division by zero
    
    Returns:
        Normalized signal
        
    Example:
        >>> signal = np.random.randn(1000) + 1j*np.random.randn(1000)
        >>> signal_norm = normalize_signal(signal)
        >>> print(np.mean(np.abs(signal_norm)**2))  # Should be ~1.0
        1.0
    """
    if method == 'power':
        # Normalize by RMS power
        power = np.mean(np.abs(signal) ** 2)
        if power > eps:
            return signal * np.sqrt(power_target / power)
        else:
            return signal
    
    elif method == 'amplitude':
        # Normalize by peak amplitude
        peak = np.max(np.abs(signal))
        if peak > eps:
            return signal * (power_target / peak)
        else:
            return signal
    
    elif method == 'std':
        # Normalize by standard deviation
        std = np.std(signal)
        if std > eps:
            return signal / std
        else:
            return signal
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def remove_dc(signal: np.ndarray) -> np.ndarray:
    """
    Remove DC offset from signal.
    
    Subtracts the mean from the signal to center it around zero.
    For complex signals, removes DC from both I and Q components.
    
    Args:
        signal: Complex or real signal array
    
    Returns:
        Signal with DC removed
        
    Example:
        >>> signal = np.array([1, 2, 3, 4, 5]) + 1j*np.array([2, 3, 4, 5, 6])
        >>> signal_dc = remove_dc(signal)
        >>> print(np.abs(np.mean(signal_dc)))  # Should be ~0
        0.0
    """
    if np.iscomplexobj(signal):
        # Remove DC from I and Q separately
        I = np.real(signal) - np.mean(np.real(signal))
        Q = np.imag(signal) - np.mean(np.imag(signal))
        return I + 1j * Q
    else:
        # Remove DC from real signal
        return signal - np.mean(signal)


def normalize_by_power(signal: np.ndarray) -> np.ndarray:
    """
    Normalize signal to unit power (legacy function name).
    
    This is a convenience wrapper for normalize_signal with method='power'.
    Kept for backward compatibility with existing code.
    
    Args:
        signal: Complex or real signal array
    
    Returns:
        Power-normalized signal
    """
    return normalize_signal(signal, method='power', power_target=1.0)
