"""
Signal resampling functions.
"""
import numpy as np
from scipy.signal import resample


def resample_signal(
    signal: np.ndarray,
    fs_old: float,
    fs_new: float
) -> np.ndarray:
    """
    Resample signal to new sampling frequency.
    
    Uses FFT-based resampling (scipy.signal.resample) which is
    appropriate for band-limited signals.
    
    Args:
        signal: Complex or real input signal
        fs_old: Original sampling frequency in Hz
        fs_new: Target sampling frequency in Hz
    
    Returns:
        Resampled signal
        
    Example:
        >>> # Downsample from 5 MHz to 2.5 MHz
        >>> signal_ds = resample_signal(signal, fs_old=5e6, fs_new=2.5e6)
        >>> print(len(signal_ds) / len(signal))
        0.5
    """
    if fs_old == fs_new:
        return signal
    
    # Calculate new number of samples
    num_samples_new = int(len(signal) * fs_new / fs_old)
    
    # Resample using FFT method
    return resample(signal, num_samples_new)
