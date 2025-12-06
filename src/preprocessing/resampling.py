"""Signal resampling utilities."""

import numpy as np
from scipy.signal import resample


def resample_signal(signal: np.ndarray, original_fs: float, 
                   target_fs: float) -> np.ndarray:
    """
    Resample signal to target sampling frequency.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    original_fs : float
        Original sampling frequency in Hz
    target_fs : float
        Target sampling frequency in Hz
    
    Returns
    -------
    np.ndarray
        Resampled signal
    """
    if original_fs == target_fs:
        return signal
    
    num_samples = int(len(signal) * target_fs / original_fs)
    return resample(signal, num_samples)
