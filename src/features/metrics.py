"""Feature metrics for signal analysis."""

import numpy as np
from typing import Union


def fwhm(signal: np.ndarray, peak_idx: Union[int, None] = None) -> float:
    """
    Compute Full Width at Half Maximum (FWHM) of a peak in signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (should contain a peak)
    peak_idx : int, optional
        Index of the peak. If None, uses argmax to find peak.
    
    Returns
    -------
    float
        FWHM in samples
    
    Notes
    -----
    FWHM is the width of the peak at half of its maximum value.
    This metric is useful for analyzing correlation peaks in GPS signals.
    """
    # Find peak if not provided
    if peak_idx is None:
        peak_idx = np.argmax(np.abs(signal))
    
    peak_value = np.abs(signal[peak_idx])
    
    if peak_value < 1e-12:
        return 0.0
    
    # Half maximum value
    half_max = peak_value / 2.0
    
    # Find left crossing
    left_idx = peak_idx
    for i in range(peak_idx, -1, -1):
        if np.abs(signal[i]) < half_max:
            left_idx = i
            break
    else:
        left_idx = 0
    
    # Find right crossing
    right_idx = peak_idx
    for i in range(peak_idx, len(signal)):
        if np.abs(signal[i]) < half_max:
            right_idx = i
            break
    else:
        right_idx = len(signal) - 1
    
    # Interpolate for more accurate crossing points
    # Left interpolation
    if left_idx < peak_idx and left_idx > 0:
        y1, y2 = np.abs(signal[left_idx]), np.abs(signal[left_idx + 1])
        if abs(y2 - y1) > 1e-12:
            left_cross = left_idx + (half_max - y1) / (y2 - y1)
        else:
            left_cross = left_idx
    else:
        left_cross = left_idx
    
    # Right interpolation
    if right_idx > peak_idx and right_idx < len(signal):
        y1, y2 = np.abs(signal[right_idx - 1]), np.abs(signal[right_idx])
        if abs(y1 - y2) > 1e-12:
            right_cross = right_idx - 1 + (half_max - y1) / (y2 - y1)
        else:
            right_cross = right_idx
    else:
        right_cross = right_idx
    
    fwhm_value = right_cross - left_cross
    
    return float(fwhm_value)
