"""
Signal filtering functions for GPS signal processing.
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirfilter, lfilter, savgol_filter, medfilt
from typing import Optional, Union


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low: float,
    high: float,
    order: int = 5
) -> np.ndarray:
    """
    Apply bandpass filter to signal.
    
    Uses a Butterworth filter to retain only frequency components
    between low and high cutoff frequencies.
    
    Args:
        signal: Complex or real input signal
        fs: Sampling frequency in Hz
        low: Low cutoff frequency in Hz
        high: High cutoff frequency in Hz
        order: Filter order (higher = sharper cutoff)
    
    Returns:
        Filtered signal
        
    Example:
        >>> # Filter to retain 0-2 MHz bandwidth
        >>> signal_filtered = bandpass_filter(signal, fs=5e6, low=0, high=2e6)
    """
    nyquist = fs / 2.0
    low_norm = low / nyquist
    high_norm = high / nyquist
    
    # Ensure frequencies are in valid range
    low_norm = max(0.001, min(low_norm, 0.999))
    high_norm = max(0.001, min(high_norm, 0.999))
    
    if low_norm >= high_norm:
        raise ValueError(f"Low cutoff ({low} Hz) must be less than high cutoff ({high} Hz)")
    
    # Design Butterworth filter
    b, a = butter(order, [low_norm, high_norm], btype='bandpass')
    
    # Apply filter
    if np.iscomplexobj(signal):
        # Filter I and Q separately
        I_filt = filtfilt(b, a, np.real(signal))
        Q_filt = filtfilt(b, a, np.imag(signal))
        return I_filt + 1j * Q_filt
    else:
        return filtfilt(b, a, signal)


def apply_notch_filter(
    signal: np.ndarray,
    fs: float,
    f0: float,
    Q: float = 30.0
) -> np.ndarray:
    """
    Apply notch filter to suppress narrow-band RFI.
    
    Removes a narrow frequency band centered at f0, useful for
    eliminating continuous wave interference.
    
    Args:
        signal: Complex or real input signal
        fs: Sampling frequency in Hz
        f0: Center frequency of notch in Hz
        Q: Quality factor (higher = narrower notch)
    
    Returns:
        Filtered signal with notch applied
        
    Example:
        >>> # Remove RFI at 1 MHz
        >>> signal_clean = apply_notch_filter(signal, fs=5e6, f0=1e6, Q=30)
    """
    # Design notch filter
    b, a = iirfilter(
        2,
        [f0 - f0/(2*Q), f0 + f0/(2*Q)],
        rs=60,
        btype='bandstop',
        fs=fs,
        output='ba'
    )
    
    # Apply filter
    if np.iscomplexobj(signal):
        I_filt = lfilter(b, a, np.real(signal))
        Q_filt = lfilter(b, a, np.imag(signal))
        return I_filt + 1j * Q_filt
    else:
        return lfilter(b, a, signal)


def remove_outliers(
    signal: np.ndarray,
    method: str = 'median',
    threshold: float = 4.0,
    window: Optional[int] = None
) -> np.ndarray:
    """
    Remove outliers from signal using robust methods.
    
    Args:
        signal: Complex or real input signal
        method: Outlier detection method
            - 'median': Median Absolute Deviation (MAD)
            - 'std': Standard deviation threshold
            - 'percentile': Clip to percentile range
        threshold: Threshold multiplier for outlier detection
        window: Window size for local outlier detection (None = global)
    
    Returns:
        Signal with outliers removed/clipped
        
    Example:
        >>> # Remove extreme amplitude spikes
        >>> signal_clean = remove_outliers(signal, method='median', threshold=4.0)
    """
    amplitude = np.abs(signal)
    
    if method == 'median':
        # Use Median Absolute Deviation (robust to outliers)
        median = np.median(amplitude)
        mad = np.median(np.abs(amplitude - median))
        threshold_val = median + threshold * mad * 1.4826  # MAD to std conversion
        
    elif method == 'std':
        # Use standard deviation (less robust)
        mean = np.mean(amplitude)
        std = np.std(amplitude)
        threshold_val = mean + threshold * std
        
    elif method == 'percentile':
        # Clip to percentile range
        low = np.percentile(amplitude, threshold)
        high = np.percentile(amplitude, 100 - threshold)
        threshold_val = high
        
    else:
        raise ValueError(f"Unknown outlier removal method: {method}")
    
    # Clip outliers
    mask = amplitude > threshold_val
    if np.any(mask):
        # Scale down outliers to threshold
        signal_clean = signal.copy()
        signal_clean[mask] = signal[mask] * (threshold_val / amplitude[mask])
        return signal_clean
    
    return signal


def smooth_signal(
    signal: np.ndarray,
    method: str = 'savgol',
    window_length: int = 11,
    **kwargs
) -> np.ndarray:
    """
    Smooth signal to reduce high-frequency noise.
    
    Args:
        signal: Complex or real input signal
        method: Smoothing method
            - 'savgol': Savitzky-Golay filter (polynomial fit)
            - 'median': Median filter
            - 'moving_average': Simple moving average
        window_length: Window length for smoothing (must be odd for savgol)
        **kwargs: Additional arguments for specific methods
            - polyorder: Polynomial order for savgol (default: 3)
    
    Returns:
        Smoothed signal
        
    Example:
        >>> # Smooth with Savitzky-Golay filter
        >>> signal_smooth = smooth_signal(signal, method='savgol', window_length=11)
    """
    if method == 'savgol':
        polyorder = kwargs.get('polyorder', 3)
        if window_length % 2 == 0:
            window_length += 1  # Must be odd
        
        if np.iscomplexobj(signal):
            I_smooth = savgol_filter(np.real(signal), window_length, polyorder)
            Q_smooth = savgol_filter(np.imag(signal), window_length, polyorder)
            return I_smooth + 1j * Q_smooth
        else:
            return savgol_filter(signal, window_length, polyorder)
    
    elif method == 'median':
        if np.iscomplexobj(signal):
            I_smooth = medfilt(np.real(signal), window_length)
            Q_smooth = medfilt(np.imag(signal), window_length)
            return I_smooth + 1j * Q_smooth
        else:
            return medfilt(signal, window_length)
    
    elif method == 'moving_average':
        kernel = np.ones(window_length) / window_length
        
        if np.iscomplexobj(signal):
            I_smooth = np.convolve(np.real(signal), kernel, mode='same')
            Q_smooth = np.convolve(np.imag(signal), kernel, mode='same')
            return I_smooth + 1j * Q_smooth
        else:
            return np.convolve(signal, kernel, mode='same')
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
