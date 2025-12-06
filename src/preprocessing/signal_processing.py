"""
Signal Processing Module for GPS Signals

This module provides essential signal processing functions for GPS signal analysis:
- Normalization
- Filtering (bandpass, notch, DC removal)
- Resampling
- Segmentation
- C/N0 estimation
- Interference mitigation
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirfilter, lfilter, resample, savgol_filter, medfilt
from scipy.fft import fft, ifft, fftfreq
from typing import Optional, Tuple


def normalize_signal(signal: np.ndarray, method: str = 'power') -> np.ndarray:
    """
    Normalize signal using various methods.
    
    Parameters
    ----------
    signal : np.ndarray
        Complex or real signal
    method : str, optional
        Normalization method: 'power', 'amplitude', 'zscore' (default: 'power')
        
    Returns
    -------
    np.ndarray
        Normalized signal
        
    Notes
    -----
    - 'power': Normalize to unit power (E[|x|^2] = 1)
    - 'amplitude': Normalize to unit amplitude (max|x| = 1)
    - 'zscore': Zero mean and unit variance
    """
    if method == 'power':
        power = np.mean(np.abs(signal) ** 2)
        # Add epsilon to prevent division by very small numbers
        return signal / np.sqrt(power + 1e-12)
    
    elif method == 'amplitude':
        max_amp = np.max(np.abs(signal))
        if max_amp > 1e-12:
            return signal / max_amp
        return signal
    
    elif method == 'zscore':
        mean = np.mean(signal)
        std = np.std(signal)
        if std > 1e-12:
            return (signal - mean) / std
        return signal - mean
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low: float,
    high: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (can be complex)
    fs : float
        Sampling frequency in Hz
    low : float
        Lower cutoff frequency in Hz
    high : float
        Upper cutoff frequency in Hz
    order : int, optional
        Filter order (default: 4)
        
    Returns
    -------
    np.ndarray
        Filtered signal
        
    Notes
    -----
    For GPS L1 C/A signals, typical bandwidth is around 2 MHz.
    The Butterworth filter provides a maximally flat passband response.
    
    Uses filtfilt for zero-phase filtering (no group delay distortion).
    """
    nyquist = fs / 2
    low_norm = low / nyquist
    high_norm = high / nyquist
    
    # Design filter
    b, a = butter(order, [low_norm, high_norm], btype='band')
    
    # Apply filter (handle complex signals)
    if np.iscomplexobj(signal):
        filtered_real = filtfilt(b, a, np.real(signal))
        filtered_imag = filtfilt(b, a, np.imag(signal))
        return filtered_real + 1j * filtered_imag
    else:
        return filtfilt(b, a, signal)


def notch_filter(
    signal: np.ndarray,
    fs: float,
    f0: float,
    Q: float = 30.0
) -> np.ndarray:
    """
    Apply notch filter to suppress narrowband interference.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (can be complex)
    fs : float
        Sampling frequency in Hz
    f0 : float
        Notch frequency (center frequency to suppress) in Hz
    Q : float, optional
        Quality factor (default: 30.0) - higher Q = narrower notch
        
    Returns
    -------
    np.ndarray
        Filtered signal
        
    Notes
    -----
    Notch filters are used to suppress Radio Frequency Interference (RFI)
    from narrowband sources like communication systems or other RF emitters.
    
    The Q factor determines the bandwidth: BW ≈ f0/Q
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
    
    # Apply filter (handle complex signals)
    if np.iscomplexobj(signal):
        filtered_real = lfilter(b, a, np.real(signal))
        filtered_imag = lfilter(b, a, np.imag(signal))
        return filtered_real + 1j * filtered_imag
    else:
        return lfilter(b, a, signal)


def remove_dc(signal: np.ndarray) -> np.ndarray:
    """
    Remove DC offset from signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (can be complex)
        
    Returns
    -------
    np.ndarray
        Signal with DC removed
        
    Notes
    -----
    DC offset can result from hardware imperfections in the RF front-end.
    Removing DC is important for proper correlation and power estimation.
    """
    if np.iscomplexobj(signal):
        return signal - np.mean(signal)
    else:
        return signal - np.mean(signal)


def resample_signal(
    signal: np.ndarray,
    fs_old: float,
    fs_new: float
) -> np.ndarray:
    """
    Resample signal to new sampling rate.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs_old : float
        Original sampling frequency in Hz
    fs_new : float
        Target sampling frequency in Hz
        
    Returns
    -------
    np.ndarray
        Resampled signal
        
    Notes
    -----
    Uses Fourier method for resampling which is appropriate for
    bandlimited signals. This maintains signal properties better
    than simple interpolation.
    """
    num_samples_new = int(len(signal) * fs_new / fs_old)
    return resample(signal, num_samples_new)


def segment_signal(
    signal: np.ndarray,
    segment_length: int,
    overlap: float = 0.0
) -> list:
    """
    Segment signal into windows of fixed length.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    segment_length : int
        Length of each segment in samples
    overlap : float, optional
        Overlap fraction between segments (0.0 to 0.99, default: 0.0)
        
    Returns
    -------
    list of np.ndarray
        List of signal segments
        
    Notes
    -----
    Segmentation is crucial for:
    - Processing large datasets in chunks
    - Time-varying analysis (detecting when spoofing starts)
    - Managing memory constraints
    
    Overlap can help capture transitions between segments.
    """
    if overlap < 0 or overlap >= 1:
        raise ValueError("Overlap must be in range [0, 1)")
    
    step = int(segment_length * (1 - overlap))
    segments = []
    
    for start in range(0, len(signal) - segment_length + 1, step):
        end = start + segment_length
        segments.append(signal[start:end])
    
    return segments


def remove_outliers(
    signal: np.ndarray,
    threshold: float = 4.0,
    method: str = 'clip'
) -> np.ndarray:
    """
    Remove or clip outliers from signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (can be complex)
    threshold : float, optional
        Threshold in standard deviations (default: 4.0)
    method : str, optional
        Method: 'clip' or 'zero' (default: 'clip')
        
    Returns
    -------
    np.ndarray
        Signal with outliers handled
        
    Notes
    -----
    Outliers can result from:
    - Impulsive interference (lightning, radar)
    - ADC saturation
    - Data corruption
    
    'clip': Limits outliers to threshold
    'zero': Sets outliers to zero
    """
    # Work with magnitude for complex signals
    magnitude = np.abs(signal)
    mean_mag = np.mean(magnitude)
    std_mag = np.std(magnitude)
    
    # Find outliers
    outlier_mask = magnitude > (mean_mag + threshold * std_mag)
    
    if method == 'clip':
        # Clip to threshold
        max_allowed = mean_mag + threshold * std_mag
        phase = np.angle(signal)
        magnitude_clipped = np.where(outlier_mask, max_allowed, magnitude)
        return magnitude_clipped * np.exp(1j * phase)
    
    elif method == 'zero':
        # Zero out outliers
        signal_clean = signal.copy()
        signal_clean[outlier_mask] = 0
        return signal_clean
    
    else:
        raise ValueError(f"Unknown method: {method}")


def apply_savgol_filter(
    signal: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing filter.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (can be complex)
    window_length : int, optional
        Length of filter window (must be odd, default: 11)
    polyorder : int, optional
        Order of polynomial fit (default: 2)
        
    Returns
    -------
    np.ndarray
        Smoothed signal
        
    Notes
    -----
    Savitzky-Golay filter smooths data while preserving shape better
    than moving average. Good for removing high-frequency noise while
    maintaining sharp features like correlation peaks.
    """
    if window_length % 2 == 0:
        window_length += 1  # Must be odd
    
    if np.iscomplexobj(signal):
        real_filtered = savgol_filter(np.real(signal), window_length, polyorder)
        imag_filtered = savgol_filter(np.imag(signal), window_length, polyorder)
        return real_filtered + 1j * imag_filtered
    else:
        return savgol_filter(signal, window_length, polyorder)


def apply_median_filter(signal: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply median filter for impulsive noise removal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (can be complex)
    kernel_size : int, optional
        Size of median filter kernel (default: 5)
        
    Returns
    -------
    np.ndarray
        Filtered signal
        
    Notes
    -----
    Median filter is very effective at removing impulsive (salt-and-pepper)
    noise while preserving edges. Non-linear filter.
    """
    if np.iscomplexobj(signal):
        real_filtered = medfilt(np.real(signal), kernel_size)
        imag_filtered = medfilt(np.imag(signal), kernel_size)
        return real_filtered + 1j * imag_filtered
    else:
        return medfilt(signal, kernel_size)


def estimate_cn0_from_signal(
    signal: np.ndarray,
    fs: float,
    correlation_peak: Optional[float] = None
) -> float:
    """
    Estimate C/N0 (Carrier-to-Noise density ratio) from signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Preprocessed complex signal
    fs : float
        Sampling frequency in Hz
    correlation_peak : float, optional
        Peak value from correlation (if available)
        
    Returns
    -------
    float
        Estimated C/N0 in dB-Hz
        
    Notes
    -----
    C/N0 is a key metric for GPS signal quality. Typical values:
    - Clear sky: 45-50 dB-Hz
    - Urban canyon: 30-40 dB-Hz
    - Indoor: 20-30 dB-Hz
    
    This is a simplified estimator. More accurate methods use
    the correlation peak and noise floor from the acquisition process.
    """
    # Total signal power
    total_power = np.mean(np.abs(signal) ** 2)
    
    if correlation_peak is not None:
        # Estimate carrier power from correlation peak
        carrier_power = (correlation_peak ** 2) / len(signal)
    else:
        # Rough estimate: assume signal is mostly noise
        carrier_power = total_power * 0.1  # Placeholder
    
    # Estimate noise power
    noise_power = total_power - carrier_power
    if noise_power <= 0:
        noise_power = 1e-12
    
    # C/N0 = Carrier power / Noise power density
    # Noise power density = noise_power / bandwidth
    # For GPS, we use the sampling rate as a proxy for bandwidth
    cn0_linear = carrier_power / (noise_power / fs)
    cn0_db = 10 * np.log10(cn0_linear) if cn0_linear > 0 else -np.inf
    
    return float(cn0_db)


def estimate_cn0_from_correlation(
    correlation: np.ndarray,
    fs: float,
    coherent_integration_time: float
) -> float:
    """
    Estimate C/N0 from correlation results (more accurate).
    
    Parameters
    ----------
    correlation : np.ndarray
        Correlation magnitude array
    fs : float
        Sampling frequency in Hz
    coherent_integration_time : float
        Integration time in seconds
        
    Returns
    -------
    float
        Estimated C/N0 in dB-Hz
        
    Notes
    -----
    This method uses the peak-to-noise ratio in the correlation
    function, which is more accurate than signal power estimation.
    
    The Narrow-Band Wide-Band Power Ratio (NWWPR) method.
    """
    # Find peak
    peak_value = np.max(correlation)
    peak_index = np.argmax(correlation)
    
    # Estimate noise floor (exclude peak region)
    window = int(0.02 * len(correlation))  # 2% window around peak
    noise_mask = np.ones(len(correlation), dtype=bool)
    noise_mask[max(0, peak_index-window):min(len(correlation), peak_index+window)] = False
    
    noise_floor = np.mean(correlation[noise_mask])
    noise_std = np.std(correlation[noise_mask])
    
    # Signal-to-noise ratio
    snr = (peak_value - noise_floor) / noise_std if noise_std > 0 else 0
    
    # Convert to C/N0
    # C/N0 ≈ SNR / T_coh (where T_coh is coherent integration time)
    cn0_linear = snr / coherent_integration_time
    cn0_db = 10 * np.log10(cn0_linear) if cn0_linear > 0 else -np.inf
    
    return float(cn0_db)


def apply_frequency_correction(
    signal: np.ndarray,
    fs: float,
    freq_correction: float
) -> np.ndarray:
    """
    Apply frequency correction to signal (Doppler/IF correction).
    
    Parameters
    ----------
    signal : np.ndarray
        Complex input signal
    fs : float
        Sampling frequency in Hz
    freq_correction : float
        Frequency to remove (IF + Doppler) in Hz
        
    Returns
    -------
    np.ndarray
        Frequency-corrected signal
        
    Notes
    -----
    This implements the frequency shifting property of Fourier Transform:
    Multiplying by exp(-j*2*pi*f*t) shifts the spectrum by -f Hz.
    
    Essential for:
    - Removing intermediate frequency (IF)
    - Compensating for Doppler shift
    - Bringing signal to baseband
    """
    t = np.arange(len(signal)) / fs
    mixer = np.exp(-1j * 2 * np.pi * freq_correction * t)
    return signal * mixer


def apply_pulse_blanking(
    signal: np.ndarray,
    threshold_factor: float = 4.0
) -> np.ndarray:
    """
    Mitigate impulsive interference using pulse blanking.
    
    Parameters
    ----------
    signal : np.ndarray
        Complex input signal
    threshold_factor : float, optional
        Threshold in standard deviations (default: 4.0)
        
    Returns
    -------
    np.ndarray
        Signal with pulses blanked
        
    Notes
    -----
    Pulse blanking is a time-domain interference mitigation technique
    that zeros out samples exceeding a threshold. Effective against
    radar pulses and other impulsive RFI.
    """
    magnitude = np.abs(signal)
    threshold = threshold_factor * np.std(magnitude)
    
    # Blank (zero) samples above threshold
    signal_blanked = signal.copy()
    signal_blanked[magnitude > threshold] = 0
    
    return signal_blanked


def apply_frequency_domain_interference_mitigation(
    signal: np.ndarray,
    threshold_factor: float = 3.5
) -> np.ndarray:
    """
    Mitigate narrowband interference in frequency domain (FDPB).
    
    Parameters
    ----------
    signal : np.ndarray
        Complex input signal
    threshold_factor : float, optional
        Threshold factor for MAD-based detection (default: 3.5)
        
    Returns
    -------
    np.ndarray
        Signal with interference mitigated
        
    Notes
    -----
    Frequency Domain Pulse Blanking (FDPB) suppresses narrowband RFI
    by zeroing out spectral bins that exceed a robust threshold.
    
    Uses Median Absolute Deviation (MAD) for robust threshold estimation.
    """
    # Transform to frequency domain
    signal_fft = fft(signal)
    magnitude_spectrum = np.abs(signal_fft)
    
    # Robust threshold using MAD
    median_mag = np.median(magnitude_spectrum)
    mad = np.median(np.abs(magnitude_spectrum - median_mag))
    threshold = median_mag + threshold_factor * 1.4826 * mad  # 1.4826 makes MAD ~ std
    
    # Zero out bins above threshold
    signal_fft[magnitude_spectrum > threshold] = 0
    
    # Transform back to time domain
    return ifft(signal_fft)
