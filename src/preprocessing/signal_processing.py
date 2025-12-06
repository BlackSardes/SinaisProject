"""
Signal preprocessing functions for GPS spoofing detection.
Includes functions from the original pre_process.py and additional utilities.
"""
import os
import numpy as np
from scipy.signal import butter, filtfilt, medfilt, savgol_filter, resample as scipy_resample
from scipy.io import loadmat
import pandas as pd
from typing import Optional, Union, Tuple, List


def read_iq_data(file_path: str, start_offset_samples: int, count_samples: int) -> Optional[np.ndarray]:
    """
    Read IQ data from binary files (TEXBAT format: int16 interleaved I/Q).
    
    Args:
        file_path: Path to binary file (.bin or .dat)
        start_offset_samples: Starting position in complex samples
        count_samples: Number of complex samples to read
    
    Returns:
        Complex numpy array (float32) with I+jQ data, or None if error
    """
    bytes_per_iq_pair = 4  # 2 bytes I + 2 bytes Q
    start_offset_bytes = start_offset_samples * bytes_per_iq_pair
    count_int16 = 2 * count_samples
    
    try:
        with open(file_path, "rb") as f:
            f.seek(start_offset_bytes)
            raw = np.fromfile(f, dtype=np.int16, count=count_int16)
        
        if raw.size < count_int16:
            return None
        
        I = raw[0::2].astype(np.float32)
        Q = raw[1::2].astype(np.float32)
        signal = I + 1j * Q
        return signal
    
    except Exception as e:
        print(f"Warning: Error reading segment from {os.path.basename(file_path)}: {e}")
        return None


def load_signal(path: str, file_format: Optional[str] = None) -> Tuple[np.ndarray, dict]:
    """
    Generic signal loader supporting multiple formats.
    
    Args:
        path: Path to signal file
        file_format: Force specific format ('bin', 'mat', 'csv'). Auto-detected if None.
    
    Returns:
        Tuple of (signal array, metadata dict)
    
    Notes:
        - For .bin/.dat: assumes TEXBAT format (int16 interleaved I/Q)
        - For .mat: loads first complex array found or combines I/Q fields
        - For .csv: expects columns 'I' and 'Q' or 'real' and 'imag'
        - FGI-SpoofRepo: provide instructions in metadata for manual download
    """
    if file_format is None:
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.bin', '.dat']:
            file_format = 'bin'
        elif ext == '.mat':
            file_format = 'mat'
        elif ext == '.csv':
            file_format = 'csv'
        else:
            raise ValueError(f"Unknown file extension: {ext}. Specify file_format explicitly.")
    
    metadata = {'path': path, 'format': file_format}
    
    if file_format == 'bin':
        # Read entire file as IQ data
        signal = read_iq_data(path, 0, os.path.getsize(path) // 4)
        if signal is None:
            raise IOError(f"Failed to read binary file: {path}")
        metadata['samples'] = len(signal)
        return signal, metadata
    
    elif file_format == 'mat':
        # Load MATLAB file
        try:
            mat_data = loadmat(path)
            # Try to find signal data (common field names)
            for key in ['signal', 'data', 'IQ', 'samples']:
                if key in mat_data:
                    signal = mat_data[key].flatten()
                    if not np.iscomplexobj(signal):
                        # Try to construct complex from I and Q
                        if len(signal) % 2 == 0:
                            signal = signal[::2] + 1j * signal[1::2]
                    metadata['samples'] = len(signal)
                    metadata['mat_key'] = key
                    return signal, metadata
            
            # If not found, look for I and Q separately
            if 'I' in mat_data and 'Q' in mat_data:
                signal = mat_data['I'].flatten() + 1j * mat_data['Q'].flatten()
                metadata['samples'] = len(signal)
                return signal, metadata
            
            raise ValueError("Could not find signal data in .mat file. Expected fields: 'signal', 'data', 'IQ', or 'I'/'Q'")
        except Exception as e:
            raise IOError(f"Error loading .mat file: {e}")
    
    elif file_format == 'csv':
        # Load CSV file
        df = pd.read_csv(path)
        if 'I' in df.columns and 'Q' in df.columns:
            signal = df['I'].values + 1j * df['Q'].values
        elif 'real' in df.columns and 'imag' in df.columns:
            signal = df['real'].values + 1j * df['imag'].values
        else:
            raise ValueError("CSV must contain columns ('I', 'Q') or ('real', 'imag')")
        metadata['samples'] = len(signal)
        return signal, metadata
    
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def normalize_signal(signal: np.ndarray, method: str = 'power') -> np.ndarray:
    """
    Normalize signal using various methods.
    
    Args:
        signal: Input signal (real or complex)
        method: Normalization method ('power', 'max', 'std')
    
    Returns:
        Normalized signal
    """
    if method == 'power':
        return normalize_by_power(signal)
    elif method == 'max':
        max_val = np.max(np.abs(signal))
        return signal / max_val if max_val > 1e-12 else signal
    elif method == 'std':
        std_val = np.std(signal)
        return signal / std_val if std_val > 1e-12 else signal
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def normalize_by_power(signal: np.ndarray) -> np.ndarray:
    """
    Normalize signal so that mean power E[|x|^2] ≈ 1 (0 dB).
    
    Args:
        signal: Complex signal array
    
    Returns:
        Power-normalized signal
    """
    power = np.mean(np.abs(signal)**2)
    if power > 1e-12:
        return signal / np.sqrt(power)
    return signal


def bandpass_filter(signal: np.ndarray, fs: float, low: float, high: float, order: int = 5) -> np.ndarray:
    """
    Apply bandpass filter to signal.
    
    Args:
        signal: Input signal (complex)
        fs: Sampling frequency (Hz)
        low: Lower cutoff frequency (Hz)
        high: Upper cutoff frequency (Hz)
        order: Filter order
    
    Returns:
        Filtered signal
    """
    nyquist = fs / 2
    low_norm = low / nyquist
    high_norm = high / nyquist
    
    if low_norm <= 0 or high_norm >= 1:
        raise ValueError("Cutoff frequencies must be in range (0, fs/2)")
    
    b, a = butter(order, [low_norm, high_norm], btype='band')
    
    if np.iscomplexobj(signal):
        # Filter I and Q separately
        I_filtered = filtfilt(b, a, np.real(signal))
        Q_filtered = filtfilt(b, a, np.imag(signal))
        return I_filtered + 1j * Q_filtered
    else:
        return filtfilt(b, a, signal)


def remove_dc(signal: np.ndarray) -> np.ndarray:
    """
    Remove DC offset from signal.
    
    Args:
        signal: Input signal
    
    Returns:
        Signal with DC removed
    """
    if np.iscomplexobj(signal):
        I = np.real(signal) - np.mean(np.real(signal))
        Q = np.imag(signal) - np.mean(np.imag(signal))
        return I + 1j * Q
    else:
        return signal - np.mean(signal)


def resample_signal(signal: np.ndarray, fs_old: float, fs_new: float) -> np.ndarray:
    """
    Resample signal to new sampling rate.
    
    Args:
        signal: Input signal
        fs_old: Original sampling frequency (Hz)
        fs_new: Target sampling frequency (Hz)
    
    Returns:
        Resampled signal
    """
    if fs_old == fs_new:
        return signal
    
    num_samples_new = int(len(signal) * fs_new / fs_old)
    return scipy_resample(signal, num_samples_new)


def window_segment(signal: np.ndarray, fs: float, window_s: float, hop_s: float) -> List[np.ndarray]:
    """
    Segment signal into overlapping windows.
    
    Args:
        signal: Input signal
        fs: Sampling frequency (Hz)
        window_s: Window size in seconds
        hop_s: Hop size in seconds
    
    Returns:
        List of signal windows
    """
    window_samples = int(window_s * fs)
    hop_samples = int(hop_s * fs)
    
    windows = []
    start = 0
    while start + window_samples <= len(signal):
        windows.append(signal[start:start + window_samples])
        start += hop_samples
    
    return windows


def align_channels(signals: List[np.ndarray], method: str = 'cross_correlation') -> List[np.ndarray]:
    """
    Align multiple signal channels (e.g., from different antennas).
    
    Args:
        signals: List of signal arrays
        method: Alignment method ('cross_correlation')
    
    Returns:
        List of aligned signals
    """
    if len(signals) < 2:
        return signals
    
    if method == 'cross_correlation':
        # Use first signal as reference
        ref = signals[0]
        aligned = [ref]
        
        for sig in signals[1:]:
            # Compute cross-correlation to find delay
            corr = np.correlate(ref, sig, mode='full')
            delay = len(sig) - 1 - np.argmax(np.abs(corr))
            
            # Align signal
            if delay > 0:
                sig_aligned = np.concatenate([np.zeros(delay, dtype=sig.dtype), sig[:-delay]])
            elif delay < 0:
                sig_aligned = np.concatenate([sig[-delay:], np.zeros(-delay, dtype=sig.dtype)])
            else:
                sig_aligned = sig
            
            aligned.append(sig_aligned)
        
        return aligned
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def remove_outliers(signal: np.ndarray, method: str = 'median', thresh: float = 3.0) -> np.ndarray:
    """
    Remove or suppress outliers in signal.
    
    Args:
        signal: Input signal
        method: Outlier removal method ('median', 'std', 'iqr')
        thresh: Threshold factor
    
    Returns:
        Signal with outliers removed/suppressed
    """
    mag = np.abs(signal)
    
    if method == 'median':
        median = np.median(mag)
        mad = np.median(np.abs(mag - median))
        threshold = median + thresh * mad
        mask = mag > threshold
        result = signal.copy()
        result[mask] = signal[mask] * (threshold / mag[mask])
        return result
    
    elif method == 'std':
        mean = np.mean(mag)
        std = np.std(mag)
        threshold = mean + thresh * std
        mask = mag > threshold
        result = signal.copy()
        result[mask] = signal[mask] * (threshold / mag[mask])
        return result
    
    elif method == 'iqr':
        q1, q3 = np.percentile(mag, [25, 75])
        iqr = q3 - q1
        threshold = q3 + thresh * iqr
        mask = mag > threshold
        result = signal.copy()
        result[mask] = signal[mask] * (threshold / mag[mask])
        return result
    
    else:
        raise ValueError(f"Unknown outlier removal method: {method}")


def smooth_signal(signal: np.ndarray, method: str = 'savgol', window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    """
    Smooth signal using various methods.
    
    Args:
        signal: Input signal
        method: Smoothing method ('savgol', 'median')
        window_length: Window length for smoothing (must be odd for savgol)
        polyorder: Polynomial order for Savitzky-Golay filter
    
    Returns:
        Smoothed signal
    """
    if np.iscomplexobj(signal):
        # Smooth I and Q separately
        I_smooth = smooth_signal(np.real(signal), method, window_length, polyorder)
        Q_smooth = smooth_signal(np.imag(signal), method, window_length, polyorder)
        return I_smooth + 1j * Q_smooth
    
    if method == 'savgol':
        if window_length % 2 == 0:
            window_length += 1
        return savgol_filter(signal, window_length, polyorder)
    elif method == 'median':
        if window_length % 2 == 0:
            window_length += 1
        return medfilt(signal, kernel_size=window_length)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def estimate_cn0_from_correlation(corr_profile: np.ndarray, fs: float, 
                                   integration_time: float = 0.001) -> float:
    """
    Estimate C/N0 from correlation profile.
    
    Args:
        corr_profile: Correlation magnitude profile
        fs: Sampling frequency (Hz)
        integration_time: Integration time in seconds
    
    Returns:
        Estimated C/N0 in dB-Hz
    
    Notes:
        This is a simplified estimation. Limitations:
        - Assumes single coherent integration
        - Does not account for squaring loss
        - May be inaccurate for low C/N0 signals
    """
    peak_value = np.max(corr_profile)
    peak_idx = np.argmax(corr_profile)
    
    # Estimate noise floor (exclude region around peak)
    exclude_samples = int(0.002 * fs)  # 2ms exclusion
    noise_profile = np.concatenate([
        corr_profile[:max(0, peak_idx - exclude_samples)],
        corr_profile[min(len(corr_profile), peak_idx + exclude_samples):]
    ])
    
    if len(noise_profile) == 0:
        return 0.0
    
    noise_power = np.mean(noise_profile**2)
    signal_power = peak_value**2
    
    # C/N0 estimation
    if noise_power > 0 and signal_power > noise_power:
        # Simplified formula: C/N0 ≈ 10*log10(SNR / integration_time)
        snr = (signal_power - noise_power) / noise_power
        cn0 = 10 * np.log10(snr / integration_time)
        return max(0.0, cn0)
    
    return 0.0


def estimate_cn0_from_signal(signal: np.ndarray, fs: float, 
                              prn_code: Optional[np.ndarray] = None) -> float:
    """
    Estimate C/N0 directly from signal.
    
    Args:
        signal: Complex IQ signal
        fs: Sampling frequency (Hz)
        prn_code: Optional PRN code for correlation-based estimation
    
    Returns:
        Estimated C/N0 in dB-Hz
    
    Notes:
        If prn_code is provided, uses correlation method.
        Otherwise, uses power-based estimation which is less accurate.
    """
    if prn_code is not None:
        # Correlation-based estimation
        from .signal_processing import generate_ca_code
        
        # Ensure PRN code matches signal length
        if len(prn_code) < len(signal):
            prn_code = np.tile(prn_code, int(np.ceil(len(signal) / len(prn_code))))
        prn_code = prn_code[:len(signal)]
        
        # Compute correlation
        fft_signal = np.fft.fft(signal)
        fft_code = np.fft.fft(prn_code)
        corr = np.fft.ifft(fft_signal * np.conj(fft_code))
        corr_mag = np.abs(corr)
        
        return estimate_cn0_from_correlation(corr_mag, fs)
    else:
        # Power-based estimation (simplified)
        total_power = np.mean(np.abs(signal)**2)
        # Assume SNR ~ 10 (very rough approximation)
        # This is a placeholder - proper C/N0 needs code correlation
        estimated_snr = 10.0
        cn0 = 10 * np.log10(estimated_snr * fs / 1000)  # Very rough estimate
        return cn0


def apply_frequency_correction(signal: np.ndarray, fs: float, freq_correction: float) -> np.ndarray:
    """
    Apply frequency correction to remove Doppler/IF offset.
    
    Args:
        signal: Complex signal
        fs: Sampling frequency (Hz)
        freq_correction: Frequency to remove (Hz)
    
    Returns:
        Frequency-corrected signal
    """
    if freq_correction == 0:
        return signal
    
    t = np.arange(len(signal)) / fs
    mixer = np.exp(-1j * 2 * np.pi * freq_correction * t)
    return signal * mixer


def generate_ca_code(prn_number: int) -> np.ndarray:
    """
    Generate GPS C/A code for specified PRN.
    
    Args:
        prn_number: PRN number (1-32)
    
    Returns:
        C/A code array of 1023 chips with values +1/-1
    """
    g2_shifts = {
        1: (2,6), 2: (3,7), 3: (4,8), 4: (5,9), 5: (1,9),
        6: (2,10), 7: (1,8), 8: (2,9), 9: (3,10), 10: (2,3),
        11: (3,4), 12: (5,6), 13: (6,7), 14: (7,8), 15: (8,9),
        16: (9,10), 17: (1,4), 18: (2,5), 19: (3,6), 20: (4,7),
        21: (5,8), 22: (6,9), 23: (1,3), 24: (4,6), 25: (5,7),
        26: (6,8), 27: (7,9), 28: (8,10), 29: (1,6), 30: (2,7),
        31: (3,8), 32: (4,9)
    }
    
    if prn_number not in g2_shifts:
        raise ValueError(f"PRN number must be 1-32, got {prn_number}")
    
    s1, s2 = g2_shifts[prn_number]
    g1 = np.ones(10, dtype=int)
    g2 = np.ones(10, dtype=int)
    ca = np.zeros(1023, dtype=int)
    
    for i in range(1023):
        ca[i] = (g1[-1] ^ (g2[-s1] ^ g2[-s2]))
        new_g1 = g1[2] ^ g1[9]
        new_g2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = np.roll(g1, 1)
        g1[0] = new_g1
        g2 = np.roll(g2, 1)
        g2[0] = new_g2
    
    return 1 - 2 * ca
