"""
Correlation computation functions for GPS signal processing.
"""
import numpy as np
from typing import Optional
from ..preprocessing.signal_io import generate_ca_code


def compute_cross_correlation(
    signal: np.ndarray,
    prn_code: np.ndarray,
    method: str = 'fft'
) -> np.ndarray:
    """
    Compute cross-correlation between signal and PRN code.
    
    Cross-correlation is fundamental to GPS signal acquisition and tracking.
    The correlation peak indicates code phase alignment.
    
    Args:
        signal: Complex I/Q signal
        prn_code: Local PRN code replica (+1/-1 values)
        method: Computation method
            - 'fft': FFT-based (fast, O(N log N))
            - 'direct': Direct computation (slow, O(N^2))
    
    Returns:
        Complex correlation values
        
    Example:
        >>> ca_code = generate_ca_code(prn=1)
        >>> # Oversample code to match signal rate
        >>> corr = compute_cross_correlation(signal, ca_code_oversampled)
        >>> peak_idx = np.argmax(np.abs(corr))
    """
    if method == 'fft':
        # FFT-based correlation (efficient)
        fft_signal = np.fft.fft(signal)
        fft_code = np.fft.fft(prn_code)
        corr_fft = fft_signal * np.conj(fft_code)
        corr = np.fft.ifft(corr_fft)
        return corr
    
    elif method == 'direct':
        # Direct correlation (for reference/debugging)
        N = len(signal)
        M = len(prn_code)
        if M > N:
            raise ValueError("PRN code length cannot exceed signal length")
        
        corr = np.zeros(N, dtype=complex)
        for lag in range(N):
            if lag + M <= N:
                corr[lag] = np.sum(signal[lag:lag+M] * np.conj(prn_code))
            else:
                # Wrap around
                part1 = signal[lag:]
                part2 = signal[:lag+M-N]
                code_part1 = prn_code[:len(part1)]
                code_part2 = prn_code[len(part1):]
                corr[lag] = np.sum(part1 * np.conj(code_part1)) + \
                           np.sum(part2 * np.conj(code_part2))
        return corr
    
    else:
        raise ValueError(f"Unknown correlation method: {method}")


def compute_autocorrelation(
    signal: np.ndarray,
    max_lag: Optional[int] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute autocorrelation of signal.
    
    Autocorrelation reveals periodicity and correlation structure,
    useful for analyzing signal self-similarity.
    
    Args:
        signal: Complex or real input signal
        max_lag: Maximum lag to compute (None = full length)
        normalize: If True, normalize to [-1, 1] range
    
    Returns:
        Autocorrelation values
        
    Example:
        >>> acf = compute_autocorrelation(signal, max_lag=1000)
        >>> # Peak at 0 lag, with periodic structure for GPS signals
    """
    N = len(signal)
    if max_lag is None:
        max_lag = N
    
    # Use FFT method for efficiency
    fft_sig = np.fft.fft(signal, n=2*N)
    power_spectrum = fft_sig * np.conj(fft_sig)
    acf = np.fft.ifft(power_spectrum)
    acf = acf[:N].real  # Take only positive lags
    
    if normalize:
        acf = acf / acf[0]
    
    return acf[:max_lag]


def generate_local_code(
    prn_number: int,
    fs: float,
    duration_s: float,
    ca_chip_rate: float = 1.023e6
) -> np.ndarray:
    """
    Generate oversampled local PRN code for correlation.
    
    Creates a PRN code replica at the signal sampling rate by
    oversampling and repeating the base C/A code.
    
    Args:
        prn_number: Satellite PRN (1-32)
        fs: Sampling frequency in Hz
        duration_s: Duration of code to generate in seconds
        ca_chip_rate: C/A code chip rate in Hz (default: 1.023 MHz)
    
    Returns:
        Oversampled PRN code array
        
    Example:
        >>> # Generate 1ms of PRN 1 at 5 MHz sampling
        >>> local_code = generate_local_code(prn=1, fs=5e6, duration_s=0.001)
        >>> print(len(local_code))
        5000
    """
    # Generate base C/A code (1023 chips)
    ca_code = generate_ca_code(prn_number)
    
    # Calculate samples per chip
    samples_per_chip = int(fs / ca_chip_rate)
    
    # Oversample by repeating each chip
    code_oversampled = np.repeat(ca_code, samples_per_chip)
    
    # Calculate total samples needed
    total_samples = int(fs * duration_s)
    
    # Repeat code to fill duration
    ca_period_samples = len(code_oversampled)
    repeats = int(np.ceil(total_samples / ca_period_samples))
    code_full = np.tile(code_oversampled, repeats)
    
    # Trim to exact length
    return code_full[:total_samples]
