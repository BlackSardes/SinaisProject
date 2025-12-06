"""
Signal I/O Module for GPS Signal Processing

This module provides functions for loading GPS signals from various formats:
- TEXBAT/FGI binary files (.bin, .dat)
- MATLAB files (.mat)
- CSV files
- Synthetic signal generation for testing
"""

import os
import numpy as np
from typing import Optional, Tuple, Union
from scipy.io import loadmat


def load_signal(
    path: str,
    start_offset_samples: int = 0,
    count_samples: Optional[int] = None,
    format: str = 'auto'
) -> Optional[np.ndarray]:
    """
    Generic signal loader that supports multiple formats.
    
    Parameters
    ----------
    path : str
        Path to the signal file
    start_offset_samples : int, optional
        Starting position in samples (default: 0)
    count_samples : int, optional
        Number of samples to read (None = read all)
    format : str, optional
        File format: 'auto', 'binary', 'mat', 'csv' (default: 'auto')
        
    Returns
    -------
    np.ndarray or None
        Complex signal array (I+jQ) or None if loading fails
    """
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return None
    
    # Auto-detect format from extension
    if format == 'auto':
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.bin', '.dat']:
            format = 'binary'
        elif ext == '.mat':
            format = 'mat'
        elif ext == '.csv':
            format = 'csv'
        else:
            print(f"Warning: Unknown file extension {ext}, trying binary format")
            format = 'binary'
    
    # Load based on format
    if format == 'binary':
        return load_iq_binary(path, start_offset_samples, count_samples)
    elif format == 'mat':
        return load_mat_file(path, start_offset_samples, count_samples)
    elif format == 'csv':
        return load_csv_file(path, start_offset_samples, count_samples)
    else:
        print(f"Error: Unsupported format: {format}")
        return None


def load_iq_binary(
    file_path: str,
    start_offset_samples: int = 0,
    count_samples: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Load I/Q data from binary file (TEXBAT/FGI format).
    
    Data format: Interleaved int16 pairs (I, Q, I, Q, ...)
    
    Parameters
    ----------
    file_path : str
        Path to binary file
    start_offset_samples : int, optional
        Starting sample position (default: 0)
    count_samples : int, optional
        Number of complex samples to read (None = read all)
        
    Returns
    -------
    np.ndarray or None
        Complex signal (I+jQ) as float32, or None on error
        
    Notes
    -----
    This function implements the standard TEXBAT dataset format where
    each I/Q pair is stored as two consecutive int16 values (4 bytes total).
    """
    bytes_per_iq_pair = 4  # 2 bytes I + 2 bytes Q
    start_offset_bytes = start_offset_samples * bytes_per_iq_pair
    
    try:
        with open(file_path, "rb") as f:
            # Seek to start position
            f.seek(start_offset_bytes)
            
            # Determine number of samples to read
            if count_samples is None:
                # Read all remaining data
                raw = np.fromfile(f, dtype=np.int16)
            else:
                count_int16 = 2 * count_samples
                raw = np.fromfile(f, dtype=np.int16, count=count_int16)
        
        # Check if we got enough data
        if raw.size < 2:
            print(f"Warning: Insufficient data in {os.path.basename(file_path)}")
            return None
        
        # Ensure even number of values
        if raw.size % 2 != 0:
            raw = raw[:-1]
        
        # Extract I and Q components
        I = raw[0::2].astype(np.float32)
        Q = raw[1::2].astype(np.float32)
        
        # Construct complex signal
        signal = I + 1j * Q
        
        return signal
    
    except Exception as e:
        print(f"Error reading binary file {os.path.basename(file_path)}: {e}")
        return None


def load_mat_file(
    file_path: str,
    start_offset_samples: int = 0,
    count_samples: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Load signal from MATLAB .mat file.
    
    Parameters
    ----------
    file_path : str
        Path to .mat file
    start_offset_samples : int, optional
        Starting sample position (default: 0)
    count_samples : int, optional
        Number of samples to read (None = read all)
        
    Returns
    -------
    np.ndarray or None
        Complex signal or None on error
        
    Notes
    -----
    Expects the .mat file to contain variables 'I' and 'Q' or 'signal'.
    """
    try:
        mat_data = loadmat(file_path)
        
        # Try to find signal data
        if 'signal' in mat_data:
            signal = mat_data['signal'].flatten()
        elif 'I' in mat_data and 'Q' in mat_data:
            I = mat_data['I'].flatten()
            Q = mat_data['Q'].flatten()
            signal = I + 1j * Q
        else:
            print(f"Error: .mat file must contain 'signal' or 'I'/'Q' variables")
            return None
        
        # Apply offset and count
        end_sample = start_offset_samples + count_samples if count_samples else len(signal)
        signal = signal[start_offset_samples:end_sample]
        
        return signal.astype(np.complex64)
    
    except Exception as e:
        print(f"Error reading .mat file {os.path.basename(file_path)}: {e}")
        return None


def load_csv_file(
    file_path: str,
    start_offset_samples: int = 0,
    count_samples: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Load signal from CSV file.
    
    Parameters
    ----------
    file_path : str
        Path to CSV file
    start_offset_samples : int, optional
        Starting sample position (default: 0)
    count_samples : int, optional
        Number of samples to read (None = read all)
        
    Returns
    -------
    np.ndarray or None
        Complex signal or None on error
        
    Notes
    -----
    Expects CSV with two columns: I and Q (real and imaginary parts).
    """
    try:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        
        if data.shape[1] < 2:
            print(f"Error: CSV must have at least 2 columns (I, Q)")
            return None
        
        I = data[:, 0]
        Q = data[:, 1]
        signal = I + 1j * Q
        
        # Apply offset and count
        end_sample = start_offset_samples + count_samples if count_samples else len(signal)
        signal = signal[start_offset_samples:end_sample]
        
        return signal.astype(np.complex64)
    
    except Exception as e:
        print(f"Error reading CSV file {os.path.basename(file_path)}: {e}")
        return None


def generate_synthetic_signal(
    num_samples: int,
    fs: float,
    snr_db: float = 10.0,
    prn: int = 1,
    doppler_hz: float = 0.0,
    add_spoofing: bool = False
) -> np.ndarray:
    """
    Generate synthetic GPS-like signal for testing.
    
    Parameters
    ----------
    num_samples : int
        Number of samples to generate
    fs : float
        Sampling frequency in Hz
    snr_db : float, optional
        Signal-to-noise ratio in dB (default: 10.0)
    prn : int, optional
        PRN code number (default: 1)
    doppler_hz : float, optional
        Doppler frequency shift in Hz (default: 0.0)
    add_spoofing : bool, optional
        If True, add a spoofing signal (default: False)
        
    Returns
    -------
    np.ndarray
        Complex synthetic signal
        
    Notes
    -----
    This is a simplified model for testing purposes. Real GPS signals
    are much more complex.
    """
    from .prn_codes import generate_ca_code
    
    # Generate PRN code
    ca_code = generate_ca_code(prn)
    
    # Oversample the code
    ca_chip_rate = 1.023e6
    samples_per_chip = int(fs / ca_chip_rate)
    local_code = np.repeat(ca_code, samples_per_chip)
    
    # Repeat code to match signal length
    repeats = int(np.ceil(num_samples / len(local_code)))
    local_code = np.tile(local_code, repeats)[:num_samples]
    
    # Create time vector
    t = np.arange(num_samples) / fs
    
    # Add carrier with Doppler
    carrier = np.exp(1j * 2 * np.pi * doppler_hz * t)
    
    # Modulate
    signal = local_code * carrier
    
    # Calculate noise power for desired SNR
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Add complex noise
    noise = np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    signal = signal + noise
    
    # Add spoofing signal if requested
    if add_spoofing:
        # Spoofing signal is similar but slightly delayed and stronger
        spoof_delay = int(0.1 * samples_per_chip)  # Small delay
        spoof_signal = 1.5 * np.roll(signal, spoof_delay)  # 50% stronger
        signal = signal + spoof_signal
    
    return signal.astype(np.complex64)
