"""
<<<<<<< HEAD
Signal I/O functions for loading GPS signals from various formats.
"""
import os
from typing import Optional, Union
import numpy as np
from pathlib import Path


def read_iq_binary(
    file_path: Union[str, Path],
    start_offset_samples: int = 0,
    count_samples: Optional[int] = None,
    dtype: str = 'int16'
) -> Optional[np.ndarray]:
    """
    Load I/Q data from binary file (interleaved int16 format).
    
    This function reads binary GPS signal data in the format used by datasets
    like TEXBAT and FGI-SpoofRepo, where I and Q samples are interleaved as
    int16 values: [I0, Q0, I1, Q1, ...].
    
    Args:
        file_path: Path to binary file (.bin or .dat)
        start_offset_samples: Starting position in complex samples (not bytes)
        count_samples: Number of complex samples to read (None = read all)
        dtype: Data type of the raw samples (default: 'int16')
    
    Returns:
        Complex numpy array (float32) with I/Q data, or None if read fails
        
    Example:
        >>> signal = read_iq_binary('data/gps_signal.bin', 0, 250000)
        >>> print(signal.shape, signal.dtype)
        (250000,) complex64
    """
    bytes_per_iq_pair = np.dtype(dtype).itemsize * 2
    start_offset_bytes = start_offset_samples * bytes_per_iq_pair
    
    try:
        with open(file_path, 'rb') as f:
            # Seek to start position
            f.seek(start_offset_bytes)
            
            # Determine count
            if count_samples is None:
                # Read all remaining data
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()
                f.seek(start_offset_bytes)  # Seek back
                count_int = (file_size - start_offset_bytes) // np.dtype(dtype).itemsize
            else:
                count_int = 2 * count_samples
            
            # Read raw data
            raw = np.fromfile(f, dtype=dtype, count=count_int)
        
        if raw.size < 2:
            return None
        
        # Separate I and Q components
        I = raw[0::2].astype(np.float32)
        Q = raw[1::2].astype(np.float32)
        
        # Return complex signal
        return I + 1j * Q
    
    except (OSError, IOError) as e:
        print(f"Warning: Failed to read {file_path}: {e}")
        return None


def load_signal(
    file_path: Union[str, Path],
    file_format: Optional[str] = None,
    **kwargs
) -> Optional[np.ndarray]:
    """
    Generic signal loader supporting multiple formats.
    
    Supports:
    - Binary I/Q files (.bin, .dat): interleaved int16 format
    - CSV files (.csv): expects columns 'I' and 'Q' or two columns
    - MAT files (.mat): expects variable 'signal' or 'data'
    
    Args:
        file_path: Path to signal file
        file_format: Force specific format ('binary', 'csv', 'mat').
                    If None, infers from extension.
        **kwargs: Additional arguments passed to format-specific loaders
    
    Returns:
        Complex numpy array with signal data, or None if loading fails
        
    Example:
        >>> # Auto-detect format
        >>> signal = load_signal('data/signal.bin')
        >>> 
        >>> # Force format and specify parameters
        >>> signal = load_signal('data/signal.dat', file_format='binary',
        ...                      start_offset_samples=1000, count_samples=50000)
    """
    file_path = Path(file_path)
    
    # Infer format from extension if not provided
    if file_format is None:
        ext = file_path.suffix.lower()
        if ext in ['.bin', '.dat']:
            file_format = 'binary'
        elif ext == '.csv':
            file_format = 'csv'
        elif ext == '.mat':
            file_format = 'mat'
        else:
            raise ValueError(f"Unknown file format: {ext}. Specify file_format explicitly.")
    
    # Load based on format
    if file_format == 'binary':
        return read_iq_binary(file_path, **kwargs)
    
    elif file_format == 'csv':
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            
            # Try to find I and Q columns
            if 'I' in df.columns and 'Q' in df.columns:
                return df['I'].values + 1j * df['Q'].values
            elif len(df.columns) >= 2:
                # Assume first two columns are I and Q
                return df.iloc[:, 0].values + 1j * df.iloc[:, 1].values
            else:
                raise ValueError("CSV must have at least 2 columns or columns named 'I' and 'Q'")
        except Exception as e:
            print(f"Warning: Failed to read CSV {file_path}: {e}")
            return None
    
    elif file_format == 'mat':
        try:
            from scipy.io import loadmat
            mat_data = loadmat(file_path)
            
            # Try common variable names
            for var_name in ['signal', 'data', 'iq', 'samples']:
                if var_name in mat_data:
                    data = mat_data[var_name].squeeze()
                    # Handle real arrays (separate I and Q)
                    if np.isrealobj(data) and data.ndim == 2 and data.shape[1] == 2:
                        return data[:, 0] + 1j * data[:, 1]
                    # Handle complex arrays
                    elif np.iscomplexobj(data):
                        return data
                    # Handle single column real data
                    elif np.isrealobj(data) and data.ndim == 1:
                        print(f"Warning: Found real-valued data in {file_path}. "
                              "Treating as I-only signal (Q=0).")
                        return data.astype(np.complex64)
            
            raise ValueError(f"Could not find signal data in {file_path}. "
                           "Expected variables: 'signal', 'data', 'iq', or 'samples'")
        except Exception as e:
            print(f"Warning: Failed to read MAT file {file_path}: {e}")
            return None
    
    else:
        raise ValueError(f"Unsupported format: {file_format}")


def generate_ca_code(prn_number: int) -> np.ndarray:
    """
    Generate GPS C/A code (Gold sequence) for given PRN.
    
    Implements the GPS C/A code generator using two 10-bit Linear Feedback
    Shift Registers (LFSRs) with specific tap configurations for each PRN.
    
    Args:
        prn_number: Satellite PRN number (1-32)
    
    Returns:
        Array of 1023 chips with values +1 or -1
        
    Raises:
        ValueError: If PRN number is outside valid range
        
    Example:
        >>> ca = generate_ca_code(1)
        >>> print(ca.shape, ca[0])
        (1023,) 1
    """
    # G2 shift tap positions for each PRN
    g2_shifts = {
        1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9), 5: (1, 9),
        6: (2, 10), 7: (1, 8), 8: (2, 9), 9: (3, 10), 10: (2, 3),
        11: (3, 4), 12: (5, 6), 13: (6, 7), 14: (7, 8), 15: (8, 9),
        16: (9, 10), 17: (1, 4), 18: (2, 5), 19: (3, 6), 20: (4, 7),
        21: (5, 8), 22: (6, 9), 23: (1, 3), 24: (4, 6), 25: (5, 7),
        26: (6, 8), 27: (7, 9), 28: (8, 10), 29: (1, 6), 30: (2, 7),
        31: (3, 8), 32: (4, 9)
    }
    
    if prn_number not in g2_shifts:
        raise ValueError(f"PRN must be in range 1-32, got {prn_number}")
    
    s1, s2 = g2_shifts[prn_number]
    
    # Initialize LFSRs with all ones
    g1 = np.ones(10, dtype=int)
    g2 = np.ones(10, dtype=int)
    ca = np.zeros(1023, dtype=int)
    
    # Generate 1023 chips
    for i in range(1023):
        # XOR outputs
        ca[i] = (g1[-1] ^ (g2[-s1] ^ g2[-s2]))
        
        # Update G1 (taps: 3, 10)
        new_g1 = g1[2] ^ g1[9]
        g1 = np.roll(g1, 1)
        g1[0] = new_g1
        
        # Update G2 (taps: 2, 3, 6, 8, 9, 10)
        new_g2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g2 = np.roll(g2, 1)
        g2[0] = new_g2
    
    # Convert to +1/-1
    return 1 - 2 * ca
=======
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
>>>>>>> main
