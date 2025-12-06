"""
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
