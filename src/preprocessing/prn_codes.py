"""
PRN Code Generation Module

This module provides functions for generating GPS C/A codes (Gold codes)
for satellite identification and correlation.
"""

import numpy as np
from typing import Optional


def generate_ca_code(prn_number: int) -> np.ndarray:
    """
    Generate GPS C/A code (Gold sequence) for a specific PRN.
    
    Parameters
    ----------
    prn_number : int
        PRN satellite number (1-32)
        
    Returns
    -------
    np.ndarray
        C/A code array of length 1023 with values +1 or -1
        
    Raises
    ------
    ValueError
        If prn_number is not in range 1-32
        
    Notes
    -----
    GPS C/A codes are 1023-chip Gold codes generated from two Linear Feedback
    Shift Registers (LFSRs). These codes have good autocorrelation properties
    (sharp peak) and low cross-correlation (orthogonality between satellites).
    
    The codes are fundamental to GPS signal processing as they enable:
    - Satellite identification
    - Range measurement via correlation
    - Multiple access (CDMA)
    
    References
    ----------
    IS-GPS-200 Interface Specification
    """
    # G2 tap positions for each PRN (phase selection)
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
        raise ValueError(f"PRN number must be in range 1-32, got {prn_number}")
    
    s1, s2 = g2_shifts[prn_number]
    
    # Initialize G1 and G2 registers (10-bit LFSRs)
    g1 = np.ones(10, dtype=int)
    g2 = np.ones(10, dtype=int)
    
    # Generate 1023 chips
    ca = np.zeros(1023, dtype=int)
    
    for i in range(1023):
        # Output: G1[10] XOR (G2[s1] XOR G2[s2])
        ca[i] = g1[-1] ^ (g2[-s1] ^ g2[-s2])
        
        # G1 feedback: tap positions 3 and 10
        new_g1 = g1[2] ^ g1[9]
        
        # G2 feedback: tap positions 2, 3, 6, 8, 9, 10
        new_g2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        
        # Shift registers
        g1 = np.roll(g1, 1)
        g1[0] = new_g1
        g2 = np.roll(g2, 1)
        g2[0] = new_g2
    
    # Convert from {0, 1} to {-1, +1} for bipolar modulation
    return 1 - 2 * ca


def generate_local_code_oversampled(
    prn_number: int,
    fs: float,
    num_samples: int,
    ca_chip_rate: float = 1.023e6
) -> np.ndarray:
    """
    Generate oversampled local PRN code for correlation.
    
    Parameters
    ----------
    prn_number : int
        PRN satellite number (1-32)
    fs : float
        Sampling frequency in Hz
    num_samples : int
        Desired length of output signal in samples
    ca_chip_rate : float, optional
        C/A code chip rate in Hz (default: 1.023e6)
        
    Returns
    -------
    np.ndarray
        Oversampled PRN code of length num_samples
        
    Notes
    -----
    This function generates a PRN code at the sampling rate of the received
    signal, which is necessary for correlation in the time domain.
    
    Each chip is repeated for samples_per_chip samples, and the code is
    tiled to match the desired signal length.
    """
    # Generate base C/A code
    ca_code = generate_ca_code(prn_number)
    
    # Calculate samples per chip
    samples_per_chip = int(fs / ca_chip_rate)
    
    # Oversample by repeating each chip
    local_code = np.repeat(ca_code, samples_per_chip)
    
    # Tile code to match or exceed desired length
    repeats = int(np.ceil(num_samples / len(local_code)))
    local_code = np.tile(local_code, repeats)
    
    # Truncate to exact length
    return local_code[:num_samples].astype(np.float32)


def verify_ca_code_properties(prn_number: int, plot: bool = False) -> dict:
    """
    Verify autocorrelation properties of C/A code.
    
    Parameters
    ----------
    prn_number : int
        PRN satellite number to verify
    plot : bool, optional
        If True, plot the autocorrelation function (default: False)
        
    Returns
    -------
    dict
        Dictionary with properties:
        - peak_value: Maximum autocorrelation
        - peak_to_sidelobe: Ratio of peak to max sidelobe
        - code_length: Length of code (should be 1023)
        
    Notes
    -----
    A good C/A code should have:
    - Peak autocorrelation at zero lag: 1023
    - Maximum sidelobe level: ±65 (about -13.9 dB)
    - Peak-to-sidelobe ratio: ~15.7 (23.9 dB)
    """
    ca_code = generate_ca_code(prn_number)
    
    # Compute autocorrelation
    autocorr = np.correlate(ca_code, ca_code, mode='full')
    center = len(autocorr) // 2
    
    # Find peak and sidelobe
    peak_value = autocorr[center]
    
    # Mask out the peak region
    mask = np.ones(len(autocorr), dtype=bool)
    mask[center] = False
    max_sidelobe = np.max(np.abs(autocorr[mask]))
    
    peak_to_sidelobe = peak_value / max_sidelobe if max_sidelobe > 0 else np.inf
    
    properties = {
        'peak_value': int(peak_value),
        'max_sidelobe': int(max_sidelobe),
        'peak_to_sidelobe': float(peak_to_sidelobe),
        'peak_to_sidelobe_db': float(20 * np.log10(peak_to_sidelobe)),
        'code_length': len(ca_code)
    }
    
    if plot:
        import matplotlib.pyplot as plt
        lags = np.arange(-len(ca_code) + 1, len(ca_code))
        plt.figure(figsize=(12, 4))
        plt.plot(lags, autocorr, linewidth=0.8)
        plt.axhline(y=max_sidelobe, color='r', linestyle='--', label=f'Max sidelobe: {max_sidelobe}')
        plt.axhline(y=-max_sidelobe, color='r', linestyle='--')
        plt.xlabel('Lag (chips)')
        plt.ylabel('Autocorrelation')
        plt.title(f'C/A Code Autocorrelation - PRN {prn_number}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return properties
