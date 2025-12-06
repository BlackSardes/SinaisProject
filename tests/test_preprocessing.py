"""
Tests for preprocessing module.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from preprocessing.signal_processing import (
    normalize_signal,
    bandpass_filter,
    remove_dc,
    generate_ca_code,
    normalize_by_power,
)


def test_normalize_by_power():
    """Test power normalization."""
    # Create signal with known power
    signal = np.random.randn(1000) + 1j * np.random.randn(1000)
    signal = signal * 5.0  # Scale to have power != 1
    
    # Normalize
    normalized = normalize_by_power(signal)
    
    # Check power is approximately 1
    power = np.mean(np.abs(normalized)**2)
    assert abs(power - 1.0) < 0.01, f"Power should be ~1.0, got {power}"


def test_normalize_signal():
    """Test signal normalization methods."""
    signal = np.random.randn(1000) + 1j * np.random.randn(1000)
    
    # Test power normalization
    norm_power = normalize_signal(signal, method='power')
    assert np.abs(np.mean(np.abs(norm_power)**2) - 1.0) < 0.01
    
    # Test max normalization
    norm_max = normalize_signal(signal, method='max')
    assert abs(np.max(np.abs(norm_max)) - 1.0) < 1e-10


def test_remove_dc():
    """Test DC removal."""
    # Create signal with DC offset
    signal = np.random.randn(1000) + 1j * np.random.randn(1000)
    signal = signal + (3.0 + 2.0j)  # Add DC offset
    
    # Remove DC
    no_dc = remove_dc(signal)
    
    # Check DC is removed
    mean_real = np.mean(np.real(no_dc))
    mean_imag = np.mean(np.imag(no_dc))
    
    assert abs(mean_real) < 1e-10, f"Real DC should be 0, got {mean_real}"
    assert abs(mean_imag) < 1e-10, f"Imag DC should be 0, got {mean_imag}"


def test_bandpass_filter():
    """Test bandpass filter."""
    fs = 5e6
    duration = 0.1
    t = np.arange(int(fs * duration)) / fs
    
    # Create signal with low and high frequency components
    signal = np.sin(2 * np.pi * 1e5 * t) + np.sin(2 * np.pi * 1e6 * t)  # 100kHz + 1MHz
    
    # Bandpass filter to keep only 1MHz component
    filtered = bandpass_filter(signal, fs, low=0.5e6, high=1.5e6, order=5)
    
    # Signal should be reduced but not zero
    assert np.std(filtered) > 0, "Filtered signal should not be zero"
    assert len(filtered) == len(signal), "Length should be preserved"


def test_generate_ca_code():
    """Test C/A code generation."""
    # Test PRN 1
    ca_code = generate_ca_code(1)
    
    # Check properties
    assert len(ca_code) == 1023, "C/A code should have 1023 chips"
    assert set(ca_code).issubset({-1, 1}), "C/A code should only contain -1 and 1"
    
    # Test different PRNs produce different codes
    ca_code_2 = generate_ca_code(2)
    assert not np.array_equal(ca_code, ca_code_2), "Different PRNs should produce different codes"
    
    # Test invalid PRN
    with pytest.raises(ValueError):
        generate_ca_code(0)
    
    with pytest.raises(ValueError):
        generate_ca_code(33)


def test_generate_ca_code_reproducibility():
    """Test that C/A code generation is reproducible."""
    ca_code_1 = generate_ca_code(1)
    ca_code_2 = generate_ca_code(1)
    
    assert np.array_equal(ca_code_1, ca_code_2), "Same PRN should produce identical codes"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
