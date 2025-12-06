"""Tests for feature metrics."""

import pytest
import numpy as np
from src.features.metrics import fwhm


def test_fwhm_simple_peak():
    """Test FWHM calculation on a simple triangular peak."""
    # Create a simple triangular peak
    signal = np.array([0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0])
    peak_idx = 6
    
    result = fwhm(signal, peak_idx)
    
    # FWHM should be around 4-6 samples for this peak
    assert result > 0
    assert result < len(signal)


def test_fwhm_gaussian_peak():
    """Test FWHM on a Gaussian-like peak."""
    # Create a Gaussian-like peak
    x = np.linspace(-5, 5, 100)
    sigma = 1.0
    signal = np.exp(-(x**2) / (2 * sigma**2))
    
    result = fwhm(signal)
    
    # For a Gaussian, FWHM ≈ 2.355 * sigma in the same units
    # Since we have 100 points over 10 units, 1 unit = 10 points
    # Expected FWHM ≈ 2.355 * 1.0 * 10 = 23.55 points
    assert 20 < result < 30


def test_fwhm_flat_signal():
    """Test FWHM on a flat signal (no peak)."""
    signal = np.ones(100)
    
    result = fwhm(signal)
    
    # For a flat signal, FWHM should be 0 or very small
    assert result >= 0


def test_fwhm_zero_signal():
    """Test FWHM on a zero signal."""
    signal = np.zeros(100)
    
    result = fwhm(signal)
    
    assert result == 0.0


def test_fwhm_complex_signal():
    """Test FWHM on a complex signal."""
    # Create a complex signal with a peak in magnitude
    x = np.linspace(-5, 5, 100)
    signal = np.exp(-(x**2) / 2) * np.exp(1j * x)
    
    result = fwhm(signal)
    
    # Should handle complex signals by using magnitude
    assert result > 0
    assert result < len(signal)


def test_fwhm_with_peak_idx():
    """Test FWHM with explicit peak index."""
    signal = np.array([0, 0, 1, 2, 3, 2, 1, 0, 0])
    peak_idx = 4
    
    result = fwhm(signal, peak_idx)
    
    assert result > 0
    assert result < len(signal)
