"""Tests for preprocessing functions."""

import pytest
import numpy as np
from src.preprocessing import (
    normalize_signal,
    remove_dc,
    window_segment,
    remove_outliers,
    smooth_signal
)


def test_normalize_signal_power():
    """Test power normalization."""
    signal = np.array([1, 2, 3, 4, 5], dtype=float)
    normalized = normalize_signal(signal, method='power')
    
    # Check that power is approximately 1
    power = np.mean(normalized ** 2)
    assert np.isclose(power, 1.0, rtol=1e-6)


def test_normalize_signal_max():
    """Test max normalization."""
    signal = np.array([1, 2, 3, 4, 5], dtype=float)
    normalized = normalize_signal(signal, method='max')
    
    # Check that max is 1
    assert np.isclose(np.max(np.abs(normalized)), 1.0)


def test_normalize_signal_std():
    """Test standard deviation normalization."""
    signal = np.array([1, 2, 3, 4, 5], dtype=float)
    normalized = normalize_signal(signal, method='std')
    
    # Check that mean is 0 and std is 1
    assert np.isclose(np.mean(normalized), 0.0, atol=1e-10)
    assert np.isclose(np.std(normalized), 1.0, rtol=1e-6)


def test_normalize_signal_complex():
    """Test normalization on complex signal."""
    signal = np.array([1+1j, 2+2j, 3+3j])
    normalized = normalize_signal(signal, method='power')
    
    power = np.mean(np.abs(normalized) ** 2)
    assert np.isclose(power, 1.0, rtol=1e-6)


def test_remove_dc():
    """Test DC removal."""
    signal = np.array([1, 2, 3, 4, 5], dtype=float)
    result = remove_dc(signal)
    
    # Check that mean is approximately 0
    assert np.isclose(np.mean(result), 0.0, atol=1e-10)


def test_remove_dc_complex():
    """Test DC removal on complex signal."""
    signal = np.array([1+2j, 2+3j, 3+4j])
    result = remove_dc(signal)
    
    # Check that mean is approximately 0
    assert np.isclose(np.abs(np.mean(result)), 0.0, atol=1e-10)


def test_window_segment_no_overlap():
    """Test window segmentation without overlap."""
    signal = np.arange(100)
    window_size = 10
    
    segments = window_segment(signal, window_size, overlap=0)
    
    assert len(segments) == 10
    assert all(len(seg) == window_size for seg in segments)
    assert np.array_equal(segments[0], signal[:10])


def test_window_segment_with_overlap():
    """Test window segmentation with overlap."""
    signal = np.arange(100)
    window_size = 20
    overlap = 10
    
    segments = window_segment(signal, window_size, overlap=overlap)
    
    # Step size = window_size - overlap = 10
    # Number of segments = (100 - 20) / 10 + 1 = 9
    assert len(segments) == 9
    assert all(len(seg) == window_size for seg in segments)


def test_remove_outliers():
    """Test outlier removal."""
    signal = np.array([1, 2, 3, 4, 100, 6, 7, 8], dtype=float)
    result = remove_outliers(signal, threshold=2.0)
    
    # The outlier (100) should be replaced
    assert result[4] != 100
    assert np.all(result[:4] == signal[:4])


def test_smooth_signal_savgol():
    """Test Savitzky-Golay smoothing."""
    signal = np.sin(np.linspace(0, 10, 100)) + 0.1 * np.random.randn(100)
    smoothed = smooth_signal(signal, method='savgol', window_length=11, polyorder=3)
    
    assert len(smoothed) == len(signal)
    # Smoothed signal should have less variance
    assert np.var(smoothed) <= np.var(signal)


def test_smooth_signal_median():
    """Test median smoothing."""
    signal = np.sin(np.linspace(0, 10, 100)) + 0.1 * np.random.randn(100)
    smoothed = smooth_signal(signal, method='median', window_length=11)
    
    assert len(smoothed) == len(signal)
    # Smoothed signal should have less variance
    assert np.var(smoothed) <= np.var(signal)
