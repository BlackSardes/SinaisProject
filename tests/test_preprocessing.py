"""
Tests for preprocessing module.
"""
import pytest
import numpy as np
from src.preprocessing.signal_io import generate_ca_code
from src.preprocessing.normalization import normalize_signal, remove_dc
from src.preprocessing.filtering import bandpass_filter
from src.preprocessing.windowing import window_segment


def test_generate_ca_code():
    """Test C/A code generation."""
    # Generate code for PRN 1
    ca_code = generate_ca_code(1)
    
    # Check length
    assert len(ca_code) == 1023, "C/A code should have 1023 chips"
    
    # Check values are +1 or -1
    assert np.all(np.isin(ca_code, [-1, 1])), "C/A code should only contain +1 and -1"
    
    # Check reproducibility
    ca_code2 = generate_ca_code(1)
    assert np.array_equal(ca_code, ca_code2), "C/A code should be deterministic"
    
    # Check different PRNs produce different codes
    ca_code_prn2 = generate_ca_code(2)
    assert not np.array_equal(ca_code, ca_code_prn2), "Different PRNs should produce different codes"


def test_generate_ca_code_invalid_prn():
    """Test that invalid PRN raises error."""
    with pytest.raises(ValueError):
        generate_ca_code(0)
    
    with pytest.raises(ValueError):
        generate_ca_code(33)


def test_normalize_signal(sample_signal):
    """Test signal normalization."""
    signal_norm = normalize_signal(sample_signal, method='power')
    
    # Check power is approximately 1
    power = np.mean(np.abs(signal_norm) ** 2)
    assert np.isclose(power, 1.0, rtol=0.01), f"Power should be ~1.0, got {power}"
    
    # Check signal length unchanged
    assert len(signal_norm) == len(sample_signal)
    
    # Check complex signal preserved
    assert np.iscomplexobj(signal_norm)


def test_remove_dc(sample_signal):
    """Test DC removal."""
    # Add DC offset
    signal_with_dc = sample_signal + (10 + 5j)
    
    # Remove DC
    signal_dc_removed = remove_dc(signal_with_dc)
    
    # Check mean is approximately zero
    mean_val = np.mean(signal_dc_removed)
    assert np.abs(mean_val) < 0.1, f"Mean should be ~0, got {mean_val}"
    
    # Check length unchanged
    assert len(signal_dc_removed) == len(sample_signal)


def test_bandpass_filter(sample_signal):
    """Test bandpass filtering."""
    fs = 5e6
    signal_filtered = bandpass_filter(sample_signal, fs, low=0, high=2e6, order=5)
    
    # Check length unchanged (filtfilt preserves length)
    assert len(signal_filtered) == len(sample_signal)
    
    # Check complex signal preserved
    assert np.iscomplexobj(signal_filtered)
    
    # Check filter reduces out-of-band noise (basic check)
    # Signal should still have reasonable power
    power_original = np.mean(np.abs(sample_signal) ** 2)
    power_filtered = np.mean(np.abs(signal_filtered) ** 2)
    assert power_filtered > 0.1 * power_original, "Filtered signal has too little power"


def test_window_segment(sample_signal):
    """Test signal segmentation."""
    fs = 5e6
    window_s = 0.02  # 20 ms
    hop_s = 0.01     # 10 ms
    
    segments = window_segment(sample_signal, fs, window_s, hop_s)
    
    # Check we got segments
    assert len(segments) > 0, "Should produce at least one segment"
    
    # Check segment length
    expected_length = int(window_s * fs)
    for seg in segments:
        assert len(seg) == expected_length, f"Each segment should have {expected_length} samples"
    
    # Check overlap
    # With 50% overlap and 0.1s signal, we should get ~9 segments
    expected_segments = int((len(sample_signal) / fs - window_s) / hop_s) + 1
    assert len(segments) >= expected_segments - 1, f"Expected ~{expected_segments} segments"


def test_window_segment_with_indices(sample_signal):
    """Test window_segment returns indices when requested."""
    fs = 5e6
    segments, indices = window_segment(
        sample_signal, fs, window_s=0.02, hop_s=0.01, return_indices=True
    )
    
    assert len(segments) == len(indices), "Should have same number of segments and indices"
    assert indices[0] == 0, "First index should be 0"
    
    # Check indices are spaced by hop
    hop_samples = int(0.01 * fs)
    for i in range(1, len(indices)):
        assert indices[i] == indices[i-1] + hop_samples, "Indices should be spaced by hop"
