"""
<<<<<<< HEAD
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
=======
Unit tests for preprocessing module.
"""

import pytest
import numpy as np
from src.preprocessing.signal_io import generate_synthetic_signal
from src.preprocessing.signal_processing import (
    normalize_signal,
    remove_dc,
    bandpass_filter,
    notch_filter,
    resample_signal,
    segment_signal,
    estimate_cn0_from_signal
)
from src.preprocessing.prn_codes import generate_ca_code, verify_ca_code_properties
from src.preprocessing.pipeline import preprocess_signal


class TestSignalIO:
    """Tests for signal I/O functions."""
    
    def test_generate_synthetic_signal(self):
        """Test synthetic signal generation."""
        num_samples = 10000
        fs = 5e6
        signal = generate_synthetic_signal(num_samples, fs, snr_db=10.0, prn=1)
        
        assert signal is not None
        assert len(signal) == num_samples
        assert np.iscomplexobj(signal)
        assert signal.dtype == np.complex64
    
    def test_synthetic_signal_with_spoofing(self):
        """Test synthetic signal with spoofing."""
        signal = generate_synthetic_signal(
            num_samples=10000,
            fs=5e6,
            snr_db=10.0,
            prn=1,
            add_spoofing=True
        )
        
        assert signal is not None
        assert len(signal) == 10000


class TestSignalProcessing:
    """Tests for signal processing functions."""
    
    def test_normalize_power(self):
        """Test power normalization."""
        signal = np.random.randn(1000) + 1j * np.random.randn(1000)
        signal_norm = normalize_signal(signal, method='power')
        
        power = np.mean(np.abs(signal_norm) ** 2)
        assert np.abs(power - 1.0) < 0.01  # Should be ~1.0
    
    def test_normalize_amplitude(self):
        """Test amplitude normalization."""
        signal = np.random.randn(1000) + 1j * np.random.randn(1000)
        signal_norm = normalize_signal(signal, method='amplitude')
        
        max_amp = np.max(np.abs(signal_norm))
        assert np.abs(max_amp - 1.0) < 0.01  # Should be ~1.0
    
    def test_remove_dc(self):
        """Test DC removal."""
        dc_offset = 5.0 + 3.0j
        signal = np.random.randn(1000) + 1j * np.random.randn(1000) + dc_offset
        signal_clean = remove_dc(signal)
        
        mean = np.mean(signal_clean)
        assert np.abs(mean) < 0.1  # Should be near zero
    
    def test_bandpass_filter(self):
        """Test bandpass filtering."""
        fs = 5e6
        num_samples = 10000
        
        # Create signal with known frequency content
        t = np.arange(num_samples) / fs
        signal = np.cos(2 * np.pi * 1e6 * t)  # 1 MHz tone
        
        # Filter: pass 0.5-1.5 MHz
        filtered = bandpass_filter(signal, fs, 0.5e6, 1.5e6)
        
        assert len(filtered) == len(signal)
        # Signal at 1 MHz should pass through
        assert np.max(np.abs(filtered)) > 0.5
    
    def test_notch_filter(self):
        """Test notch filtering."""
        fs = 5e6
        num_samples = 10000
        
        # Create signal with interference at 1 MHz
        t = np.arange(num_samples) / fs
        interference = 10 * np.cos(2 * np.pi * 1e6 * t)
        signal = np.random.randn(num_samples) + interference
        
        # Notch at 1 MHz
        filtered = notch_filter(signal, fs, 1e6, Q=30)
        
        assert len(filtered) == len(signal)
        # Power should be reduced
        assert np.mean(np.abs(filtered) ** 2) < np.mean(np.abs(signal) ** 2)
    
    def test_resample_signal(self):
        """Test signal resampling."""
        signal = np.random.randn(1000)
        fs_old = 5e6
        fs_new = 10e6
        
        resampled = resample_signal(signal, fs_old, fs_new)
        
        # Should have 2x samples
        assert len(resampled) == 2000
    
    def test_segment_signal(self):
        """Test signal segmentation."""
        signal = np.random.randn(10000)
        segment_length = 1000
        
        segments = segment_signal(signal, segment_length, overlap=0.0)
        
        assert len(segments) == 10
        assert all(len(seg) == segment_length for seg in segments)
    
    def test_segment_signal_with_overlap(self):
        """Test signal segmentation with overlap."""
        signal = np.random.randn(10000)
        segment_length = 1000
        
        segments = segment_signal(signal, segment_length, overlap=0.5)
        
        # With 50% overlap, should have more segments
        assert len(segments) > 10
    
    def test_cn0_estimation(self):
        """Test C/N0 estimation."""
        signal = generate_synthetic_signal(10000, 5e6, snr_db=10.0, prn=1)
        cn0 = estimate_cn0_from_signal(signal, 5e6)
        
        # Should return a reasonable value
        assert cn0 > 0
        assert cn0 < 100  # Reasonable upper bound


class TestPRNCodes:
    """Tests for PRN code generation."""
    
    def test_generate_ca_code(self):
        """Test C/A code generation."""
        for prn in [1, 10, 20, 32]:
            ca_code = generate_ca_code(prn)
            
            assert len(ca_code) == 1023
            assert set(ca_code) == {-1, 1}  # Only -1 and +1
    
    def test_ca_code_invalid_prn(self):
        """Test error handling for invalid PRN."""
        with pytest.raises(ValueError):
            generate_ca_code(0)
        
        with pytest.raises(ValueError):
            generate_ca_code(33)
    
    def test_ca_code_autocorrelation(self):
        """Test C/A code autocorrelation properties."""
        properties = verify_ca_code_properties(1, plot=False)
        
        assert properties['code_length'] == 1023
        assert properties['peak_value'] == 1023
        
        # Check peak-to-sidelobe ratio is reasonable
        assert properties['peak_to_sidelobe'] > 10  # Should be ~15.7
    
    def test_different_prns_are_different(self):
        """Test that different PRNs generate different codes."""
        code1 = generate_ca_code(1)
        code2 = generate_ca_code(2)
        
        # Codes should be different
        assert not np.array_equal(code1, code2)


class TestPreprocessingPipeline:
    """Tests for complete preprocessing pipeline."""
    
    def test_preprocess_signal_minimal(self):
        """Test minimal preprocessing."""
        signal = generate_synthetic_signal(10000, 5e6, snr_db=10.0, prn=1)
        
        config = {
            'remove_dc': True,
            'normalize': 'power'
        }
        
        processed = preprocess_signal(signal, 5e6, config)
        
        assert processed is not None
        assert len(processed) == len(signal)
        
        # Check power normalization
        power = np.mean(np.abs(processed) ** 2)
        assert np.abs(power - 1.0) < 0.1
    
    def test_preprocess_signal_full(self):
        """Test full preprocessing pipeline."""
        signal = generate_synthetic_signal(10000, 5e6, snr_db=10.0, prn=1)
        
        config = {
            'remove_dc': True,
            'freq_correction': 0.0,
            'apply_notch': True,
            'notch_freq': 1e6,
            'notch_q': 30.0,
            'pulse_blanking': True,
            'pb_threshold': 4.0,
            'normalize': 'power'
        }
        
        processed = preprocess_signal(signal, 5e6, config)
        
        assert processed is not None
        assert len(processed) == len(signal)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
>>>>>>> main
