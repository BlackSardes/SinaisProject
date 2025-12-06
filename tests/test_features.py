"""
Tests for feature extraction module.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from features.correlation import compute_cross_correlation, extract_correlation_features
from features.temporal import extract_temporal_features
from features.pipeline import build_feature_vector
from preprocessing.signal_processing import generate_ca_code


def test_compute_cross_correlation():
    """Test cross-correlation computation."""
    # Create simple signal and code
    signal = np.random.randn(1000) + 1j * np.random.randn(1000)
    code = np.random.choice([-1, 1], size=1023)
    
    # Compute correlation
    corr = compute_cross_correlation(signal, code)
    
    # Check properties
    assert len(corr) == len(signal), "Correlation should have same length as signal"
    assert np.all(corr >= 0), "Correlation magnitude should be non-negative"
    assert np.max(corr) > 0, "Correlation should have non-zero peak"


def test_extract_correlation_features():
    """Test correlation feature extraction."""
    # Create synthetic correlation profile with clear peak
    profile = np.random.randn(1000) * 0.1 + 10  # Noise floor
    peak_idx = 500
    profile[peak_idx] = 100  # Strong peak
    profile[peak_idx-10:peak_idx+10] = 80  # Wider peak
    profile = np.abs(profile)
    
    # Extract features
    fs = 5e6
    features = extract_correlation_features(profile, fs)
    
    # Check expected features
    assert 'peak_height' in features
    assert 'peak_index' in features
    assert 'peak_to_secondary' in features
    assert 'fwhm' in features
    assert 'asymmetry' in features
    
    # Check values are reasonable
    assert features['peak_height'] > 0
    assert features['peak_index'] == peak_idx
    assert features['peak_to_secondary'] > 1  # Primary should be larger than secondary


def test_extract_temporal_features():
    """Test temporal feature extraction."""
    # Create signal
    signal = np.random.randn(1000) + 1j * np.random.randn(1000)
    fs = 5e6
    
    # Extract features
    features = extract_temporal_features(signal, fs, correlation_peak=100.0)
    
    # Check expected features
    assert 'mean_real' in features
    assert 'mean_imag' in features
    assert 'var_real' in features
    assert 'var_imag' in features
    assert 'total_power' in features
    assert 'cn0_estimate' in features
    
    # Check values are finite
    for key, value in features.items():
        assert np.isfinite(value), f"Feature {key} should be finite"


def test_build_feature_vector():
    """Test complete feature vector building."""
    # Generate synthetic signal
    fs = 5e6
    duration = 0.1
    t = np.arange(int(fs * duration)) / fs
    
    # Simple signal
    signal = np.exp(1j * 2 * np.pi * 1000 * t)  # 1kHz carrier
    signal += (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) * 0.1
    
    # Generate PRN code
    prn_code = generate_ca_code(1)
    
    # Build feature vector
    features = build_feature_vector(
        signal=signal,
        prn_code=prn_code,
        fs=fs,
        label=0,
        metadata={'prn': 1}
    )
    
    # Check it's a dict
    assert isinstance(features, dict)
    
    # Check key features exist
    assert 'peak_height' in features
    assert 'cn0_estimate' in features
    assert 'total_power' in features
    assert 'label' in features
    assert 'prn' in features
    
    # Check label
    assert features['label'] == 0


def test_feature_extraction_with_synthetic_spoofing():
    """Test that features differ between authentic and spoofed signals."""
    from utils.synthetic_data import generate_synthetic_gps_signal
    
    fs = 5e6
    duration = 0.1
    prn = 1
    
    # Generate authentic signal
    auth_signal = generate_synthetic_gps_signal(
        prn=prn, fs=fs, duration=duration,
        cn0=45.0, spoofed=False
    )
    
    # Generate spoofed signal (with higher power)
    spoof_signal = generate_synthetic_gps_signal(
        prn=prn, fs=fs, duration=duration,
        cn0=50.0, spoofed=True,
        spoof_params={'power_increase': 10.0}
    )
    
    # Extract features
    prn_code = generate_ca_code(prn)
    auth_features = build_feature_vector(auth_signal, prn_code, fs)
    spoof_features = build_feature_vector(spoof_signal, prn_code, fs)
    
    # Spoofed signal should have higher peak
    assert spoof_features['peak_height'] > auth_features['peak_height'], \
        "Spoofed signal should have higher correlation peak"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
