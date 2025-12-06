"""
<<<<<<< HEAD
Tests for feature extraction module.
"""
import pytest
import numpy as np
from src.features.correlation import compute_cross_correlation, generate_local_code
from src.features.correlation_features import (
    compute_peak_height,
    compute_fwhm,
    compute_peak_ratio,
    extract_correlation_features
)
from src.features.feature_pipeline import build_feature_vector


def test_compute_cross_correlation(sample_signal):
    """Test cross-correlation computation."""
    fs = 5e6
    prn = 1
    duration_s = len(sample_signal) / fs
    
    # Generate local code
    local_code = generate_local_code(prn, fs, duration_s)
    
    # Compute correlation
    corr = compute_cross_correlation(sample_signal, local_code, method='fft')
    
    # Check output length
    assert len(corr) == len(sample_signal), "Correlation should have same length as signal"
    
    # Check it's complex
    assert np.iscomplexobj(corr), "Correlation should be complex"
    
    # Check there's a peak (correlation should find something)
    corr_mag = np.abs(corr)
    peak = np.max(corr_mag)
    mean = np.mean(corr_mag)
    assert peak > 2 * mean, "Should have distinguishable peak"


def test_generate_local_code():
    """Test local code generation."""
    prn = 1
    fs = 5e6
    duration_s = 0.001  # 1 ms
    
    code = generate_local_code(prn, fs, duration_s)
    
    # Check length
    expected_length = int(fs * duration_s)
    assert len(code) == expected_length, f"Expected {expected_length} samples"
    
    # Check values are +1 or -1
    assert np.all(np.isin(code, [-1, 1])), "Code should only contain +1 and -1"


def test_compute_peak_height():
    """Test peak height computation."""
    # Create synthetic correlation profile with clear peak
    corr_mag = np.random.randn(1000) * 0.1  # Noise
    corr_mag[500] = 10.0  # Peak
    
    peak_height = compute_peak_height(corr_mag)
    
    assert peak_height == 10.0, f"Peak height should be 10.0, got {peak_height}"


def test_compute_fwhm():
    """Test FWHM computation."""
    # Create Gaussian-like peak
    x = np.linspace(-50, 50, 1001)
    corr_mag = np.exp(-x**2 / (2 * 5**2))  # Gaussian with sigma=5 samples
    
    samples_per_chip = 5
    fwhm = compute_fwhm(corr_mag, samples_per_chip, fraction=0.5)
    
    # For Gaussian, FWHM ≈ 2.355 * sigma
    # With sigma=5 samples and samples_per_chip=5, FWHM ≈ 2.355*5/5 = 2.355 chips
    # But actual computation may vary, so be more lenient
    assert 20.0 < fwhm < 25.0, f"FWHM should be ~23 chips for this test case, got {fwhm}"


def test_compute_fwhm_narrow_peak():
    """Test FWHM with narrow peak."""
    # Very narrow peak
    corr_mag = np.zeros(1000)
    corr_mag[500:502] = 1.0  # 2-sample peak
    
    samples_per_chip = 5
    fwhm = compute_fwhm(corr_mag, samples_per_chip, fraction=0.5)
    
    # Should be close to 2 samples = 0.4 chips
    assert fwhm < 1.0, f"Narrow peak should have small FWHM, got {fwhm}"


def test_compute_peak_ratio():
    """Test peak ratio computation."""
    # Create profile with clear primary and secondary peaks
    corr_mag = np.random.randn(1000) * 0.1
    corr_mag[500] = 10.0   # Primary peak
    corr_mag[300] = 2.0    # Secondary peak
    
    samples_per_chip = 5
    ratio = compute_peak_ratio(corr_mag, samples_per_chip)
    
    # Ratio should be 10/2 = 5
    assert 4.5 < ratio < 5.5, f"Peak ratio should be ~5.0, got {ratio}"


def test_extract_correlation_features():
    """Test extraction of all correlation features."""
    # Create synthetic correlation profile
    x = np.linspace(-50, 50, 1001)
    corr_mag = np.exp(-x**2 / 50) + np.random.randn(1001) * 0.01
    
    samples_per_chip = 5
    features = extract_correlation_features(corr_mag, samples_per_chip)
    
    # Check all expected features are present
    expected_features = [
        'peak_height', 'fwhm', 'peak_ratio', 'peak_offset',
        'asymmetry', 'energy_window', 'skewness', 'kurtosis'
    ]
    
    for feat in expected_features:
        assert feat in features, f"Missing feature: {feat}"
        assert not np.isnan(features[feat]), f"Feature {feat} is NaN"


def test_build_feature_vector(sample_dataset):
    """Test building feature vector from signals."""
    signals, labels = sample_dataset
    fs = 5e6
    prn = 1
    
    # Build features
    features_df = build_feature_vector(
        signals[:5],  # Use first 5 signals
        fs=fs,
        prn=prn,
        include_correlation=True,
        include_temporal=True,
        include_cn0_variation=False  # Skip for short signals
    )
    
    # Check DataFrame structure
    assert len(features_df) == 5, "Should have one row per signal"
    assert 'segment_id' in features_df.columns
    
    # Check correlation features are present
    corr_features = [col for col in features_df.columns if col.startswith('corr_')]
    assert len(corr_features) > 0, "Should have correlation features"
    
    # Check temporal features are present
    temp_features = [col for col in features_df.columns if col.startswith('temp_')]
    assert len(temp_features) > 0, "Should have temporal features"
    
    # Check no NaN values in features
    assert not features_df.isnull().any().any(), "Features should not contain NaN"


def test_build_feature_vector_single_signal(sample_signal):
    """Test feature extraction from single signal."""
    fs = 5e6
    prn = 1
    
    features_df = build_feature_vector(
        sample_signal,
        fs=fs,
        prn=prn
    )
    
    # Should return DataFrame with one row
    assert len(features_df) == 1, "Should have one row for single signal"
    assert features_df.iloc[0]['segment_id'] == 0
=======
Unit tests for feature extraction module.
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing.signal_io import generate_synthetic_signal
from src.preprocessing.prn_codes import generate_local_code_oversampled
from src.features.correlation import (
    compute_correlation_fft,
    extract_peak_metrics,
    compute_autocorrelation,
    compute_crosscorrelation
)
from src.features.statistical import (
    extract_power_features,
    extract_statistical_features,
    extract_spectral_features,
    extract_temporal_features
)
from src.features.pipeline import extract_features_from_segment


class TestCorrelation:
    """Tests for correlation-based feature extraction."""
    
    def test_compute_correlation_fft(self):
        """Test FFT-based correlation."""
        signal = generate_synthetic_signal(10000, 5e6, snr_db=10.0, prn=1)
        local_code = generate_local_code_oversampled(1, 5e6, 10000)
        
        correlation = compute_correlation_fft(signal, local_code)
        
        assert len(correlation) == len(signal)
        assert np.all(correlation >= 0)  # Magnitude should be non-negative
        assert np.max(correlation) > 0
    
    def test_correlation_length_mismatch(self):
        """Test error handling for length mismatch."""
        signal = np.random.randn(1000)
        local_code = np.random.randn(500)
        
        with pytest.raises(ValueError):
            compute_correlation_fft(signal, local_code)
    
    def test_extract_peak_metrics(self):
        """Test peak metrics extraction."""
        # Create synthetic correlation with clear peak
        correlation = np.zeros(1000)
        peak_idx = 500
        correlation[peak_idx] = 100  # Main peak
        correlation[peak_idx-1:peak_idx+2] = [50, 100, 50]  # Triangular shape
        correlation[300] = 10  # Secondary peak
        
        samples_per_chip = 5
        metrics = extract_peak_metrics(correlation, samples_per_chip)
        
        assert 'peak_value' in metrics
        assert 'peak_index' in metrics
        assert 'peak_to_secondary' in metrics
        assert 'fwhm' in metrics
        assert 'asymmetry' in metrics
        
        assert metrics['peak_value'] == 100
        assert metrics['peak_index'] == peak_idx
        assert metrics['peak_to_secondary'] > 1  # Peak should be higher than secondary
    
    def test_autocorrelation(self):
        """Test autocorrelation computation."""
        signal = np.random.randn(1000)
        autocorr = compute_autocorrelation(signal, max_lag=100)
        
        assert len(autocorr) == 101  # 0 to max_lag inclusive
        assert autocorr[0] > 0  # Zero-lag should be positive (signal power)
    
    def test_crosscorrelation(self):
        """Test cross-correlation computation."""
        signal1 = np.random.randn(1000)
        signal2 = np.random.randn(1000)
        
        crosscorr = compute_crosscorrelation(signal1, signal2)
        
        # Full correlation has length 2*N-1
        assert len(crosscorr) == 2 * len(signal1) - 1


class TestStatisticalFeatures:
    """Tests for statistical feature extraction."""
    
    def test_extract_power_features(self):
        """Test power features extraction."""
        signal = generate_synthetic_signal(10000, 5e6, snr_db=10.0, prn=1)
        
        features = extract_power_features(signal, 5e6)
        
        assert 'total_power' in features
        assert 'cn0_estimate' in features
        assert 'snr_estimate' in features
        assert 'mean_real' in features
        assert 'std_amplitude' in features
        
        # Check reasonable values
        assert features['total_power'] > 0
        assert features['cn0_estimate'] > 0
    
    def test_extract_statistical_features(self):
        """Test general statistical features."""
        signal = generate_synthetic_signal(10000, 5e6, snr_db=10.0, prn=1)
        
        features = extract_statistical_features(signal)
        
        assert 'mean_magnitude' in features
        assert 'std_magnitude' in features
        assert 'skewness_magnitude' in features
        assert 'kurtosis_magnitude' in features
        assert 'entropy_magnitude' in features
        
        # Check reasonable values
        assert features['mean_magnitude'] > 0
        assert features['std_magnitude'] > 0
    
    def test_extract_spectral_features(self):
        """Test spectral features extraction."""
        signal = generate_synthetic_signal(10000, 5e6, snr_db=10.0, prn=1)
        
        features = extract_spectral_features(signal, 5e6)
        
        assert 'spectral_centroid' in features
        assert 'spectral_spread' in features
        assert 'spectral_flatness' in features
        assert 'peak_frequency' in features
        
        # Spectral flatness should be in [0, 1]
        assert 0 <= features['spectral_flatness'] <= 1
    
    def test_extract_temporal_features(self):
        """Test temporal features extraction."""
        signal = generate_synthetic_signal(10000, 5e6, snr_db=10.0, prn=1)
        
        features = extract_temporal_features(signal)
        
        assert 'zero_crossing_rate' in features
        assert 'autocorr_lag1' in features
        assert 'energy' in features
        assert 'peak_to_average' in features
        
        # Zero crossing rate should be in [0, 1]
        assert 0 <= features['zero_crossing_rate'] <= 1
        assert features['energy'] > 0


class TestFeaturePipeline:
    """Tests for complete feature extraction pipeline."""
    
    def test_extract_features_from_segment(self):
        """Test feature extraction from single segment."""
        signal = generate_synthetic_signal(50000, 5e6, snr_db=10.0, prn=1)
        
        # Preprocess first
        from src.preprocessing.pipeline import preprocess_signal
        signal_processed = preprocess_signal(signal, 5e6)
        
        # Extract features
        features = extract_features_from_segment(
            signal_processed,
            fs=5e6,
            prn=1,
            include_statistical=True
        )
        
        # Check that we got a comprehensive set of features
        assert len(features) > 20  # Should have many features
        
        # Check for key features
        assert 'peak_value' in features
        assert 'peak_to_secondary' in features
        assert 'cn0_estimate' in features
        assert 'asymmetry' in features
    
    def test_extract_features_correlation_only(self):
        """Test feature extraction without statistical features."""
        signal = generate_synthetic_signal(50000, 5e6, snr_db=10.0, prn=1)
        
        from src.preprocessing.pipeline import preprocess_signal
        signal_processed = preprocess_signal(signal, 5e6)
        
        features = extract_features_from_segment(
            signal_processed,
            fs=5e6,
            prn=1,
            include_statistical=False
        )
        
        # Should have fewer features (only correlation-based)
        assert len(features) > 5
        assert len(features) < 20
        
        # Should have correlation features but not statistical
        assert 'peak_value' in features
        assert 'mean_magnitude' not in features  # Statistical feature
    
    def test_features_are_numeric(self):
        """Test that all features are numeric."""
        signal = generate_synthetic_signal(50000, 5e6, snr_db=10.0, prn=1)
        
        from src.preprocessing.pipeline import preprocess_signal
        signal_processed = preprocess_signal(signal, 5e6)
        
        features = extract_features_from_segment(
            signal_processed,
            fs=5e6,
            prn=1
        )
        
        # All values should be numeric
        for key, value in features.items():
            assert isinstance(value, (int, float, np.number))
            assert not np.isnan(value)
            assert not np.isinf(value)


class TestFeaturePipelineTransform:
    """Tests for feature transformation pipeline."""
    
    def test_feature_pipeline_fit_transform(self):
        """Test feature pipeline fit and transform."""
        from src.features.pipeline import create_feature_pipeline
        
        # Create synthetic feature matrix
        n_samples = 100
        n_features = 20
        X = pd.DataFrame(np.random.randn(n_samples, n_features))
        y = np.random.randint(0, 2, n_samples)
        
        # Create and fit pipeline
        pipeline = create_feature_pipeline(
            normalize=True,
            handle_missing=True
        )
        
        X_transformed = pipeline.fit_transform(X, y)
        
        assert X_transformed.shape[0] == n_samples
        
        # Check normalization (mean should be ~0, std should be ~1)
        means = np.mean(X_transformed, axis=0)
        stds = np.std(X_transformed, axis=0)
        
        assert np.allclose(means, 0, atol=0.1)
        assert np.allclose(stds, 1, atol=0.1)
    
    def test_feature_selection(self):
        """Test feature selection in pipeline."""
        from src.features.pipeline import create_feature_pipeline
        
        # Create synthetic feature matrix
        n_samples = 100
        n_features = 50
        X = pd.DataFrame(np.random.randn(n_samples, n_features))
        y = np.random.randint(0, 2, n_samples)
        
        # Create pipeline with feature selection
        pipeline = create_feature_pipeline(
            feature_selection=True,
            n_features=10,
            normalize=True
        )
        
        X_transformed = pipeline.fit_transform(X, y)
        
        # Should have reduced to 10 features
        assert X_transformed.shape[1] == 10
    
    def test_pca_dimensionality_reduction(self):
        """Test PCA dimensionality reduction."""
        from src.features.pipeline import create_feature_pipeline
        
        # Create synthetic feature matrix
        n_samples = 100
        n_features = 50
        X = pd.DataFrame(np.random.randn(n_samples, n_features))
        y = np.random.randint(0, 2, n_samples)
        
        # Create pipeline with PCA
        pipeline = create_feature_pipeline(
            pca=True,
            n_components=5,
            normalize=True
        )
        
        X_transformed = pipeline.fit_transform(X, y)
        
        # Should have 5 components
        assert X_transformed.shape[1] == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
>>>>>>> main
