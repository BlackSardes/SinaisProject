"""
Feature pipeline for building complete feature vectors from GPS signals.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from .correlation import compute_cross_correlation, generate_local_code
from .correlation_features import extract_correlation_features
from .temporal_features import extract_temporal_features, compute_cn0_variation_features


def build_feature_vector(
    signals: Union[np.ndarray, List[np.ndarray]],
    fs: float,
    prn: int = 1,
    include_correlation: bool = True,
    include_temporal: bool = True,
    include_cn0_variation: bool = True,
    ca_chip_rate: float = 1.023e6
) -> pd.DataFrame:
    """
    Build feature vectors from windowed signals.
    
    This is the main feature extraction pipeline that processes
    one or more signal segments and returns a DataFrame ready
    for machine learning.
    
    Args:
        signals: Single signal array or list of signal arrays
        fs: Sampling frequency in Hz
        prn: PRN number for correlation (default: 1)
        include_correlation: Extract correlation features
        include_temporal: Extract temporal features
        include_cn0_variation: Extract C/N0 variation features
        ca_chip_rate: C/A code chip rate in Hz
    
    Returns:
        pandas DataFrame with features (one row per signal segment)
        
    Example:
        >>> from src.preprocessing import window_segment
        >>> segments = window_segment(signal, fs=5e6, window_s=0.5, hop_s=0.25)
        >>> features_df = build_feature_vector(segments, fs=5e6, prn=1)
        >>> print(features_df.shape)
        (240, 25)  # 240 segments, 25 features
    """
    # Handle single signal
    if isinstance(signals, np.ndarray) and signals.ndim == 1:
        signals = [signals]
    
    all_features = []
    samples_per_chip = int(fs / ca_chip_rate)
    
    for idx, signal in enumerate(signals):
        feature_dict = {'segment_id': idx}
        
        # Correlation features
        if include_correlation and len(signal) > 0:
            # Generate local code
            duration_s = len(signal) / fs
            local_code = generate_local_code(prn, fs, duration_s, ca_chip_rate)
            
            # Compute correlation
            corr = compute_cross_correlation(signal, local_code, method='fft')
            corr_mag = np.abs(corr)
            
            # Extract correlation features
            corr_features = extract_correlation_features(corr_mag, samples_per_chip)
            for key, val in corr_features.items():
                feature_dict[f'corr_{key}'] = val
        
        # Temporal features
        if include_temporal and len(signal) > 0:
            temp_features = extract_temporal_features(signal, fs)
            for key, val in temp_features.items():
                feature_dict[f'temp_{key}'] = val
        
        # C/N0 variation features (only if segment is long enough)
        if include_cn0_variation and len(signal) > fs * 0.2:  # At least 0.2 seconds
            try:
                cn0_features = compute_cn0_variation_features(
                    signal, fs, window_s=0.05, hop_s=0.025
                )
                for key, val in cn0_features.items():
                    feature_dict[f'cn0var_{key}'] = val
            except Exception as e:
                print(f"Warning: Failed to compute C/N0 variation for segment {idx}: {e}")
        
        all_features.append(feature_dict)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    return df


def preprocess_features(
    X: Union[pd.DataFrame, np.ndarray],
    impute_strategy: str = 'median',
    scale: bool = True,
    pca_components: Optional[int] = None,
    fit: bool = True,
    imputer: Optional[SimpleImputer] = None,
    scaler: Optional[StandardScaler] = None,
    pca: Optional[PCA] = None
) -> tuple:
    """
    Preprocess features for machine learning.
    
    Handles missing values, scaling, and optional dimensionality reduction.
    
    Args:
        X: Feature matrix (DataFrame or array)
        impute_strategy: Strategy for imputing missing values
            ('mean', 'median', 'most_frequent', 'constant')
        scale: Whether to standardize features (zero mean, unit variance)
        pca_components: Number of PCA components (None = no PCA)
        fit: If True, fit transformers; if False, use provided transformers
        imputer: Pre-fitted imputer (used when fit=False)
        scaler: Pre-fitted scaler (used when fit=False)
        pca: Pre-fitted PCA (used when fit=False)
    
    Returns:
        Tuple of (X_processed, imputer, scaler, pca)
        Where transformers are None if not used
        
    Example:
        >>> # Training
        >>> X_train_proc, imp, scl, pca_model = preprocess_features(
        ...     X_train, scale=True, pca_components=10, fit=True
        ... )
        >>> 
        >>> # Testing
        >>> X_test_proc, _, _, _ = preprocess_features(
        ...     X_test, fit=False, imputer=imp, scaler=scl, pca=pca_model
        ... )
    """
    # Convert to numpy if DataFrame
    feature_names = None
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values
    
    X_processed = X.copy()
    
    # Impute missing values
    if fit:
        imputer = SimpleImputer(strategy=impute_strategy)
        X_processed = imputer.fit_transform(X_processed)
    elif imputer is not None:
        X_processed = imputer.transform(X_processed)
    
    # Scale features
    if scale:
        if fit:
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed)
        elif scaler is not None:
            X_processed = scaler.transform(X_processed)
    
    # Apply PCA
    if pca_components is not None:
        if fit:
            pca = PCA(n_components=pca_components, random_state=42)
            X_processed = pca.fit_transform(X_processed)
        elif pca is not None:
            X_processed = pca.transform(X_processed)
    
    return X_processed, imputer, scaler, pca


def extract_features_from_file(
    file_path: str,
    fs: float,
    window_s: float = 0.5,
    hop_s: float = 0.25,
    prn: int = 1,
    max_duration_s: Optional[float] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Extract features directly from a signal file.
    
    Convenience function that loads, windows, and extracts features
    in one call.
    
    Args:
        file_path: Path to signal file
        fs: Sampling frequency in Hz
        window_s: Window duration in seconds
        hop_s: Hop size in seconds
        prn: PRN number for correlation
        max_duration_s: Maximum signal duration to process (None = all)
        **kwargs: Additional arguments for build_feature_vector
    
    Returns:
        DataFrame with features
        
    Example:
        >>> features = extract_features_from_file(
        ...     'data/signal.bin', fs=5e6, window_s=0.5, hop_s=0.25, prn=1
        ... )
    """
    from ..preprocessing.signal_io import load_signal
    from ..preprocessing.windowing import window_segment
    from ..preprocessing.normalization import normalize_signal
    
    # Load signal
    if max_duration_s is not None:
        count_samples = int(max_duration_s * fs)
        signal = load_signal(file_path, count_samples=count_samples)
    else:
        signal = load_signal(file_path)
    
    if signal is None:
        raise ValueError(f"Failed to load signal from {file_path}")
    
    # Normalize
    signal = normalize_signal(signal)
    
    # Window
    segments = window_segment(signal, fs, window_s, hop_s)
    
    # Extract features
    features_df = build_feature_vector(segments, fs, prn, **kwargs)
    
    # Add metadata
    features_df['file_path'] = file_path
    features_df['window_s'] = window_s
    features_df['hop_s'] = hop_s
    features_df['prn'] = prn
    
    return features_df
