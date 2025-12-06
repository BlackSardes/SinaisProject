"""
Feature Extraction Pipeline

This module provides a complete pipeline for extracting features from
GPS signals for spoofing detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from ..preprocessing.prn_codes import generate_local_code_oversampled
from .correlation import compute_correlation_fft, extract_peak_metrics, extract_temporal_gradient
from .statistical import extract_all_statistical_features


def extract_features_from_segment(
    signal: np.ndarray,
    fs: float,
    prn: int,
    ca_chip_rate: float = 1.023e6,
    include_statistical: bool = True
) -> Dict[str, float]:
    """
    Extract all features from a signal segment.
    
    Parameters
    ----------
    signal : np.ndarray
        Preprocessed complex signal segment
    fs : float
        Sampling frequency in Hz
    prn : int
        PRN number for local code generation
    ca_chip_rate : float, optional
        C/A chip rate in Hz (default: 1.023e6)
    include_statistical : bool, optional
        Include statistical/spectral/temporal features (default: True)
        
    Returns
    -------
    dict
        Feature dictionary with all extracted features
        
    Notes
    -----
    This function combines:
    1. Correlation-based features (peak metrics)
    2. Power-based features (C/N0, SNR)
    3. Statistical features (mean, std, skewness, kurtosis)
    4. Spectral features (frequency domain)
    5. Temporal features (time domain)
    """
    features = {}
    
    # Generate local PRN code
    local_code = generate_local_code_oversampled(prn, fs, len(signal), ca_chip_rate)
    
    # Compute correlation
    correlation = compute_correlation_fft(signal, local_code)
    
    # Extract correlation peak metrics
    samples_per_chip = int(fs / ca_chip_rate)
    peak_metrics = extract_peak_metrics(correlation, samples_per_chip, fs)
    features.update(peak_metrics)
    
    # Extract statistical features if requested
    if include_statistical:
        stat_features = extract_all_statistical_features(
            signal,
            fs,
            peak_metrics.get('peak_value'),
            peak_metrics.get('secondary_peak_value')
        )
        features.update(stat_features)
    
    return features


def extract_features_from_file(
    file_path: str,
    fs: float,
    prn: int,
    segment_duration: float,
    total_duration: Optional[float] = None,
    overlap: float = 0.5,
    preprocess_config: Optional[Dict] = None,
    label_func: Optional[callable] = None,
    ca_chip_rate: float = 1.023e6,
    include_statistical: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract features from entire file using sliding windows.
    
    Parameters
    ----------
    file_path : str
        Path to signal file
    fs : float
        Sampling frequency in Hz
    prn : int
        PRN number
    segment_duration : float
        Duration of each segment in seconds
    total_duration : float, optional
        Total duration to process in seconds (None = entire file)
    overlap : float, optional
        Overlap fraction between segments (default: 0.5)
    preprocess_config : dict, optional
        Preprocessing configuration
    label_func : callable, optional
        Function to assign labels: label = func(segment_start_time)
    ca_chip_rate : float, optional
        C/A chip rate in Hz
    include_statistical : bool, optional
        Include statistical features
    verbose : bool, optional
        Print progress (default: True)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with features for each segment
        
    Notes
    -----
    This function processes an entire file in segments, extracting
    features from each segment and organizing them into a DataFrame
    for machine learning.
    """
    from ..preprocessing.signal_io import load_signal
    from ..preprocessing.pipeline import preprocess_signal
    
    # Calculate segment parameters
    num_samples_per_segment = int(fs * segment_duration)
    step_samples = int(num_samples_per_segment * (1 - overlap))
    
    if total_duration is not None:
        total_samples = int(fs * total_duration)
    else:
        # Try to determine file size (for binary files)
        import os
        file_size = os.path.getsize(file_path)
        # Assume int16 I/Q pairs (4 bytes per complex sample)
        total_samples = file_size // 4
    
    # Extract features for each segment
    features_list = []
    segment_idx = 0
    
    for start_sample in range(0, total_samples - num_samples_per_segment + 1, step_samples):
        # Load segment
        signal = load_signal(file_path, start_sample, num_samples_per_segment)
        
        if signal is None or len(signal) < num_samples_per_segment:
            continue
        
        # Preprocess
        if preprocess_config is not None:
            signal = preprocess_signal(signal, fs, preprocess_config)
        
        # Extract features
        features = extract_features_from_segment(
            signal, fs, prn, ca_chip_rate, include_statistical
        )
        
        # Add metadata
        segment_start_time = start_sample / fs
        features['segment_start_time'] = segment_start_time
        features['segment_index'] = segment_idx
        features['prn'] = prn
        features['file_path'] = file_path
        
        # Add label if function provided
        if label_func is not None:
            features['label'] = label_func(segment_start_time)
        
        features_list.append(features)
        segment_idx += 1
        
        if verbose and segment_idx % 10 == 0:
            print(f"Processed {segment_idx} segments ({segment_start_time:.1f}s)...")
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    
    if verbose:
        print(f"Feature extraction complete: {len(df)} segments")
    
    return df


def create_feature_pipeline(
    feature_selection: bool = False,
    n_features: Optional[int] = None,
    pca: bool = False,
    n_components: Optional[int] = None,
    handle_missing: bool = True,
    normalize: bool = True
) -> 'FeaturePipeline':
    """
    Create a feature processing pipeline.
    
    Parameters
    ----------
    feature_selection : bool, optional
        Apply feature selection (default: False)
    n_features : int, optional
        Number of features to select (if feature_selection=True)
    pca : bool, optional
        Apply PCA dimensionality reduction (default: False)
    n_components : int, optional
        Number of PCA components (if pca=True)
    handle_missing : bool, optional
        Handle missing values (default: True)
    normalize : bool, optional
        Normalize features (default: True)
        
    Returns
    -------
    FeaturePipeline
        Configured feature pipeline
    """
    return FeaturePipeline(
        feature_selection=feature_selection,
        n_features=n_features,
        pca=pca,
        n_components=n_components,
        handle_missing=handle_missing,
        normalize=normalize
    )


class FeaturePipeline:
    """
    Feature processing pipeline for GPS spoofing detection.
    
    This pipeline handles:
    - Missing value imputation
    - Feature normalization
    - Feature selection
    - Dimensionality reduction (PCA)
    """
    
    def __init__(
        self,
        feature_selection: bool = False,
        n_features: Optional[int] = None,
        pca: bool = False,
        n_components: Optional[int] = None,
        handle_missing: bool = True,
        normalize: bool = True
    ):
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.pca = pca
        self.n_components = n_components
        self.handle_missing = handle_missing
        self.normalize = normalize
        
        # Components (initialized during fit)
        self.imputer = None
        self.scaler = None
        self.feature_selector = None
        self.pca_transformer = None
        self.feature_names = None
        self.selected_features = None
    
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'FeaturePipeline':
        """
        Fit the pipeline to training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray, optional
            Labels (for supervised feature selection)
            
        Returns
        -------
        self
        """
        self.feature_names = X.columns.tolist()
        
        # Handle missing values
        if self.handle_missing:
            self.imputer = SimpleImputer(strategy='mean')
            X_imputed = self.imputer.fit_transform(X)
            X = pd.DataFrame(X_imputed, columns=self.feature_names)
        
        # Normalize features
        if self.normalize:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # Feature selection
        if self.feature_selection and y is not None:
            from sklearn.feature_selection import SelectKBest, f_classif
            
            if self.n_features is None:
                self.n_features = min(20, X.shape[1])
            
            self.feature_selector = SelectKBest(f_classif, k=self.n_features)
            self.feature_selector.fit(X, y)
            self.selected_features = [self.feature_names[i] for i in self.feature_selector.get_support(indices=True)]
        
        # PCA
        if self.pca:
            if self.n_components is None:
                self.n_components = min(10, X.shape[1])
            
            self.pca_transformer = PCA(n_components=self.n_components)
            self.pca_transformer.fit(X)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted pipeline.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Transformed features
        """
        # Handle missing values
        if self.handle_missing and self.imputer is not None:
            X_imputed = self.imputer.transform(X)
            X = pd.DataFrame(X_imputed, columns=self.feature_names)
        
        # Normalize
        if self.normalize and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # Feature selection
        if self.feature_selection and self.feature_selector is not None:
            X = X[self.selected_features]
        
        # PCA
        if self.pca and self.pca_transformer is not None:
            X = self.pca_transformer.transform(X)
        
        return X if isinstance(X, np.ndarray) else X.values
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray, optional
            Labels
            
        Returns
        -------
        np.ndarray
            Transformed features
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from selection.
        
        Returns
        -------
        pd.DataFrame or None
            Feature importance scores (if selection was used)
        """
        if self.feature_selector is None:
            return None
        
        scores = self.feature_selector.scores_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'score': scores
        })
        importance_df = importance_df.sort_values('score', ascending=False)
        
        return importance_df
