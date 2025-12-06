"""
Feature extraction pipeline and preprocessing.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from .correlation import compute_cross_correlation, extract_correlation_features
from .temporal import extract_temporal_features


def build_feature_vector(signal: np.ndarray, prn_code: np.ndarray, fs: float,
                         ca_chip_rate: float = 1.023e6, 
                         label: Optional[int] = None,
                         metadata: Optional[Dict] = None) -> Dict[str, float]:
    """
    Build complete feature vector from a signal window.
    
    Args:
        signal: Complex IQ signal window
        prn_code: PRN code for correlation
        fs: Sampling frequency (Hz)
        ca_chip_rate: C/A code chip rate (Hz)
        label: Optional ground truth label
        metadata: Optional metadata (filename, segment info, etc.)
    
    Returns:
        Dictionary containing all features
    """
    features = {}
    
    # Add metadata if provided
    if metadata is not None:
        features.update(metadata)
    
    # Compute correlation
    corr_profile = compute_cross_correlation(signal, prn_code)
    
    # Extract correlation features
    corr_features = extract_correlation_features(corr_profile, fs, ca_chip_rate)
    features.update(corr_features)
    
    # Extract temporal features
    temporal_features = extract_temporal_features(
        signal, fs, 
        correlation_peak=corr_features.get('peak_height', 0.0),
        correlation_secondary=corr_features.get('secondary_peak_value', 0.0)
    )
    features.update(temporal_features)
    
    # Add label if provided
    if label is not None:
        features['label'] = label
    
    return features


def build_feature_dataframe(signals: List[np.ndarray], prn_code: np.ndarray, 
                            fs: float, labels: Optional[List[int]] = None,
                            metadata_list: Optional[List[Dict]] = None) -> pd.DataFrame:
    """
    Build feature DataFrame from multiple signal windows.
    
    Args:
        signals: List of signal windows
        prn_code: PRN code for correlation
        fs: Sampling frequency (Hz)
        labels: Optional list of labels
        metadata_list: Optional list of metadata dicts
    
    Returns:
        DataFrame with features for all signals
    """
    feature_dicts = []
    
    for i, signal in enumerate(signals):
        label = labels[i] if labels is not None else None
        metadata = metadata_list[i] if metadata_list is not None else None
        
        features = build_feature_vector(signal, prn_code, fs, label=label, metadata=metadata)
        feature_dicts.append(features)
    
    return pd.DataFrame(feature_dicts)


def preprocess_features(X: pd.DataFrame, y: Optional[pd.Series] = None,
                       imputer: Optional[SimpleImputer] = None,
                       scaler: Optional[StandardScaler] = None,
                       pca: Optional[PCA] = None,
                       feature_columns: Optional[List[str]] = None,
                       fit: bool = True) -> Tuple[np.ndarray, Optional[SimpleImputer], 
                                                   Optional[StandardScaler], Optional[PCA]]:
    """
    Preprocess features with imputation, scaling, and optional PCA.
    
    Args:
        X: Feature DataFrame or array
        y: Optional target (for stratified operations, not used currently)
        imputer: Fitted imputer (or None to create new)
        scaler: Fitted scaler (or None to create new)
        pca: Fitted PCA (or None to skip PCA)
        feature_columns: List of feature column names to use
        fit: Whether to fit transformers (True for training, False for inference)
    
    Returns:
        Tuple of (transformed features, imputer, scaler, pca)
    """
    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Select feature columns
    if feature_columns is not None:
        X = X[feature_columns]
    else:
        # Auto-select numeric columns, exclude metadata
        exclude_cols = ['label', 'filename', 'segment_start_s', 'prn', 'segment_index']
        feature_columns = [col for col in X.columns if col not in exclude_cols and 
                          pd.api.types.is_numeric_dtype(X[col])]
        X = X[feature_columns]
    
    X_array = X.values
    
    # Imputation
    if imputer is None and fit:
        imputer = SimpleImputer(strategy='median')
        X_array = imputer.fit_transform(X_array)
    elif imputer is not None:
        if fit:
            X_array = imputer.fit_transform(X_array)
        else:
            X_array = imputer.transform(X_array)
    
    # Scaling
    if scaler is None and fit:
        scaler = StandardScaler()
        X_array = scaler.fit_transform(X_array)
    elif scaler is not None:
        if fit:
            X_array = scaler.fit_transform(X_array)
        else:
            X_array = scaler.transform(X_array)
    
    # PCA (optional)
    if pca is not None:
        if fit:
            X_array = pca.fit_transform(X_array)
        else:
            X_array = pca.transform(X_array)
    
    return X_array, imputer, scaler, pca


def select_features(X: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """
    Select specific features from DataFrame.
    
    Args:
        X: Feature DataFrame
        feature_names: List of feature names to select
    
    Returns:
        DataFrame with selected features
    """
    available_features = [f for f in feature_names if f in X.columns]
    if len(available_features) < len(feature_names):
        missing = set(feature_names) - set(available_features)
        print(f"Warning: Missing features: {missing}")
    
    return X[available_features]
