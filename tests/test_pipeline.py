"""
Integration tests for the complete pipeline.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from utils.synthetic_data import generate_synthetic_dataset
from preprocessing.signal_processing import generate_ca_code
from features.pipeline import build_feature_vector, preprocess_features
from models.train import train_model, evaluate_model
import pandas as pd


def test_complete_pipeline_synthetic():
    """Test complete pipeline with synthetic data."""
    # Generate small dataset
    signals, labels, metadata = generate_synthetic_dataset(
        num_authentic=20,
        num_spoofed=20,
        fs=5e6,
        duration=0.1,  # Shorter for speed
        random_state=42
    )
    
    assert len(signals) == 40
    assert len(labels) == 40
    assert sum(labels) == 20  # Half should be spoofed
    
    # Extract features
    all_features = []
    for i, signal in enumerate(signals):
        prn = metadata[i]['prn']
        ca_code = generate_ca_code(prn)
        features = build_feature_vector(signal, ca_code, 5e6, label=labels[i])
        all_features.append(features)
    
    df_features = pd.DataFrame(all_features)
    assert df_features.shape[0] == 40
    assert 'label' in df_features.columns
    
    # Prepare for training
    X = df_features.drop(columns=['label', 'prn', 'segment_index'], errors='ignore')
    y = df_features['label'].values
    
    # Preprocess
    X_processed, imputer, scaler, _ = preprocess_features(X, y, fit=True)
    assert X_processed.shape[0] == 40
    
    # Train model (quick test with small dataset)
    model, train_metrics = train_model(
        X_processed, y,
        model_name='random_forest',
        params={'n_estimators': 10, 'max_depth': 5},  # Small for speed
        cv=2,  # Fewer folds for speed
        random_state=42
    )
    
    assert model is not None
    assert 'cv_mean' in train_metrics
    assert 0 <= train_metrics['cv_mean'] <= 1
    
    # Evaluate
    metrics = evaluate_model(model, X_processed, y)
    assert 'accuracy' in metrics
    assert 'confusion_matrix' in metrics
    assert 0 <= metrics['accuracy'] <= 1


def test_pipeline_handles_missing_values():
    """Test that pipeline handles missing/NaN values."""
    # Create feature DataFrame with some NaN values
    data = {
        'feature1': [1.0, 2.0, np.nan, 4.0],
        'feature2': [5.0, np.nan, 7.0, 8.0],
        'label': [0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    X = df[['feature1', 'feature2']]
    y = df['label'].values
    
    # Preprocess should handle NaN
    X_processed, imputer, scaler, _ = preprocess_features(X, y, fit=True)
    
    # Check no NaN in output
    assert not np.any(np.isnan(X_processed))


def test_model_reproducibility():
    """Test that models produce reproducible results."""
    # Generate data
    signals, labels, metadata = generate_synthetic_dataset(
        num_authentic=10,
        num_spoofed=10,
        fs=5e6,
        duration=0.1,
        random_state=42
    )
    
    # Extract features
    all_features = []
    for i, signal in enumerate(signals):
        prn = metadata[i]['prn']
        ca_code = generate_ca_code(prn)
        features = build_feature_vector(signal, ca_code, 5e6, label=labels[i])
        all_features.append(features)
    
    df = pd.DataFrame(all_features)
    X = df.drop(columns=['label', 'prn', 'segment_index'], errors='ignore')
    y = df['label'].values
    
    X_proc, _, _, _ = preprocess_features(X, y, fit=True)
    
    # Train twice with same seed
    model1, _ = train_model(X_proc, y, model_name='random_forest', 
                           params={'n_estimators': 10}, cv=2, random_state=42)
    model2, _ = train_model(X_proc, y, model_name='random_forest',
                           params={'n_estimators': 10}, cv=2, random_state=42)
    
    # Predictions should be identical
    pred1 = model1.predict(X_proc)
    pred2 = model2.predict(X_proc)
    
    assert np.array_equal(pred1, pred2), "Models with same seed should produce identical predictions"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
