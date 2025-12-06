"""
<<<<<<< HEAD
Integration tests for complete pipeline.
"""
import pytest
import numpy as np
import pandas as pd
from src.utils.synthetic_data import (
    generate_synthetic_gps_signal,
    generate_synthetic_dataset,
    create_synthetic_features_dataframe
)
from src.features.feature_pipeline import build_feature_vector
from src.models.training import train_model
from src.models.evaluation import evaluate_model


def test_end_to_end_synthetic_pipeline():
    """Test complete pipeline with synthetic data."""
    # Step 1: Generate dataset
    signals, labels = generate_synthetic_dataset(
        n_authentic=20,
        n_spoofed=20,
        duration_s=0.1,
        fs=5e6,
        seed=42
=======
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
>>>>>>> main
    )
    
    assert len(signals) == 40
    assert len(labels) == 40
<<<<<<< HEAD
    assert np.sum(labels == 0) == 20  # Authentic
    assert np.sum(labels == 1) == 20  # Spoofed
    
    # Step 2: Extract features
    features_df = build_feature_vector(signals, fs=5e6, prn=1)
    features_df['label'] = labels
    
    assert len(features_df) == 40
    assert 'label' in features_df.columns
    
    # Step 3: Prepare data
    X = features_df.drop(['segment_id', 'label'], axis=1, errors='ignore').values
    y = features_df['label'].values
    
    assert X.shape[0] == 40
    assert X.shape[1] > 10  # Should have multiple features
    
    # Step 4: Train model
    model, info = train_model(
        X, y,
        model_name='random_forest',
        test_size=0.3,
        random_state=42
    )
    
    # Step 5: Evaluate
    metrics = evaluate_model(
        model,
        info['X_test'],
        info['y_test']
    )
    
    # Check that model achieves reasonable performance on synthetic data
    # (should be easy to classify)
    assert metrics['test_accuracy'] > 0.5  # Better than random
    assert 'confusion_matrix' in metrics


def test_create_synthetic_features_dataframe():
    """Test convenience function for creating features DataFrame."""
    df = create_synthetic_features_dataframe(
        n_authentic=15,
        n_spoofed=15,
        fs=5e6,
        prn=1,
        seed=42
    )
    
    # Check structure
    assert len(df) == 30
    assert 'label' in df.columns
    assert 'segment_id' in df.columns
    
    # Check labels
    assert np.sum(df['label'] == 0) == 15
    assert np.sum(df['label'] == 1) == 15
    
    # Check features are numeric
    feature_cols = [col for col in df.columns if col not in ['segment_id', 'label']]
    assert len(feature_cols) > 10
    
    for col in feature_cols:
        assert df[col].dtype in [np.float64, np.float32, np.int64, np.int32]


def test_authentic_vs_spoofed_distinguishable():
    """Test that authentic and spoofed signals produce different features."""
    # Generate one of each
    signal_auth = generate_synthetic_gps_signal(
        duration_s=0.1,
        fs=5e6,
        prn=1,
        cn0_db=45,
        add_spoofing=False,
        seed=42
    )
    
    signal_spoof = generate_synthetic_gps_signal(
        duration_s=0.1,
        fs=5e6,
        prn=1,
        cn0_db=45,
        add_spoofing=True,
        spoofing_delay_chips=0.5,
        spoofing_power_ratio=2.0,
        seed=43
    )
    
    # Extract features
    features_auth = build_feature_vector([signal_auth], fs=5e6, prn=1)
    features_spoof = build_feature_vector([signal_spoof], fs=5e6, prn=1)
    
    # Drop metadata columns
    feat_cols = [col for col in features_auth.columns if col not in ['segment_id']]
    
    # Compare key features (should be different)
    # At least one feature should be significantly different
    # Check that features are not identical
    differences = []
    for col in feat_cols:
        if col != 'segment_id':
            val_auth = features_auth[col].iloc[0]
            val_spoof = features_spoof[col].iloc[0]
            if not np.isnan(val_auth) and not np.isnan(val_spoof):
                diff = abs(val_auth - val_spoof)
                differences.append(diff)
    
    # At least some features should be different
    assert len(differences) > 0, "Should have extracted some features"
    # Check that there's variation (not all zeros)
    assert np.sum(np.array(differences) > 0.001) > 0, "Some features should differ between authentic and spoofed"


def test_pipeline_reproducibility():
    """Test that pipeline produces same results with same seed."""
    # Run pipeline twice with same seed
    df1 = create_synthetic_features_dataframe(
        n_authentic=10,
        n_spoofed=10,
        seed=42
    )
    
    df2 = create_synthetic_features_dataframe(
        n_authentic=10,
        n_spoofed=10,
        seed=42
    )
    
    # Should be identical
    pd.testing.assert_frame_equal(df1, df2)
=======
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
>>>>>>> main
