"""
Tests for models module.
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from src.models.classifiers import get_classifier
from src.models.training import train_model, train_with_smote
from src.models.evaluation import evaluate_model
from src.models.persistence import save_model, load_model


def test_get_classifier():
    """Test classifier instantiation."""
    # Random Forest
    clf_rf = get_classifier('random_forest', random_state=42)
    assert clf_rf is not None
    assert hasattr(clf_rf, 'fit')
    assert hasattr(clf_rf, 'predict')
    
    # SVM
    clf_svm = get_classifier('svm', random_state=42)
    assert clf_svm is not None
    
    # MLP
    clf_mlp = get_classifier('mlp', random_state=42)
    assert clf_mlp is not None


def test_get_classifier_with_params():
    """Test classifier with custom parameters."""
    params = {'n_estimators': 50, 'max_depth': 10}
    clf = get_classifier('random_forest', params=params, random_state=42)
    
    assert clf.n_estimators == 50
    assert clf.max_depth == 10


def test_get_classifier_invalid_name():
    """Test that invalid classifier name raises error."""
    with pytest.raises(ValueError):
        get_classifier('invalid_model')


def test_train_model_simple():
    """Test basic model training."""
    # Create simple dataset
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    # Train model
    model, info = train_model(
        X, y,
        model_name='random_forest',
        test_size=0.3,
        random_state=42
    )
    
    # Check model is trained
    assert hasattr(model, 'predict')
    
    # Check info dict
    assert 'X_train' in info
    assert 'X_test' in info
    assert 'y_train' in info
    assert 'y_test' in info
    assert info['n_train_samples'] == 70
    assert info['n_test_samples'] == 30
    
    # Check predictions work
    y_pred = model.predict(info['X_test'])
    assert len(y_pred) == len(info['y_test'])


def test_train_with_smote():
    """Test training with SMOTE."""
    # Create imbalanced dataset
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.concatenate([np.zeros(80), np.ones(20)])
    
    # Train with SMOTE
    model, info = train_with_smote(
        X, y,
        model_name='random_forest',
        test_size=0.3,
        random_state=42
    )
    
    # Check SMOTE was used
    assert info['smote_used'] == True
    assert 'smote_params' in info
    
    # Check model works
    y_pred = model.predict(info['X_test'])
    assert len(y_pred) == len(info['y_test'])


def test_evaluate_model():
    """Test model evaluation."""
    # Create simple dataset
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    # Train model
    model, info = train_model(X, y, random_state=42)
    
    # Evaluate
    metrics = evaluate_model(
        model,
        info['X_test'],
        info['y_test'],
        info['X_train'],
        info['y_train']
    )
    
    # Check metrics are present
    assert 'test_accuracy' in metrics
    assert 'test_precision' in metrics
    assert 'test_recall' in metrics
    assert 'test_f1' in metrics
    assert 'confusion_matrix' in metrics
    assert 'train_accuracy' in metrics
    
    # Check metrics are in valid range
    assert 0 <= metrics['test_accuracy'] <= 1
    assert 0 <= metrics['test_f1'] <= 1


def test_save_and_load_model():
    """Test model persistence."""
    # Create and train simple model
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    model, _ = train_model(X, y, random_state=42, test_size=0.2)
    
    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / 'test_model.pkl'
        metadata = {'test_key': 'test_value', 'accuracy': 0.95}
        
        save_model(model, str(filepath), metadata=metadata)
        
        # Check files exist
        assert filepath.exists()
        assert filepath.with_suffix('.json').exists()
        
        # Load model
        loaded_model, loaded_metadata = load_model(str(filepath), load_metadata=True)
        
        # Check model works
        y_pred_original = model.predict(X)
        y_pred_loaded = loaded_model.predict(X)
        np.testing.assert_array_equal(y_pred_original, y_pred_loaded)
        
        # Check metadata
        assert loaded_metadata['test_key'] == 'test_value'
        assert loaded_metadata['accuracy'] == 0.95


def test_save_model_creates_directory():
    """Test that save_model creates output directory."""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    model, _ = train_model(X, y, random_state=42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / 'subdir' / 'nested' / 'model.pkl'
        
        # Directory doesn't exist yet
        assert not filepath.parent.exists()
        
        # Save should create it
        save_model(model, str(filepath))
        
        assert filepath.exists()
        assert filepath.parent.exists()


def test_load_model_not_found():
    """Test that loading non-existent model raises error."""
    with pytest.raises(FileNotFoundError):
        load_model('nonexistent_model.pkl')
