"""
Model Training Module for GPS Spoofing Detection

This module provides functions for training machine learning models
with proper cross-validation, class balancing, and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = 'random_forest',
    params: Optional[Dict[str, Any]] = None,
    cv: int = 5,
    balance_method: str = 'class_weight',
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a classification model with cross-validation.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Labels (n_samples,)
    model_name : str, optional
        Model type: 'random_forest', 'svm', 'mlp' (default: 'random_forest')
    params : dict, optional
        Model hyperparameters (None = use defaults)
    cv : int, optional
        Number of cross-validation folds (default: 5)
    balance_method : str, optional
        Class balancing method: 'class_weight', 'smote', 'none' (default: 'class_weight')
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    verbose : bool, optional
        Print training progress (default: True)
        
    Returns
    -------
    model : trained model
        Fitted model on full training data
    cv_results : dict
        Cross-validation results with metrics
        
    Notes
    -----
    Supports three model types with optimized hyperparameters for
    GPS spoofing detection:
    
    - RandomForest: Ensemble method, handles non-linear relationships well
    - SVM: Kernel-based method, good for high-dimensional data
    - MLP: Neural network, can learn complex patterns
    
    Class balancing is crucial as spoofing events may be rare in data.
    """
    # Create model
    model = create_model(model_name, params, balance_method, random_state)
    
    # Handle SMOTE balancing
    if balance_method == 'smote':
        # Create pipeline with SMOTE
        smote = SMOTE(random_state=random_state)
        model = ImbPipeline([
            ('smote', smote),
            ('classifier', model)
        ])
    
    # Cross-validation
    if verbose:
        print(f"Training {model_name} with {cv}-fold cross-validation...")
        print(f"Balance method: {balance_method}")
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: {np.bincount(y)}")
    
    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='binary', zero_division=0),
        'recall': make_scorer(recall_score, average='binary', zero_division=0),
        'f1': make_scorer(f1_score, average='binary', zero_division=0)
    }
    
    # Stratified K-Fold for imbalanced data
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y,
        cv=cv_splitter,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Train final model on full data
    model.fit(X, y)
    
    if verbose:
        print(f"\nCross-Validation Results:")
        print(f"  Accuracy:  {cv_results['test_accuracy'].mean():.4f} (+/- {cv_results['test_accuracy'].std():.4f})")
        print(f"  Precision: {cv_results['test_precision'].mean():.4f} (+/- {cv_results['test_precision'].std():.4f})")
        print(f"  Recall:    {cv_results['test_recall'].mean():.4f} (+/- {cv_results['test_recall'].std():.4f})")
        print(f"  F1-Score:  {cv_results['test_f1'].mean():.4f} (+/- {cv_results['test_f1'].std():.4f})")
    
    # Convert results to simple dict
    results_dict = {
        'model_name': model_name,
        'cv_folds': cv,
        'balance_method': balance_method,
        'accuracy_mean': float(cv_results['test_accuracy'].mean()),
        'accuracy_std': float(cv_results['test_accuracy'].std()),
        'precision_mean': float(cv_results['test_precision'].mean()),
        'precision_std': float(cv_results['test_precision'].std()),
        'recall_mean': float(cv_results['test_recall'].mean()),
        'recall_std': float(cv_results['test_recall'].std()),
        'f1_mean': float(cv_results['test_f1'].mean()),
        'f1_std': float(cv_results['test_f1'].std())
    }
    
    return model, results_dict


def create_model(
    model_name: str,
    params: Optional[Dict[str, Any]] = None,
    balance_method: str = 'class_weight',
    random_state: int = 42
) -> Any:
    """
    Create a model with specified hyperparameters.
    
    Parameters
    ----------
    model_name : str
        Model type: 'random_forest', 'svm', 'mlp'
    params : dict, optional
        Custom hyperparameters (None = use defaults)
    balance_method : str, optional
        Class balancing method
    random_state : int, optional
        Random seed
        
    Returns
    -------
    model
        Initialized model
    """
    if params is None:
        params = get_default_params(model_name, balance_method)
    
    if model_name == 'random_forest':
        model = RandomForestClassifier(random_state=random_state, **params)
    
    elif model_name == 'svm':
        model = SVC(random_state=random_state, probability=True, **params)
    
    elif model_name == 'mlp':
        model = MLPClassifier(random_state=random_state, **params)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def get_default_params(model_name: str, balance_method: str = 'class_weight') -> Dict[str, Any]:
    """
    Get default hyperparameters for each model type.
    
    Parameters
    ----------
    model_name : str
        Model type
    balance_method : str, optional
        Class balancing method
        
    Returns
    -------
    dict
        Default hyperparameters
    """
    if model_name == 'random_forest':
        params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'n_jobs': -1
        }
        if balance_method == 'class_weight':
            params['class_weight'] = 'balanced'
    
    elif model_name == 'svm':
        params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'cache_size': 1000
        }
        if balance_method == 'class_weight':
            params['class_weight'] = 'balanced'
    
    elif model_name == 'mlp':
        params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 'auto',
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'max_iter': 500,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10
        }
    
    else:
        params = {}
    
    return params


def train_multiple_models(
    X: np.ndarray,
    y: np.ndarray,
    model_configs: Optional[List[Dict[str, Any]]] = None,
    cv: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    Train multiple models and compare results.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    model_configs : list of dict, optional
        List of model configurations (None = use defaults)
    cv : int, optional
        Cross-validation folds
    random_state : int, optional
        Random seed
    verbose : bool, optional
        Print progress
        
    Returns
    -------
    dict
        Dictionary mapping model names to (model, cv_results) tuples
        
    Examples
    --------
    >>> configs = [
    ...     {'model_name': 'random_forest', 'balance_method': 'class_weight'},
    ...     {'model_name': 'svm', 'balance_method': 'smote'},
    ...     {'model_name': 'mlp'}
    ... ]
    >>> results = train_multiple_models(X, y, configs)
    """
    if model_configs is None:
        # Default configurations
        model_configs = [
            {'model_name': 'random_forest', 'balance_method': 'class_weight'},
            {'model_name': 'random_forest', 'balance_method': 'smote'},
            {'model_name': 'svm', 'balance_method': 'class_weight'},
            {'model_name': 'mlp', 'balance_method': 'none'}
        ]
    
    results = {}
    
    for config in model_configs:
        model_name = config.get('model_name', 'random_forest')
        balance_method = config.get('balance_method', 'class_weight')
        params = config.get('params', None)
        
        key = f"{model_name}_{balance_method}"
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Training: {key}")
            print(f"{'='*70}")
        
        model, cv_results = train_model(
            X, y,
            model_name=model_name,
            params=params,
            cv=cv,
            balance_method=balance_method,
            random_state=random_state,
            verbose=verbose
        )
        
        results[key] = (model, cv_results)
    
    return results


def save_model(model: Any, filepath: str, metadata: Optional[Dict] = None) -> None:
    """
    Save trained model to disk.
    
    Parameters
    ----------
    model : trained model
        Model to save
    filepath : str
        Path to save model (should end in .pkl or .joblib)
    metadata : dict, optional
        Additional metadata to save with model
        
    Notes
    -----
    Uses joblib for efficient serialization of scikit-learn models.
    """
    save_dict = {
        'model': model,
        'metadata': metadata or {}
    }
    
    joblib.dump(save_dict, filepath, compress=3)
    print(f"Model saved to: {filepath}")


def load_model(filepath: str) -> Tuple[Any, Dict]:
    """
    Load trained model from disk.
    
    Parameters
    ----------
    filepath : str
        Path to saved model
        
    Returns
    -------
    model : trained model
        Loaded model
    metadata : dict
        Metadata saved with model
    """
    save_dict = joblib.load(filepath)
    model = save_dict['model']
    metadata = save_dict.get('metadata', {})
    
    print(f"Model loaded from: {filepath}")
    return model, metadata


def create_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create stratified train-test split.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    test_size : float, optional
        Fraction for test set (default: 0.3)
    random_state : int, optional
        Random seed (default: 42)
    stratify : bool, optional
        Use stratified split (default: True)
        
    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )
