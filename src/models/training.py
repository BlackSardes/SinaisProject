"""
Model training functions.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler

from .classifiers import get_classifier


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = 'random_forest',
    params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.3,
    random_state: int = 42,
    scale: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a classification model.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        model_name: Name of classifier ('random_forest', 'svm', 'mlp')
        params: Classifier hyperparameters (None = use defaults)
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        scale: Whether to scale features before training
    
    Returns:
        Tuple of (trained_model, info_dict)
        where info_dict contains train/test splits and metadata
        
    Example:
        >>> model, info = train_model(X, y, model_name='random_forest')
        >>> y_pred = model.predict(info['X_test'])
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Get classifier
    clf = get_classifier(model_name, params, random_state)
    
    # Build pipeline with optional scaling
    if scale:
        from sklearn.pipeline import Pipeline
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
    else:
        model = clf
    
    # Train
    model.fit(X_train, y_train)
    
    # Prepare info dict
    info = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'model_name': model_name,
        'params': params,
        'random_state': random_state,
        'test_size': test_size,
        'n_train_samples': len(y_train),
        'n_test_samples': len(y_test),
        'n_features': X.shape[1],
    }
    
    return model, info


def train_with_smote(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = 'random_forest',
    params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.3,
    random_state: int = 42,
    smote_params: Optional[Dict[str, Any]] = None,
    scale: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a classification model with SMOTE oversampling.
    
    SMOTE (Synthetic Minority Over-sampling Technique) generates
    synthetic samples for the minority class to balance the dataset.
    Useful when class imbalance is severe.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        model_name: Name of classifier
        params: Classifier hyperparameters
        test_size: Fraction of data for testing
        random_state: Random seed
        smote_params: SMOTE parameters (None = defaults)
        scale: Whether to scale features
    
    Returns:
        Tuple of (trained_pipeline, info_dict)
        
    Example:
        >>> # Train with SMOTE when minority class is < 10%
        >>> model, info = train_with_smote(X, y, model_name='random_forest')
    """
    # Split data (SMOTE applied only to training set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Configure SMOTE
    if smote_params is None:
        smote_params = {
            'random_state': random_state,
            'k_neighbors': 5,
        }
    else:
        smote_params.setdefault('random_state', random_state)
    
    smote = SMOTE(**smote_params)
    
    # Get classifier
    clf = get_classifier(model_name, params, random_state)
    
    # Build imbalanced-learn pipeline
    pipeline_steps = []
    
    if scale:
        pipeline_steps.append(('scaler', StandardScaler()))
    
    pipeline_steps.extend([
        ('smote', smote),
        ('classifier', clf)
    ])
    
    model = ImbPipeline(pipeline_steps)
    
    # Train
    model.fit(X_train, y_train)
    
    # Prepare info dict
    info = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'model_name': model_name,
        'params': params,
        'random_state': random_state,
        'test_size': test_size,
        'smote_used': True,
        'smote_params': smote_params,
        'n_train_samples_original': len(y_train),
        'n_test_samples': len(y_test),
        'n_features': X.shape[1],
        'class_distribution_train_original': {
            int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))
        },
    }
    
    return model, info


def train_with_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = 'random_forest',
    params: Optional[Dict[str, Any]] = None,
    cv: int = 5,
    random_state: int = 42,
    scale: bool = True,
    use_smote: bool = False
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train model on full dataset and report cross-validation scores.
    
    Unlike train_model, this doesn't hold out a test set but instead
    uses cross-validation to estimate performance.
    
    Args:
        X: Feature matrix
        y: Target labels
        model_name: Name of classifier
        params: Classifier parameters
        cv: Number of cross-validation folds
        random_state: Random seed
        scale: Whether to scale features
        use_smote: Whether to use SMOTE in pipeline
    
    Returns:
        Tuple of (trained_model, info_dict)
        
    Example:
        >>> model, info = train_with_cross_validation(X, y, cv=10)
        >>> print(f"CV Accuracy: {info['cv_scores'].mean():.3f}")
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    
    # Get classifier
    clf = get_classifier(model_name, params, random_state)
    
    # Build pipeline
    pipeline_steps = []
    
    if scale:
        pipeline_steps.append(('scaler', StandardScaler()))
    
    if use_smote:
        from imblearn.over_sampling import SMOTE
        pipeline_steps.append(('smote', SMOTE(random_state=random_state)))
        model = ImbPipeline(pipeline_steps + [('classifier', clf)])
    else:
        pipeline_steps.append(('classifier', clf))
        model = Pipeline(pipeline_steps)
    
    # Perform cross-validation before fitting on full data
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    # Train on full dataset
    model.fit(X, y)
    
    info = {
        'model_name': model_name,
        'params': params,
        'cv_folds': cv,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'random_state': random_state,
        'n_samples': len(y),
        'n_features': X.shape[1],
        'use_smote': use_smote,
    }
    
    return model, info
