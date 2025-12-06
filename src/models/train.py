"""Model training utilities."""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def train_model(X: np.ndarray, y: np.ndarray, 
               model_name: str = 'random_forest',
               use_smote: bool = False,
               cv: int = 5,
               random_state: int = 42,
               **model_params) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train a classification model for spoofing detection.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Labels (n_samples,)
    model_name : str, default='random_forest'
        Model type: 'random_forest', 'svm', or 'mlp'
    use_smote : bool, default=False
        Whether to use SMOTE for handling class imbalance
    cv : int, default=5
        Number of cross-validation folds
    random_state : int, default=42
        Random state for reproducibility
    **model_params : dict
        Additional parameters for the model
    
    Returns
    -------
    pipeline : Pipeline
        Trained sklearn Pipeline
    metrics : dict
        Training metrics including cross-validation scores
    """
    # Select base model
    if model_name == 'random_forest':
        base_model = RandomForestClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', None),
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
    elif model_name == 'svm':
        base_model = SVC(
            kernel=model_params.get('kernel', 'rbf'),
            C=model_params.get('C', 1.0),
            gamma=model_params.get('gamma', 'scale'),
            class_weight='balanced',
            random_state=random_state,
            probability=True
        )
    elif model_name == 'mlp':
        base_model = MLPClassifier(
            hidden_layer_sizes=model_params.get('hidden_layer_sizes', (100, 50)),
            activation=model_params.get('activation', 'relu'),
            max_iter=model_params.get('max_iter', 500),
            random_state=random_state,
            early_stopping=True
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Build pipeline
    if use_smote:
        pipeline = ImbPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=random_state)),
            ('classifier', base_model)
        ])
    else:
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', base_model)
        ])
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    
    # Train on full dataset
    pipeline.fit(X, y)
    
    # Collect metrics
    metrics = {
        'model_name': model_name,
        'cv_mean': float(np.mean(cv_scores)),
        'cv_std': float(np.std(cv_scores)),
        'cv_scores': cv_scores.tolist(),
        'use_smote': use_smote,
        'random_state': random_state
    }
    
    return pipeline, metrics
