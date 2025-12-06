"""Model evaluation utilities."""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.pipeline import Pipeline


def evaluate_model(pipeline: Pipeline, X: np.ndarray, y: np.ndarray,
                  labels: Optional[list] = None) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Parameters
    ----------
    pipeline : Pipeline
        Trained sklearn Pipeline
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        True labels (n_samples,)
    labels : list, optional
        Label names for classification report
    
    Returns
    -------
    metrics : dict
        Dictionary containing:
        - accuracy: Overall accuracy
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
        - confusion_matrix: Confusion matrix
        - classification_report: Detailed classification report
        - roc_auc: ROC AUC score (if applicable)
    """
    # Make predictions
    y_pred = pipeline.predict(X)
    
    # Basic metrics
    metrics = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y, y_pred, average='weighted', zero_division=0)),
        'f1': float(f1_score(y, y_pred, average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
    }
    
    # Classification report
    if labels is None:
        labels = ['authentic', 'spoofed']
    
    report = classification_report(y, y_pred, target_names=labels, 
                                  output_dict=True, zero_division=0)
    metrics['classification_report'] = report
    
    # ROC AUC (if binary classification and probabilities available)
    try:
        if hasattr(pipeline, 'predict_proba'):
            y_proba = pipeline.predict_proba(X)
            if y_proba.shape[1] == 2:  # Binary classification
                metrics['roc_auc'] = float(roc_auc_score(y, y_proba[:, 1]))
    except Exception:
        pass  # Skip ROC AUC if not applicable
    
    return metrics
