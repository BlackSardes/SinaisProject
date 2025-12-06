"""
Model evaluation functions.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import cross_val_score


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    class_names: Optional[list] = None
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Computes various classification metrics including accuracy, precision,
    recall, F1-score, confusion matrix, and ROC-AUC.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        X_train: Training features (optional, for train score)
        y_train: Training labels (optional)
        class_names: List of class names for report
    
    Returns:
        Dictionary with evaluation metrics
        
    Example:
        >>> metrics = evaluate_model(model, X_test, y_test)
        >>> print(f"Test Accuracy: {metrics['test_accuracy']:.3f}")
        >>> print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    """
    # Predictions
    y_pred = model.predict(X_test)
    
    # Probabilities (if available)
    try:
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = None
    except:
        y_prob = None
    
    # Basic metrics
    metrics = {
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'test_recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'test_f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }
    
    # Binary classification metrics
    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        metrics['test_precision_binary'] = precision_score(y_test, y_pred, zero_division=0)
        metrics['test_recall_binary'] = recall_score(y_test, y_pred, zero_division=0)
        metrics['test_f1_binary'] = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Classification report
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    metrics['classification_report'] = report
    metrics['classification_report_str'] = classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0
    )
    
    # ROC-AUC
    if y_prob is not None:
        try:
            if n_classes == 2:
                # Binary classification
                if y_prob.ndim == 2:
                    y_prob_pos = y_prob[:, 1]
                else:
                    y_prob_pos = y_prob
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob_pos)
                
                # ROC curve
                fpr, tpr, thresholds = roc_curve(y_test, y_prob_pos)
                metrics['roc_curve'] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': thresholds
                }
            else:
                # Multi-class
                metrics['roc_auc_ovr'] = roc_auc_score(
                    y_test, y_prob, average='weighted', multi_class='ovr'
                )
                metrics['roc_auc_ovo'] = roc_auc_score(
                    y_test, y_prob, average='weighted', multi_class='ovo'
                )
        except Exception as e:
            print(f"Warning: Could not compute ROC-AUC: {e}")
    
    # Training set performance (if provided)
    if X_train is not None and y_train is not None:
        y_train_pred = model.predict(X_train)
        metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        metrics['train_precision'] = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
        metrics['train_recall'] = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
        metrics['train_f1'] = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
    
    return metrics


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform cross-validation on model.
    
    Args:
        model: Model to evaluate (can be untrained)
        X: Feature matrix
        y: Labels
        cv: Number of cross-validation folds
        scoring: Scoring metric (None = accuracy)
    
    Returns:
        Dictionary with cross-validation results
        
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> clf = RandomForestClassifier(n_estimators=100, random_state=42)
        >>> cv_results = cross_validate_model(clf, X, y, cv=10)
        >>> print(f"CV Accuracy: {cv_results['mean']:.3f} ± {cv_results['std']:.3f}")
    """
    if scoring is None:
        scoring = 'accuracy'
    
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    return {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max(),
        'cv_folds': cv,
        'scoring': scoring,
    }


def generate_evaluation_report(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    class_names: Optional[list] = None
) -> str:
    """
    Generate a comprehensive text evaluation report.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        X_train: Training features (optional)
        y_train: Training labels (optional)
        class_names: Class names
    
    Returns:
        Formatted text report
        
    Example:
        >>> report = generate_evaluation_report(model, X_test, y_test, X_train, y_train)
        >>> print(report)
    """
    metrics = evaluate_model(model, X_test, y_test, X_train, y_train, class_names)
    
    lines = []
    lines.append("=" * 70)
    lines.append("MODEL EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Test performance
    lines.append("Test Set Performance:")
    lines.append(f"  Accuracy:  {metrics['test_accuracy']:.4f}")
    lines.append(f"  Precision: {metrics['test_precision']:.4f}")
    lines.append(f"  Recall:    {metrics['test_recall']:.4f}")
    lines.append(f"  F1-Score:  {metrics['test_f1']:.4f}")
    
    if 'roc_auc' in metrics:
        lines.append(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    lines.append("")
    
    # Training performance (if available)
    if 'train_accuracy' in metrics:
        lines.append("Training Set Performance:")
        lines.append(f"  Accuracy:  {metrics['train_accuracy']:.4f}")
        lines.append(f"  Precision: {metrics['train_precision']:.4f}")
        lines.append(f"  Recall:    {metrics['train_recall']:.4f}")
        lines.append(f"  F1-Score:  {metrics['train_f1']:.4f}")
        lines.append("")
    
    # Confusion Matrix
    lines.append("Confusion Matrix:")
    cm = metrics['confusion_matrix']
    for row in cm:
        lines.append("  " + "  ".join(f"{val:6d}" for val in row))
    lines.append("")
    
    # Classification Report
    lines.append("Classification Report:")
    lines.append(metrics['classification_report_str'])
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def compute_feature_importance(
    model: Any,
    feature_names: Optional[list] = None,
    top_n: int = 20
) -> Dict[str, Any]:
    """
    Compute and rank feature importance.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: Names of features
        top_n: Number of top features to return
    
    Returns:
        Dictionary with importance scores
        
    Example:
        >>> importance = compute_feature_importance(model, feature_names)
        >>> for name, score in importance['top_features']:
        ...     print(f"{name}: {score:.4f}")
    """
    # Extract feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        # Try to get from pipeline
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            clf = model.named_steps['classifier']
            if hasattr(clf, 'feature_importances_'):
                importances = clf.feature_importances_
            elif hasattr(clf, 'coef_'):
                importances = np.abs(clf.coef_).mean(axis=0)
            else:
                raise ValueError("Model does not support feature importance")
        else:
            raise ValueError("Model does not support feature importance")
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    top_indices = indices[:top_n]
    top_features = [(feature_names[i], importances[i]) for i in top_indices]
    
    return {
        'importances': importances,
        'feature_names': feature_names,
        'top_features': top_features,
        'top_n': top_n,
    }
