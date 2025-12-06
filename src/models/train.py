"""
Model training and evaluation functions.
"""
import numpy as np
from typing import Dict, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, accuracy_score, precision_recall_fscore_support
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def train_model(X: np.ndarray, y: np.ndarray, 
                model_name: str = 'random_forest',
                params: Optional[Dict] = None,
                use_smote: bool = False,
                cv: int = 5,
                random_state: int = 42) -> Tuple[Any, Dict]:
    """
    Train a classification model with optional SMOTE balancing.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        model_name: Model type ('random_forest', 'svm', 'mlp')
        params: Model parameters (None = defaults)
        use_smote: Whether to apply SMOTE for balancing
        cv: Number of cross-validation folds
        random_state: Random state for reproducibility
    
    Returns:
        Tuple of (trained model, training metrics dict)
    """
    # Default parameters for each model
    default_params = {
        'random_forest': {
            'n_estimators': 100,
            'class_weight': 'balanced',
            'random_state': random_state,
            'n_jobs': -1,
        },
        'svm': {
            'kernel': 'rbf',
            'class_weight': 'balanced',
            'random_state': random_state,
            'probability': True,
        },
        'mlp': {
            'hidden_layer_sizes': (100, 50),
            'max_iter': 500,
            'random_state': random_state,
        }
    }
    
    # Get model parameters
    if params is None:
        params = default_params.get(model_name, {})
    else:
        # Merge with defaults
        params = {**default_params.get(model_name, {}), **params}
    
    # Create model
    if model_name == 'random_forest':
        model = RandomForestClassifier(**params)
    elif model_name == 'svm':
        model = SVC(**params)
    elif model_name == 'mlp':
        model = MLPClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Apply SMOTE if requested
    if use_smote:
        smote = SMOTE(random_state=random_state)
        pipeline = ImbPipeline([
            ('smote', smote),
            ('classifier', model)
        ])
        final_model = pipeline
    else:
        final_model = model
    
    # Cross-validation
    cv_scores = cross_val_score(
        final_model, X, y, 
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring='accuracy'
    )
    
    # Train on full dataset
    final_model.fit(X, y)
    
    # Training metrics
    y_pred_train = final_model.predict(X)
    train_accuracy = accuracy_score(y, y_pred_train)
    
    metrics = {
        'model_name': model_name,
        'use_smote': use_smote,
        'cv_mean': float(np.mean(cv_scores)),
        'cv_std': float(np.std(cv_scores)),
        'train_accuracy': float(train_accuracy),
        'params': params,
    }
    
    return final_model, metrics


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray, 
                   class_names: Optional[list] = None) -> Dict:
    """
    Evaluate trained model on test set.
    
    Args:
        model: Trained model
        X: Test feature matrix
        y: True labels
        class_names: Optional class names for reporting
    
    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    if class_names is None:
        class_names = ['Authentic', 'Spoofed']
    
    # Predictions
    y_pred = model.predict(X)
    
    # Basic metrics
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # ROC AUC (if model supports probability)
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)[:, 1]
            roc_auc = roc_auc_score(y, y_proba)
        else:
            roc_auc = None
    except:
        roc_auc = None
    
    # Classification report
    report = classification_report(y, y_pred, target_names=class_names, output_dict=True)
    
    # Detailed metrics from confusion matrix
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        specificity = None
        sensitivity = None
        false_alarm_rate = None
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'confusion_matrix': cm.tolist(),
        'specificity': float(specificity) if specificity is not None else None,
        'sensitivity': float(sensitivity) if sensitivity is not None else None,
        'false_alarm_rate': float(false_alarm_rate) if false_alarm_rate is not None else None,
        'classification_report': report,
    }
    
    return metrics


def print_evaluation_report(metrics: Dict):
    """
    Print formatted evaluation report.
    
    Args:
        metrics: Metrics dictionary from evaluate_model()
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION REPORT")
    print("="*70)
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    if metrics['roc_auc'] is not None:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    if metrics['specificity'] is not None:
        print(f"Specificity: {metrics['specificity']:.4f}")
    
    if metrics['false_alarm_rate'] is not None:
        print(f"False Alarm Rate: {metrics['false_alarm_rate']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(cm)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"\nTrue Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")
    
    print("\n" + "="*70)
