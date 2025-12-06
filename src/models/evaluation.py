"""
Model Evaluation Module for GPS Spoofing Detection

This module provides comprehensive evaluation metrics and reporting
for classification models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a trained model.
    
    Parameters
    ----------
    model : trained model
        Model to evaluate
    X_test : np.ndarray
        Test feature matrix
    y_test : np.ndarray
        Test labels
    verbose : bool, optional
        Print evaluation results (default: True)
        
    Returns
    -------
    dict
        Evaluation metrics including:
        - confusion_matrix: 2x2 confusion matrix
        - accuracy: Overall accuracy
        - precision: Precision score
        - recall: Recall (sensitivity) score
        - f1: F1-score
        - specificity: True negative rate
        - false_alarm_rate: False positive rate
        - roc_auc: ROC AUC score (if probabilities available)
        - average_precision: Average precision score
        - classification_report: Detailed report string
        
    Notes
    -----
    For spoofing detection:
    - Recall (sensitivity): How many spoofing attacks are detected
    - Precision: How many detected attacks are real
    - False alarm rate: How often genuine signals are flagged
    - Specificity: How often genuine signals are correctly identified
    """
    results = {}
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Probabilities (if available)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = None
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    
    # Basic metrics
    results['accuracy'] = float(accuracy_score(y_test, y_pred))
    results['precision'] = float(precision_score(y_test, y_pred, zero_division=0))
    results['recall'] = float(recall_score(y_test, y_pred, zero_division=0))
    results['f1'] = float(f1_score(y_test, y_pred, zero_division=0))
    
    # Confusion matrix components
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        results['true_negatives'] = int(tn)
        results['false_positives'] = int(fp)
        results['false_negatives'] = int(fn)
        results['true_positives'] = int(tp)
        
        # Specificity (true negative rate)
        results['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        # False alarm rate
        results['false_alarm_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    # ROC AUC (if probabilities available)
    if y_proba is not None:
        try:
            results['roc_auc'] = float(roc_auc_score(y_test, y_proba))
            results['average_precision'] = float(average_precision_score(y_test, y_proba))
        except:
            results['roc_auc'] = None
            results['average_precision'] = None
    else:
        results['roc_auc'] = None
        results['average_precision'] = None
    
    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=['Authentic', 'Spoofed'],
        zero_division=0
    )
    results['classification_report'] = report
    
    if verbose:
        print("\n" + "="*70)
        print("MODEL EVALUATION RESULTS")
        print("="*70)
        print(f"\nAccuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1']:.4f}")
        
        if 'specificity' in results:
            print(f"Specificity: {results['specificity']:.4f}")
            print(f"False Alarm Rate: {results['false_alarm_rate']:.4f}")
        
        if results['roc_auc'] is not None:
            print(f"ROC AUC:   {results['roc_auc']:.4f}")
        
        print("\n" + "-"*70)
        print("Confusion Matrix:")
        print("-"*70)
        print(cm)
        
        if cm.shape == (2, 2):
            print(f"\nTrue Negatives:  {results['true_negatives']}")
            print(f"False Positives: {results['false_positives']}")
            print(f"False Negatives: {results['false_negatives']}")
            print(f"True Positives:  {results['true_positives']}")
        
        print("\n" + "-"*70)
        print("Classification Report:")
        print("-"*70)
        print(report)
    
    return results


def compare_models(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Compare multiple models on the same test set.
    
    Parameters
    ----------
    models_dict : dict
        Dictionary mapping model names to trained models
    X_test : np.ndarray
        Test feature matrix
    y_test : np.ndarray
        Test labels
        
    Returns
    -------
    pd.DataFrame
        Comparison table with metrics for each model
    """
    results_list = []
    
    for name, model in models_dict.items():
        print(f"\nEvaluating: {name}")
        print("-" * 50)
        
        results = evaluate_model(model, X_test, y_test, verbose=False)
        
        results_summary = {
            'Model': name,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1'],
            'Specificity': results.get('specificity', np.nan),
            'False Alarm': results.get('false_alarm_rate', np.nan),
            'ROC AUC': results.get('roc_auc', np.nan)
        }
        
        results_list.append(results_summary)
    
    comparison_df = pd.DataFrame(results_list)
    comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def get_roc_curve_data(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get ROC curve data for plotting.
    
    Parameters
    ----------
    model : trained model
        Model with predict_proba method
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
        
    Returns
    -------
    fpr : np.ndarray
        False positive rates
    tpr : np.ndarray
        True positive rates
    thresholds : np.ndarray
        Decision thresholds
    """
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_proba = model.decision_function(X_test)
    else:
        raise ValueError("Model must have predict_proba or decision_function method")
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    return fpr, tpr, thresholds


def get_precision_recall_curve_data(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get precision-recall curve data for plotting.
    
    Parameters
    ----------
    model : trained model
        Model with predict_proba method
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
        
    Returns
    -------
    precision : np.ndarray
        Precision values
    recall : np.ndarray
        Recall values
    thresholds : np.ndarray
        Decision thresholds
    """
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_proba = model.decision_function(X_test)
    else:
        raise ValueError("Model must have predict_proba or decision_function method")
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    return precision, recall, thresholds


def get_feature_importance(
    model: Any,
    feature_names: Optional[list] = None,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extract and rank feature importance from model.
    
    Parameters
    ----------
    model : trained model
        Model with feature_importances_ attribute (e.g., RandomForest)
    feature_names : list, optional
        Names of features
    top_n : int, optional
        Number of top features to return (default: 20)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with features ranked by importance
        
    Notes
    -----
    Only works for models that have feature importance (RandomForest, etc.)
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importances = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importances))]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    if top_n is not None:
        importance_df = importance_df.head(top_n)
    
    return importance_df


def analyze_errors(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze model errors in detail.
    
    Parameters
    ----------
    model : trained model
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    feature_names : list, optional
        Feature names
        
    Returns
    -------
    false_positives : pd.DataFrame
        Cases where model predicted spoofing but was authentic
    false_negatives : pd.DataFrame
        Cases where model missed spoofing attacks
        
    Notes
    -----
    Helps understand what types of signals the model struggles with.
    """
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = np.ones(len(y_test)) * 0.5
    
    # Create DataFrame
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]
    
    error_df = pd.DataFrame(X_test, columns=feature_names)
    error_df['true_label'] = y_test
    error_df['predicted_label'] = y_pred
    error_df['probability'] = y_proba
    error_df['error'] = y_test != y_pred
    
    # False positives (predicted spoofed, actually authentic)
    false_positives = error_df[(error_df['true_label'] == 0) & (error_df['predicted_label'] == 1)]
    
    # False negatives (predicted authentic, actually spoofed)
    false_negatives = error_df[(error_df['true_label'] == 1) & (error_df['predicted_label'] == 0)]
    
    print(f"\nError Analysis:")
    print(f"  False Positives: {len(false_positives)}")
    print(f"  False Negatives: {len(false_negatives)}")
    print(f"  Total Errors: {len(false_positives) + len(false_negatives)}")
    print(f"  Error Rate: {(len(false_positives) + len(false_negatives)) / len(y_test):.2%}")
    
    return false_positives, false_negatives


def generate_evaluation_report(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None
) -> str:
    """
    Generate comprehensive text report of model evaluation.
    
    Parameters
    ----------
    model : trained model
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    model_name : str, optional
        Name of model for report
    save_path : str, optional
        Path to save report (if None, just return string)
        
    Returns
    -------
    str
        Report text
    """
    results = evaluate_model(model, X_test, y_test, verbose=False)
    
    report = []
    report.append("="*70)
    report.append(f"EVALUATION REPORT: {model_name}")
    report.append("="*70)
    report.append("")
    
    report.append("PERFORMANCE METRICS")
    report.append("-"*70)
    report.append(f"Accuracy:            {results['accuracy']:.4f}")
    report.append(f"Precision:           {results['precision']:.4f}")
    report.append(f"Recall (Sensitivity): {results['recall']:.4f}")
    report.append(f"F1-Score:            {results['f1']:.4f}")
    
    if 'specificity' in results:
        report.append(f"Specificity:         {results['specificity']:.4f}")
        report.append(f"False Alarm Rate:    {results['false_alarm_rate']:.4f}")
    
    if results['roc_auc'] is not None:
        report.append(f"ROC AUC:             {results['roc_auc']:.4f}")
        report.append(f"Average Precision:   {results['average_precision']:.4f}")
    
    report.append("")
    report.append("CONFUSION MATRIX")
    report.append("-"*70)
    cm = results['confusion_matrix']
    report.append(str(cm))
    
    if cm.shape == (2, 2):
        report.append("")
        report.append(f"True Negatives:  {results['true_negatives']} (Authentic correctly identified)")
        report.append(f"False Positives: {results['false_positives']} (Authentic incorrectly flagged)")
        report.append(f"False Negatives: {results['false_negatives']} (Spoofing missed)")
        report.append(f"True Positives:  {results['true_positives']} (Spoofing detected)")
    
    report.append("")
    report.append("DETAILED CLASSIFICATION REPORT")
    report.append("-"*70)
    report.append(results['classification_report'])
    
    report_text = "\n".join(report)
    
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {save_path}")
    
    return report_text
