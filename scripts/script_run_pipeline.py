#!/usr/bin/env python3
"""
End-to-end GPS spoofing detection pipeline.

This script runs the complete pipeline from signal loading to model evaluation.
It can work with synthetic data (for testing) or real FGI-SpoofRepo data.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import (
    normalize_signal,
    window_segment,
    remove_dc,
)
from src.features import build_feature_vector
from src.models import train_model, evaluate_model, save_model
from src.utils import plot_confusion_matrix, plot_feature_importance
from sklearn.model_selection import train_test_split


def generate_synthetic_data(n_samples: int = 100000, 
                            fs: float = 5e6,
                            spoofing_ratio: float = 0.3) -> tuple:
    """
    Generate synthetic GPS-like signals for testing.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    fs : float
        Sampling frequency
    spoofing_ratio : float
        Fraction of samples that are spoofed
    
    Returns
    -------
    signal : np.ndarray
        Synthetic signal
    labels : np.ndarray
        Ground truth labels (0=authentic, 1=spoofed)
    """
    print("Generating synthetic GPS-like signals...")
    
    t = np.arange(n_samples) / fs
    
    # Authentic signal: carrier + noise
    carrier_freq = 1e6  # 1 MHz carrier (for illustration)
    authentic = np.exp(1j * 2 * np.pi * carrier_freq * t)
    noise = 0.5 * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    
    # Create spoofed portion with slightly different characteristics
    spoof_start = int(n_samples * (1 - spoofing_ratio))
    signal = authentic + noise
    
    # Modify spoofed portion (different power, phase)
    signal[spoof_start:] *= 1.2  # Increased power
    signal[spoof_start:] *= np.exp(1j * 0.1)  # Phase shift
    signal[spoof_start:] += 0.3 * (np.random.randn(n_samples - spoof_start) + 
                                    1j * np.random.randn(n_samples - spoof_start))
    
    # Create labels
    labels = np.zeros(n_samples, dtype=int)
    labels[spoof_start:] = 1
    
    print(f"Generated {n_samples} samples ({spoofing_ratio*100:.1f}% spoofed)")
    
    return signal, labels


def generate_reference_code(length: int) -> np.ndarray:
    """Generate a simple pseudo-random reference code."""
    np.random.seed(42)  # For reproducibility
    return np.random.choice([-1, 1], size=length)


def run_pipeline(data_path: Optional[Path] = None,
                synthetic: bool = True,
                output_dir: Path = Path('output'),
                model_name: str = 'random_forest',
                use_smote: bool = False) -> dict:
    """
    Run the complete GPS spoofing detection pipeline.
    
    Parameters
    ----------
    data_path : Path, optional
        Path to real data directory
    synthetic : bool
        Use synthetic data if True
    output_dir : Path
        Directory to save outputs
    model_name : str
        Model type ('random_forest', 'svm', 'mlp')
    use_smote : bool
        Use SMOTE for handling class imbalance
    
    Returns
    -------
    results : dict
        Dictionary containing pipeline results and metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================
    # Step 1: Load or Generate Data
    # ============================================
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING")
    print("="*70)
    
    fs = 5e6  # Sampling frequency
    
    if synthetic:
        signal, segment_labels = generate_synthetic_data(n_samples=100000, fs=fs)
    else:
        if data_path is None:
            raise ValueError("data_path must be provided when synthetic=False")
        print(f"Loading data from {data_path}...")
        # In a real implementation, load from data_path
        # For now, fall back to synthetic
        print("Real data loading not implemented yet, using synthetic data")
        signal, segment_labels = generate_synthetic_data(n_samples=100000, fs=fs)
    
    # ============================================
    # Step 2: Preprocessing
    # ============================================
    print("\n" + "="*70)
    print("STEP 2: PREPROCESSING")
    print("="*70)
    
    # Remove DC
    signal = remove_dc(signal)
    print("✓ DC component removed")
    
    # Normalize
    signal = normalize_signal(signal, method='power')
    print("✓ Signal normalized")
    
    # ============================================
    # Step 3: Segmentation
    # ============================================
    print("\n" + "="*70)
    print("STEP 3: SEGMENTATION")
    print("="*70)
    
    window_size = int(0.001 * fs)  # 1 ms windows
    overlap = window_size // 2
    
    windows = window_segment(signal, window_size, overlap)
    print(f"✓ Signal segmented into {len(windows)} windows")
    
    # Create labels for each window (use majority label)
    step = window_size - overlap
    window_labels = []
    for i in range(len(windows)):
        start_idx = i * step
        end_idx = start_idx + window_size
        # Majority vote for window label
        window_label = np.median(segment_labels[start_idx:end_idx])
        window_labels.append(int(window_label))
    
    window_labels = np.array(window_labels)
    print(f"  Authentic windows: {np.sum(window_labels == 0)}")
    print(f"  Spoofed windows: {np.sum(window_labels == 1)}")
    
    # ============================================
    # Step 4: Feature Extraction
    # ============================================
    print("\n" + "="*70)
    print("STEP 4: FEATURE EXTRACTION")
    print("="*70)
    
    # Generate reference code
    ref_code = generate_reference_code(window_size)
    
    # Extract features
    features_df = build_feature_vector(windows, prn=1, fs=fs, reference_code=ref_code)
    features_df['label'] = window_labels
    
    print(f"✓ Extracted {len(features_df.columns)-1} features")
    print(f"  Feature columns: {list(features_df.columns[1:-1])}")
    
    # Save features
    features_path = output_dir / 'features.csv'
    features_df.to_csv(features_path, index=False)
    print(f"✓ Features saved to {features_path}")
    
    # ============================================
    # Step 5: Model Training
    # ============================================
    print("\n" + "="*70)
    print("STEP 5: MODEL TRAINING")
    print("="*70)
    
    # Prepare data
    feature_cols = [col for col in features_df.columns if col not in ['window_idx', 'prn', 'label']]
    X = features_df[feature_cols].values
    y = features_df['label'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    print(f"\nTraining {model_name} model...")
    pipeline, train_metrics = train_model(
        X_train, y_train,
        model_name=model_name,
        use_smote=use_smote,
        cv=5,
        random_state=42
    )
    
    print(f"✓ Model trained")
    print(f"  Cross-validation accuracy: {train_metrics['cv_mean']:.3f} ± {train_metrics['cv_std']:.3f}")
    
    # ============================================
    # Step 6: Model Evaluation
    # ============================================
    print("\n" + "="*70)
    print("STEP 6: MODEL EVALUATION")
    print("="*70)
    
    eval_metrics = evaluate_model(pipeline, X_test, y_test, labels=['Authentic', 'Spoofed'])
    
    print(f"Test Set Performance:")
    print(f"  Accuracy:  {eval_metrics['accuracy']:.3f}")
    print(f"  Precision: {eval_metrics['precision']:.3f}")
    print(f"  Recall:    {eval_metrics['recall']:.3f}")
    print(f"  F1-Score:  {eval_metrics['f1']:.3f}")
    
    if 'roc_auc' in eval_metrics:
        print(f"  ROC AUC:   {eval_metrics['roc_auc']:.3f}")
    
    print("\nConfusion Matrix:")
    cm = np.array(eval_metrics['confusion_matrix'])
    print(cm)
    
    # ============================================
    # Step 7: Save Results
    # ============================================
    print("\n" + "="*70)
    print("STEP 7: SAVING RESULTS")
    print("="*70)
    
    # Save model
    model_path = output_dir / f'{model_name}_model.joblib'
    save_model(pipeline, model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Save metrics
    metrics_path = output_dir / 'metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("GPS Spoofing Detection - Pipeline Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Use SMOTE: {use_smote}\n\n")
        f.write("Training Metrics:\n")
        f.write(f"  CV Accuracy: {train_metrics['cv_mean']:.3f} ± {train_metrics['cv_std']:.3f}\n\n")
        f.write("Test Metrics:\n")
        f.write(f"  Accuracy:  {eval_metrics['accuracy']:.3f}\n")
        f.write(f"  Precision: {eval_metrics['precision']:.3f}\n")
        f.write(f"  Recall:    {eval_metrics['recall']:.3f}\n")
        f.write(f"  F1-Score:  {eval_metrics['f1']:.3f}\n")
        if 'roc_auc' in eval_metrics:
            f.write(f"  ROC AUC:   {eval_metrics['roc_auc']:.3f}\n")
    
    print(f"✓ Metrics saved to {metrics_path}")
    
    # ============================================
    # Summary
    # ============================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nOutputs saved to: {output_dir.absolute()}")
    print(f"  - Features: features.csv")
    print(f"  - Model: {model_name}_model.joblib")
    print(f"  - Metrics: metrics.txt")
    
    return {
        'pipeline': pipeline,
        'features_df': features_df,
        'train_metrics': train_metrics,
        'eval_metrics': eval_metrics,
        'feature_cols': feature_cols,
    }


def main():
    """Main entry point for the pipeline script."""
    parser = argparse.ArgumentParser(
        description='GPS Spoofing Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with synthetic data (default)
  python script_run_pipeline.py --synthetic
  
  # Run with real data
  python script_run_pipeline.py --data-path ../data/scenario1/
  
  # Use SVM model with SMOTE
  python script_run_pipeline.py --synthetic --model svm --use-smote
        """
    )
    
    parser.add_argument(
        '--data-path',
        type=Path,
        help='Path to data directory (for real data)'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        default=True,
        help='Use synthetic data (default: True)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output'),
        help='Output directory for results (default: output/)'
    )
    parser.add_argument(
        '--model',
        choices=['random_forest', 'svm', 'mlp'],
        default='random_forest',
        help='Model type (default: random_forest)'
    )
    parser.add_argument(
        '--use-smote',
        action='store_true',
        help='Use SMOTE for handling class imbalance'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_pipeline(
            data_path=args.data_path,
            synthetic=args.synthetic,
            output_dir=args.output_dir,
            model_name=args.model,
            use_smote=args.use_smote
        )
        
        print("\n✓ Pipeline execution completed successfully!")
        return 0
    
    except Exception as e:
        print(f"\n✗ Pipeline execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
