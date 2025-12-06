#!/usr/bin/env python3

"""
GPS Spoofing Detection Pipeline - Complete Execution Script

This script runs the complete pipeline:
1. Load or generate data
2. Preprocess signals
3. Extract features
4. Train model
5. Evaluate and save results
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from utils.synthetic_data import generate_synthetic_dataset
from utils.data_loader import load_fgi_dataset, load_texbat_dataset
from preprocessing.signal_processing import generate_ca_code
from features.pipeline import build_feature_vector, preprocess_features
from models.train import train_model, evaluate_model, print_evaluation_report
from models.persistence import save_model
from utils.plots import plot_confusion_matrix, plot_roc_curves, plot_feature_distributions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='GPS Spoofing Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with synthetic data (default)
  python script_run_pipeline.py --mode synthetic --num-samples 200
  
  # Run with FGI-SpoofRepo data
  python script_run_pipeline.py --mode fgi --data-dir ../data/raw/fgi-spoof-repo
  
  # Run with TEXBAT data
  python script_run_pipeline.py --mode texbat --data-dir ../data/raw/texbat --spoof-time 17.0
        """
    )
    
    parser.add_argument('--mode', type=str, default='synthetic',
                       choices=['synthetic', 'fgi', 'texbat'],
                       help='Data source mode')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Path to data directory (for fgi/texbat modes)')
    parser.add_argument('--num-samples', type=int, default=200,
                       help='Number of samples per class (synthetic mode)')
    parser.add_argument('--fs', type=float, default=5e6,
                       help='Sampling frequency in Hz')
    parser.add_argument('--duration', type=float, default=0.5,
                       help='Signal duration in seconds')
    parser.add_argument('--spoof-time', type=float, default=17.0,
                       help='Spoofing start time in seconds (TEXBAT mode)')
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['random_forest', 'svm', 'mlp'],
                       help='Model type')
    parser.add_argument('--use-smote', action='store_true',
                       help='Use SMOTE for class balancing')
    parser.add_argument('--output-dir', type=str, default='../data/processed',
                       help='Output directory for results')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def load_data(args):
    """Load data based on mode."""
    print(f"\n{'='*70}")
    print("STEP 1: Loading Data")
    print('='*70)
    
    if args.mode == 'synthetic':
        print(f"Generating synthetic dataset ({args.num_samples} samples per class)...")
        signals, labels, metadata = generate_synthetic_dataset(
            num_authentic=args.num_samples,
            num_spoofed=args.num_samples,
            fs=args.fs,
            duration=args.duration,
            prn_range=(1, 5),
            random_state=args.random_seed
        )
    
    elif args.mode == 'fgi':
        if args.data_dir is None:
            raise ValueError("--data-dir required for FGI mode")
        print(f"Loading FGI-SpoofRepo dataset from {args.data_dir}...")
        signals, labels, metadata = load_fgi_dataset(args.data_dir)
    
    elif args.mode == 'texbat':
        if args.data_dir is None:
            raise ValueError("--data-dir required for TEXBAT mode")
        print(f"Loading TEXBAT dataset from {args.data_dir}...")
        signals, labels, metadata = load_texbat_dataset(
            args.data_dir,
            fs=args.fs,
            segment_duration=args.duration,
            spoof_start_time=args.spoof_time,
            max_segments=args.num_samples * 2 if args.num_samples else None
        )
    
    print(f"\nLoaded {len(signals)} signals")
    print(f"Class distribution: Authentic={sum(1 for l in labels if l == 0)}, "
          f"Spoofed={sum(1 for l in labels if l == 1)}")
    
    return signals, labels, metadata


def extract_features(signals, labels, metadata, fs):
    """Extract features from signals."""
    print(f"\n{'='*70}")
    print("STEP 2: Feature Extraction")
    print('='*70)
    
    all_features = []
    for i, signal in enumerate(signals):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing signal {i+1}/{len(signals)}...")
        
        prn = metadata[i].get('prn', 1)
        ca_code = generate_ca_code(prn)
        
        features = build_feature_vector(
            signal=signal,
            prn_code=ca_code,
            fs=fs,
            label=labels[i],
            metadata={'prn': prn, 'segment_index': i}
        )
        all_features.append(features)
    
    df_features = pd.DataFrame(all_features)
    print(f"\nFeature extraction complete. Shape: {df_features.shape}")
    print(f"Features: {[col for col in df_features.columns if col not in ['label', 'prn', 'segment_index']]}")
    
    return df_features


def train_and_evaluate(df_features, args):
    """Train and evaluate model."""
    print(f"\n{'='*70}")
    print("STEP 3: Model Training and Evaluation")
    print('='*70)
    
    # Prepare data
    X = df_features.drop(columns=['label', 'prn', 'segment_index'], errors='ignore')
    y = df_features['label'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=args.random_seed, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Preprocess
    print("\nPreprocessing features...")
    X_train_proc, imputer, scaler, _ = preprocess_features(X_train, y_train, fit=True)
    X_test_proc, _, _, _ = preprocess_features(X_test, y_test, imputer=imputer, scaler=scaler, fit=False)
    
    # Train
    print(f"\nTraining {args.model} model...")
    model, train_metrics = train_model(
        X_train_proc, y_train,
        model_name=args.model,
        use_smote=args.use_smote,
        cv=5,
        random_state=args.random_seed
    )
    
    print(f"\nTraining CV Accuracy: {train_metrics['cv_mean']:.4f} (+/- {train_metrics['cv_std']:.4f})")
    
    # Evaluate
    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, X_test_proc, y_test)
    print_evaluation_report(metrics)
    
    return model, metrics, imputer, scaler, X, X_test_proc, y_test


def save_results(model, metrics, imputer, scaler, feature_cols, args, df_features, X_test_proc, y_test):
    """Save model and generate visualizations."""
    print(f"\n{'='*70}")
    print("STEP 4: Saving Results")
    print('='*70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(args.output_dir, f'{args.model}_model.pkl')
    metadata = {
        'model_name': args.model,
        'use_smote': args.use_smote,
        'metrics': metrics,
        'features': list(feature_cols),
        'random_state': args.random_seed,
        'data_mode': args.mode,
    }
    save_model(model, model_path, metadata)
    
    # Save preprocessors
    import joblib
    joblib.dump(imputer, os.path.join(args.output_dir, 'imputer.pkl'))
    joblib.dump(scaler, os.path.join(args.output_dir, 'scaler.pkl'))
    print("Preprocessors saved.")
    
    # Save features
    features_path = os.path.join(args.output_dir, 'features.csv')
    df_features.to_csv(features_path, index=False)
    print(f"Features saved to: {features_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        class_names=['Authentic', 'Spoofed'],
        title=f'{args.model.upper()} - Confusion Matrix',
        save_path=cm_path
    )
    plt.close()
    
    # ROC curve
    if metrics['roc_auc'] is not None:
        try:
            y_proba = model.predict_proba(X_test_proc)[:, 1]
            roc_path = os.path.join(args.output_dir, 'roc_curve.png')
            plot_roc_curves(
                y_test,
                {args.model: y_proba},
                title='ROC Curve',
                save_path=roc_path
            )
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate ROC curve: {e}")
    
    # Feature distributions
    try:
        key_features = ['peak_height', 'peak_to_secondary', 'fpw', 'asymmetry', 
                       'cn0_estimate', 'total_power']
        available = [f for f in key_features if f in df_features.columns]
        if available:
            dist_path = os.path.join(args.output_dir, 'feature_distributions.png')
            plot_feature_distributions(
                df_features,
                features=available,
                label_col='label',
                save_path=dist_path
            )
            plt.close()
    except Exception as e:
        print(f"Warning: Could not generate feature distributions: {e}")
    
    # Save metrics to JSON
    import json
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    # Convert numpy types to native Python types
    metrics_clean = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_clean[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            # Use .item() for scalar numpy values
            metrics_clean[key] = value.item() if hasattr(value, 'item') else float(value)
        else:
            metrics_clean[key] = value
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_clean, f, indent=2, default=str)
    print(f"Metrics saved to: {metrics_path}")
    
    print(f"\nAll results saved to: {args.output_dir}")


def main():
    """Main pipeline execution."""
    args = parse_args()
    
    print("="*70)
    print("GPS SPOOFING DETECTION PIPELINE")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"SMOTE: {args.use_smote}")
    print(f"Random seed: {args.random_seed}")
    
    try:
        # Load data
        signals, labels, metadata = load_data(args)
        
        # Extract features
        df_features = extract_features(signals, labels, metadata, args.fs)
        
        # Train and evaluate
        model, metrics, imputer, scaler, feature_cols, X_test_proc, y_test = train_and_evaluate(df_features, args)
        
        # Save results
        save_results(model, metrics, imputer, scaler, feature_cols, args, df_features, X_test_proc, y_test)
        
        print(f"\n{'='*70}")
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print('='*70)
        print(f"\nFinal Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        if metrics['roc_auc'] is not None:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
