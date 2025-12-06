#!/usr/bin/env python3
"""
Complete GPS Spoofing Detection Pipeline

This script executes the full pipeline:
1. Generate or load GPS signals
2. Extract features
3. Train classification models
4. Evaluate and save results

Usage:
    python scripts/run_pipeline.py --synthetic
    python scripts/run_pipeline.py --data-dir data/ --output-dir results/
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.synthetic_data import generate_synthetic_dataset
from src.features.feature_pipeline import build_feature_vector
from src.models.training import train_model, train_with_smote
from src.models.evaluation import evaluate_model, generate_evaluation_report
from src.models.persistence import save_model
from src.utils.plots import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_distributions,
    save_figure
)


def main():
    parser = argparse.ArgumentParser(description='GPS Spoofing Detection Pipeline')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data instead of real data')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing GPS signal files')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--n-authentic', type=int, default=200,
                       help='Number of authentic samples (synthetic mode)')
    parser.add_argument('--n-spoofed', type=int, default=200,
                       help='Number of spoofed samples (synthetic mode)')
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['random_forest', 'svm', 'mlp'],
                       help='Model to train')
    parser.add_argument('--use-smote', action='store_true',
                       help='Use SMOTE for class balancing')
    parser.add_argument('--fs', type=float, default=5e6,
                       help='Sampling frequency in Hz')
    parser.add_argument('--prn', type=int, default=1,
                       help='PRN number for correlation')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("=" * 70)
    print("GPS SPOOFING DETECTION PIPELINE")
    print("=" * 70)
    print(f"Mode: {'Synthetic Data' if args.synthetic else 'Real Data'}")
    print(f"Model: {args.model}")
    print(f"Use SMOTE: {args.use_smote}")
    print(f"Random State: {args.random_state}")
    print(f"Output Directory: {output_path}")
    print("=" * 70)
    print()
    
    # Step 1: Load or generate data
    print("Step 1: Loading/Generating Data...")
    print("-" * 70)
    
    if args.synthetic:
        # Generate synthetic data
        print(f"Generating {args.n_authentic} authentic and {args.n_spoofed} spoofed signals...")
        signals, labels = generate_synthetic_dataset(
            n_authentic=args.n_authentic,
            n_spoofed=args.n_spoofed,
            duration_s=0.5,
            fs=args.fs,
            prn=args.prn,
            seed=args.random_state
        )
        print(f"Generated {len(signals)} signals")
    else:
        # Load real data (placeholder - would need actual implementation)
        print(f"Loading data from {args.data_dir}...")
        print("WARNING: Real data loading not implemented. Use --synthetic flag.")
        print("To use real data, implement signal loading in this script.")
        return
    
    print()
    
    # Step 2: Extract features
    print("Step 2: Extracting Features...")
    print("-" * 70)
    
    features_df = build_feature_vector(
        signals,
        fs=args.fs,
        prn=args.prn,
        include_correlation=True,
        include_temporal=True,
        include_cn0_variation=True
    )
    
    # Add labels
    features_df['label'] = labels
    
    print(f"Extracted features shape: {features_df.shape}")
    print(f"Features: {[col for col in features_df.columns if col not in ['segment_id', 'label']]}")
    print()
    
    # Save features
    features_path = output_path / f'features_{timestamp}.csv'
    features_df.to_csv(features_path, index=False)
    print(f"Features saved to: {features_path}")
    print()
    
    # Step 3: Prepare data for training
    print("Step 3: Preparing Data for Training...")
    print("-" * 70)
    
    # Drop non-feature columns
    X = features_df.drop(['segment_id', 'label'], axis=1, errors='ignore').values
    y = features_df['label'].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: Authentic={np.sum(y==0)}, Spoofed={np.sum(y==1)}")
    print()
    
    # Step 4: Train model
    print("Step 4: Training Model...")
    print("-" * 70)
    
    if args.use_smote:
        print("Training with SMOTE...")
        model, info = train_with_smote(
            X, y,
            model_name=args.model,
            random_state=args.random_state,
            test_size=0.3
        )
    else:
        print("Training without SMOTE (using class_weight='balanced')...")
        model, info = train_model(
            X, y,
            model_name=args.model,
            random_state=args.random_state,
            test_size=0.3
        )
    
    print(f"Training completed!")
    print(f"Training samples: {info['n_train_samples']}")
    print(f"Test samples: {info['n_test_samples']}")
    print()
    
    # Step 5: Evaluate model
    print("Step 5: Evaluating Model...")
    print("-" * 70)
    
    metrics = evaluate_model(
        model,
        info['X_test'],
        info['y_test'],
        info['X_train'],
        info['y_train'],
        class_names=['Authentic', 'Spoofed']
    )
    
    # Generate and print report
    report = generate_evaluation_report(
        model,
        info['X_test'],
        info['y_test'],
        info['X_train'],
        info['y_train'],
        class_names=['Authentic', 'Spoofed']
    )
    print(report)
    
    # Save report
    report_path = output_path / f'evaluation_report_{args.model}_{timestamp}.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    print()
    
    # Step 6: Generate visualizations
    print("Step 6: Generating Visualizations...")
    print("-" * 70)
    
    # Confusion matrix
    fig_cm = plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names=['Authentic', 'Spoofed'],
        title=f'{args.model.title()} - Confusion Matrix'
    )
    save_figure(fig_cm, f'confusion_matrix_{args.model}_{timestamp}',
                output_dir=str(output_path))
    
    # ROC curve
    if 'roc_curve' in metrics:
        roc_data = {
            args.model.title(): {
                'fpr': metrics['roc_curve']['fpr'],
                'tpr': metrics['roc_curve']['tpr'],
                'auc': metrics.get('roc_auc', 0)
            }
        }
        fig_roc = plot_roc_curves(roc_data, title='ROC Curve')
        save_figure(fig_roc, f'roc_curve_{args.model}_{timestamp}',
                    output_dir=str(output_path))
    
    # Feature distributions
    fig_dist = plot_feature_distributions(
        features_df,
        label_column='label',
        class_names={0: 'Authentic', 1: 'Spoofed'}
    )
    save_figure(fig_dist, f'feature_distributions_{timestamp}',
                output_dir=str(output_path))
    
    print(f"Visualizations saved to: {output_path}")
    print()
    
    # Step 7: Save model
    print("Step 7: Saving Model...")
    print("-" * 70)
    
    model_path = output_path / f'model_{args.model}_{timestamp}.pkl'
    metadata = {
        'model_name': args.model,
        'timestamp': timestamp,
        'test_accuracy': metrics['test_accuracy'],
        'test_f1': metrics['test_f1'],
        'roc_auc': metrics.get('roc_auc', None),
        'n_features': X.shape[1],
        'feature_names': [col for col in features_df.columns 
                         if col not in ['segment_id', 'label']],
        'random_state': args.random_state,
        'use_smote': args.use_smote,
    }
    
    save_model(model, str(model_path), metadata=metadata)
    print()
    
    # Summary
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test F1-Score: {metrics['test_f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print()
    print(f"All results saved to: {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
