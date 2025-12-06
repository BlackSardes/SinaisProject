#!/usr/bin/env python3
"""
Complete GPS Spoofing Detection Pipeline

This script runs the complete pipeline from signal loading to model evaluation.
Can be used with real data files or synthetic signals for testing.

Usage:
    python scripts/run_pipeline.py --data-dir data/raw --output-dir results
    python scripts/run_pipeline.py --synthetic --output-dir results
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.signal_io import load_signal, generate_synthetic_signal
from src.preprocessing.pipeline import preprocess_signal, create_preprocessing_config
from src.features.pipeline import extract_features_from_segment, extract_features_from_file
from src.models.training import train_model, train_multiple_models, save_model, create_train_test_split
from src.models.evaluation import evaluate_model, compare_models, generate_evaluation_report, get_feature_importance
from src.utils.plots import (
    plot_correlation_profile,
    plot_feature_distributions,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_cn0_over_time,
    plot_model_comparison,
    create_results_directory
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GPS Spoofing Detection Pipeline')
    
    parser.add_argument('--data-dir', type=str, default='data/raw',
                        help='Directory containing signal files')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory for output results')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic signals instead of real data')
    parser.add_argument('--fs', type=float, default=5e6,
                        help='Sampling frequency in Hz (default: 5 MHz)')
    parser.add_argument('--prn', type=int, default=1,
                        help='PRN satellite number (default: 1)')
    parser.add_argument('--segment-duration', type=float, default=0.5,
                        help='Segment duration in seconds (default: 0.5)')
    parser.add_argument('--spoof-start-time', type=float, default=17.0,
                        help='Time when spoofing starts in seconds (default: 17.0)')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['random_forest', 'svm', 'mlp'],
                        help='Models to train (default: all)')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Cross-validation folds (default: 5)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')
    
    return parser.parse_args()


def create_synthetic_dataset(args):
    """Create synthetic dataset for testing."""
    print("\n" + "="*70)
    print("CREATING SYNTHETIC DATASET")
    print("="*70)
    
    num_segments = 100
    segment_samples = int(args.fs * args.segment_duration)
    features_list = []
    
    for i in range(num_segments):
        # Determine if this segment is spoofed
        time_s = i * args.segment_duration
        is_spoofed = time_s >= args.spoof_start_time
        
        # Generate signal
        signal = generate_synthetic_signal(
            num_samples=segment_samples,
            fs=args.fs,
            snr_db=10.0 if not is_spoofed else 15.0,  # Higher SNR for spoofed
            prn=args.prn,
            add_spoofing=is_spoofed
        )
        
        # Preprocess
        config = create_preprocessing_config('default')
        signal_processed = preprocess_signal(signal, args.fs, config)
        
        # Extract features
        features = extract_features_from_segment(
            signal_processed,
            fs=args.fs,
            prn=args.prn,
            include_statistical=True
        )
        
        # Add metadata
        features['segment_start_time'] = time_s
        features['label'] = 1 if is_spoofed else 0
        
        features_list.append(features)
        
        if args.verbose and (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_segments} segments...")
    
    df = pd.DataFrame(features_list)
    print(f"\nSynthetic dataset created: {len(df)} segments")
    print(f"  Authentic: {sum(df['label'] == 0)}")
    print(f"  Spoofed:   {sum(df['label'] == 1)}")
    
    return df


def process_real_data(args):
    """Process real data files."""
    print("\n" + "="*70)
    print("PROCESSING REAL DATA FILES")
    print("="*70)
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return None
    
    # Find data files
    file_patterns = ['*.bin', '*.dat', '*.mat', '*.csv']
    data_files = []
    for pattern in file_patterns:
        data_files.extend(data_dir.glob(pattern))
    
    if not data_files:
        print(f"Error: No data files found in {data_dir}")
        print(f"  Looked for: {', '.join(file_patterns)}")
        return None
    
    print(f"Found {len(data_files)} data file(s):")
    for f in data_files:
        print(f"  - {f.name}")
    
    # Define labeling function (customize based on your dataset)
    def label_func(time_s):
        return 1 if time_s >= args.spoof_start_time else 0
    
    # Process files
    all_features = []
    
    for file_path in data_files:
        print(f"\nProcessing: {file_path.name}")
        
        config = create_preprocessing_config('default')
        
        try:
            df = extract_features_from_file(
                str(file_path),
                fs=args.fs,
                prn=args.prn,
                segment_duration=args.segment_duration,
                preprocess_config=config,
                label_func=label_func,
                verbose=args.verbose
            )
            
            all_features.append(df)
            
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
            continue
    
    if not all_features:
        print("Error: No features extracted")
        return None
    
    # Combine all features
    df = pd.concat(all_features, ignore_index=True)
    
    print(f"\nTotal features extracted: {len(df)} segments")
    print(f"  Authentic: {sum(df['label'] == 0)}")
    print(f"  Spoofed:   {sum(df['label'] == 1)}")
    
    return df


def train_and_evaluate_models(df, args, output_dir):
    """Train and evaluate models."""
    print("\n" + "="*70)
    print("TRAINING AND EVALUATION")
    print("="*70)
    
    # Prepare data
    feature_columns = [col for col in df.columns if col not in ['label', 'segment_start_time', 'segment_index', 'prn', 'file_path']]
    X = df[feature_columns].values
    y = df['label'].values
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Features: {len(feature_columns)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = create_train_test_split(
        X, y,
        test_size=0.3,
        random_state=args.random_state
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")
    
    # Train models
    model_configs = []
    for model_name in args.models:
        model_configs.append({
            'model_name': model_name,
            'balance_method': 'class_weight'
        })
    
    models_results = train_multiple_models(
        X_train, y_train,
        model_configs=model_configs,
        cv=args.cv_folds,
        random_state=args.random_state,
        verbose=args.verbose
    )
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    
    test_results = {}
    for name, (model, cv_results) in models_results.items():
        print(f"\n{name}")
        print("-" * 50)
        
        eval_results = evaluate_model(model, X_test, y_test, verbose=True)
        test_results[name] = eval_results
        
        # Save model
        model_path = output_dir / 'models' / f'{name}.pkl'
        metadata = {
            'model_name': name,
            'cv_results': cv_results,
            'test_results': {k: v for k, v in eval_results.items() if k != 'classification_report'},
            'feature_columns': feature_columns,
            'training_date': datetime.now().isoformat()
        }
        save_model(model, str(model_path), metadata)
        
        # Generate report
        report_path = output_dir / 'reports' / f'{name}_report.txt'
        report = generate_evaluation_report(
            model, X_test, y_test,
            model_name=name,
            save_path=str(report_path)
        )
    
    # Compare models
    comparison_df = compare_models(
        {name: model for name, (model, _) in models_results.items()},
        X_test, y_test
    )
    
    # Save comparison
    comparison_df.to_csv(output_dir / 'reports' / 'model_comparison.csv', index=False)
    
    return models_results, test_results, comparison_df, feature_columns


def generate_visualizations(df, models_results, test_results, comparison_df, feature_columns, X_test, y_test, output_dir, args):
    """Generate all visualizations."""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    figures_dir = output_dir / 'figures'
    
    # Feature distributions
    print("  - Feature distributions...")
    key_features = ['peak_to_secondary', 'asymmetry', 'cn0_estimate', 'fwhm', 'peak_value']
    available_features = [f for f in key_features if f in df.columns]
    
    if available_features:
        plot_feature_distributions(
            df, available_features,
            save_path=str(figures_dir / 'feature_distributions.png')
        )
    
    # C/N0 over time
    print("  - C/N0 evolution...")
    if 'cn0_estimate' in df.columns:
        plot_cn0_over_time(
            df,
            save_path=str(figures_dir / 'cn0_over_time.png')
        )
    
    # Model comparison
    print("  - Model comparison...")
    plot_model_comparison(
        comparison_df,
        save_path=str(figures_dir / 'model_comparison.png')
        )
    
    # For each model
    for name, (model, cv_results) in models_results.items():
        print(f"  - {name} visualizations...")
        
        # Confusion matrix
        if name in test_results:
            cm = test_results[name]['confusion_matrix']
            plot_confusion_matrix(
                cm,
                title=f'Confusion Matrix - {name}',
                save_path=str(figures_dir / f'{name}_confusion_matrix.png')
            )
        
        # ROC curve (use actual test data)
        if hasattr(model, 'predict_proba') and X_test is not None:
            from src.models.evaluation import get_roc_curve_data
            
            try:
                fpr, tpr, _ = get_roc_curve_data(model, X_test, y_test)
                roc_auc = test_results[name].get('roc_auc', 0.0)
                plot_roc_curve(
                    fpr, tpr, roc_auc,
                    model_name=name,
                    save_path=str(figures_dir / f'{name}_roc_curve.png')
                )
            except Exception as e:
                print(f"    Warning: Could not generate ROC curve: {e}")
        
        # Feature importance (for Random Forest)
        if hasattr(model, 'feature_importances_'):
            importance_df = get_feature_importance(model, feature_columns)
            plot_feature_importance(
                importance_df,
                title=f'Feature Importance - {name}',
                save_path=str(figures_dir / f'{name}_feature_importance.png')
            )
    
    print(f"\nAll visualizations saved to: {figures_dir}")


def main():
    """Main pipeline execution."""
    args = parse_args()
    
    print("="*70)
    print("GPS SPOOFING DETECTION PIPELINE")
    print("="*70)
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.random_state}")
    
    # Set random seeds
    np.random.seed(args.random_state)
    
    # Create output directory
    output_dir = create_results_directory(args.output_dir)
    
    # Step 1: Get or generate data
    if args.synthetic:
        df = create_synthetic_dataset(args)
    else:
        df = process_real_data(args)
    
    if df is None or len(df) == 0:
        print("\nError: No data available. Exiting.")
        return 1
    
    # Save features
    features_path = output_dir / 'features.csv'
    df.to_csv(features_path, index=False)
    print(f"\nFeatures saved to: {features_path}")
    
    # Step 2: Train and evaluate models
    models_results, test_results, comparison_df, feature_columns = train_and_evaluate_models(
        df, args, output_dir
    )
    
    # Step 3: Generate visualizations
    # Extract test data for visualizations
    feature_cols = [col for col in df.columns if col not in ['label', 'segment_start_time', 'segment_index', 'prn', 'file_path']]
    X = df[feature_cols].values
    y = df['label'].values
    _, X_test_viz, _, y_test_viz = create_train_test_split(X, y, test_size=0.3, random_state=args.random_state)
    
    generate_visualizations(
        df, models_results, test_results, comparison_df,
        feature_columns, X_test_viz, y_test_viz, output_dir, args
    )
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Features: features.csv")
    print(f"  - Models: models/")
    print(f"  - Reports: reports/")
    print(f"  - Figures: figures/")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
