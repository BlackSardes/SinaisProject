"""
Visualization Module for GPS Spoofing Detection

This module provides plotting functions for:
- Correlation profiles
- Feature distributions
- Confusion matrices
- ROC curves
- Feature importance
- C/N0 analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Any
from pathlib import Path


# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_correlation_profile(
    correlation: np.ndarray,
    samples_per_chip: int,
    title: str = "Correlation Profile",
    peak_index: Optional[int] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    """
    Plot correlation profile with peak highlighting.
    
    Parameters
    ----------
    correlation : np.ndarray
        Correlation magnitude array
    samples_per_chip : int
        Samples per chip for x-axis scaling
    title : str, optional
        Plot title
    peak_index : int, optional
        Index of peak (if None, auto-detect)
    save_path : str, optional
        Path to save figure
    figsize : tuple, optional
        Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to chip units
    x_chips = np.arange(len(correlation)) / samples_per_chip
    
    # Plot correlation
    ax.plot(x_chips, correlation, 'b-', linewidth=1.2, label='Correlation')
    
    # Highlight peak
    if peak_index is None:
        peak_index = np.argmax(correlation)
    
    peak_chip = peak_index / samples_per_chip
    peak_value = correlation[peak_index]
    
    ax.plot(peak_chip, peak_value, 'r*', markersize=15, label=f'Peak ({peak_value:.2f})')
    
    # Mark half-maximum
    half_max = peak_value / 2
    ax.axhline(y=half_max, color='g', linestyle='--', alpha=0.5, label=f'FWHM level')
    
    # Labels
    ax.set_xlabel('Code Phase (chips)')
    ax.set_ylabel('Correlation Magnitude')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()


def plot_feature_distributions(
    df: pd.DataFrame,
    features: List[str],
    label_column: str = 'label',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Plot distributions of features by class.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and labels
    features : list of str
        Feature names to plot
    label_column : str, optional
        Name of label column
    save_path : str, optional
        Path to save figure
    figsize : tuple, optional
        Figure size
    """
    n_features = len(features)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Plot distributions for each class
        for label in df[label_column].unique():
            data = df[df[label_column] == label][feature]
            label_name = 'Spoofed' if label == 1 else 'Authentic'
            ax.hist(data, bins=30, alpha=0.6, label=label_name, density=True)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(feature, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ['Authentic', 'Spoofed'],
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot confusion matrix as heatmap.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (2x2 or NxN)
    class_names : list of str, optional
        Class names for labels
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
    figsize : tuple, optional
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        ax=ax
    )
    
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot ROC curve.
    
    Parameters
    ----------
    fpr : np.ndarray
        False positive rates
    tpr : np.ndarray
        True positive rates
    roc_auc : float
        ROC AUC score
    model_name : str, optional
        Model name for legend
    save_path : str, optional
        Path to save figure
    figsize : tuple, optional
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Curve', fontweight='bold', pad=20)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()


def plot_multiple_roc_curves(
    roc_data: dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 7)
) -> None:
    """
    Plot multiple ROC curves for comparison.
    
    Parameters
    ----------
    roc_data : dict
        Dictionary mapping model names to (fpr, tpr, auc) tuples
    save_path : str, optional
        Path to save figure
    figsize : tuple, optional
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(roc_data)))
    
    for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, linewidth=2, color=color, label=f'{name} (AUC = {auc:.3f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Curve Comparison', fontweight='bold', pad=20)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot feature importance as horizontal bar chart.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int, optional
        Number of top features to show
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
    figsize : tuple, optional
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get top N features
    plot_df = importance_df.head(top_n).sort_values('importance', ascending=True)
    
    # Create horizontal bar chart
    ax.barh(plot_df['feature'], plot_df['importance'], color='steelblue')
    
    ax.set_xlabel('Importance', fontweight='bold')
    ax.set_ylabel('Feature', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()


def plot_cn0_over_time(
    df: pd.DataFrame,
    time_column: str = 'segment_start_time',
    cn0_column: str = 'cn0_estimate',
    label_column: str = 'label',
    title: str = "C/N0 Over Time",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """
    Plot C/N0 evolution over time with spoofing indication.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time, C/N0, and label columns
    time_column : str, optional
        Time column name
    cn0_column : str, optional
        C/N0 column name
    label_column : str, optional
        Label column name
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
    figsize : tuple, optional
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot C/N0
    authentic = df[df[label_column] == 0]
    spoofed = df[df[label_column] == 1]
    
    if len(authentic) > 0:
        ax.plot(authentic[time_column], authentic[cn0_column],
                'go', markersize=4, alpha=0.6, label='Authentic')
    
    if len(spoofed) > 0:
        ax.plot(spoofed[time_column], spoofed[cn0_column],
                'ro', markersize=4, alpha=0.6, label='Spoofed')
    
    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('C/N0 (dB-Hz)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()


def plot_signal_spectrum(
    signal: np.ndarray,
    fs: float,
    title: str = "Signal Spectrum",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    """
    Plot signal spectrum.
    
    Parameters
    ----------
    signal : np.ndarray
        Complex signal
    fs : float
        Sampling frequency in Hz
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
    figsize : tuple, optional
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute FFT
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/fs))
    magnitude_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
    
    # Plot
    ax.plot(freqs / 1e6, magnitude_db, 'b-', linewidth=0.8)
    
    ax.set_xlabel('Frequency (MHz)', fontweight='bold')
    ax.set_ylabel('Magnitude (dB)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'F1-Score',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot bar chart comparing multiple models.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with model comparison results
    metric : str, optional
        Metric to plot (column name)
    save_path : str, optional
        Path to save figure
    figsize : tuple, optional
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by metric
    plot_df = comparison_df.sort_values(metric, ascending=False)
    
    # Create bar chart
    bars = ax.bar(range(len(plot_df)), plot_df[metric], color='steelblue')
    
    # Color best model differently
    bars[0].set_color('darkgreen')
    
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df['Model'], rotation=45, ha='right')
    ax.set_ylabel(metric, fontweight='bold')
    ax.set_title(f'Model Comparison - {metric}', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()


def create_results_directory(base_path: str = 'results') -> Path:
    """
    Create directory for saving results and figures.
    
    Parameters
    ----------
    base_path : str, optional
        Base path for results directory
        
    Returns
    -------
    Path
        Path object for results directory
    """
    results_dir = Path(base_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (results_dir / 'figures').mkdir(exist_ok=True)
    (results_dir / 'models').mkdir(exist_ok=True)
    (results_dir / 'reports').mkdir(exist_ok=True)
    
    return results_dir
