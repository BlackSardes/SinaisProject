"""
Visualization utilities for GPS spoofing detection.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict
import pandas as pd
from sklearn.metrics import roc_curve, auc


def plot_correlation_profile(corr_profile: np.ndarray, fs: float, 
                             title: str = "Correlation Profile",
                             save_path: Optional[str] = None,
                             figsize: tuple = (12, 6)):
    """
    Plot correlation profile.
    
    Args:
        corr_profile: Correlation magnitude array
        fs: Sampling frequency (Hz)
        title: Plot title
        save_path: Path to save figure (None = don't save)
        figsize: Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Time axis (in chips for GPS)
    ca_chip_rate = 1.023e6
    samples_per_chip = fs / ca_chip_rate
    chips = np.arange(len(corr_profile)) / samples_per_chip
    
    ax.plot(chips, corr_profile, 'b-', linewidth=1.5)
    ax.set_xlabel('Code Phase (chips)', fontsize=12)
    ax.set_ylabel('Correlation Magnitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Mark peak
    peak_idx = np.argmax(corr_profile)
    ax.plot(chips[peak_idx], corr_profile[peak_idx], 'ro', markersize=10, label='Peak')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax


def plot_feature_distributions(df: pd.DataFrame, features: List[str],
                               label_col: str = 'label',
                               save_path: Optional[str] = None,
                               figsize: tuple = (15, 10)):
    """
    Plot feature distributions by class.
    
    Args:
        df: Feature DataFrame
        features: List of feature names to plot
        label_col: Column name for labels
        save_path: Path to save figure
        figsize: Figure size
    """
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature in enumerate(features):
        if i >= len(axes):
            break
        
        ax = axes[i]
        
        if label_col in df.columns:
            for label in df[label_col].unique():
                data = df[df[label_col] == label][feature].dropna()
                label_name = 'Spoofed' if label == 1 else 'Authentic'
                ax.hist(data, bins=30, alpha=0.6, label=label_name, density=True)
            ax.legend()
        else:
            ax.hist(df[feature].dropna(), bins=30, alpha=0.7)
        
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, axes


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str] = None,
                         title: str = "Confusion Matrix",
                         save_path: Optional[str] = None,
                         figsize: tuple = (8, 6)):
    """
    Plot confusion matrix with annotations.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    if class_names is None:
        class_names = ['Authentic', 'Spoofed']
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax


def plot_roc_curves(y_true: np.ndarray, y_scores: Dict[str, np.ndarray],
                   title: str = "ROC Curves",
                   save_path: Optional[str] = None,
                   figsize: tuple = (10, 8)):
    """
    Plot ROC curves for multiple models.
    
    Args:
        y_true: True labels
        y_scores: Dictionary of {model_name: prediction_scores}
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    for model_name, scores in y_scores.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax


def plot_cn0_by_channel(cn0_values: Dict[str, List[float]], 
                       time_axis: Optional[np.ndarray] = None,
                       title: str = "C/N0 Over Time by Channel",
                       save_path: Optional[str] = None,
                       figsize: tuple = (14, 6)):
    """
    Plot C/N0 values over time for different channels/PRNs.
    
    Args:
        cn0_values: Dictionary of {channel_name: [cn0_values]}
        time_axis: Optional time axis (seconds)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    for channel_name, values in cn0_values.items():
        if time_axis is None:
            x = np.arange(len(values))
            xlabel = 'Sample Index'
        else:
            x = time_axis[:len(values)]
            xlabel = 'Time (s)'
        
        ax.plot(x, values, linewidth=2, marker='o', markersize=4, label=channel_name)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('C/N0 (dB-Hz)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax


def plot_signal_spectrum(signal: np.ndarray, fs: float,
                        title: str = "Signal Spectrum",
                        save_path: Optional[str] = None,
                        figsize: tuple = (12, 6)):
    """
    Plot signal spectrum (FFT).
    
    Args:
        signal: Complex signal
        fs: Sampling frequency (Hz)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Compute FFT
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    
    # Plot positive frequencies only
    mask = freqs > 0
    ax.semilogy(freqs[mask]/1e6, np.abs(fft_vals[mask]), 'b-', linewidth=0.8)
    
    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax
