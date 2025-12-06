"""
Visualization functions for GPS spoofing detection analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd


# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: str = 'results',
    formats: List[str] = ['png'],
    dpi: int = 300,
    bbox_inches: str = 'tight'
) -> None:
    """
    Save figure in high resolution.
    
    Args:
        fig: Matplotlib figure object
        filename: Output filename (without extension)
        output_dir: Output directory
        formats: List of formats to save (['png'], ['pdf'], ['png', 'pdf'])
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box setting ('tight' or None)
    
    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> save_figure(fig, 'my_plot', formats=['png', 'pdf'])
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        filepath = output_path / f"{filename}.{fmt}"
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, format=fmt)
        print(f"Figure saved: {filepath}")


def plot_correlation_profile(
    corr_magnitude: np.ndarray,
    fs: float = 5e6,
    ca_chip_rate: float = 1.023e6,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    highlight_peak: bool = True
) -> plt.Figure:
    """
    Plot correlation profile with peak annotation.
    
    Args:
        corr_magnitude: Magnitude of correlation
        fs: Sampling frequency in Hz
        ca_chip_rate: C/A code chip rate
        title: Plot title
        ax: Matplotlib axes (None = create new figure)
        highlight_peak: Whether to highlight peak position
    
    Returns:
        Figure object
        
    Example:
        >>> fig = plot_correlation_profile(corr_mag, fs=5e6)
        >>> save_figure(fig, 'correlation_profile')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure
    
    # Convert samples to chips
    samples_per_chip = fs / ca_chip_rate
    chips = np.arange(len(corr_magnitude)) / samples_per_chip
    
    # Plot correlation
    ax.plot(chips, corr_magnitude, 'b-', linewidth=1)
    ax.set_xlabel('Code Phase (chips)')
    ax.set_ylabel('Correlation Magnitude')
    ax.set_title(title or 'GPS Correlation Profile')
    ax.grid(True, alpha=0.3)
    
    # Highlight peak
    if highlight_peak:
        peak_idx = np.argmax(corr_magnitude)
        peak_chip = peak_idx / samples_per_chip
        peak_val = corr_magnitude[peak_idx]
        
        ax.plot(peak_chip, peak_val, 'r*', markersize=15, label='Peak')
        ax.axvline(peak_chip, color='r', linestyle='--', alpha=0.5)
        ax.legend()
    
    return fig


def plot_feature_distributions(
    features_df: pd.DataFrame,
    label_column: str = 'label',
    feature_columns: Optional[List[str]] = None,
    class_names: Optional[Dict[int, str]] = None,
    ncols: int = 3
) -> plt.Figure:
    """
    Plot feature distributions by class.
    
    Args:
        features_df: DataFrame with features and labels
        label_column: Name of label column
        feature_columns: Features to plot (None = all numeric columns)
        class_names: Dictionary mapping labels to names
        ncols: Number of columns in subplot grid
    
    Returns:
        Figure object
        
    Example:
        >>> fig = plot_feature_distributions(
        ...     df, label_column='label',
        ...     class_names={0: 'Authentic', 1: 'Spoofed'}
        ... )
    """
    # Select features
    if feature_columns is None:
        feature_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
        if label_column in feature_columns:
            feature_columns.remove(label_column)
    
    n_features = len(feature_columns)
    nrows = (n_features + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.array(axes).flatten()
    
    # Map labels to names
    if class_names is None:
        class_names = {label: f"Class {label}" for label in features_df[label_column].unique()}
    
    # Plot each feature
    for idx, feature in enumerate(feature_columns):
        ax = axes[idx]
        
        for label in sorted(features_df[label_column].unique()):
            data = features_df[features_df[label_column] == label][feature]
            ax.hist(data, bins=30, alpha=0.6, label=class_names.get(label, str(label)))
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {feature}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    cmap: str = 'Blues',
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        normalize: Whether to normalize values
        title: Plot title
        cmap: Color map
        ax: Matplotlib axes
    
    Returns:
        Figure object
        
    Example:
        >>> from sklearn.metrics import confusion_matrix
        >>> cm = confusion_matrix(y_test, y_pred)
        >>> fig = plot_confusion_matrix(cm, class_names=['Authentic', 'Spoofed'])
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap=cmap,
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)
    
    return fig


def plot_roc_curves(
    roc_data: Dict[str, Dict[str, np.ndarray]],
    title: str = 'ROC Curves',
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot ROC curves for multiple models or classes.
    
    Args:
        roc_data: Dictionary of {name: {'fpr': fpr, 'tpr': tpr, 'auc': auc}}
        title: Plot title
        ax: Matplotlib axes
    
    Returns:
        Figure object
        
    Example:
        >>> roc_data = {
        ...     'RandomForest': {'fpr': fpr1, 'tpr': tpr1, 'auc': 0.95},
        ...     'SVM': {'fpr': fpr2, 'tpr': tpr2, 'auc': 0.93}
        ... }
        >>> fig = plot_roc_curves(roc_data)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')
    
    # Plot each ROC curve
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))
    
    for idx, (name, data) in enumerate(roc_data.items()):
        fpr = data['fpr']
        tpr = data['tpr']
        auc = data.get('auc', np.trapz(tpr, fpr))
        
        ax.plot(fpr, tpr, color=colors[idx], linewidth=2,
                label=f'{name} (AUC={auc:.3f})')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return fig


def plot_cn0_by_channel(
    cn0_data: Dict[str, np.ndarray],
    time_points: Optional[np.ndarray] = None,
    title: str = 'C/N0 by Channel',
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot C/N0 time series for multiple channels/PRNs.
    
    Args:
        cn0_data: Dictionary of {channel_name: cn0_values}
        time_points: Time points for x-axis (None = sample index)
        title: Plot title
        ax: Matplotlib axes
    
    Returns:
        Figure object
        
    Example:
        >>> cn0_data = {'PRN 1': cn0_prn1, 'PRN 5': cn0_prn5}
        >>> fig = plot_cn0_by_channel(cn0_data, time_points=times)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(cn0_data)))
    
    for idx, (channel, cn0_vals) in enumerate(cn0_data.items()):
        if time_points is None:
            x = np.arange(len(cn0_vals))
            xlabel = 'Sample Index'
        else:
            x = time_points[:len(cn0_vals)]
            xlabel = 'Time (s)'
        
        ax.plot(x, cn0_vals, color=colors[idx], linewidth=2,
                marker='o', markersize=4, label=channel)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('C/N0 (dB-Hz)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_feature_importance(
    importance_data: Dict[str, float],
    top_n: int = 20,
    title: str = 'Feature Importance',
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot feature importance as horizontal bar chart.
    
    Args:
        importance_data: Dictionary of {feature_name: importance_score}
        top_n: Number of top features to show
        title: Plot title
        ax: Matplotlib axes
    
    Returns:
        Figure object
        
    Example:
        >>> importance = {'corr_peak_height': 0.25, 'temp_cn0': 0.18, ...}
        >>> fig = plot_feature_importance(importance, top_n=15)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    else:
        fig = ax.figure
    
    # Sort by importance
    sorted_items = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:top_n]
    
    features = [item[0] for item in top_items]
    importances = [item[1] for item in top_items]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # Top feature at top
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    return fig
