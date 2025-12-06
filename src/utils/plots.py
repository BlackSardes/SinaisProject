"""Plotting utilities for signal analysis and model evaluation."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
from pathlib import Path


def plot_signal(signal: np.ndarray, fs: float, 
               title: str = 'Signal', 
               save_path: Optional[Path] = None) -> None:
    """
    Plot time-domain signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Signal to plot
    fs : float
        Sampling frequency in Hz
    title : str
        Plot title
    save_path : Path, optional
        Path to save figure
    """
    t = np.arange(len(signal)) / fs
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    if np.iscomplexobj(signal):
        ax.plot(t, np.real(signal), label='Real', alpha=0.7)
        ax.plot(t, np.imag(signal), label='Imaginary', alpha=0.7)
        ax.legend()
    else:
        ax.plot(t, signal)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()


def plot_spectrum(signal: np.ndarray, fs: float,
                 title: str = 'Spectrum',
                 save_path: Optional[Path] = None) -> None:
    """
    Plot frequency spectrum of signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Signal to analyze
    fs : float
        Sampling frequency in Hz
    title : str
        Plot title
    save_path : Path, optional
        Path to save figure
    """
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    magnitude = np.abs(fft_result)
    
    # Plot only positive frequencies
    mask = freqs >= 0
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.semilogy(freqs[mask] / 1e6, magnitude[mask])
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Magnitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()


def plot_correlation(correlation: np.ndarray, fs: float,
                    title: str = 'Correlation Profile',
                    save_path: Optional[Path] = None) -> None:
    """
    Plot correlation profile.
    
    Parameters
    ----------
    correlation : np.ndarray
        Correlation result
    fs : float
        Sampling frequency in Hz
    title : str
        Plot title
    save_path : Path, optional
        Path to save figure
    """
    t = np.arange(len(correlation)) / fs
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t * 1e6, np.abs(correlation))  # Time in microseconds
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Correlation Magnitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, labels: List[str],
                         title: str = 'Confusion Matrix',
                         save_path: Optional[Path] = None) -> None:
    """
    Plot confusion matrix as heatmap.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    labels : List[str]
        Class labels
    title : str
        Plot title
    save_path : Path, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=labels, yticklabels=labels,
               ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importances: np.ndarray, feature_names: List[str],
                           top_n: int = 10,
                           title: str = 'Feature Importance',
                           save_path: Optional[Path] = None) -> None:
    """
    Plot feature importance from tree-based models.
    
    Parameters
    ----------
    importances : np.ndarray
        Feature importances
    feature_names : List[str]
        Names of features
    top_n : int
        Number of top features to display
    title : str
        Plot title
    save_path : Path, optional
        Path to save figure
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importances[indices])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
