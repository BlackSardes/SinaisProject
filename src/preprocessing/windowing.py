"""
Signal windowing and segmentation functions.
"""
import numpy as np
from typing import List, Tuple, Optional


def window_segment(
    signal: np.ndarray,
    fs: float,
    window_s: float,
    hop_s: float,
    return_indices: bool = False
) -> List[np.ndarray]:
    """
    Segment signal into overlapping windows.
    
    Divides a long signal into shorter segments with optional overlap,
    which is useful for processing signals in chunks and for data augmentation.
    
    Args:
        signal: Input signal array
        fs: Sampling frequency in Hz
        window_s: Window duration in seconds
        hop_s: Hop size (time between window starts) in seconds
        return_indices: If True, return (segments, start_indices)
    
    Returns:
        List of signal segments, or tuple of (segments, indices) if return_indices=True
        
    Example:
        >>> # Create 1-second windows with 0.5-second hop (50% overlap)
        >>> segments = window_segment(signal, fs=5e6, window_s=1.0, hop_s=0.5)
        >>> print(len(segments))
        119  # For a 60-second signal
    """
    window_samples = int(window_s * fs)
    hop_samples = int(hop_s * fs)
    
    segments = []
    indices = []
    
    start = 0
    while start + window_samples <= len(signal):
        segments.append(signal[start:start + window_samples])
        indices.append(start)
        start += hop_samples
    
    if return_indices:
        return segments, indices
    return segments


def align_channels(
    signals: List[np.ndarray],
    method: str = 'correlation'
) -> List[np.ndarray]:
    """
    Align multiple signal channels by removing time offsets.
    
    Useful when processing multi-antenna or multi-frequency data
    where signals may have different time delays.
    
    Args:
        signals: List of signal arrays to align
        method: Alignment method
            - 'correlation': Cross-correlation based alignment (default)
            - 'energy': Align by first energy peak
    
    Returns:
        List of aligned signals (same length as input)
        
    Example:
        >>> # Align two antenna channels
        >>> aligned = align_channels([antenna1, antenna2])
    """
    if len(signals) < 2:
        return signals
    
    # Use first signal as reference
    reference = signals[0]
    aligned = [reference]
    
    for signal in signals[1:]:
        if method == 'correlation':
            # Find delay using cross-correlation
            corr = np.correlate(np.abs(reference), np.abs(signal), mode='full')
            delay = len(signal) - 1 - np.argmax(corr)
            
            # Shift signal to align
            if delay > 0:
                aligned_signal = np.concatenate([np.zeros(delay, dtype=signal.dtype), signal])
            elif delay < 0:
                aligned_signal = signal[-delay:]
            else:
                aligned_signal = signal
            
            # Trim to reference length
            aligned_signal = aligned_signal[:len(reference)]
            
        elif method == 'energy':
            # Find first significant energy peak in each signal
            ref_energy = np.abs(reference) ** 2
            sig_energy = np.abs(signal) ** 2
            
            threshold_ref = 0.1 * np.max(ref_energy)
            threshold_sig = 0.1 * np.max(sig_energy)
            
            ref_start = np.argmax(ref_energy > threshold_ref)
            sig_start = np.argmax(sig_energy > threshold_sig)
            
            delay = sig_start - ref_start
            
            # Shift and trim
            if delay > 0:
                aligned_signal = signal[delay:]
            elif delay < 0:
                aligned_signal = np.concatenate([np.zeros(-delay, dtype=signal.dtype), signal])
            else:
                aligned_signal = signal
            
            aligned_signal = aligned_signal[:len(reference)]
        
        else:
            raise ValueError(f"Unknown alignment method: {method}")
        
        # Pad if necessary
        if len(aligned_signal) < len(reference):
            pad_length = len(reference) - len(aligned_signal)
            aligned_signal = np.concatenate([
                aligned_signal,
                np.zeros(pad_length, dtype=signal.dtype)
            ])
        
        aligned.append(aligned_signal)
    
    return aligned
