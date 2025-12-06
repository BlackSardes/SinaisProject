"""Signal I/O utilities for loading signals from various formats."""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
from pathlib import Path


def load_signal(path: Union[str, Path]) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Load signal from file (.mat or .csv).
    
    Parameters
    ----------
    path : str or Path
        Path to the signal file (.mat or .csv)
    
    Returns
    -------
    signal : np.ndarray or tuple
        For .mat files: tuple of (signal, metadata)
        For .csv files: signal as numpy array
    
    Raises
    ------
    ValueError
        If file format is not supported
    FileNotFoundError
        If file does not exist
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    suffix = path.suffix.lower()
    
    if suffix == '.mat':
        try:
            from scipy.io import loadmat
            mat_data = loadmat(str(path))
            # Try to find the main signal array
            # Common keys: 'signal', 'data', 'sig', 'x'
            possible_keys = ['signal', 'data', 'sig', 'x', 'iq_data']
            signal_key = None
            
            for key in possible_keys:
                if key in mat_data:
                    signal_key = key
                    break
            
            if signal_key is None:
                # Get first non-metadata key
                non_meta_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                if not non_meta_keys:
                    raise ValueError("No valid signal data found in .mat file")
                signal_key = non_meta_keys[0]
            
            signal = mat_data[signal_key]
            metadata = {k: v for k, v in mat_data.items() if k.startswith('__') or k != signal_key}
            
            return signal.squeeze(), metadata
        except ImportError:
            raise ImportError("scipy is required to load .mat files")
    
    elif suffix == '.csv':
        df = pd.read_csv(path)
        # Assume first column is the signal, or use all numeric columns
        if df.shape[1] == 1:
            return df.values.squeeze()
        else:
            # Check if there are I/Q columns
            if 'I' in df.columns and 'Q' in df.columns:
                return (df['I'].values + 1j * df['Q'].values)
            # Otherwise return all numeric data
            return df.select_dtypes(include=[np.number]).values
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .mat, .csv")
