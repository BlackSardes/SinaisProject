"""
<<<<<<< HEAD
Model persistence functions for saving and loading trained models.
"""
import joblib
from pathlib import Path
from typing import Any, Optional, Dict
import json


def save_model(
    model: Any,
    filepath: str,
    metadata: Optional[Dict] = None,
    compress: int = 3
) -> None:
    """
    Save trained model to disk using joblib.
    
    Args:
        model: Trained model to save
        filepath: Output file path (will add .pkl extension if missing)
        metadata: Optional metadata dictionary to save alongside model
        compress: Compression level (0-9, higher = more compression)
    
    Example:
        >>> save_model(model, 'models/rf_model.pkl', metadata={'accuracy': 0.95})
    """
    filepath = Path(filepath)
    
    # Add .pkl extension if missing
    if filepath.suffix not in ['.pkl', '.joblib']:
        filepath = filepath.with_suffix('.pkl')
    
    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, filepath, compress=compress)
    
    # Save metadata if provided
    if metadata is not None:
        meta_path = filepath.with_suffix('.json')
        with open(meta_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            metadata_serializable = _make_json_serializable(metadata)
            json.dump(metadata_serializable, f, indent=2)
    
    print(f"Model saved to: {filepath}")
    if metadata is not None:
        print(f"Metadata saved to: {meta_path}")


def load_model(filepath: str, load_metadata: bool = True) -> Any:
    """
    Load trained model from disk.
    
    Args:
        filepath: Path to saved model file
        load_metadata: Whether to also load metadata file if it exists
    
    Returns:
        Loaded model, or tuple of (model, metadata) if load_metadata=True
        and metadata file exists
        
    Example:
        >>> model = load_model('models/rf_model.pkl')
        >>> # or with metadata
        >>> model, metadata = load_model('models/rf_model.pkl', load_metadata=True)
    """
    filepath = Path(filepath)
    
    # Add .pkl extension if missing
    if filepath.suffix not in ['.pkl', '.joblib'] and not filepath.exists():
        filepath = filepath.with_suffix('.pkl')
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    # Load model
    model = joblib.load(filepath)
    
    # Load metadata if requested
    if load_metadata:
        meta_path = filepath.with_suffix('.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            return model, metadata
    
    return model


def _make_json_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
=======
Model persistence utilities.
"""
import joblib
import json
import os
from typing import Any, Dict, Optional


def save_model(model: Any, path: str, metadata: Optional[Dict] = None):
    """
    Save model to disk using joblib.
    
    Args:
        model: Trained model to save
        path: Save path (with .pkl or .joblib extension)
        metadata: Optional metadata to save alongside model
    """
    # Save model
    joblib.dump(model, path)
    print(f"Model saved to: {path}")
    
    # Save metadata if provided
    if metadata is not None:
        meta_path = path.rsplit('.', 1)[0] + '_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {meta_path}")


def load_model(path: str) -> Any:
    """
    Load model from disk.
    
    Args:
        path: Path to saved model
    
    Returns:
        Loaded model
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    model = joblib.load(path)
    print(f"Model loaded from: {path}")
    
    # Try to load metadata
    meta_path = path.rsplit('.', 1)[0] + '_metadata.json'
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        print(f"Metadata loaded from: {meta_path}")
        return model, metadata
    
    return model
>>>>>>> main
