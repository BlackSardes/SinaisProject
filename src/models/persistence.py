"""
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
