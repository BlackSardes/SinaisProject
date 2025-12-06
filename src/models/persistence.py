"""Model persistence utilities using joblib."""

import joblib
from pathlib import Path
from typing import Any, Union


def save_model(model: Any, path: Union[str, Path]) -> None:
    """
    Save model to disk using joblib.
    
    Parameters
    ----------
    model : Any
        Model object to save (typically sklearn Pipeline)
    path : str or Path
        Path where to save the model
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Union[str, Path]) -> Any:
    """
    Load model from disk using joblib.
    
    Parameters
    ----------
    path : str or Path
        Path to the saved model
    
    Returns
    -------
    model : Any
        Loaded model object
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    return joblib.load(path)
