"""
GPS Spoofing Classification Models Module

This module provides training, evaluation, and model management functions
for GPS spoofing detection using machine learning.
"""

from .training import train_model, train_with_smote
from .evaluation import evaluate_model, cross_validate_model
from .persistence import save_model, load_model
from .classifiers import get_classifier

__all__ = [
    'train_model',
    'train_with_smote',
    'evaluate_model',
    'cross_validate_model',
    'save_model',
    'load_model',
    'get_classifier',
]
