"""
Classifier definitions and factory functions.
"""
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def get_classifier(
    model_name: str = 'random_forest',
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42
):
    """
    Get classifier instance with specified parameters.
    
    Args:
        model_name: Name of classifier
            - 'random_forest': RandomForestClassifier (default)
            - 'svm': Support Vector Machine
            - 'mlp': Multi-Layer Perceptron
        params: Dictionary of classifier parameters (overrides defaults)
        random_state: Random seed for reproducibility
    
    Returns:
        Configured classifier instance
        
    Example:
        >>> clf = get_classifier('random_forest', {'n_estimators': 200})
        >>> clf.fit(X_train, y_train)
    """
    if params is None:
        params = {}
    
    # Default parameters for each classifier
    if model_name == 'random_forest':
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': 'balanced',  # Handle class imbalance
            'random_state': random_state,
            'n_jobs': -1,  # Use all CPU cores
        }
        default_params.update(params)
        return RandomForestClassifier(**default_params)
    
    elif model_name == 'svm':
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'class_weight': 'balanced',
            'random_state': random_state,
            'probability': True,  # Enable probability estimates for ROC curves
        }
        default_params.update(params)
        return SVC(**default_params)
    
    elif model_name == 'mlp':
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 'auto',
            'learning_rate': 'adaptive',
            'max_iter': 500,
            'random_state': random_state,
        }
        default_params.update(params)
        return MLPClassifier(**default_params)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                        f"Supported: 'random_forest', 'svm', 'mlp'")


def get_default_params(model_name: str) -> Dict[str, Any]:
    """
    Get default hyperparameters for a classifier.
    
    Useful for hyperparameter tuning as a starting point.
    
    Args:
        model_name: Name of classifier
    
    Returns:
        Dictionary of default parameters
        
    Example:
        >>> params = get_default_params('random_forest')
        >>> print(params['n_estimators'])
        100
    """
    if model_name == 'random_forest':
        return {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
        }
    elif model_name == 'svm':
        return {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'class_weight': 'balanced',
        }
    elif model_name == 'mlp':
        return {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'learning_rate': 'adaptive',
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_param_grid(model_name: str) -> Dict[str, list]:
    """
    Get hyperparameter grid for cross-validation search.
    
    Args:
        model_name: Name of classifier
    
    Returns:
        Dictionary suitable for GridSearchCV or RandomizedSearchCV
        
    Example:
        >>> from sklearn.model_selection import GridSearchCV
        >>> param_grid = get_param_grid('random_forest')
        >>> clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    """
    if model_name == 'random_forest':
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None],
        }
    elif model_name == 'svm':
        return {
            'kernel': ['rbf', 'linear'],
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'class_weight': ['balanced', None],
        }
    elif model_name == 'mlp':
        return {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")
