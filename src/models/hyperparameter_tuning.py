import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


def get_hyperparameter_grids():
    """Define hyperparameter grids for all models."""
    return {
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        },
        'gradient_boosting': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'linear_sgd': {
            'alpha': [0.0001, 0.001, 0.01],
            'l1_ratio': [0.15, 0.5, 0.85],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'max_iter': [1000, 2000],
            'tol': [1e-4, 1e-3]
        },
        'lstm': {
            'units': [32, 64, 128],
            'dropout': [0.1, 0.2, 0.3],
            'batch_size': [32, 64, 128],
            'learning_rate': [0.001, 0.0005, 0.0001],
            'sequence_length': [24, 48, 72]
        }
    }


def rmse_scorer(y_true, y_pred):
    """Calculate RMSE score for model evaluation."""
    return -np.sqrt(mean_squared_error(y_true, y_pred))


def tune_model_hyperparameters(
        model: Any,
        param_grid: Dict[str, Any],
        X: Any,
        y: Any,
        cv: int = 5,
        n_iter: int = 10
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Tune model hyperparameters using time series cross-validation.

    Args:
        model: Model instance to tune
        param_grid: Dictionary of parameters to tune
        X: Training features
        y: Target variable
        cv: Number of cross-validation folds
        n_iter: Number of iterations for randomized search

    Returns:
        Tuple containing:
        - Best model
        - Best parameters
        - Best score
    """
    try:
        # Set up time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv)

        # Define scoring metrics
        scoring = {
            'rmse': make_scorer(rmse_scorer),
            'r2': make_scorer(r2_score)
        }

        # Configure grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=tscv,
            scoring=scoring,
            refit='r2',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        # Fit grid search
        logging.info("Starting hyperparameter tuning...")
        grid_search.fit(X, y)

        # Log results
        logging.info(f"Best parameters: {grid_search.best_params_}")
        logging.info(f"Best R² score: {grid_search.best_score_:.4f}")

        # Save detailed CV results
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results.to_csv('model_results/hyperparameter_tuning/cv_results.csv', index=False)

        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

    except Exception as e:
        logging.error(f"Error in hyperparameter tuning: {str(e)}")
        raise


def validate_hyperparameters(model, params):
    """Validate hyperparameters are within acceptable ranges."""
    if 'batch_size' in params and params['batch_size'] <= 0:
        raise ValueError("batch_size must be positive")

    if 'learning_rate' in params and params['learning_rate'] <= 0:
        raise ValueError("learning_rate must be positive")

    if 'n_estimators' in params and params['n_estimators'] <= 0:
        raise ValueError("n_estimators must be positive")

    if hasattr(model, 'validate_params'):
        model.validate_params(params)


class HyperparameterTuningResult:
    """Store and analyze hyperparameter tuning results."""

    def __init__(self):
        self.results = []

    def add_result(self, model_name, params, scores, fold):
        """Add a tuning result."""
        self.results.append({
            'model_name': model_name,
            'params': params,
            'scores': scores,
            'fold': fold
        })

    def get_summary(self):
        """Get summary of tuning results."""
        summary = []
        for result in self.results:
            summary.append({
                'model_name': result['model_name'],
                'fold': result['fold'],
                'r2': result['scores'].get('r2', None),
                'rmse': result['scores'].get('rmse', None),
                'mae': result['scores'].get('mae', None),
                'best_params': str(result['params'])
            })
        return pd.DataFrame(summary)

    def save_results(self, filename):
        """Save tuning results to file."""
        summary = self.get_summary()
        summary.to_csv(filename, index=False)
        logging.info(f"Saved tuning results to {filename}")
