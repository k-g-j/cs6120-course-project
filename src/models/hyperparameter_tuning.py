import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


def get_hyperparameter_grids():
    """Define reduced hyperparameter grids for model tuning."""
    return {
        'random_forest': {
            'n_estimators': [50, 100],
            'max_depth': [10],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['sqrt']
        },
        'gradient_boosting': {
            'n_estimators': [100],
            'max_depth': [3],
            'learning_rate': [0.05],
            'subsample': [0.8],
            'min_samples_split': [2]
        },
        'linear_sgd': {
            'alpha': [0.0001],
            'l1_ratio': [0.15],
            'penalty': ['elasticnet'],
            'max_iter': [1000],
            'tol': [1e-3]
        },
        'lstm': {
            'units': [32],
            'dropout': [0.1],
            'batch_size': [32],
            'learning_rate': [0.001]
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
        cv: int = 3,
        n_iter: int = 10
) -> Tuple[Any, Dict[str, Any], float]:
    """Tune model hyperparameters using RandomizedSearchCV."""
    try:
        # Set up time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv)

        # Define scoring metrics
        scoring = {
            'rmse': make_scorer(rmse_scorer),
            'r2': make_scorer(r2_score)
        }

        # Calculate parameter combinations
        n_params = np.product([len(v) for v in param_grid.values()])
        n_iter = min(n_iter, n_params)

        # Configure randomized search with memory-efficient settings
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=tscv,
            scoring=scoring,
            refit='r2',
            n_jobs=2,  # Limit parallel jobs
            verbose=1,
            random_state=42,
            return_train_score=True,
            pre_dispatch='2*n_jobs',  # Limit number of pre-dispatched jobs
            error_score='raise'
        )

        # Clear memory before fitting
        import gc
        gc.collect()

        # Fit random search
        logging.info("Starting hyperparameter tuning...")
        random_search.fit(X, y)

        # Log results
        logging.info(f"Best parameters: {random_search.best_params_}")
        logging.info(f"Best R² score: {random_search.best_score_:.4f}")

        # Get best estimator and clear memory
        best_estimator = random_search.best_estimator_
        best_params = random_search.best_params_
        best_score = random_search.best_score_

        del random_search
        gc.collect()

        return best_estimator, best_params, best_score

    except Exception as e:
        logging.error(f"Error in hyperparameter tuning: {str(e)}")
        return None, None, None
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
