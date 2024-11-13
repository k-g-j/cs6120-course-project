from scipy.stats import randint, uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def get_hyperparameter_grids():
    """Define hyperparameter search spaces for each model"""

    # Grid search parameters for ensemble
    ensemble_params = {
        'n_folds': [3, 5, 7]
    }

    # Random search parameters for other models
    rf_params = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(5, 30),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    }

    gb_params = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4)
    }

    lstm_params = {
        'units': randint(20, 100),
        'dropout': uniform(0.1, 0.5),
        'batch_size': [16, 32, 64],
        'epochs': [50, 100, 150]
    }

    return {
        'ensemble': ensemble_params,
        'random_forest': rf_params,
        'gradient_boosting': gb_params,
        'lstm': lstm_params
    }


def tune_model_hyperparameters(model, param_grid, X_train, y_train, cv=5, n_iter=20):
    """Perform hyperparameter tuning using appropriate search strategy."""

    # Determine number of parameter combinations
    n_combinations = 1
    for param_values in param_grid.values():
        n_combinations *= len(param_values) if isinstance(param_values, list) else 10

    # Use GridSearchCV for small parameter spaces, RandomizedSearchCV for large ones
    if n_combinations <= n_iter:
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
    else:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1
        )

    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_, search.best_score_
