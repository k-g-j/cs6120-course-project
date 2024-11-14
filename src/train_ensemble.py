import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from src.models.advanced_ensemble import StackedEnsembleRegressor


def create_ensemble_features(data):
    """Create features needed for ensemble model."""
    df = data.copy()

    # Basic time components
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month

    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    # Time of day features
    df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
    df['is_peak_sun'] = ((df['hour'] >= 10) & (df['hour'] <= 14)).astype(int)
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

    # Lag features
    df['kWh_lag_1h'] = df['kWh'].shift(1)
    df['kWh_lag_24h'] = df['kWh'].shift(24)
    df['kWh_lag_168h'] = df['kWh'].shift(168)  # 1 week

    # Rolling statistics
    df['kWh_rolling_mean_24h'] = df['kWh'].rolling(window=24, min_periods=1).mean()
    df['kWh_rolling_std_24h'] = df['kWh'].rolling(window=24, min_periods=1).std()
    df['kWh_rolling_max_24h'] = df['kWh'].rolling(window=24, min_periods=1).max()

    # Drop temporary columns
    df = df.drop(['hour', 'day', 'month'], axis=1)

    # Fill NaN values using forward fill then backward fill
    df = df.ffill().bfill()

    return df


def train_and_evaluate_ensemble(solar_data, config):
    """Train and evaluate the stacked ensemble model."""
    logging.info("Training stacked ensemble model...")

    # Create features for ensemble model
    logging.info("Creating ensemble features...")
    data = create_ensemble_features(solar_data)

    # Define feature columns
    feature_columns = [
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'day_sin', 'day_cos', 'is_weekend', 'is_daytime',
        'is_peak_sun', 'kWh_lag_1h', 'kWh_lag_24h',
        'kWh_lag_168h', 'kWh_rolling_mean_24h',
        'kWh_rolling_std_24h', 'kWh_rolling_max_24h'
    ]

    # Verify all required columns exist
    missing_cols = [col for col in feature_columns if col not in data.columns]
    if missing_cols:
        logging.error(f"Missing required columns: {missing_cols}")
        return None, float('inf')

    # Prepare features and target
    X = data[feature_columns]
    y = data['kWh']

    # Check for any remaining non-numeric columns
    non_numeric_cols = X.select_dtypes(exclude=['float64', 'int64']).columns
    if len(non_numeric_cols) > 0:
        logging.error(f"Non-numeric columns found: {non_numeric_cols}")
        return None, float('inf')

    logging.info(f"Feature columns used for ensemble: {feature_columns}")
    logging.info(f"Features shape: {X.shape}")

    try:
        # Train with different numbers of folds
        results = []
        for n_folds in [3, 5]:
            logging.info(f"Training ensemble with {n_folds} folds...")
            ensemble = StackedEnsembleRegressor(n_folds=n_folds)
            ensemble.fit(X, y)
            predictions = ensemble.predict(X)

            # Calculate metrics
            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)
            results.append((ensemble, mse, r2, n_folds))

            logging.info(f"Ensemble with {n_folds} folds - MSE: {mse:.4f}, R²: {r2:.4f}")

        # Select best model based on MSE
        best_ensemble, best_mse, best_r2, best_folds = min(results, key=lambda x: x[1])
        logging.info(
            f"Best ensemble model has {best_folds} folds - MSE: {best_mse:.4f}, R²: {best_r2:.4f}")

        # Save results
        tuning_results = pd.DataFrame({
            'model': ['stacked_ensemble'],
            'best_score': [best_mse],
            'best_params': [{'n_folds': best_folds}]
        })
        tuning_results.to_csv(config.TUNING_RESULTS_DIR / 'ensemble_tuning.csv', index=False)

        return best_ensemble, best_mse

    except Exception as e:
        logging.error(f"Error in ensemble training: {str(e)}")
        return None, float('inf')
