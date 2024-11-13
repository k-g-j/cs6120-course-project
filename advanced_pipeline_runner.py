import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

from config import CONFIG
from pipeline_runner import PipelineConfig, prepare_data_for_modeling
from src.data_preprocessing import SolarDataPreprocessor
from src.models.advanced_ensemble import StackedEnsembleRegressor
from src.models.deep_learning import LSTMRegressor
from src.models.hyperparameter_tuning import (
    get_hyperparameter_grids,
    tune_model_hyperparameters
)
from src.train_advanced_models import train_and_evaluate_advanced_models
from src.train_models import train_and_evaluate_baseline_models
from src.visualization.model_evaluation import generate_model_report


class AdvancedPipelineConfig(PipelineConfig):
    """Extended pipeline configuration for advanced models."""

    def __init__(self):
        super().__init__()
        self.ENSEMBLE_DIR = self.MODEL_DIR / 'ensemble'
        self.DEEP_LEARNING_DIR = self.MODEL_DIR / 'deep_learning'
        self.TUNING_RESULTS_DIR = self.RESULTS_DIR / 'hyperparameter_tuning'

        # Create additional directories
        self._create_advanced_directories()

    def _create_advanced_directories(self):
        """Create directories for advanced models."""
        advanced_directories = [
            self.ENSEMBLE_DIR,
            self.DEEP_LEARNING_DIR,
            self.TUNING_RESULTS_DIR
        ]

        for directory in advanced_directories:
            directory.mkdir(parents=True, exist_ok=True)


def setup_advanced_logging(config):
    """Set up logging for advanced pipeline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(config.LOGS_DIR) / f'advanced_pipeline_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


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
        logging.error("Falling back to best advanced model")
        return None, float('inf')


def train_and_evaluate_lstm(solar_data, config):
    """Train and evaluate the LSTM model."""
    logging.info("Training LSTM model...")

    try:
        # Check if 'kWh' is in columns
        if 'kWh' not in solar_data.columns:
            logging.error("Target column 'kWh' not found in data")
            return None, float('inf')

        # Get numeric columns except target
        numeric_data = solar_data.select_dtypes(include=['float64', 'int64'])

        # Drop location and any other object columns if they exist
        object_columns = solar_data.select_dtypes(include=['object']).columns
        if not object_columns.empty:
            logging.info(f"Dropping non-numeric columns: {list(object_columns)}")
            solar_data = solar_data.drop(columns=object_columns)

        # Split features and target
        X = numeric_data.drop('kWh', axis=1, errors='ignore')
        y = solar_data['kWh']

        logging.info(f"LSTM feature shape: {X.shape}")
        logging.info(f"LSTM target shape: {y.shape}")
        logging.info(f"Feature columns: {list(X.columns)}")

        # Initialize LSTM model
        lstm = LSTMRegressor(
            units=50,
            dropout=0.2,
            batch_size=32,
            epochs=100
        )

        # Get hyperparameter grid
        param_grids = get_hyperparameter_grids()

        # Tune hyperparameters
        logging.info("Starting hyperparameter tuning for LSTM...")
        tuned_lstm, best_params, best_score = tune_model_hyperparameters(
            lstm,
            param_grids['lstm'],
            X, y
        )

        logging.info(f"Best LSTM parameters: {best_params}")
        logging.info(f"Best LSTM score: {best_score}")

        return tuned_lstm, abs(best_score)

    except Exception as e:
        logging.error(f"Error in LSTM training: {str(e)}", exc_info=True)
        return None, float('inf')


def main():
    """Run the complete advanced pipeline."""
    config = AdvancedPipelineConfig()
    setup_advanced_logging(config)

    try:
        logging.info("Starting advanced pipeline...")

        # Run preprocessing
        logging.info("Running data preprocessing...")
        preprocessor = SolarDataPreprocessor(output_dir=str(config.PROCESSED_DIR))
        processed_data = preprocessor.process_all_datasets(CONFIG)

        # Extract and prepare solar production data
        solar_data = processed_data['solar_production']
        logging.info(f"Initial data shape: {solar_data.shape}")

        # Prepare data for modeling
        solar_data = prepare_data_for_modeling(solar_data)
        logging.info(f"Processed data shape: {solar_data.shape}")

        # Train baseline and standard advanced models
        baseline_metrics = train_and_evaluate_baseline_models(solar_data)
        baseline_metrics['model_type'] = 'baseline'

        advanced_metrics, best_standard_model = train_and_evaluate_advanced_models(
            solar_data,
            config
        )
        advanced_metrics['model_type'] = 'advanced'

        # Train ensemble model
        ensemble_model, ensemble_mse = train_and_evaluate_ensemble(solar_data, config)

        # Train LSTM model
        lstm_metrics = train_and_evaluate_lstm(solar_data, config)

        # Combine all metrics
        metrics_df = pd.concat([
            pd.DataFrame(baseline_metrics),
            pd.DataFrame(advanced_metrics),
            lstm_metrics
        ])

        if ensemble_model is not None:
            ensemble_metrics = pd.DataFrame([{
                'model_name': 'stacked_ensemble',
                'model_type': 'ensemble',
                'rmse': np.sqrt(ensemble_mse),
                'r2': 1 - ensemble_mse,
                'fold': 'all'
            }])
            metrics_df = pd.concat([metrics_df, ensemble_metrics])

        # Save metrics
        metrics_file = Path(config.RESULTS_DIR) / 'final_model_metrics.csv'
        metrics_df.to_csv(metrics_file, index=False)

        # Generate report
        generate_model_report(metrics_df, config)

        # Find best model across all models
        best_idx = metrics_df['r2'].idxmax()
        best_model_metrics = metrics_df.iloc[best_idx]

        # Log best model performance
        logging.info("\nBest model performance:")
        logging.info(
            f"Model: {best_model_metrics['model_name']} ({best_model_metrics['model_type']})"
        )
        logging.info(f"R²: {float(best_model_metrics['r2']):.4f}")
        logging.info(f"RMSE: {float(best_model_metrics['rmse']):.4f}")

    except Exception as e:
        logging.error(f"Advanced pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
