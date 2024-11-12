import logging
import os
from datetime import datetime

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.models.baseline_models import BaselineModels


def setup_logging():
    """Set up logging configuration."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'model_training_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_processed_data():
    """Load processed solar production data."""
    logging.info("Loading processed solar production data...")

    solar_prod = pd.read_csv('processed_data/processed_solar_production.csv')

    # Convert date to datetime and set as index
    solar_prod['datetime'] = pd.to_datetime(solar_prod['date'])
    solar_prod.set_index('datetime', inplace=True)

    # Clean up columns
    if 'Unnamed: 0' in solar_prod.columns:
        solar_prod = solar_prod.drop('Unnamed: 0', axis=1)
    if 'date' in solar_prod.columns:
        solar_prod = solar_prod.drop('date', axis=1)

    logging.info(f"Data shape: {solar_prod.shape}")
    logging.info(f"Columns: {solar_prod.columns.tolist()}")
    logging.info(f"Date range: {solar_prod.index.min()} to {solar_prod.index.max()}")

    return solar_prod


def train_and_evaluate_baseline_models(data):
    """Train and evaluate baseline models using time series cross-validation."""
    logging.info("Starting baseline model training...")

    tscv = TimeSeriesSplit(n_splits=5)
    all_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(data), 1):
        logging.info(f"Training fold {fold}/5...")

        # Split data
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        # Initialize and train baseline models
        baseline = BaselineModels(train_data, test_data, target_col='kWh')
        baseline.prepare_data()

        # Train linear models
        fold_metrics = baseline.train_linear_models()

        # Add fold number to metrics
        for model_metrics in fold_metrics.values():
            model_metrics['fold'] = fold
            all_metrics.append(model_metrics)

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    # Save metrics
    os.makedirs('model_results', exist_ok=True)
    metrics_df.to_csv('model_results/baseline_metrics.csv', index=False)

    return metrics_df


def main():
    # Set up logging
    setup_logging()

    try:
        # Create results directory
        os.makedirs('model_results', exist_ok=True)

        # Load data
        logging.info("Loading processed data...")
        solar_data = load_processed_data()

        # Train and evaluate baseline models
        metrics_df = train_and_evaluate_baseline_models(solar_data)

        # Log summary metrics
        logging.info("\nAverage metrics across folds:")
        summary = metrics_df.groupby('model_name').mean()
        logging.info("\n" + str(summary))

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
