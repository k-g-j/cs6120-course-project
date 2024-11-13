import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.models.advanced_models import AdvancedModels
from src.visualization.model_evaluation import create_visualizations, save_model_artifacts


def setup_logging():
    """Set up logging configuration."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'advanced_model_training_{timestamp}.log')

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


def collect_predictions(model_predictions, actual_values, timestamps):
    """Collect predictions and actual values for visualization."""
    return pd.DataFrame({
        'timestamp': timestamps,
        'actual': actual_values,
        'predicted': model_predictions,
    })


def train_and_evaluate_advanced_models(data, config):
    """Train and evaluate advanced models using time series cross-validation."""
    logging.info("Starting advanced model training...")

    tscv = TimeSeriesSplit(n_splits=5)
    all_metrics = []
    all_predictions = []
    best_model = None
    best_r2 = -float('inf')
    feature_importance_data = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(data), 1):
        logging.info(f"Training fold {fold}/5...")

        try:
            # Split data
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            # Initialize and train advanced models
            advanced = AdvancedModels(train_data, test_data, target_col='kWh')
            advanced.prepare_data()

            # Train models and get metrics
            fold_metrics = advanced.train_models()

            for name, metrics in fold_metrics.items():
                model = advanced.models[name]
                predictions = advanced.predictions[name]

                # Add fold number to metrics
                metrics['fold'] = fold
                all_metrics.append(metrics)

                # Collect predictions
                fold_predictions = collect_predictions(
                    predictions['test'],
                    advanced.y_test,
                    test_data.index
                )
                fold_predictions['model'] = name
                fold_predictions['fold'] = fold
                all_predictions.append(fold_predictions)

                # Save model artifacts
                model_dir = str(config.MODEL_DIR) if hasattr(config, 'MODEL_DIR') else 'models'
                save_model_artifacts(model, name, fold, metrics, output_dir=model_dir)

                # Track best model
                if metrics['r2'] > best_r2:
                    best_r2 = metrics['r2']
                    best_model = (name, model)
                    if hasattr(model, 'feature_importances_'):
                        feature_importance_data = pd.DataFrame({
                            'feature': advanced.feature_cols,
                            'importance': model.feature_importances_
                        })

        except Exception as e:
            logging.error(f"Error in fold {fold}: {str(e)}")
            continue

    if not all_predictions:
        raise ValueError("No models were successfully trained")

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    # Combine all predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
    predictions_df.set_index('timestamp', inplace=True)

    # Create visualizations
    vis_dir = str(config.VISUALIZATIONS_DIR) if hasattr(config,
                                                        'VISUALIZATIONS_DIR') else 'visualizations'
    create_visualizations(
        metrics_df,
        predictions_df,
        feature_importance_data,
        output_dir=vis_dir
    )

    # Save metrics
    results_dir = str(config.RESULTS_DIR) if hasattr(config, 'RESULTS_DIR') else 'model_results'
    os.makedirs(results_dir, exist_ok=True)
    metrics_file = os.path.join(results_dir, 'advanced_metrics.csv')
    metrics_df.to_csv(metrics_file, index=False)

    return metrics_df, best_model


def main():
    setup_logging()

    try:
        # Create necessary directories
        for dir_name in ['visualizations', 'models', 'reports', 'model_results']:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

        # Load data
        solar_data = load_processed_data()

        # Create minimal config
        class Config:
            def __init__(self):
                self.MODEL_DIR = Path('models')
                self.RESULTS_DIR = Path('model_results')
                self.VISUALIZATIONS_DIR = Path('visualizations')
                self.REPORTS_DIR = Path('reports')

        config = Config()

        # Train and evaluate advanced models
        metrics_df, (best_model_name, best_model) = train_and_evaluate_advanced_models(solar_data,
                                                                                       config)

        # Log summary metrics
        logging.info("\nAverage metrics across folds:")
        summary = metrics_df.groupby('model_name').mean()
        logging.info("\n" + str(summary))

        # Calculate best R² from metrics
        best_r2 = metrics_df.loc[metrics_df['model_name'] == best_model_name, 'r2'].max()

        # Log best model
        logging.info(f"\nBest model: {best_model_name} (R² = {best_r2:.4f})")

        # Save best model separately
        save_model_artifacts(
            best_model,
            f"{best_model_name}_best",
            'final',
            {'r2': best_r2},
            output_dir='models'
        )

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
