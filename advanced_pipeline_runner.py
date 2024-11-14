import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG
from pipeline_runner import PipelineConfig, prepare_data_for_modeling
from src.data_preprocessing import SolarDataPreprocessor
from src.models.hyperparameter_tuning import get_hyperparameter_grids
from src.train_advanced_models import train_and_evaluate_advanced_models
from src.train_ensemble import train_and_evaluate_ensemble
from src.train_lstm import train_and_evaluate_lstm
from src.train_models import train_and_evaluate_baseline_models
from src.visualization.model_evaluation import generate_model_report


class AdvancedPipelineConfig(PipelineConfig):
    """Extended pipeline configuration for advanced models."""

    def __init__(self):
        super().__init__()
        self.ENSEMBLE_DIR = self.MODEL_DIR / 'ensemble'
        self.DEEP_LEARNING_DIR = self.MODEL_DIR / 'deep_learning'
        self.TUNING_RESULTS_DIR = self.RESULTS_DIR / 'hyperparameter_tuning'
        self.CHECKPOINT_DIR = self.MODEL_DIR / 'checkpoints'

        # Create additional directories
        self._create_advanced_directories()

    def _create_advanced_directories(self):
        """Create directories for advanced models."""
        advanced_directories = [
            self.ENSEMBLE_DIR,
            self.DEEP_LEARNING_DIR,
            self.TUNING_RESULTS_DIR,
            self.CHECKPOINT_DIR
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


def train_advanced_pipeline(solar_data, config):
    """Execute advanced training pipeline with hyperparameter tuning."""
    try:
        # Get hyperparameter grids
        param_grids = get_hyperparameter_grids()

        # Train baseline models first
        baseline_metrics = train_and_evaluate_baseline_models(solar_data)
        baseline_metrics['model_type'] = 'baseline'

        # Train advanced models with hyperparameter tuning
        advanced_metrics, best_standard_model = train_and_evaluate_advanced_models(
            solar_data,
            config,
            param_grids
        )
        advanced_metrics['model_type'] = 'advanced'

        # Train ensemble model
        ensemble_model, ensemble_mse = train_and_evaluate_ensemble(solar_data, config)

        # Train LSTM with tuning
        lstm_model, lstm_score = train_and_evaluate_lstm(
            solar_data,
            config,
            param_grids['lstm']
        )

        # Combine all metrics
        all_metrics = []
        all_metrics.append(pd.DataFrame(baseline_metrics))
        all_metrics.append(pd.DataFrame(advanced_metrics))

        # Add ensemble metrics
        if ensemble_model is not None:
            ensemble_metrics = pd.DataFrame([{
                'model_name': 'stacked_ensemble',
                'model_type': 'ensemble',
                'rmse': np.sqrt(ensemble_mse),
                'r2': 1 - ensemble_mse / np.var(solar_data['kWh']),
                'mae': np.nan,
                'mape': np.nan,
                'fold': 'all'
            }])
            all_metrics.append(ensemble_metrics)

        # Add LSTM metrics
        if lstm_model is not None:
            lstm_metrics = pd.DataFrame([{
                'model_name': 'lstm',
                'model_type': 'deep_learning',
                'rmse': np.sqrt(lstm_score),
                'r2': 1 - lstm_score / np.var(solar_data['kWh']),
                'mae': np.nan,
                'mape': np.nan,
                'fold': 'all'
            }])
            all_metrics.append(lstm_metrics)

        return pd.concat(all_metrics, ignore_index=True)

    except Exception as e:
        logging.error(f"Error in advanced training pipeline: {str(e)}")
        raise


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

        # Run advanced training pipeline
        metrics_df = train_advanced_pipeline(solar_data, config)

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
        logging.error(f"Advanced pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
