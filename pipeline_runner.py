import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import CONFIG
from src.data_preprocessing import SolarDataPreprocessor
from src.models.advanced_models import PreprocessedData
from src.train_advanced_models import train_and_evaluate_advanced_models
from src.train_models import train_and_evaluate_baseline_models
from src.visualization.model_evaluation import generate_model_report


class PipelineConfig:
    """Pipeline configuration and directory setup."""

    def __init__(self):
        # Directory paths
        self.BASE_DIR = Path.cwd()
        self.DATA_DIR = self.BASE_DIR / 'data'
        self.PROCESSED_DIR = self.BASE_DIR / 'processed_data'
        self.MODEL_DIR = self.BASE_DIR / 'models'
        self.RESULTS_DIR = self.BASE_DIR / 'model_results'
        self.REPORTS_DIR = self.BASE_DIR / 'reports'
        self.VISUALIZATIONS_DIR = self.BASE_DIR / 'visualizations'
        self.LOGS_DIR = self.BASE_DIR / 'logs'

        # Create directories
        self._create_directories()

        # Update CONFIG with new paths
        self._update_config()

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.PROCESSED_DIR,
            self.MODEL_DIR,
            self.RESULTS_DIR,
            self.REPORTS_DIR,
            self.VISUALIZATIONS_DIR,
            self.LOGS_DIR
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _update_config(self):
        """Update CONFIG dictionary with path information."""
        CONFIG.update({
            'processed_dir': str(self.PROCESSED_DIR),
            'model_dir': str(self.MODEL_DIR),
            'results_dir': str(self.RESULTS_DIR),
            'reports_dir': str(self.REPORTS_DIR),
            'visualizations_dir': str(self.VISUALIZATIONS_DIR),
            'logs_dir': str(self.LOGS_DIR)
        })


def setup_logging(config):
    """Set up logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(config.LOGS_DIR) / f'pipeline_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def analyze_dataset(df, name):
    """Perform basic analysis on a dataset."""
    return {
        'shape': df.shape,
        'missing_values': df.isnull().sum(),
        'descriptive_stats': df.describe()
    }


def save_analysis_results(analysis_results, output_file):
    """Save analysis results to Excel."""
    with pd.ExcelWriter(output_file) as writer:
        for dataset_name, results in analysis_results.items():
            sheet_base = dataset_name[:25]

            pd.DataFrame(results['descriptive_stats']).to_excel(
                writer, sheet_name=f'{sheet_base}_stats')
            pd.DataFrame(results['missing_values']).to_excel(
                writer, sheet_name=f'{sheet_base}_miss')


def generate_summary_report(metrics, config):
    """Generate a summary report of the entire pipeline."""
    report_file = Path(config.REPORTS_DIR) / 'pipeline_summary.md'

    with open(report_file, 'w') as f:
        f.write("# Solar Energy Production Prediction Pipeline Summary\n\n")

        # Model Performance Summary
        f.write("## Model Performance Summary\n\n")
        summary = metrics.groupby(['model_type', 'model_name']).mean()
        f.write(summary.to_markdown())

        # Best Model Details
        f.write("\n\n## Best Model Performance\n\n")
        best_model_metrics = metrics.loc[metrics['r2'].idxmax()]
        f.write(f"- Model: {best_model_metrics['model_name']}\n")
        f.write(f"- Type: {best_model_metrics['model_type']}\n")
        f.write(f"- R²: {best_model_metrics['r2']:.4f}\n")
        f.write(f"- RMSE: {best_model_metrics['rmse']:.4f}\n")

        logging.info(f"Summary report generated: {report_file}")


def prepare_data_for_modeling(solar_data):
    """Prepare data for modeling by ensuring proper datetime index."""
    preprocessed = PreprocessedData(solar_data)
    return preprocessed.get_data()


def main():
    """Run the complete pipeline."""
    config = PipelineConfig()
    setup_logging(config)

    try:
        logging.info("Starting pipeline...")

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
        logging.info(f"Date range: {solar_data.index.min()} to {solar_data.index.max()}")

        # Train models
        logging.info("Training models...")

        # Train and evaluate baseline models
        logging.info("Training baseline models...")
        baseline_metrics = train_and_evaluate_baseline_models(solar_data)
        baseline_metrics['model_type'] = 'baseline'

        # Train and evaluate advanced models
        logging.info("Training advanced models...")
        advanced_metrics, best_model = train_and_evaluate_advanced_models(solar_data, config)
        advanced_metrics['model_type'] = 'advanced'

        # Combine metrics
        metrics_df = pd.concat([
            pd.DataFrame(baseline_metrics),
            pd.DataFrame(advanced_metrics)
        ])

        # Save metrics
        metrics_file = Path(config.RESULTS_DIR) / 'model_metrics.csv'
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
        logging.info(f"MAE: {float(best_model_metrics['mae']):.4f}")
        if 'mape' in best_model_metrics:
            logging.info(f"MAPE: {float(best_model_metrics['mape']):.4f}")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
