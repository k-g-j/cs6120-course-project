import logging
from datetime import datetime

import pandas as pd

from config import CONFIG
from pipeline_runner import PipelineConfig
from src.ablation_studies import AblationStudy
from src.data_preprocessing import SolarDataPreprocessor


def setup_logging(config):
    """Set up logging for ablation studies."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOGS_DIR / f'ablation_studies_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def run_and_save_study(study_func, study_name, results_dir, ablation):
    """Run a specific ablation study and save results."""
    logging.info(f"\nRunning {study_name} ablation study...")
    try:
        results = study_func()
        results.to_csv(results_dir / f'{study_name}_results.csv', index=False)
        logging.info(f"Completed {study_name} study. Results saved.")
        return results
    except Exception as e:
        logging.error(f"Error in {study_name} study: {str(e)}")
        return None


def main():
    """Run all ablation studies."""
    # Initialize configuration
    config = PipelineConfig()
    setup_logging(config)

    try:
        logging.info("Starting ablation studies...")

        # Load and preprocess data
        preprocessor = SolarDataPreprocessor(output_dir=str(config.PROCESSED_DIR))
        processed_data = preprocessor.process_all_datasets(CONFIG)
        solar_data = processed_data['solar_production']

        # Create results directory
        results_dir = config.RESULTS_DIR / 'ablation_studies'
        results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ablation study
        ablation = AblationStudy(solar_data, config)

        # Run all studies
        study_results = {}

        # 1. Input Dimension Ablation
        study_results['input_dimension'] = run_and_save_study(
            ablation.run_input_dimension_ablation,
            'input_dimension',
            results_dir,
            ablation
        )

        # 2. Preprocessing Ablation
        study_results['preprocessing'] = run_and_save_study(
            ablation.run_preprocessing_ablation,
            'preprocessing',
            results_dir,
            ablation
        )

        # 3. Temporal Resolution Ablation
        study_results['temporal_resolution'] = run_and_save_study(
            ablation.run_temporal_resolution_ablation,
            'temporal_resolution',
            results_dir,
            ablation
        )

        # 4. Data Volume Ablation
        study_results['data_volume'] = run_and_save_study(
            ablation.run_data_volume_ablation,
            'data_volume',
            results_dir,
            ablation
        )

        # Save summary
        summary_data = []
        for study_name, results in study_results.items():
            if results is not None and 'r2' in results.columns:
                summary_data.append({
                    'study': study_name,
                    'max_r2': results['r2'].max(),
                    'min_r2': results['r2'].min(),
                    'impact': ((results['r2'].max() - results['r2'].min()) /
                               results['r2'].max() * 100)
                })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(results_dir / 'ablation_summary.csv', index=False)

        logging.info("All ablation studies completed successfully")

    except Exception as e:
        logging.error(f"Error in ablation studies: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
