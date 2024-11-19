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
    log_file = config.LOGS_DIR / f'preprocessing_ablation_{timestamp}.log'

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main():
    """Run preprocessing ablation study."""
    config = PipelineConfig()
    setup_logging(config)

    try:
        logging.info("Starting preprocessing ablation study...")

        # Load and preprocess data
        preprocessor = SolarDataPreprocessor(output_dir=str(config.PROCESSED_DIR))
        processed_data = preprocessor.process_all_datasets(CONFIG)
        solar_data = processed_data['solar_production']

        if not isinstance(solar_data.index, pd.DatetimeIndex):
            logging.info("Converting index to datetime...")
            if 'datetime' in solar_data.columns:
                solar_data.set_index('datetime', inplace=True)
            elif 'date' in solar_data.columns:
                solar_data['datetime'] = pd.to_datetime(solar_data['date'])
                solar_data.set_index('datetime', inplace=True)
                solar_data.drop('date', axis=1, errors='ignore', inplace=True)

        # Initialize ablation study
        ablation = AblationStudy(solar_data, config)

        # Run only preprocessing ablation
        results = ablation.run_preprocessing_ablation()

        # Save results
        results_dir = config.RESULTS_DIR / 'ablation_studies'
        results_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(results_dir / 'preprocessing_results.csv', index=False)

        logging.info("Preprocessing ablation study completed successfully")

    except Exception as e:
        logging.error(f"Error in preprocessing ablation study: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
