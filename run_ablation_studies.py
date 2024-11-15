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
        level=logging.DEBUG,  # Set to DEBUG level for more detailed logging
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main():
    """Run ablation studies on the solar prediction models."""
    # Initialize configuration
    config = PipelineConfig()
    setup_logging(config)

    try:
        logging.info("Starting ablation studies...")

        # Load and preprocess data
        preprocessor = SolarDataPreprocessor(output_dir=str(config.PROCESSED_DIR))
        processed_data = preprocessor.process_all_datasets(CONFIG)
        solar_data = processed_data['solar_production']

        # Log data information
        logging.debug(f"Loaded data shape: {solar_data.shape}")
        logging.debug(f"Data columns: {solar_data.columns.tolist()}")
        logging.debug(f"Data index type: {type(solar_data.index)}")
        logging.debug(f"First few rows:\n{solar_data.head()}")

        if not isinstance(solar_data.index, pd.DatetimeIndex):
            logging.info("Converting index to datetime...")
            if 'datetime' in solar_data.columns:
                solar_data.set_index('datetime', inplace=True)
            elif 'date' in solar_data.columns:
                solar_data['datetime'] = pd.to_datetime(solar_data['date'])
                solar_data.set_index('datetime', inplace=True)
                solar_data.drop('date', axis=1, errors='ignore', inplace=True)
            logging.debug(f"After conversion - index type: {type(solar_data.index)}")

        # Initialize ablation study
        ablation = AblationStudy(solar_data, config)

        # Run all studies
        results = ablation.run_all_studies()

        # Log completion
        logging.info("Ablation studies completed successfully")
        logging.info(f"Results saved in {config.RESULTS_DIR}/ablation_studies/")
        logging.info(f"Visualizations saved in {config.VISUALIZATIONS_DIR}/ablation_studies/")

    except Exception as e:
        logging.error(f"Error in ablation studies: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
