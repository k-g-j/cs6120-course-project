import logging

import pandas as pd

from src.config import BASE_DIR, DATA_DIR, PROCESSED_DIR, MODEL_DIR, RESULTS_DIR, REPORTS_DIR, \
    VISUALIZATIONS_DIR, LOGS_DIR


def prepare_data_for_modeling(solar_data):
    """Prepare data for modeling by ensuring proper datetime index."""
    try:
        # Ensure we have a datetime index
        if not isinstance(solar_data.index, pd.DatetimeIndex):
            if 'datetime' in solar_data.columns:
                solar_data.set_index('datetime', inplace=True)
            elif 'date' in solar_data.columns:
                solar_data['datetime'] = pd.to_datetime(solar_data['date'])
                solar_data.set_index('datetime', inplace=True)
                solar_data.drop('date', axis=1, errors='ignore', inplace=True)

        # Sort index
        solar_data.sort_index(inplace=True)

        # Remove any duplicate indices
        solar_data = solar_data[~solar_data.index.duplicated()]

        # Fill missing values if any
        solar_data = solar_data.ffill().bfill()
        logging.info(f"Prepared data shape: {solar_data.shape}")
        logging.info(f"Date range: {solar_data.index.min()} to {solar_data.index.max()}")

        return solar_data

    except Exception as e:
        logging.error(f"Error preparing data: {str(e)}")
        raise


class PipelineConfig:
    """Pipeline configuration and directory setup."""

    def __init__(self):
        # Base directories
        self.BASE_DIR = BASE_DIR
        self.DATA_DIR = DATA_DIR
        self.PROCESSED_DIR = PROCESSED_DIR
        self.MODEL_DIR = MODEL_DIR
        self.RESULTS_DIR = RESULTS_DIR
        self.REPORTS_DIR = REPORTS_DIR
        self.VISUALIZATIONS_DIR = VISUALIZATIONS_DIR
        self.LOGS_DIR = LOGS_DIR

        # Add checkpoint directory
        self.CHECKPOINT_DIR = self.MODEL_DIR / 'checkpoints'

        # Create all directories
        self._create_directories()

    def _create_directories(self):
        """Create all necessary directories with error handling."""
        directories = [
            # Base directories
            self.PROCESSED_DIR,
            self.MODEL_DIR,
            self.RESULTS_DIR,
            self.REPORTS_DIR,
            self.VISUALIZATIONS_DIR,
            self.LOGS_DIR,

            # Additional directories
            self.CHECKPOINT_DIR,
        ]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logging.info(f"✓ Created/verified directory: {directory}")
            except Exception as e:
                logging.error(f"✗ Error creating directory {directory}: {str(e)}")
                raise

    def prepare_data_for_modeling(solar_data):
        """Prepare data for modeling by ensuring proper datetime index."""
        try:
            # Ensure we have a datetime index
            if not isinstance(solar_data.index, pd.DatetimeIndex):
                if 'datetime' in solar_data.columns:
                    solar_data.set_index('datetime', inplace=True)
                elif 'date' in solar_data.columns:
                    solar_data['datetime'] = pd.to_datetime(solar_data['date'])
                    solar_data.set_index('datetime', inplace=True)
                    solar_data.drop('date', axis=1, errors='ignore', inplace=True)

            # Sort index
            solar_data.sort_index(inplace=True)

            # Remove any duplicate indices
            solar_data = solar_data[~solar_data.index.duplicated()]

            # Fill missing values using forward fill then backward fill
            solar_data = solar_data.ffill().bfill()  # Updated this line

            logging.info(f"Prepared data shape: {solar_data.shape}")
            logging.info(f"Date range: {solar_data.index.min()} to {solar_data.index.max()}")

            return solar_data

        except Exception as e:
            logging.error(f"Error preparing data: {str(e)}")
            raise
