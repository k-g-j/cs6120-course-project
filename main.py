import logging
import os
from datetime import datetime

import pandas as pd

from config import CONFIG
from src.data_preprocessing import SolarDataPreprocessor


def setup_logging():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'preprocessing_{timestamp}.log')

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
    analysis = {
        'shape': df.shape,
        'missing_values': df.isnull().sum(),
        'descriptive_stats': df.describe()
    }
    return analysis


def save_analysis_results(analysis_results):
    """Save analysis results to Excel with shortened sheet names."""
    with pd.ExcelWriter('analysis_results.xlsx') as writer:
        for dataset_name, results in analysis_results.items():
            # Shorten sheet names to comply with Excel's 31-character limit
            sheet_base = dataset_name[:25]  # Leave room for suffix

            # Save descriptive statistics
            stats_sheet = f'{sheet_base}_stats'
            pd.DataFrame(results['descriptive_stats']).to_excel(
                writer, sheet_name=stats_sheet)

            # Save missing values analysis
            missing_sheet = f'{sheet_base}_miss'
            pd.DataFrame(results['missing_values']).to_excel(
                writer, sheet_name=missing_sheet)


def main():
    setup_logging()
    logging.info("Starting data preprocessing pipeline")

    try:
        # Initialize preprocessor
        preprocessor = SolarDataPreprocessor(output_dir=CONFIG['output_dir'])

        # Process all datasets
        logging.info("Processing datasets...")
        processed_data = preprocessor.process_all_datasets(CONFIG)

        # Perform analysis on processed data
        logging.info("Analyzing processed datasets...")
        analysis_results = {}

        # Analyze each dataset
        for category, data in processed_data.items():
            if isinstance(data, pd.DataFrame):
                analysis_results[category] = analyze_dataset(data, category)
            elif isinstance(data, dict):
                for sub_category, df in data.items():
                    analysis_results[f"{category}_{sub_category}"] = analyze_dataset(
                        df, f"{category}_{sub_category}")

        # Save analysis results
        logging.info("Saving analysis results...")
        save_analysis_results(analysis_results)

        logging.info("Data preprocessing and analysis completed successfully")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
