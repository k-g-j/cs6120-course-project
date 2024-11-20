import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.ablation_studies import AblationStudy
from src.config import CONFIG
from src.data_preprocessing import SolarDataPreprocessor
from src.pipeline_runner import PipelineConfig


def setup_logging(config):
    """Set up logging for ablation studies."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(config.LOGS_DIR) / f'preprocessing_ablation_{timestamp}.log'

    logging.basicConfig(
        level=logging.DEBUG,
        format=CONFIG['log_format'],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main():
    """Run preprocessing ablation study with enhanced error handling."""
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

        # Create results directory
        results_dir = Path(CONFIG['ablation_results'])
        results_dir.mkdir(parents=True, exist_ok=True)

        # Run preprocessing ablation
        results = ablation.run_preprocessing_ablation()

        # Save results
        output_file = results_dir / 'preprocessing_results.csv'
        results.to_csv(output_file, index=False)
        logging.info(f"✓ Saved preprocessing ablation results to {output_file}")

        # Generate analysis report
        report_file = results_dir / 'preprocessing_analysis.md'
        try:
            with open(report_file, 'w') as f:
                f.write("# Preprocessing Ablation Study Analysis\n\n")

                # Overview
                f.write("## Overview\n")
                f.write(f"Total configurations tested: {len(results)}\n")

                # Performance metrics
                if 'r2' in results.columns:
                    f.write("\n## Performance Metrics\n")
                    best_config = results.loc[results['r2'].idxmax()]
                    f.write(f"### Best Configuration\n")
                    f.write(f"- R² Score: {best_config['r2']:.4f}\n")
                    f.write(f"- Configuration: {best_config['preprocessing_config']}\n\n")

                    # Impact analysis
                    baseline = results['r2'].max()
                    impact = ((baseline - results['r2'].min()) / baseline * 100)
                    f.write(f"### Impact Analysis\n")
                    f.write(f"- Maximum impact on R² score: {impact:.1f}%\n")

                    # Results table
                    f.write("\n## Detailed Results\n")
                    f.write(results.to_markdown())

            logging.info(f"✓ Saved analysis report to {report_file}")

        except Exception as e:
            logging.error(f"✗ Error generating analysis report: {str(e)}")

        logging.info("✓ Preprocessing ablation study completed successfully")

    except Exception as e:
        logging.error(f"✗ Error in preprocessing ablation study: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
