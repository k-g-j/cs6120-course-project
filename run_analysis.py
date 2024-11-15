import logging
from datetime import datetime

from analysis_report import AnalysisReport
from pipeline_runner import PipelineConfig


def setup_logging(config):
    """Set up logging for analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOGS_DIR / f'analysis_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main():
    """Run comprehensive analysis of model results and ablation studies."""
    # Initialize configuration
    config = PipelineConfig()
    setup_logging(config)

    try:
        logging.info("Starting comprehensive analysis...")

        # Generate analysis report
        analyzer = AnalysisReport(config)
        report_path = analyzer.generate_report()

        logging.info(f"Analysis completed. Report generated at: {report_path}")
        logging.info(f"Visualizations saved in: {config.VISUALIZATIONS_DIR}/analysis/")

    except Exception as e:
        logging.error(f"Error in analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
