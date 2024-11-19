import logging
from datetime import datetime

from src.comprehensive_report import generate_comprehensive_report

from pipeline_runner import PipelineConfig


def setup_logging(config):
    """Set up logging for report generation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOGS_DIR / f'report_generation_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main():
    """Generate comprehensive evaluation report."""
    # Initialize configuration
    config = PipelineConfig()
    setup_logging(config)

    try:
        logging.info("Starting comprehensive report generation...")

        # Generate report
        report_path = generate_comprehensive_report(config)

        logging.info(f"Report generated successfully: {report_path}")
        logging.info("Report generation completed.")

    except Exception as e:
        logging.error(f"Error generating report: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
