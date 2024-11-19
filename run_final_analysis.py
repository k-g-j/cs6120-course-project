import logging
from datetime import datetime

from pipeline_runner import PipelineConfig
from src.final_analysis.analysis_compiler import FinalAnalysisCompiler
from src.final_analysis.visualization_generator import FinalVisualizationGenerator


def setup_logging(config):
    """Set up logging for final analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOGS_DIR / f'final_analysis_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main():
    """Run final analysis compilation and visualization."""
    # Initialize configuration
    config = PipelineConfig()
    setup_logging(config)

    try:
        logging.info("Starting final analysis...")

        # Initialize analysis compiler
        compiler = FinalAnalysisCompiler(config)

        # Load all results (this also computes summary statistics)
        logging.info("Loading results from all studies...")
        compiler.load_all_results()

        # Print loaded results summary
        compiler.print_loaded_results()

        # Generate final analysis report
        logging.info("Generating final analysis report...")
        report_path = compiler.generate_final_analysis()

        # Initialize visualization generator
        viz_generator = FinalVisualizationGenerator(config)

        # Generate all visualizations
        logging.info("Generating visualizations...")
        viz_generator.generate_all_visualizations(compiler.results)

        # Create summary dashboard
        logging.info("Creating summary dashboard...")
        viz_generator.create_summary_dashboard(compiler.results)

        logging.info("Final analysis completed successfully")
        logging.info(f"Report generated: {report_path}")
        logging.info(f"Visualizations saved in: {config.VISUALIZATIONS_DIR}/final_analysis/")

    except Exception as e:
        logging.error(f"Error in final analysis: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
