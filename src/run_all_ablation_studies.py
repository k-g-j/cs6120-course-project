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
    log_file = Path(config.LOGS_DIR) / f'ablation_studies_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format=CONFIG['log_format'],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def run_and_save_study(study_func, study_name, results_dir, ablation):
    """Run a specific ablation study and save results with error handling."""
    logging.info(f"\nRunning {study_name} ablation study...")
    try:
        # Run the study
        results = study_func()

        # Create the output file path
        output_file = results_dir / f'{study_name}_results.csv'

        # Save results
        results.to_csv(output_file, index=False)
        logging.info(f"✓ Completed {study_name} study. Results saved to {output_file}")

        # Save detailed analysis if available
        try:
            analysis_file = results_dir / f'{study_name}_analysis.md'
            with open(analysis_file, 'w') as f:
                f.write(f"# {study_name.replace('_', ' ').title()} Ablation Study Analysis\n\n")
                f.write("## Overview\n")
                f.write(f"Total configurations tested: {len(results)}\n")
                if 'r2' in results.columns:
                    best_config = results.loc[results['r2'].idxmax()]
                    f.write(f"\n## Best Configuration\n")
                    f.write(f"R² Score: {best_config['r2']:.4f}\n")
                    for col in results.columns:
                        if col != 'r2':
                            f.write(f"{col}: {best_config[col]}\n")
            logging.info(f"✓ Saved detailed analysis to {analysis_file}")
        except Exception as e:
            logging.warning(f"⚠ Could not save detailed analysis: {str(e)}")

        return results

    except Exception as e:
        logging.error(f"✗ Error in {study_name} study: {str(e)}")
        return None


def save_summary_report(study_results, results_dir):
    """Generate and save a summary report of all ablation studies."""
    summary_file = results_dir / 'ablation_summary.md'
    try:
        with open(summary_file, 'w') as f:
            f.write("# Ablation Studies Summary Report\n\n")

            for study_name, results in study_results.items():
                if results is not None and isinstance(results, pd.DataFrame):
                    f.write(f"\n## {study_name.replace('_', ' ').title()}\n")
                    f.write(f"Total configurations tested: {len(results)}\n")

                    if 'r2' in results.columns:
                        f.write("\n### Performance Metrics\n")
                        f.write(f"- Best R² Score: {results['r2'].max():.4f}\n")
                        f.write(f"- Average R² Score: {results['r2'].mean():.4f}\n")
                        f.write(f"- Score Range: {results['r2'].max() - results['r2'].min():.4f}\n")

                        impact = ((results['r2'].max() - results['r2'].min()) /
                                  results['r2'].max() * 100)
                        f.write(f"- Overall Impact: {impact:.1f}%\n")

        logging.info(f"✓ Saved summary report to {summary_file}")
    except Exception as e:
        logging.error(f"✗ Error saving summary report: {str(e)}")


def main():
    """Run all ablation studies with enhanced error handling and reporting."""
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
        results_dir = Path(CONFIG['ablation_results'])
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
            logging.info(f"✓ Saved summary metrics to {results_dir / 'ablation_summary.csv'}")

            # Generate comprehensive summary report
            save_summary_report(study_results, results_dir)

        logging.info("✓ All ablation studies completed successfully")

    except Exception as e:
        logging.error(f"✗ Error in ablation studies: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
