# src/comprehensive_report.py

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_comprehensive_report(config):
    """Generate comprehensive evaluation report combining all results."""
    reports_dir = Path(config.REPORTS_DIR)
    results_dir = Path(config.RESULTS_DIR)
    viz_dir = Path(config.VISUALIZATIONS_DIR) / 'comprehensive'

    # Create directories if they don't exist
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results(results_dir)

    # Generate report sections
    sections = []

    # Executive Summary
    sections.append(generate_executive_summary(results))

    # Model Performance Analysis
    sections.append(generate_model_analysis(results, viz_dir))

    # Feature Importance Analysis
    sections.append(generate_feature_analysis(results, viz_dir))

    # Results Analysis
    sections.append(generate_results_analysis(results, viz_dir))

    # Write report
    report_path = reports_dir / 'comprehensive_report.md'
    with open(report_path, 'w') as f:
        f.write('\n\n'.join(sections))

    logging.info(f"Comprehensive report generated: {report_path}")
    return report_path


def load_results(results_dir):
    """Load all available results."""
    results = {}

    # Load model metrics if available
    model_metrics = results_dir / 'model_metrics.csv'
    if model_metrics.exists():
        results['model_metrics'] = pd.read_csv(model_metrics)

    # Load ensemble results if available
    ensemble_dir = results_dir / 'ensemble'
    if ensemble_dir.exists():
        ensemble_metrics = ensemble_dir / 'ensemble_metrics.csv'
        if ensemble_metrics.exists():
            results['ensemble_metrics'] = pd.read_csv(ensemble_metrics)

    # Load ablation results if available
    ablation_dir = results_dir / 'ablation_studies'
    if ablation_dir.exists():
        results['ablation'] = {}
        for result_file in ablation_dir.glob('*_results.csv'):
            study_name = result_file.stem.replace('_results', '')
            results['ablation'][study_name] = pd.read_csv(result_file)

    return results


def generate_executive_summary(results):
    """Generate executive summary section."""
    summary = ["# Solar Energy Production Prediction - Comprehensive Report\n"]
    summary.append("## Executive Summary\n")

    # Model performance summary
    if 'model_metrics' in results:
        metrics = results['model_metrics']
        best_model = metrics.loc[metrics['r2'].idxmax()]
        summary.append("### Model Performance\n")
        summary.append(f"- Best performing model: {best_model['model_name']}")
        summary.append(f"- Best R² score: {best_model['r2']:.4f}")
        summary.append(f"- Best RMSE: {best_model['rmse']:.4f}")

    # Ensemble results if available
    if 'ensemble_metrics' in results:
        ensemble = results['ensemble_metrics']
        summary.append("\n### Ensemble Model Performance\n")
        summary.append(f"- Average R² score: {ensemble['r2'].mean():.4f}")
        summary.append(f"- Average RMSE: {ensemble['rmse'].mean():.4f}")

    # Ablation studies summary
    if 'ablation' in results:
        summary.append("\n### Key Findings from Ablation Studies\n")
        for study_name, study_results in results['ablation'].items():
            if 'r2' in study_results.columns:
                impact = ((study_results['r2'].max() - study_results['r2'].min()) /
                          study_results['r2'].max() * 100)
                summary.append(
                    f"- {study_name.replace('_', ' ').title()}: {impact:.1f}% maximum impact")

    return '\n'.join(summary)


def generate_model_analysis(results, viz_dir):
    """Generate model performance analysis section."""
    analysis = ["\n## Model Performance Analysis\n"]

    if 'model_metrics' in results:
        metrics = results['model_metrics']

        # Model type comparison
        analysis.append("### Performance by Model Type\n")
        type_metrics = metrics.groupby('model_type').agg({
            'r2': ['mean', 'std'],
            'rmse': ['mean', 'std']
        }).round(4)

        analysis.append(type_metrics.to_markdown())

        # Create visualization
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=metrics, x='model_type', y='r2')
        plt.title('R² Score Distribution by Model Type')
        plt.xlabel('Model Type')
        plt.ylabel('R² Score')
        plt.tight_layout()
        plt.savefig(viz_dir / 'model_type_performance.png')
        plt.close()

        analysis.append("\n![Model Type Performance](model_type_performance.png)\n")

    if 'ensemble_metrics' in results:
        analysis.append("\n### Ensemble Model Analysis\n")
        ensemble = results['ensemble_metrics']
        analysis.append(
            f"- Consistent performance across folds (R² std: {ensemble['r2'].std():.4f})")
        analysis.append(f"- Stable predictions (RMSE std: {ensemble['rmse'].std():.4f})")

    return '\n'.join(analysis)


def generate_feature_analysis(results, viz_dir):
    """Generate feature importance analysis section."""
    analysis = ["\n## Feature Importance Analysis\n"]

    if 'ablation' in results and 'input_dimension' in results['ablation']:
        feature_results = results['ablation']['input_dimension']
        if 'r2' in feature_results.columns:
            # Sort features by importance
            feature_importance = feature_results.sort_values('r2', ascending=False)

            analysis.append("### Most Important Features\n")
            for idx, row in feature_importance.head().iterrows():
                analysis.append(f"- {idx}: R² impact of {row['r2']:.4f}")

            # Create visualization
            plt.figure(figsize=(10, 6))
            plt.bar(feature_importance.index, feature_importance['r2'])
            plt.title('Feature Importance by R² Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(viz_dir / 'feature_importance.png')
            plt.close()

            analysis.append("\n![Feature Importance](feature_importance.png)\n")

    return '\n'.join(analysis)


def generate_results_analysis(results, viz_dir):
    """Generate detailed results analysis section."""
    analysis = ["\n## Detailed Results Analysis\n"]

    # Ablation studies analysis
    if 'ablation' in results:
        analysis.append("### Ablation Studies Results\n")

        for study_name, study_results in results['ablation'].items():
            if 'r2' in study_results.columns:
                analysis.append(f"\n#### {study_name.replace('_', ' ').title()}\n")

                # Calculate impact
                baseline = study_results['r2'].max()
                impact = ((baseline - study_results['r2']) / baseline * 100).mean()

                analysis.append(f"- Average impact: {impact:.1f}%")
                analysis.append(f"- Best configuration R²: {baseline:.4f}")
                analysis.append(f"- Worst configuration R²: {study_results['r2'].min():.4f}")

    # Model stability analysis
    if 'model_metrics' in results:
        analysis.append("\n### Model Stability Analysis\n")
        stability = results['model_metrics'].groupby('model_name').agg({
            'r2': 'std',
            'rmse': 'std'
        }).round(4)

        analysis.append("Standard deviation across folds:\n")
        analysis.append(stability.to_markdown())

    return '\n'.join(analysis)
