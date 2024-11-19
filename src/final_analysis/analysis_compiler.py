# src/final_analysis/analysis_compiler.py

import logging
from pathlib import Path

import pandas as pd


class FinalAnalysisCompiler:
    """Compile and analyze all results from different studies and evaluations."""

    def __init__(self, config):
        self.config = config
        self.results = {}
        self.summary_stats = {}

    def load_all_results(self):
        """Load results from all studies and evaluations."""
        results_dir = Path(self.config.RESULTS_DIR)

        # Load ablation studies results
        ablation_dir = results_dir / 'ablation_studies'
        if ablation_dir.exists():
            self.results['ablation'] = {}
            for result_file in ablation_dir.glob('*_results.csv'):
                study_name = result_file.stem.replace('_results', '')
                self.results['ablation'][study_name] = pd.read_csv(result_file)

        # Load ensemble results
        ensemble_dir = results_dir / 'ensemble'
        if ensemble_dir.exists():
            ensemble_metrics = ensemble_dir / 'ensemble_metrics.csv'
            if ensemble_metrics.exists():
                self.results['ensemble'] = pd.read_csv(ensemble_metrics)

        # Load final model metrics
        model_metrics = results_dir / 'final_model_metrics.csv'
        if model_metrics.exists():
            self.results['model_metrics'] = pd.read_csv(model_metrics)

        logging.info("Loaded results from all studies")

    def compute_summary_statistics(self):
        """Compute summary statistics for all results."""
        self.summary_stats = {}

        # Analyze model metrics
        if 'model_metrics' in self.results:
            metrics = self.results['model_metrics']
            self.summary_stats['models'] = {
                'best_model': {
                    'name': metrics.loc[metrics['r2'].idxmax(), 'model_name'],
                    'r2': metrics['r2'].max(),
                    'rmse': metrics.loc[metrics['r2'].idxmax(), 'rmse']
                },
                'model_comparison': metrics.groupby('model_type').agg({
                    'r2': ['mean', 'std'],
                    'rmse': ['mean', 'std']
                })
            }

        # Analyze ablation studies
        if 'ablation' in self.results:
            self.summary_stats['ablation'] = {}
            for study_name, results in self.results['ablation'].items():
                if 'r2' in results.columns:
                    baseline = results['r2'].max()
                    impacts = ((baseline - results['r2']) / baseline * 100)
                    self.summary_stats['ablation'][study_name] = {
                        'max_impact': impacts.max(),
                        'mean_impact': impacts.mean(),
                        'best_config': results.loc[results['r2'].idxmax()].to_dict()
                    }

        # Analyze ensemble results
        if 'ensemble' in self.results:
            ensemble = self.results['ensemble']
            self.summary_stats['ensemble'] = {
                'mean_r2': ensemble['r2'].mean(),
                'mean_rmse': ensemble['rmse'].mean(),
                'improvement_over_base': None  # Will be calculated if base metrics exist
            }

            if 'model_metrics' in self.results:
                base_r2 = self.results['model_metrics'].groupby('model_type')['r2'].mean()
                self.summary_stats['ensemble']['improvement_over_base'] = {
                    model_type: ((ensemble['r2'].mean() - r2) / r2 * 100)
                    for model_type, r2 in base_r2.items()
                }

    def generate_final_analysis(self):
        """Generate final analysis report combining all results."""
        report_sections = []

        # 1. Executive Summary
        report_sections.append(self._generate_executive_summary())

        # 2. Detailed Analysis by Component
        report_sections.append(self._generate_detailed_analysis())

        # 3. Key Findings and Insights
        report_sections.append(self._generate_key_findings())

        # 4. Technical Implementation Details
        report_sections.append(self._generate_technical_details())

        # 5. Future Work and Recommendations
        report_sections.append(self._generate_recommendations())

        # Combine all sections
        final_report = "\n\n".join(report_sections)

        # Save report
        report_path = self.config.REPORTS_DIR / 'final_analysis_report.md'
        with open(report_path, 'w') as f:
            f.write(final_report)

        logging.info(f"Final analysis report generated: {report_path}")
        return report_path

    def _generate_executive_summary(self):
        """Generate executive summary section."""
        summary = ["# Final Analysis Report: Solar Energy Production Prediction\n"]
        summary.append("## Executive Summary\n")

        if 'models' in self.summary_stats:
            best_model = self.summary_stats['models']['best_model']
            summary.append("### Model Performance")
            summary.append(f"- Best performing model: {best_model['name']}")
            summary.append(f"- Achieved R² score: {best_model['r2']:.4f}")
            summary.append(f"- RMSE: {best_model['rmse']:.4f}")

        if 'ensemble' in self.summary_stats:
            ensemble = self.summary_stats['ensemble']
            summary.append("\n### Ensemble Performance")
            summary.append(f"- Ensemble R² score: {ensemble['mean_r2']:.4f}")
            summary.append(f"- Ensemble RMSE: {ensemble['mean_rmse']:.4f}")

        if 'ablation' in self.summary_stats:
            summary.append("\n### Key Impact Factors")
            for study, results in self.summary_stats['ablation'].items():
                summary.append(
                    f"- {study.replace('_', ' ').title()}: {results['max_impact']:.1f}% maximum impact")

        return "\n".join(summary)

    def _generate_detailed_analysis(self):
        """Generate detailed analysis section."""
        analysis = ["\n## Detailed Analysis\n"]

        # Model performance analysis
        if 'models' in self.summary_stats:
            analysis.append("### Model Performance Analysis\n")
            model_comparison = self.summary_stats['models']['model_comparison']
            analysis.append("Performance metrics by model type:\n")
            analysis.append(model_comparison.to_markdown())

        # Ablation studies analysis
        if 'ablation' in self.summary_stats:
            analysis.append("\n### Ablation Studies Analysis\n")
            for study, results in self.summary_stats['ablation'].items():
                analysis.append(f"\n#### {study.replace('_', ' ').title()}")
                analysis.append(f"- Maximum performance impact: {results['max_impact']:.1f}%")
                analysis.append(f"- Average performance impact: {results['mean_impact']:.1f}%")
                analysis.append(
                    f"- Best configuration achieved R² = {results['best_config']['r2']:.4f}")

        # Ensemble analysis
        if 'ensemble' in self.summary_stats:
            analysis.append("\n### Ensemble Model Analysis\n")
            ensemble = self.summary_stats['ensemble']
            analysis.append(f"- Mean R² score: {ensemble['mean_r2']:.4f}")
            if ensemble['improvement_over_base']:
                analysis.append("\nImprovement over baseline models:")
                for model_type, improvement in ensemble['improvement_over_base'].items():
                    analysis.append(f"- vs {model_type}: {improvement:.1f}% improvement")

        return "\n".join(analysis)

    def _generate_key_findings(self):
        """Generate key findings section."""
        findings = ["\n## Key Findings and Insights\n"]

        # Best performing approaches
        findings.append("### Best Performing Approaches")
        if 'models' in self.summary_stats:
            best_model = self.summary_stats['models']['best_model']
            findings.append(f"\n1. {best_model['name']}")
            findings.append(f"   - Achieved highest R² of {best_model['r2']:.4f}")
            findings.append(f"   - RMSE: {best_model['rmse']:.4f}")

        # Critical factors
        findings.append("\n### Critical Factors")
        if 'ablation' in self.summary_stats:
            sorted_impacts = sorted(
                [(study, results['max_impact'])
                 for study, results in self.summary_stats['ablation'].items()],
                key=lambda x: x[1],
                reverse=True
            )
            for study, impact in sorted_impacts:
                findings.append(
                    f"- {study.replace('_', ' ').title()}: {impact:.1f}% maximum impact")

        return "\n".join(findings)

    def _generate_technical_details(self):
        """Generate technical details section."""
        details = ["\n## Technical Implementation Details\n"]

        # Model architectures
        details.append("### Model Architectures")
        if 'models' in self.summary_stats:
            model_types = self.summary_stats['models']['model_comparison'].index.tolist()
            for model_type in model_types:
                details.append(f"\n#### {model_type}")
                if model_type == 'baseline':
                    details.append("- Linear regression and regularized variants")
                    details.append("- Optimized using standard solvers")
                elif model_type == 'advanced':
                    details.append("- Ensemble methods (Random Forest, Gradient Boosting)")
                    details.append("- Neural network architectures (LSTM, CNN)")
                elif model_type == 'ensemble':
                    details.append("- Stacked ensemble combining multiple models")
                    details.append("- Weighted combination using cross-validation")

        # Preprocessing pipeline
        if 'ablation' in self.summary_stats and 'preprocessing' in self.summary_stats['ablation']:
            details.append("\n### Preprocessing Pipeline")
            prep = self.summary_stats['ablation']['preprocessing']
            details.append(f"- Best configuration: {prep['best_config']}")

        return "\n".join(details)

    def _generate_recommendations(self):
        """Generate recommendations section."""
        recommendations = ["\n## Recommendations and Future Work\n"]

        # Model improvements
        recommendations.append("### Model Improvements")
        recommendations.append("1. Architecture Enhancements")
        if 'models' in self.summary_stats:
            best_type = self.summary_stats['models']['model_comparison']['r2']['mean'].idxmax()
            recommendations.append(f"   - Focus on {best_type} architectures")
            recommendations.append("   - Explore deeper architectures and additional features")

        # Data improvements
        recommendations.append("\n2. Data Collection and Processing")
        if 'ablation' in self.summary_stats:
            for study, results in self.summary_stats['ablation'].items():
                if results['max_impact'] > 10:  # Significant impact
                    recommendations.append(
                        f"   - Improve {study.replace('_', ' ')} based on {results['max_impact']:.1f}% potential impact")

        # Deployment considerations
        recommendations.append("\n3. Deployment Considerations")
        recommendations.append("   - Implement real-time prediction pipeline")
        recommendations.append("   - Set up monitoring and retraining framework")
        recommendations.append("   - Develop error handling and fallback mechanisms")

        return "\n".join(recommendations)
