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

        # Load model metrics
        model_metrics = results_dir / 'model_metrics.csv'
        if model_metrics.exists():
            self.results['model_metrics'] = pd.read_csv(model_metrics)
            self._compute_model_statistics()

        # Load ablation studies
        ablation_dir = results_dir / 'ablation_studies'
        if ablation_dir.exists():
            self.results['ablation'] = {}
            for file in ablation_dir.glob('*_results.csv'):
                study_name = file.stem.replace('_results', '')
                self.results['ablation'][study_name] = pd.read_csv(file)
            self._compute_ablation_statistics()

        # Load ensemble results
        ensemble_dir = results_dir / 'ensemble'
        if ensemble_dir.exists():
            metrics_file = ensemble_dir / 'ensemble_metrics.csv'
            if metrics_file.exists():
                self.results['ensemble'] = {}
                self.results['ensemble']['metrics'] = pd.read_csv(metrics_file)
                self._compute_ensemble_statistics()

            predictions_file = ensemble_dir / 'ensemble_predictions.csv'
            if predictions_file.exists():
                self.results['ensemble']['predictions'] = pd.read_csv(predictions_file)

        logging.info("Loaded results from all studies")

    def _compute_model_statistics(self):
        """Compute summary statistics for model performance."""
        if 'model_metrics' in self.results:
            metrics = self.results['model_metrics']

            # Find best model
            best_idx = metrics['r2'].idxmax()
            best_model = metrics.loc[best_idx]

            # Compute model type comparisons
            model_comparison = metrics.groupby('model_type').agg({
                'r2': ['mean', 'std'],
                'rmse': ['mean', 'std']
            })

            self.summary_stats['models'] = {
                'best_model': {
                    'name': best_model['model_name'],
                    'r2': best_model['r2'],
                    'rmse': best_model['rmse']
                },
                'model_comparison': model_comparison
            }

    def _compute_ablation_statistics(self):
        """Compute statistics for ablation studies."""
        if 'ablation' in self.results:
            self.summary_stats['ablation'] = {}

            for study_name, results in self.results['ablation'].items():
                if 'r2' in results.columns:
                    baseline = results['r2'].max()
                    if baseline > 0:
                        impact = ((baseline - results['r2'].min()) / baseline * 100)
                        self.summary_stats['ablation'][study_name] = {
                            'max_impact': impact,
                            'best_config': results.loc[results['r2'].idxmax()].to_dict(),
                            'baseline_r2': baseline
                        }

    def _compute_ensemble_statistics(self):
        """Compute statistics for ensemble results."""
        if 'ensemble' in self.results and 'metrics' in self.results['ensemble']:
            metrics = self.results['ensemble']['metrics']

            ensemble_stats = {
                'mean_r2': metrics['r2'].mean(),
                'mean_rmse': metrics['rmse'].mean(),
                'improvement_over_base': {}
            }

            # Calculate improvement over other model types
            if 'model_metrics' in self.results:
                base_metrics = self.results['model_metrics']
                base_r2 = base_metrics.groupby('model_type')['r2'].mean()

                for model_type, r2 in base_r2.items():
                    if r2 > 0:
                        improvement = ((ensemble_stats['mean_r2'] - r2) / r2) * 100
                        ensemble_stats['improvement_over_base'][model_type] = improvement

            self.summary_stats['ensemble'] = ensemble_stats

    def print_loaded_results(self):
        """Print summary of loaded results."""
        print("\nLoaded Results Summary:")
        print("-----------------------")

        if 'model_metrics' in self.results:
            print("\nModel Metrics:")
            print(
                f"- Number of models: {len(self.results['model_metrics']['model_name'].unique())}")
            print(
                f"- Model types: {', '.join(self.results['model_metrics']['model_type'].unique())}")

        if 'ablation' in self.results:
            print("\nAblation Studies:")
            for study_name, results in self.results['ablation'].items():
                print(f"- {study_name}: {len(results)} configurations")

        if 'ensemble' in self.results:
            print("\nEnsemble Results:")
            if 'metrics' in self.results['ensemble']:
                print("- Ensemble metrics available")
            if 'predictions' in self.results['ensemble']:
                print("- Ensemble predictions available")

        print("\nSummary Statistics:")
        for category, stats in self.summary_stats.items():
            print(f"\n{category.title()}:")
            if isinstance(stats, dict):
                for key, value in stats.items():
                    if isinstance(value, dict):
                        print(f"- {key}: {len(value)} items")
                    else:
                        print(f"- {key}: {value}")

    def generate_final_analysis(self):
        """Generate final analysis report."""
        report_sections = []

        # 1. Executive Summary
        report_sections.append(self._generate_executive_summary())

        # 2. Detailed Analysis
        report_sections.append(self._generate_detailed_analysis())

        # 3. Key Findings
        report_sections.append(self._generate_key_findings())

        # 4. Ablation Studies Analysis
        report_sections.append(self._generate_ablation_analysis())

        # 5. Technical Details
        report_sections.append(self._generate_technical_details())

        # 6. Recommendations
        report_sections.append(self._generate_recommendations())

        # Combine all sections
        report = "\n\n".join(report_sections)

        # Save report
        report_path = self.config.REPORTS_DIR / 'final_analysis_report.md'
        with open(report_path, 'w') as f:
            f.write(report)

        return report_path

    def _generate_executive_summary(self):
        """Generate executive summary section."""
        summary = ["# Final Analysis Report: Solar Energy Production Prediction\n"]
        summary.append("## Executive Summary\n")

        if 'models' in self.summary_stats:
            best_model = self.summary_stats['models']['best_model']
            summary.append("### Model Performance")
            summary.append(f"- Best Model: {best_model['name']}")
            summary.append(f"- R² Score: {best_model['r2']:.4f}")
            summary.append(f"- RMSE: {best_model['rmse']:.4f}\n")

        if 'ensemble' in self.summary_stats:
            ensemble = self.summary_stats['ensemble']
            summary.append("### Ensemble Performance")
            summary.append(f"- Average R² Score: {ensemble['mean_r2']:.4f}")
            summary.append(f"- Average RMSE: {ensemble['mean_rmse']:.4f}\n")

        if 'ablation' in self.summary_stats:
            summary.append("### Impact Analysis")
            for study, stats in self.summary_stats['ablation'].items():
                summary.append(
                    f"- {study.replace('_', ' ').title()}: {stats['max_impact']:.1f}% maximum impact")

        return "\n".join(summary)

    def _generate_detailed_analysis(self):
        """Generate detailed analysis section."""
        analysis = ["\n## Detailed Analysis\n"]

        # Model Performance Analysis
        analysis.append("### Model Performance Analysis\n")
        if 'models' in self.summary_stats:
            model_comparison = self.summary_stats['models']['model_comparison']
            analysis.append(model_comparison.to_markdown())
            analysis.append("\n")

        # Ensemble Analysis
        if 'ensemble' in self.summary_stats:
            analysis.append("### Ensemble Model Analysis\n")
            ensemble = self.summary_stats['ensemble']
            analysis.append(f"- Mean R² Score: {ensemble['mean_r2']:.4f}")
            analysis.append(f"- Mean RMSE: {ensemble['mean_rmse']:.4f}\n")

            if ensemble['improvement_over_base']:
                analysis.append("Performance Improvements:")
                for model_type, improvement in ensemble['improvement_over_base'].items():
                    analysis.append(f"- vs {model_type}: {improvement:.1f}% improvement")

        return "\n".join(analysis)

    def _generate_key_findings(self):
        """Generate key findings section."""
        findings = ["\n## Key Findings and Insights\n"]

        if 'models' in self.summary_stats:
            findings.append("### Model Performance")
            best_model = self.summary_stats['models']['best_model']
            findings.append(f"1. {best_model['name']}")
            findings.append(f"   - Best R² Score: {best_model['r2']:.4f}")
            findings.append(f"   - Best RMSE: {best_model['rmse']:.4f}\n")

            model_comp = self.summary_stats['models']['model_comparison']
            findings.append("2. Model Type Effectiveness")
            for model_type in model_comp.index:
                r2_mean = model_comp.loc[model_type, ('r2', 'mean')]
                r2_std = model_comp.loc[model_type, ('r2', 'std')]
                findings.append(f"   - {model_type}: R² = {r2_mean:.4f} (±{r2_std:.4f})")

        return "\n".join(findings)

    def _generate_ablation_analysis(self):
        """Generate ablation studies analysis."""
        analysis = ["\n## Ablation Studies Analysis\n"]

        if 'ablation' in self.summary_stats:
            for study_name, stats in self.summary_stats['ablation'].items():
                analysis.append(f"### {study_name.replace('_', ' ').title()}")
                analysis.append(f"- Maximum Impact: {stats['max_impact']:.1f}%")
                analysis.append(f"- Best R² Score: {stats['baseline_r2']:.4f}")

                # Add specific findings for each study type
                if study_name == 'input_dimension':
                    analysis.append("- Most important features by impact:")
                    best_config = stats['best_config']
                    for feature, value in best_config.items():
                        if feature != 'r2':
                            analysis.append(f"  * {feature}: {value:.4f}")

                analysis.append("")  # Add spacing between studies

        return "\n".join(analysis)

    def _generate_technical_details(self):
        """Generate technical details section."""
        details = ["\n## Technical Implementation Details\n"]

        # Model Architectures
        details.append("### Model Architectures\n")
        if 'models' in self.summary_stats:
            model_types = self.summary_stats['models']['model_comparison'].index
            for model_type in model_types:
                details.append(f"\n#### {model_type}")
                # Add specific architecture details
                if model_type == 'baseline':
                    details.append("- Linear regression variants")
                    details.append("- Regularized models (Ridge, Lasso)")
                elif model_type == 'advanced':
                    details.append("- Ensemble methods (Random Forest, Gradient Boosting)")
                    details.append("- Deep learning architectures (LSTM, CNN)")
                elif model_type == 'ensemble':
                    details.append("- Stacked ensemble approach")
                    details.append("- Weighted model combination")

        return "\n".join(details)

    def _generate_recommendations(self):
        """Generate recommendations section."""
        recommendations = ["\n## Recommendations and Future Work\n"]

        # Model Improvements
        recommendations.append("### Model Improvements")
        if 'models' in self.summary_stats:
            best_type = self.summary_stats['models']['model_comparison']['r2']['mean'].idxmax()
            recommendations.append(f"1. Focus on {best_type} architectures")
            recommendations.append("2. Explore deeper/wider architectures")
            recommendations.append("3. Implement ensemble combinations\n")

        # Data Improvements
        recommendations.append("### Data Collection")
        recommendations.append("1. Gather additional features")
        recommendations.append("2. Increase temporal resolution")
        recommendations.append("3. Expand training dataset\n")

        # Deployment
        recommendations.append("### Deployment Strategy")
        recommendations.append("1. Implement real-time prediction pipeline")
        recommendations.append("2. Set up model monitoring")
        recommendations.append("3. Establish retraining schedule")

        return "\n".join(recommendations)
