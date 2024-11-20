import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


class AnalysisReport:
    """Generate comprehensive analysis report of model performance and ablation studies."""

    def __init__(self, config):
        self.config = config
        self.results_dir = Path(config.RESULTS_DIR)
        self.reports_dir = Path(config.REPORTS_DIR)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def load_results(self):
        """Load all results from various studies."""
        results = {}

        # Load model metrics
        try:
            results['model_metrics'] = pd.read_csv(self.results_dir / 'final_model_metrics.csv')
        except FileNotFoundError:
            logging.warning("Final model metrics file not found")

        # Load ablation study results
        ablation_dir = self.results_dir / 'ablation_studies'
        if ablation_dir.exists():
            for study_file in ablation_dir.glob('*_results.csv'):
                study_name = study_file.stem.replace('_results', '')
                results[f'ablation_{study_name}'] = pd.read_csv(study_file)

        return results

    def generate_statistical_analysis(self, results):
        """Perform statistical analysis on model results."""
        analysis = []

        if 'model_metrics' in results:
            metrics = results['model_metrics']

            # Perform statistical tests
            model_groups = metrics.groupby('model_type')

            # ANOVA test between model types
            model_types = metrics['model_type'].unique()
            if len(model_types) > 2:
                r2_by_type = [group['r2'].values for name, group in model_groups]
                f_stat, p_value = stats.f_oneway(*r2_by_type)
                analysis.append("# Statistical Analysis\n")
                analysis.append(f"## ANOVA Test Results (R² across model types)")
                analysis.append(f"- F-statistic: {f_stat:.4f}")
                analysis.append(f"- p-value: {p_value:.4f}\n")

            # Descriptive statistics by model type
            analysis.append("## Model Performance Statistics by Type\n")
            for model_type, group in model_groups:
                analysis.append(f"### {model_type.title()} Models")
                stats_df = group.describe()
                analysis.append(stats_df.to_markdown())
                analysis.append("\n")

        return "\n".join(analysis)

    def analyze_ablation_results(self, results):
        """Analyze and summarize ablation study results."""
        analysis = ["# Ablation Studies Analysis\n"]

        for key, data in results.items():
            if key.startswith('ablation_'):
                study_name = key.replace('ablation_', '')
                analysis.append(f"## {study_name.replace('_', ' ').title()} Study\n")

                # Calculate impact of each factor
                if 'r2' in data.columns:
                    baseline_r2 = data['r2'].max()
                    analysis.append(
                        f"### Impact Analysis (relative to best R² score: {baseline_r2:.4f})\n")

                    impact_analysis = []
                    for _, row in data.iterrows():
                        factor_name = row.get('feature_group') or row.get('preprocessing_config') or \
                                      str(row.get('forecast_horizon')) or str(
                            row.get('data_fraction'))
                        if factor_name and 'r2' in row:
                            impact = ((baseline_r2 - row['r2']) / baseline_r2) * 100
                            impact_analysis.append({
                                'Factor': factor_name,
                                'R²': row['r2'],
                                'Performance Impact (%)': impact
                            })

                    if impact_analysis:
                        impact_df = pd.DataFrame(impact_analysis)
                        analysis.append(impact_df.to_markdown(index=False))
                        analysis.append("\n")

        return "\n".join(analysis)

    def create_summary_visualizations(self, results):
        """Create summary visualizations of all results."""
        viz_dir = self.config.VISUALIZATIONS_DIR / 'analysis'
        viz_dir.mkdir(parents=True, exist_ok=True)

        if 'model_metrics' in results:
            metrics = results['model_metrics']

            # Overall model comparison
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=metrics, x='model_type', y='r2')
            plt.title('Model Performance Distribution by Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(viz_dir / 'model_performance_distribution.png')
            plt.close()

            # Performance trend across folds
            if 'fold' in metrics.columns:
                plt.figure(figsize=(12, 6))
                sns.lineplot(data=metrics, x='fold', y='r2', hue='model_type', marker='o')
                plt.title('Model Performance Across Folds')
                plt.tight_layout()
                plt.savefig(viz_dir / 'performance_across_folds.png')
                plt.close()

        # Ablation study visualizations
        for key, data in results.items():
            if key.startswith('ablation_'):
                study_name = key.replace('ablation_', '')
                if 'r2' in data.columns:
                    plt.figure(figsize=(10, 6))
                    if 'feature_group' in data.columns:
                        sns.barplot(data=data, x='feature_group', y='r2')
                        plt.title(f'Feature Group Impact on Model Performance')
                    elif 'forecast_horizon' in data.columns:
                        sns.lineplot(data=data, x='forecast_horizon', y='r2', marker='o')
                        plt.title(f'Performance vs Forecast Horizon')
                    elif 'data_fraction' in data.columns:
                        sns.lineplot(data=data, x='data_fraction', y='r2', marker='o')
                        plt.title(f'Learning Curve (Performance vs Training Data Size)')

                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(viz_dir / f'{study_name}_analysis.png')
                    plt.close()

    def generate_report(self):
        """Generate comprehensive analysis report."""
        results = self.load_results()

        report_sections = [
            "# Solar Energy Production Prediction - Analysis Report\n",
            self.generate_statistical_analysis(results),
            self.analyze_ablation_results(results)
        ]

        # Create visualizations
        self.create_summary_visualizations(results)

        # Add methodology section
        report_sections.append("""
## Methodology

### Model Evaluation
- Comprehensive evaluation using multiple metrics (RMSE, MAE, R², MAPE)
- Cross-validation using time series splits
- Statistical significance testing between model types

### Ablation Studies
1. Input Dimension Analysis
   - Evaluated impact of different feature groups
   - Identified most crucial features for prediction accuracy

2. Preprocessing Impact
   - Tested various preprocessing configurations
   - Quantified importance of each preprocessing step

3. Temporal Resolution Analysis
   - Evaluated performance across different forecast horizons
   - Identified optimal prediction timeframes

4. Data Volume Impact
   - Analyzed learning curves with varying training data sizes
   - Determined minimum data requirements for reliable predictions

## Key Findings
""")

        # Add key findings based on results
        if 'model_metrics' in results:
            best_model = results['model_metrics'].loc[results['model_metrics']['r2'].idxmax()]
            report_sections.append(f"""
### Model Performance
- Best performing model: {best_model['model_name']} ({best_model['model_type']})
- Achieved R² score: {best_model['r2']:.4f}
- RMSE: {best_model['rmse']:.4f}
""")

        # Write report
        report_path = self.reports_dir / 'analysis_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_sections))

        logging.info(f"Analysis report generated: {report_path}")
        return report_path
