# src/final_analysis/visualization_generator.py

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class FinalVisualizationGenerator:
    """Generate comprehensive visualizations for final analysis."""

    def __init__(self, config):
        self.config = config
        self.viz_dir = Path(config.VISUALIZATIONS_DIR) / 'final_analysis'
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Set style for all plots
        plt.style.use('seaborn')
        sns.set_palette("husl")

    def generate_all_visualizations(self, results):
        """Generate all visualizations for final analysis."""
        logging.info("Generating final analysis visualizations...")

        # Create model performance visualizations
        if 'model_metrics' in results:
            self.create_model_performance_plots(results['model_metrics'])

        # Create ablation study visualizations
        if 'ablation' in results:
            self.create_ablation_visualizations(results['ablation'])

        # Create ensemble analysis visualizations
        if 'ensemble' in results:
            self.create_ensemble_visualizations(results['ensemble'])

        logging.info(f"Visualizations saved to {self.viz_dir}")

    def create_model_performance_plots(self, metrics_df):
        """Create visualizations for model performance comparison."""
        # 1. Overall Performance Comparison
        plt.figure(figsize=(12, 6))
        metrics_summary = metrics_df.groupby('model_type')[['r2', 'rmse']].mean()

        ax = metrics_summary.plot(kind='bar', rot=45)
        plt.title('Model Performance by Type')
        plt.ylabel('Score')
        plt.legend(title='Metric')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'model_performance_comparison.png')
        plt.close()

        # 2. Performance Distribution
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        sns.boxplot(data=metrics_df, x='model_type', y='r2')
        plt.title('R² Score Distribution')
        plt.xticks(rotation=45)

        plt.subplot(132)
        sns.boxplot(data=metrics_df, x='model_type', y='rmse')
        plt.title('RMSE Distribution')
        plt.xticks(rotation=45)

        plt.subplot(133)
        sns.boxplot(data=metrics_df, x='model_type', y='mae')
        plt.title('MAE Distribution')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'performance_distributions.png')
        plt.close()

        # 3. Performance Timeline
        if 'fold' in metrics_df.columns:
            plt.figure(figsize=(10, 6))
            for model_type in metrics_df['model_type'].unique():
                model_data = metrics_df[metrics_df['model_type'] == model_type]
                plt.plot(model_data['fold'], model_data['r2'],
                         marker='o', label=model_type)

            plt.title('Model Performance Across Folds')
            plt.xlabel('Fold')
            plt.ylabel('R² Score')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'performance_timeline.png')
            plt.close()

    def create_ablation_visualizations(self, ablation_results):
        """Create visualizations for ablation studies."""
        # 1. Impact Analysis Overview
        plt.figure(figsize=(12, 6))
        impact_data = []

        for study_name, results in ablation_results.items():
            if 'r2' in results.columns:
                baseline = results['r2'].max()
                impact = ((baseline - results['r2']) / baseline * 100).max()
                impact_data.append({
                    'study': study_name.replace('_', ' ').title(),
                    'impact': impact
                })

        impact_df = pd.DataFrame(impact_data)
        impact_df = impact_df.sort_values('impact', ascending=True)

        sns.barplot(data=impact_df, x='impact', y='study')
        plt.title('Maximum Impact of Each Ablation Study')
        plt.xlabel('Performance Impact (%)')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'ablation_impact_overview.png')
        plt.close()

        # 2. Detailed Study Visualizations
        for study_name, results in ablation_results.items():
            if 'r2' in results.columns:
                plt.figure(figsize=(10, 6))

                # Sort by R² score
                results_sorted = results.sort_values('r2', ascending=False)

                # Create bar plot
                sns.barplot(data=results_sorted,
                            x=results_sorted.index,
                            y='r2')

                plt.title(f'{study_name.replace("_", " ").title()} Study Results')
                plt.xlabel('Configuration')
                plt.ylabel('R² Score')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.viz_dir / f'{study_name}_detailed.png')
                plt.close()

    def create_ensemble_visualizations(self, ensemble_results):
        """Create visualizations for ensemble analysis."""
        # 1. Predictions vs Actual
        if 'predictions' in ensemble_results:
            plt.figure(figsize=(12, 6))
            predictions = ensemble_results['predictions']

            plt.scatter(predictions['actual'], predictions['predicted'],
                        alpha=0.5, label='Predictions')

            # Add perfect prediction line
            min_val = min(predictions['actual'].min(), predictions['predicted'].min())
            max_val = max(predictions['actual'].max(), predictions['predicted'].max())
            plt.plot([min_val, max_val], [min_val, max_val],
                     'r--', label='Perfect Prediction')

            plt.title('Ensemble Predictions vs Actual Values')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'ensemble_prediction_scatter.png')
            plt.close()

            # 2. Error Distribution
            plt.figure(figsize=(10, 6))
            errors = predictions['predicted'] - predictions['actual']

            sns.histplot(errors, kde=True)
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            plt.title('Ensemble Prediction Error Distribution')
            plt.xlabel('Prediction Error')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'ensemble_error_distribution.png')
            plt.close()

        # 3. Model Weights (if available)
        if 'model_weights' in ensemble_results:
            plt.figure(figsize=(8, 8))
            weights = ensemble_results['model_weights']

            plt.pie(weights.values(), labels=weights.keys(), autopct='%1.1f%%')
            plt.title('Ensemble Model Weight Distribution')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'ensemble_weights.png')
            plt.close()

    def create_summary_dashboard(self, results):
        """Create a summary dashboard combining key visualizations."""
        plt.figure(figsize=(15, 10))

        # 1. Model Performance (top left)
        plt.subplot(221)
        if 'model_metrics' in results:
            metrics_summary = results['model_metrics'].groupby('model_type')['r2'].mean()
            metrics_summary.plot(kind='bar')
            plt.title('Average R² by Model Type')
            plt.xticks(rotation=45)

        # 2. Ablation Impact (top right)
        plt.subplot(222)
        if 'ablation' in results:
            impact_data = []
            for study, data in results['ablation'].items():
                if 'r2' in data.columns:
                    impact = ((data['r2'].max() - data['r2'].min()) /
                              data['r2'].max() * 100)
                    impact_data.append({
                        'study': study,
                        'impact': impact
                    })
            impact_df = pd.DataFrame(impact_data)
            sns.barplot(data=impact_df, x='study', y='impact')
            plt.title('Ablation Study Impacts')
            plt.xticks(rotation=45)

        # 3. Ensemble Performance (bottom left)
        plt.subplot(223)
        if 'ensemble' in results and 'predictions' in results['ensemble']:
            predictions = results['ensemble']['predictions']
            plt.scatter(predictions['actual'], predictions['predicted'], alpha=0.5)
            plt.title('Ensemble Predictions')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')

        # 4. Performance Timeline (bottom right)
        plt.subplot(224)
        if 'model_metrics' in results and 'fold' in results['model_metrics'].columns:
            for model_type in results['model_metrics']['model_type'].unique():
                model_data = results['model_metrics'][
                    results['model_metrics']['model_type'] == model_type
                    ]
                plt.plot(model_data['fold'], model_data['r2'],
                         marker='o', label=model_type)
            plt.title('Performance Across Folds')
            plt.legend()

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'summary_dashboard.png')
        plt.close()
