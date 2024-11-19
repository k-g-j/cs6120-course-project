import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class FinalVisualizationGenerator:
    """Generate comprehensive visualizations for final analysis."""

    def __init__(self, config):
        self.config = config
        self.viz_dir = Path(config.VISUALIZATIONS_DIR) / 'final_analysis'
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Set style for all plots - using default style instead of seaborn
        plt.style.use('default')
        # Just set the color palette from seaborn
        sns.set_palette("husl")

        # Additional style configurations
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10

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
        if not ablation_results:  # Check if we have any ablation results
            logging.info("No ablation results to visualize")
            return

        try:
            # 1. Impact Analysis Overview
            plt.figure(figsize=(12, 6))
            impact_data = []

            for study_name, results in ablation_results.items():
                if isinstance(results, pd.DataFrame) and 'r2' in results.columns:
                    # Calculate baseline and impact only if we have R² scores
                    baseline = results['r2'].max()
                    if baseline > 0:  # Avoid division by zero
                        impact = ((baseline - results['r2'].min()) / baseline * 100)
                        impact_data.append({
                            'Study': study_name.replace('_', ' ').title(),
                            'Impact (%)': impact
                        })

            if impact_data:  # Only create visualization if we have data
                impact_df = pd.DataFrame(impact_data)
                if not impact_df.empty:
                    impact_df = impact_df.sort_values('Impact (%)', ascending=True)

                    sns.barplot(data=impact_df, x='Impact (%)', y='Study')
                    plt.title('Maximum Impact of Each Ablation Study')
                    plt.xlabel('Performance Impact (%)')
                    plt.tight_layout()
                    plt.savefig(self.viz_dir / 'ablation_impact_overview.png')
                    plt.close()

            # 2. Detailed Study Visualizations
            for study_name, results in ablation_results.items():
                if isinstance(results, pd.DataFrame) and 'r2' in results.columns:
                    plt.figure(figsize=(10, 6))

                    # Create meaningful x-axis labels
                    if 'configuration' in results.columns:
                        x_col = 'configuration'
                    else:
                        x_col = results.index

                    # Sort and plot
                    results_sorted = results.sort_values('r2', ascending=False)

                    # Create bar plot
                    sns.barplot(data=results_sorted,
                                x=range(len(results_sorted)),  # Use numeric x-axis
                                y='r2')

                    plt.title(f'{study_name.replace("_", " ").title()} Study Results')
                    plt.xlabel('Configuration')
                    plt.ylabel('R² Score')

                    # Only show a subset of x-ticks if there are many configurations
                    if len(results_sorted) > 10:
                        plt.xticks(range(0, len(results_sorted), 2), rotation=45)
                    else:
                        plt.xticks(rotation=45)

                    plt.tight_layout()
                    plt.savefig(self.viz_dir / f'{study_name}_detailed.png')
                    plt.close()

        except Exception as e:
            logging.error(f"Error creating ablation visualizations: {str(e)}")
            plt.close('all')  # Close any open figures in case of error

    def create_ensemble_visualizations(self, ensemble_results):
        """Create visualizations for ensemble analysis."""

        try:
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

                # Convert weights to list if it's a numpy array
                if isinstance(weights, np.ndarray):
                    values = weights
                    labels = [f'Model {i + 1}' for i in range(len(weights))]
                else:
                    values = list(weights.values())
                    labels = list(weights.keys())

                plt.pie(values, labels=labels, autopct='%1.1f%%')
                plt.title('Ensemble Model Weight Distribution')
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(self.viz_dir / 'ensemble_weights.png')
                plt.close()

        except Exception as e:
            logging.error(f"Error creating ensemble visualizations: {str(e)}")
            plt.close('all')  # Close any open figures in case of error

    def create_summary_dashboard(self, results):
        """Create a summary dashboard combining key visualizations."""
        plt.figure(figsize=(15, 10))

        try:
            # 1. Model Performance (top left)
            plt.subplot(221)
            if ('model_metrics' in results and
                    isinstance(results['model_metrics'], pd.DataFrame) and
                    'model_type' in results['model_metrics'].columns and
                    'r2' in results['model_metrics'].columns):
                metrics_summary = results['model_metrics'].groupby('model_type')['r2'].mean()
                metrics_summary.plot(kind='bar')
                plt.title('Average R² by Model Type')
                plt.xticks(rotation=45)

            # 2. Ablation Impact (top right)
            plt.subplot(222)
            if 'ablation' in results:
                impact_data = []
                for study, data in results['ablation'].items():
                    if isinstance(data, pd.DataFrame) and 'r2' in data.columns:
                        baseline = data['r2'].max()
                        if baseline > 0:  # Avoid division by zero
                            impact = ((baseline - data['r2'].min()) / baseline * 100)
                            impact_data.append({
                                'Study': study,
                                'Impact (%)': impact
                            })

                if impact_data:
                    impact_df = pd.DataFrame(impact_data)
                    if not impact_df.empty:
                        sns.barplot(data=impact_df, x='Study', y='Impact (%)')
                        plt.title('Ablation Study Impacts')
                        plt.xticks(rotation=45)

            # 3. Ensemble Performance (bottom left)
            plt.subplot(223)
            if ('ensemble' in results and
                    'predictions' in results['ensemble'] and
                    isinstance(results['ensemble']['predictions'], pd.DataFrame)):

                predictions = results['ensemble']['predictions']
                if 'actual' in predictions.columns and 'predicted' in predictions.columns:
                    plt.scatter(predictions['actual'], predictions['predicted'], alpha=0.5)
                    plt.title('Ensemble Predictions')
                    plt.xlabel('Actual')
                    plt.ylabel('Predicted')

            # 4. Performance Timeline (bottom right)
            plt.subplot(224)
            if ('model_metrics' in results and
                    isinstance(results['model_metrics'], pd.DataFrame) and
                    'fold' in results['model_metrics'].columns):

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

        except Exception as e:
            logging.error(f"Error creating summary dashboard: {str(e)}")
            plt.close('all')  # Close any open figures in case of error
