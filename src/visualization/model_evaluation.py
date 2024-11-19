import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_visualizations(metrics_df, predictions_df, feature_importance, output_dir):
    """Create visualizations for model evaluation."""
    os.makedirs(output_dir, exist_ok=True)

    # Check if this is ensemble data or regular model data
    is_ensemble = 'model_name' not in metrics_df.columns

    if is_ensemble:
        create_ensemble_visualizations(metrics_df, predictions_df, output_dir)
    else:
        create_model_visualizations(metrics_df, predictions_df, feature_importance, output_dir)


def create_ensemble_visualizations(metrics_df, predictions_df, output_dir):
    """Create visualizations specific to ensemble evaluation."""
    # 1. Performance across folds
    plt.figure(figsize=(10, 6))
    metrics_df[['rmse', 'r2']].plot(marker='o')
    plt.title('Ensemble Performance Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.legend(['RMSE', 'R²'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_performance.png'))
    plt.close()

    # 2. Predictions vs Actual
    if predictions_df is not None:
        plt.figure(figsize=(10, 6))
        predictions_df = predictions_df.sort_values('timestamp')
        plt.plot(predictions_df['timestamp'], predictions_df['actual'],
                 label='Actual', alpha=0.7)
        plt.plot(predictions_df['timestamp'], predictions_df['predicted'],
                 label='Predicted', alpha=0.7)
        plt.title('Ensemble Predictions vs Actual Values')
        plt.xlabel('Time')
        plt.ylabel('kWh')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ensemble_predictions.png'))
        plt.close()

        # 3. Prediction Error Distribution
        plt.figure(figsize=(10, 6))
        errors = predictions_df['predicted'] - predictions_df['actual']
        sns.histplot(errors, kde=True)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.title('Ensemble Prediction Error Distribution')
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ensemble_error_distribution.png'))
        plt.close()

        # 4. Error by Fold
        plt.figure(figsize=(10, 6))
        error_by_fold = predictions_df.groupby('fold').apply(
            lambda x: np.sqrt(((x['predicted'] - x['actual']) ** 2).mean())
        )
        error_by_fold.plot(kind='bar')
        plt.title('RMSE by Fold')
        plt.xlabel('Fold')
        plt.ylabel('RMSE')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ensemble_error_by_fold.png'))
        plt.close()


def create_model_visualizations(metrics_df, predictions_df, feature_importance, output_dir):
    """Create visualizations for regular model evaluation."""
    plt.figure(figsize=(12, 6))

    # 1. Model Performance Comparison
    numeric_metrics = ['rmse', 'mae', 'r2', 'mape']
    metrics_summary = metrics_df.groupby('model_name')[numeric_metrics].mean()

    ax = metrics_summary.plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png')
    plt.close()

    # 2. Feature Importance Plot
    if feature_importance is not None:
        plt.figure(figsize=(12, 8))
        importance_df = pd.DataFrame(feature_importance)
        importance_df = importance_df.sort_values('importance', ascending=True)

        plt.barh(y=importance_df['feature'], width=importance_df['importance'])
        plt.title('Feature Importance (Best Model)')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png')
        plt.close()

    # 3. Performance by Model Type
    if 'model_type' in metrics_df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=metrics_df, x='model_type', y='r2')
        plt.title('R² Score by Model Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_by_type.png')
        plt.close()


def save_model_artifacts(model, model_name, fold, metrics, output_dir='models'):
    """Save model and its associated artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    import joblib
    # Save model
    model_path = os.path.join(output_dir, f"{model_name}_fold_{fold}.joblib")
    joblib.dump(model, model_path)

    # Save metrics
    metrics_path = os.path.join(output_dir, f"{model_name}_fold_{fold}_metrics.json")
    pd.Series(metrics).to_json(metrics_path)

    logging.info(f"Saved {model_name} artifacts for fold {fold}")


def generate_model_report(metrics_df, config):
    """Generate a comprehensive model evaluation report."""
    try:
        import tabulate
        markdown_available = True
    except ImportError:
        markdown_available = False
        logging.warning("tabulate package not found. Generating simple text report.")

    # Get reports directory as string
    reports_dir = str(config.REPORTS_DIR)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    report = []
    report.append("# Solar Energy Production Prediction - Model Evaluation Report\n\n")

    # Check if this is ensemble data
    is_ensemble = 'model_name' not in metrics_df.columns

    if is_ensemble:
        report.extend(_generate_ensemble_report(metrics_df))
    else:
        report.extend(_generate_model_report(metrics_df, markdown_available))

    # Save report
    report_path = Path(reports_dir) / "model_evaluation_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    logging.info(f"Saved model evaluation report to {report_path}")


def _generate_ensemble_report(metrics_df):
    """Generate report sections for ensemble evaluation."""
    report = []

    # Overall Performance
    report.append("## Ensemble Model Performance\n")
    summary = metrics_df.agg({
        'rmse': ['mean', 'std'],
        'r2': ['mean', 'std']
    }).round(4)

    report.append("### Overall Metrics\n")
    report.append(f"- Average RMSE: {summary['rmse']['mean']:.4f} (± {summary['rmse']['std']:.4f})")
    report.append(f"- Average R²: {summary['r2']['mean']:.4f} (± {summary['r2']['std']:.4f})")

    # Performance by Fold
    report.append("\n### Performance by Fold\n")
    report.append(metrics_df[['fold', 'rmse', 'r2']].to_markdown(index=False))

    return report


def _generate_model_report(metrics_df, markdown_available):
    """Generate report sections for regular model evaluation."""
    report = []

    # Model Performance Summary
    report.append("## Model Performance Summary\n")
    numeric_metrics = ['rmse', 'mae', 'r2', 'mape']
    summary = metrics_df.groupby(['model_type', 'model_name'])[numeric_metrics].mean()

    if markdown_available:
        report.append(summary.to_markdown())
    else:
        report.append("\n```")
        report.append(summary.to_string())
        report.append("```\n")

    # Add detailed metrics by fold
    report.append("\n## Detailed Metrics by Fold\n")
    for model_type in metrics_df['model_type'].unique():
        report.append(f"\n### {model_type.title()} Models\n")
        model_metrics = metrics_df[metrics_df['model_type'] == model_type]
        if markdown_available:
            report.append(model_metrics.to_markdown(index=False))
        else:
            report.append("\n```")
            report.append(model_metrics.to_string())
            report.append("```\n")

    return report
