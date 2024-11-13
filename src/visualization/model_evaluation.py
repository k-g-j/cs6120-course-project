import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_visualizations(metrics_df, predictions, feature_importance, output_dir='visualizations'):
    """Create and save model evaluation visualizations."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Model Performance Comparison
    plt.figure(figsize=(12, 6))

    # Calculate mean of numeric metrics only
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

    # 3. Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions['actual'], predictions['predicted'], alpha=0.5)
    plt.plot([predictions['actual'].min(), predictions['actual'].max()],
             [predictions['actual'].min(), predictions['actual'].max()],
             'r--', lw=2)
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/actual_vs_predicted.png')
    plt.close()

    # 4. Error Distribution
    plt.figure(figsize=(10, 6))
    errors = predictions['predicted'] - predictions['actual']
    sns.histplot(errors, kde=True)
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_distribution.png')
    plt.close()

    # 5. Model Performance by Fold
    plt.figure(figsize=(12, 6))
    for metric in numeric_metrics:
        performance_by_fold = metrics_df.pivot(columns='model_name', values=metric, index='fold')
        plt.figure(figsize=(10, 6))
        performance_by_fold.plot(marker='o')
        plt.title(f'{metric.upper()} by Fold')
        plt.xlabel('Fold')
        plt.ylabel(metric.upper())
        plt.legend(title='Model')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_{metric}_by_fold.png')
        plt.close()

    logging.info(f"Saved visualizations to {output_dir}/")


def save_model_artifacts(model, model_name, fold, metrics, output_dir='models'):
    """Save model and its associated artifacts."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = Path(output_dir) / f"{model_name}_fold_{fold}.joblib"
    joblib.dump(model, model_path)

    # Save metrics
    metrics_path = Path(output_dir) / f"{model_name}_fold_{fold}_metrics.json"
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

    # Model Performance Summary
    report.append("## Model Performance Summary\n\n")
    numeric_metrics = ['rmse', 'mae', 'r2', 'mape']
    summary = metrics_df.groupby('model_name')[numeric_metrics].mean()

    # Add model type information
    model_types = metrics_df.groupby('model_name')['model_type'].first()
    summary['model_type'] = model_types

    # Reorder columns to put model_type first
    cols = ['model_type'] + numeric_metrics
    summary = summary[cols]

    if markdown_available:
        report.append(summary.to_markdown())
    else:
        report.append("\n```")
        report.append(summary.to_string())
        report.append("```\n")

    # Add detailed metrics by fold
    report.append("\n## Detailed Metrics by Fold\n\n")
    for model_type in metrics_df['model_type'].unique():
        report.append(f"\n### {model_type.title()} Models\n\n")
        model_metrics = metrics_df[metrics_df['model_type'] == model_type]
        if markdown_available:
            report.append(model_metrics.to_markdown(index=False))
        else:
            report.append("\n```")
            report.append(model_metrics.to_string())
            report.append("```\n")

    # Save report
    report_path = Path(reports_dir) / "model_evaluation_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    logging.info(f"Saved model evaluation report to {report_path}")
