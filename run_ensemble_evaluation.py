import logging
from datetime import datetime

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config import CONFIG
from pipeline_runner import PipelineConfig
from src.data_preprocessing import SolarDataPreprocessor
from src.models.feature_engineering import FeatureEngineer
from src.models.stacked_ensemble import EnhancedStackedEnsemble
from src.visualization.model_evaluation import create_visualizations


def setup_logging(config):
    """Set up logging for ensemble evaluation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOGS_DIR / f'ensemble_evaluation_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def evaluate_ensemble_model(data, config):
    """Evaluate the enhanced stacked ensemble model."""
    # Create features
    feature_engineer = FeatureEngineer()
    processed_data = feature_engineer.create_all_features(data)
    feature_sets = feature_engineer.get_feature_sets()

    # Initialize metrics collection
    all_metrics = []
    all_predictions = []

    # Use time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(processed_data), 1):
        logging.info(f"\nEvaluating fold {fold}...")

        # Split data
        train_data = processed_data.iloc[train_idx]
        test_data = processed_data.iloc[test_idx]

        # Prepare features and target
        X_train = train_data[feature_sets['all']]
        y_train = train_data['kWh']
        X_test = test_data[feature_sets['all']]
        y_test = test_data['kWh']

        # Train and evaluate ensemble
        ensemble = EnhancedStackedEnsemble(n_folds=3)
        ensemble.fit(X_train, y_train)

        # Get metrics
        metrics = ensemble.get_metrics(X_test, y_test)
        metrics['fold'] = fold
        all_metrics.append(metrics)

        # Get predictions
        predictions = ensemble.predict(X_test)
        pred_df = pd.DataFrame({
            'timestamp': test_data.index,
            'actual': y_test,
            'predicted': predictions,
            'fold': fold
        })
        all_predictions.append(pred_df)

        # Log fold results
        logging.info(f"Fold {fold} Results:")
        logging.info(f"RMSE: {metrics['rmse']:.4f}")
        logging.info(f"R²: {metrics['r2']:.4f}")
        logging.info("\nModel Weights:")
        for model, weight in metrics['model_weights'].items():
            logging.info(f"{model}: {weight:.4f}")

        if ensemble.feature_importances_ is not None:
            logging.info("\nTop 5 Feature Importances:")
            importance_df = pd.DataFrame({
                'feature': feature_sets['all'],
                'importance': ensemble.feature_importances_
            }).sort_values('importance', ascending=False).head()
            logging.info(importance_df)

    # Combine all results
    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    # Calculate average metrics
    avg_metrics = {
        'rmse': metrics_df['rmse'].mean(),
        'r2': metrics_df['r2'].mean()
    }

    logging.info("\nOverall Results:")
    logging.info(f"Average RMSE: {avg_metrics['rmse']:.4f}")
    logging.info(f"Average R²: {avg_metrics['r2']:.4f}")

    # Save results
    results_dir = config.RESULTS_DIR / 'ensemble'
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(results_dir / 'ensemble_metrics.csv', index=False)
    predictions_df.to_csv(results_dir / 'ensemble_predictions.csv', index=False)

    # Create visualizations
    viz_dir = config.VISUALIZATIONS_DIR / 'ensemble'
    viz_dir.mkdir(parents=True, exist_ok=True)

    create_visualizations(
        metrics_df,
        predictions_df,
        None,  # No feature importance visualization for now
        output_dir=str(viz_dir)
    )

    return metrics_df, predictions_df


def main():
    """Run ensemble model evaluation."""
    # Initialize configuration
    config = PipelineConfig()
    setup_logging(config)

    try:
        logging.info("Starting ensemble model evaluation...")

        # Load and preprocess data
        preprocessor = SolarDataPreprocessor(output_dir=str(config.PROCESSED_DIR))
        processed_data = preprocessor.process_all_datasets(CONFIG)
        solar_data = processed_data['solar_production']

        # Evaluate ensemble model
        metrics_df, predictions_df = evaluate_ensemble_model(solar_data, config)

        logging.info("Ensemble evaluation completed successfully")

    except Exception as e:
        logging.error(f"Error in ensemble evaluation: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
