import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.config import CONFIG
from src.data_preprocessing import SolarDataPreprocessor
from src.models.advanced_models import AdvancedModels
from src.models.feature_engineering import FeatureEngineer
from src.models.hyperparameter_tuning import tune_model_hyperparameters
from src.pipeline_runner import PipelineConfig
from src.pipeline_runner import prepare_data_for_modeling
from src.visualization.model_evaluation import generate_model_report, save_model_artifacts, \
    create_visualizations


class AdvancedPipelineConfig(PipelineConfig):
    """Extended pipeline configuration for advanced models."""

    def __init__(self):
        super().__init__()
        # Use paths from parent and config
        self.ENSEMBLE_DIR = self.MODEL_DIR / 'ensemble'
        self.DEEP_LEARNING_DIR = self.MODEL_DIR / 'deep_learning'
        self.TUNING_RESULTS_DIR = self.RESULTS_DIR / 'hyperparameter_tuning'
        self.CHECKPOINT_DIR = self.MODEL_DIR / 'checkpoints'

        # Create additional directories
        self._create_advanced_directories()

    def _create_advanced_directories(self):
        """Create directories for advanced models."""
        advanced_directories = [
            self.ENSEMBLE_DIR,
            self.DEEP_LEARNING_DIR,
            self.TUNING_RESULTS_DIR,
            self.CHECKPOINT_DIR
        ]

        for directory in advanced_directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logging.info(f"✓ Created/verified directory: {directory}")
            except Exception as e:
                logging.error(f"✗ Error creating directory {directory}: {str(e)}")
                raise


def setup_advanced_logging(config):
    """Set up logging for advanced pipeline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(config.LOGS_DIR) / f'advanced_pipeline_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format=CONFIG['log_format'],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def train_and_evaluate_advanced_models(data, config, param_grids):
    """Train and evaluate advanced models with checkpointing."""
    logging.info("Starting advanced model training with hyperparameter tuning...")

    # Create features using FeatureEngineer
    feature_engineer = FeatureEngineer()
    processed_data = feature_engineer.create_all_features(data)
    feature_sets = feature_engineer.get_feature_sets()

    tscv = TimeSeriesSplit(n_splits=5)
    all_metrics = []
    all_predictions = []
    best_model = None
    best_r2 = -float('inf')
    feature_importance_data = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(processed_data), 1):
        # Add checkpoint file
        checkpoint_file = Path(config.CHECKPOINT_DIR) / f'fold_{fold}_checkpoint.pkl'
        fold_metrics = []
        fold_predictions = []

        try:
            # Split data
            train_data = processed_data.iloc[train_idx]
            test_data = processed_data.iloc[test_idx]

            # Initialize models
            advanced = AdvancedModels(train_data, test_data, target_col='kWh')
            advanced.prepare_data(feature_columns=feature_sets['base'])

            # Train models first
            model_metrics = advanced.train_models()

            if not model_metrics:
                logging.warning(f"No models trained successfully in fold {fold}")
                continue

            # Tune hyperparameters for trained models that have parameter grids
            for model_name, metrics in model_metrics.items():
                if model_name in param_grids:
                    logging.info(f"\nTuning {model_name}...")
                    param_grid = param_grids[model_name]

                    try:
                        current_model = advanced.models[model_name]
                        feature_names = advanced.X_train.columns.tolist()

                        # Create DataFrames with feature names first
                        X_train_with_names = pd.DataFrame(
                            advanced.X_train,
                            columns=feature_names,
                            index=advanced.X_train.index
                        )
                        X_test_with_names = pd.DataFrame(
                            advanced.X_test,
                            columns=feature_names,
                            index=advanced.X_test.index
                        )

                        # Use X_train_with_names in hyperparameter tuning
                        best_model_fold, best_params, best_score = tune_model_hyperparameters(
                            current_model,
                            param_grid,
                            X_train_with_names,
                            advanced.y_train,
                            cv=3
                        )

                        if best_model_fold is not None:
                            logging.info(f"Best parameters for {model_name}: {best_params}")
                            advanced.models[model_name] = best_model_fold

                            best_model_fold.fit(X_train_with_names, advanced.y_train)
                            predictions = best_model_fold.predict(X_test_with_names)

                            if predictions is not None:
                                new_metrics = advanced.get_metrics(model_name)
                                new_metrics['fold'] = fold
                                fold_metrics.append(new_metrics)

                                save_model_artifacts(
                                    best_model_fold,
                                    model_name,
                                    fold,
                                    new_metrics,
                                    output_dir=str(config.MODEL_DIR)
                                )

                                if new_metrics['r2'] > best_r2:
                                    best_r2 = new_metrics['r2']
                                    best_model = (model_name, best_model_fold)

                                    if hasattr(best_model_fold, 'feature_importances_'):
                                        feature_importance_data = pd.DataFrame({
                                            'feature': feature_names,
                                            'importance': best_model_fold.feature_importances_
                                        })

                                pred_df = pd.DataFrame({
                                    'timestamp': test_data.index,
                                    'actual': advanced.y_test,
                                    'predicted': predictions,
                                    'model': model_name,
                                    'fold': fold
                                })
                                fold_predictions.append(pred_df)

                    except Exception as e:
                        logging.error(f"Error tuning {model_name}: {str(e)}")
                        continue

            # Save checkpoint after successful fold
            if fold_metrics and fold_predictions:
                all_metrics.extend(fold_metrics)
                all_predictions.extend(fold_predictions)
                checkpoint_data = {
                    'metrics': fold_metrics,
                    'predictions': fold_predictions,
                    'best_model': best_model
                }
                pd.to_pickle(checkpoint_data, checkpoint_file)
                logging.info(f"Saved checkpoint for fold {fold}")

        except Exception as e:
            logging.error(f"Error in fold {fold}: {str(e)}")
            continue

    if not all_metrics:
        raise ValueError("No models were successfully trained")

    # Create metrics DataFrame and combine predictions
    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
    predictions_df.set_index('timestamp', inplace=True)

    # Create visualizations
    create_visualizations(
        metrics_df,
        predictions_df,
        feature_importance_data,
        output_dir=str(config.VISUALIZATIONS_DIR)
    )

    return metrics_df, best_model


def main():
    """Run the complete advanced pipeline."""
    # Set worker timeout and configure multiprocessing
    import multiprocessing
    import os

    # Set timeout and configure multiprocessing
    os.environ["LOKY_MAX_CPU_TIME"] = "1800"  # 30 minutes timeout
    multiprocessing.set_start_method('spawn', force=True)

    config = AdvancedPipelineConfig()
    setup_advanced_logging(config)

    try:
        logging.info("Starting advanced pipeline...")

        # Run preprocessing
        logging.info("Running data preprocessing...")
        preprocessor = SolarDataPreprocessor(output_dir=str(config.PROCESSED_DIR))
        processed_data = preprocessor.process_all_datasets(CONFIG)

        # Extract and prepare solar production data
        solar_data = processed_data['solar_production']
        logging.info(f"Initial data shape: {solar_data.shape}")

        # Prepare data for modeling
        solar_data = prepare_data_for_modeling(solar_data)
        logging.info(f"Processed data shape: {solar_data.shape}")

        # Get hyperparameter grids
        from src.models.hyperparameter_tuning import get_hyperparameter_grids
        param_grids = get_hyperparameter_grids()

        # Run advanced model training and evaluation
        metrics_df, best_model = train_and_evaluate_advanced_models(
            solar_data,
            config,
            param_grids
        )

        # Save metrics
        metrics_file = Path(CONFIG['final_metrics'])
        metrics_df.to_csv(metrics_file, index=False)
        logging.info(f"✓ Saved final metrics to: {metrics_file}")

        # Generate report
        generate_model_report(metrics_df, config)
        logging.info("✓ Generated model report")

        # Find best model across all models
        best_idx = metrics_df['r2'].idxmax()
        best_model_metrics = metrics_df.iloc[best_idx]

        # Log best model performance
        logging.info("\nBest model performance:")
        logging.info(
            f"Model: {best_model_metrics['model_name']} ({best_model_metrics['model_type']})"
        )
        logging.info(f"R²: {float(best_model_metrics['r2']):.4f}")
        logging.info(f"RMSE: {float(best_model_metrics['rmse']):.4f}")

        logging.info("✓ Advanced pipeline completed successfully")

    except Exception as e:
        logging.error(f"✗ Advanced pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
