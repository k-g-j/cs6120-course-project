import logging

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.models.advanced_models import AdvancedModels
from src.models.feature_engineering import FeatureEngineer
from src.models.hyperparameter_tuning import tune_model_hyperparameters
from src.visualization.model_evaluation import create_visualizations, save_model_artifacts


def train_and_evaluate_advanced_models(data, config, param_grids):
    """Train and evaluate advanced models with hyperparameter tuning."""
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
        logging.info(f"Training fold {fold}/5...")

        try:
            # Split data
            train_data = processed_data.iloc[train_idx]
            test_data = processed_data.iloc[test_idx]

            # Initialize models
            advanced = AdvancedModels(train_data, test_data, target_col='kWh')
            advanced.prepare_data(feature_columns=feature_sets['base'])

            # Train models first
            fold_metrics = advanced.train_models()

            if not fold_metrics:
                logging.warning(f"No models trained successfully in fold {fold}")
                continue

            # Tune hyperparameters for trained models
            for model_name in fold_metrics.keys():
                if model_name in param_grids:
                    logging.info(f"\nTuning {model_name}...")
                    param_grid = param_grids[model_name]

                    # Tune hyperparameters
                    best_model_fold, best_params, best_score = tune_model_hyperparameters(
                        advanced.models[model_name],
                        param_grid,
                        advanced.X_train,
                        advanced.y_train,
                        cv=3
                    )

                    if best_model_fold is not None:
                        logging.info(f"Best parameters for {model_name}: {best_params}")

                        # Update model and get new predictions
                        advanced.models[model_name] = best_model_fold
                        predictions = advanced.evaluate_model(model_name)

                        if predictions is not None:
                            # Get updated metrics
                            metrics = advanced.get_metrics(model_name)
                            metrics['fold'] = fold
                            all_metrics.append(metrics)

                            # Save model artifacts
                            save_model_artifacts(
                                best_model_fold,
                                model_name,
                                fold,
                                metrics,
                                output_dir=str(config.MODEL_DIR)
                            )

                            # Track best model
                            if metrics['r2'] > best_r2:
                                best_r2 = metrics['r2']
                                best_model = (model_name, best_model_fold)

                                if hasattr(best_model_fold, 'feature_importances_'):
                                    feature_importance_data = pd.DataFrame({
                                        'feature': advanced.feature_cols,
                                        'importance': best_model_fold.feature_importances_
                                    })

                            # Collect predictions
                            predictions_df = pd.DataFrame({
                                'timestamp': test_data.index,
                                'actual': advanced.y_test,
                                'predicted': predictions,
                                'model': model_name,
                                'fold': fold
                            })
                            all_predictions.append(predictions_df)

        except Exception as e:
            logging.error(f"Error in fold {fold}: {str(e)}")
            continue

    if not all_metrics:
        raise ValueError("No models were successfully trained")

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    # Combine predictions
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
