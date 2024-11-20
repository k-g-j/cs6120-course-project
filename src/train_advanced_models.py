import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.models.advanced_models import AdvancedModels
from src.models.feature_engineering import FeatureEngineer
from src.models.hyperparameter_tuning import tune_model_hyperparameters
from src.visualization.model_evaluation import create_visualizations, save_model_artifacts


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
                        current_model = advanced.models[
                            model_name]  # Get the current model instance
                        # Tune hyperparameters
                        best_model_fold, best_params, best_score = tune_model_hyperparameters(
                            current_model,
                            param_grid,
                            advanced.X_train,
                            advanced.y_train,
                            cv=3
                        )

                        # Before model fitting
                        feature_names = X_train.columns.tolist()
                        model.set_params(feature_names_in_=feature_names)


                        if best_model_fold is not None:
                            logging.info(f"Best parameters for {model_name}: {best_params}")

                            # Update model and get new predictions
                            advanced.models[model_name] = best_model_fold
                            predictions = advanced.evaluate_model(model_name)

                            if predictions is not None:
                                # Get updated metrics
                                new_metrics = advanced.get_metrics(model_name)
                                new_metrics['fold'] = fold
                                fold_metrics.append(new_metrics)

                                # Save model artifacts
                                save_model_artifacts(
                                    best_model_fold,
                                    model_name,
                                    fold,
                                    new_metrics,
                                    output_dir=str(config.MODEL_DIR)
                                )

                                # Track best model
                                if new_metrics['r2'] > best_r2:
                                    best_r2 = new_metrics['r2']
                                    best_model = (model_name, best_model_fold)

                                    if hasattr(best_model_fold, 'feature_importances_'):
                                        feature_importance_data = pd.DataFrame({
                                            'feature': advanced.feature_cols,
                                            'importance': best_model_fold.feature_importances_
                                        })

                                # Collect predictions
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
