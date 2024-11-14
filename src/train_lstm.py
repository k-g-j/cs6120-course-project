import logging

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.models.deep_learning import LSTMRegressor
from src.models.hyperparameter_tuning import tune_model_hyperparameters


def prepare_lstm_features(data):
    """Prepare features for LSTM model."""
    features_df = pd.DataFrame(index=data.index)

    # Calculate rolling features
    kWh_series = data['kWh']
    rolling = kWh_series.rolling(24, min_periods=1)

    # Create feature dictionary
    feature_dict = {
        'kWh': kWh_series,
        'kWh_rolling_mean_24h': rolling.mean(),
        'hour_cos': np.cos(2 * np.pi * data.index.hour / 24),
        'kWh_rolling_std_24h': rolling.std(),
        'kWh_lag_1h': kWh_series.shift(1)
    }

    # Create DataFrame and handle NaN values
    features_df = pd.DataFrame(feature_dict)
    features_df = features_df.ffill().bfill()

    return features_df


def train_and_evaluate_lstm(solar_data, config, param_grid):
    """Train and evaluate LSTM model with hyperparameter tuning."""
    logging.info("Training LSTM model...")

    try:
        # Use 70% of data for training
        train_size = int(len(solar_data) * 0.7)
        train_data = solar_data.iloc[:train_size]

        # Prepare features
        features_df = prepare_lstm_features(train_data)
        X = features_df.drop('kWh', axis=1)
        y = features_df['kWh']

        logging.info(f"LSTM feature shape: {X.shape}")
        logging.info(f"Feature columns: {list(X.columns)}")

        # Initialize LSTM
        lstm = LSTMRegressor(
            units=32,
            dropout=0.2,
            batch_size=64,
            epochs=20,
            sequence_length=24
        )

        # Tune hyperparameters
        logging.info("Starting hyperparameter tuning for LSTM...")
        tuned_lstm, best_params, best_score = tune_model_hyperparameters(
            lstm,
            param_grid,
            X, y,
            cv=3,
            n_iter=1
        )

        logging.info(f"Best LSTM parameters: {best_params}")

        # Evaluate on full dataset
        if tuned_lstm is not None:
            features_df_full = prepare_lstm_features(solar_data)
            X_full = features_df_full.drop('kWh', axis=1)
            y_full = features_df_full['kWh']

            # Train on full dataset
            tuned_lstm.fit(X_full, y_full)

            # Make predictions and calculate score
            predictions = tuned_lstm.predict(X_full)
            final_score = r2_score(y_full, predictions)

            # Save model checkpoint
            model_path = config.CHECKPOINT_DIR / 'lstm_best.h5'
            tuned_lstm.model.save(model_path)
            logging.info(f"Saved LSTM model to {model_path}")

            return tuned_lstm, final_score

        return None, float('inf')

    except Exception as e:
        logging.error(f"Error in LSTM training: {str(e)}", exc_info=True)
        return None, float('inf')
