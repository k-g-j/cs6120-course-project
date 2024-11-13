import logging

from src.models.hyperparameter_tuning import get_hyperparameter_grids, tune_model_hyperparameters
from src.models.lstm_model import LSTMRegressor


def prepare_data(self, X):
    """Prepare data for LSTM by handling non-numeric columns."""
    # Drop or encode location columns
    location_columns = X.select_dtypes(include=['object']).columns
    X = X.drop(columns=location_columns)

    # Scale numeric features
    X_scaled = self.scaler.fit_transform(X)
    return X_scaled


def train_and_evaluate_lstm(solar_data, config):
    """Train and evaluate the LSTM model."""
    logging.info("Training LSTM model...")

    # Only select numeric columns and drop target column for features
    numeric_cols = solar_data.select_dtypes(include=['float64', 'int64']).columns
    feature_cols = [col for col in numeric_cols if col != 'kWh']

    X = solar_data[feature_cols]
    y = solar_data['kWh']

    logging.info(f"LSTM features shape: {X.shape}")
    logging.info(f"Feature columns: {feature_cols}")

    # Initialize LSTM model
    lstm = LSTMRegressor()

    # Get hyperparameter grid
    param_grids = get_hyperparameter_grids()

    try:
        # Tune hyperparameters
        logging.info("Starting hyperparameter tuning for LSTM...")
        tuned_lstm, best_params, best_score = tune_model_hyperparameters(
            lstm,
            param_grids['lstm'],
            X, y
        )

        logging.info(f"Best LSTM parameters: {best_params}")
        return tuned_lstm, abs(best_score)

    except Exception as e:
        logging.error(f"Error in LSTM training: {str(e)}")
        return None, float('inf')
