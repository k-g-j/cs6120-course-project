import logging

import numpy as np
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LSTM, Dense, Dropout
from keras.src.models import Sequential
from keras.src.optimizers import Adam
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler


class LSTMRegressor(BaseEstimator, RegressorMixin):
    """LSTM model for time series forecasting of solar energy production."""

    def __init__(self, units=50, dropout=0.2, learning_rate=0.001, batch_size=32, epochs=100):
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.scaler = MinMaxScaler()
        self.sequence_length = 24  # Use 24 hours of history
        self.model = None

    def create_sequences(self, X, y=None):
        """Create sequences for LSTM input."""
        sequences = []
        targets = []

        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i:i + self.sequence_length])
            if y is not None:
                targets.append(y[i + self.sequence_length])

        if y is not None:
            return np.array(sequences), np.array(targets)
        return np.array(sequences)

    def build_model(self, input_shape):
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(self.units, return_sequences=True,
                 input_shape=input_shape),
            Dropout(self.dropout),
            LSTM(self.units // 2),
            Dropout(self.dropout),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )

        return model

    def fit(self, X, y):
        """Fit LSTM model."""
        logging.info("Preparing data for LSTM...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y)

        logging.info(f"Created sequences shape: {X_seq.shape}")

        # Build model
        self.model = self.build_model((self.sequence_length, X.shape[1]))

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        logging.info("Training LSTM model...")

        # Train model
        history = self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

        # Log training results
        val_loss = min(history.history['val_loss'])
        logging.info(f"Best validation loss: {val_loss:.4f}")

        return self

    def predict(self, X):
        """Generate predictions using LSTM model."""
        # Scale features
        X_scaled = self.scaler.transform(X)

        # Create sequences
        X_seq = self.create_sequences(X_scaled)

        # Make predictions
        y_pred = self.model.predict(X_seq)

        # Pad predictions to match input length
        padded_predictions = np.full(len(X), np.nan)
        padded_predictions[self.sequence_length:] = y_pred.flatten()

        # Fill initial values using the mean
        padded_predictions[:self.sequence_length] = np.mean(y_pred)

        return padded_predictions

    def get_params(self, deep=True):
        """Get parameters for the estimator."""
        return {
            'units': self.units,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }

    def set_params(self, **parameters):
        """Set parameters for the estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
