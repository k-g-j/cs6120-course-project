import logging

import numpy as np
import pandas as pd
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LSTM, Dense, Dropout
from keras.src.models import Sequential
from keras.src.optimizers import Adam
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler


class LSTMRegressor(BaseEstimator, RegressorMixin):
    """LSTM model for time series forecasting."""

    def __init__(self, units=50, dropout=0.2, learning_rate=0.001,
                 batch_size=32, epochs=100, sequence_length=24):
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()

    def create_sequences(self, X):
        """Create sequences for LSTM input"""
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        sequences = []
        for i in range(len(X_scaled) - self.sequence_length):
            sequences.append(X_scaled[i:(i + self.sequence_length)])
        return np.array(sequences)

    def fit(self, X, y):
        """Fit LSTM model."""
        try:
            # Create sequences
            X_seq = self.create_sequences(X)
            y_seq = y[self.sequence_length:]

            # Build model if not already built
            if self.model is None:
                self.model = self._build_model(input_shape=(self.sequence_length, X.shape[1]))

            # Add early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            # Train model
            self.model.fit(
                X_seq, y_seq,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1
            )

            return self

        except Exception as e:
            logging.error(f"Error in LSTM fit: {str(e)}", exc_info=True)
            raise

    def predict(self, X):
        """Generate predictions."""
        # Create sequences
        X_seq = self.create_sequences(X)

        # Make predictions
        predictions = self.model.predict(X_seq)

        # Pad beginning with the mean prediction to match input length
        full_predictions = np.full(len(X), np.nan)
        full_predictions[self.sequence_length:] = predictions.flatten()
        full_predictions[:self.sequence_length] = predictions.mean()

        return full_predictions

    def _build_model(self, input_shape):
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=input_shape),
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
