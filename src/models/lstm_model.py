import numpy as np
import pandas as pd
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense, LSTM, Dropout, Input, LayerNormalization, Concatenate
from keras.src.models import Model
from keras.src.optimizers import Adam
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler


class LSTMRegressor(BaseEstimator, RegressorMixin):
    """LSTM model with attention and residual connections."""

    def __init__(self, units=64, dropout=0.2, learning_rate=0.001,
                 batch_size=64, epochs=20, sequence_length=24):
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()

    def build_model(self, n_features):
        """Build improved LSTM model."""
        # Sequential features
        seq_input = Input(shape=(self.sequence_length, n_features), name='sequential_input')

        # First LSTM layer with residual connection
        x1 = LSTM(self.units, return_sequences=True)(seq_input)
        x1 = LayerNormalization()(x1)
        x1 = Dropout(self.dropout)(x1)

        # Second LSTM layer
        x2 = LSTM(self.units // 2)(x1)
        x2 = LayerNormalization()(x2)
        x2 = Dropout(self.dropout)(x2)

        # Dense layers with residual connections
        x3 = Dense(32, activation='relu')(x2)
        x4 = Dense(16, activation='relu')(x3)
        x4 = Concatenate()([x4, Dense(16)(x2)])  # Residual connection

        # Output layer
        outputs = Dense(1)(x4)

        # Create model
        model = Model(inputs=seq_input, outputs=outputs)

        # Compile
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model

    def create_sequences(self, X):
        """Create sequences efficiently."""
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        n_samples = len(X_scaled) - self.sequence_length
        sequences = np.zeros((n_samples, self.sequence_length, X.shape[1]))

        for i in range(self.sequence_length):
            sequences[:, i, :] = X_scaled[i:i + n_samples]

        return sequences

    def fit(self, X, y):
        """Train the model with early stopping."""
        try:
            # Create sequences
            X_seq = self.create_sequences(X)
            y_seq = y[self.sequence_length:]

            # Build model
            if self.model is None:
                self.model = self.build_model(n_features=X.shape[1])

            # Early stopping
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    mode='min'
                )
            ]

            # Train model
            self.model.fit(
                X_seq, y_seq,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.1,
                callbacks=callbacks,
                verbose=1
            )

            return self

        except Exception as e:
            print(f"Error in LSTM fit: {str(e)}")
            raise

    def predict(self, X):
        """Generate predictions."""
        X_seq = self.create_sequences(X)
        predictions = self.model.predict(
            X_seq,
            batch_size=self.batch_size,
            verbose=0
        )

        # Create full predictions array
        full_predictions = np.full(len(X), np.nan)
        full_predictions[self.sequence_length:] = predictions.flatten()
        full_predictions[:self.sequence_length] = predictions.mean()

        return full_predictions
