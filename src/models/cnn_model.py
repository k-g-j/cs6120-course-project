import numpy as np
import pandas as pd
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense, Conv1D, GlobalAveragePooling1D, Input, Dropout
from keras.src.models import Model
from keras.src.optimizers import Adam
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler


class CNNRegressor(BaseEstimator, RegressorMixin):
    """CNN model for time series forecasting."""

    def __init__(self, filters=64, kernel_size=3, dropout=0.2,
                 learning_rate=0.001, batch_size=64, epochs=20,
                 sequence_length=24):
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()

    def build_model(self, n_features):
        """Build CNN model using functional API."""
        inputs = Input(shape=(self.sequence_length, n_features))

        # First Conv layer
        x = Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation='relu',
            padding='same'
        )(inputs)
        x = Dropout(self.dropout)(x)

        # Second Conv layer
        x = Conv1D(
            filters=self.filters // 2,
            kernel_size=self.kernel_size,
            activation='relu',
            padding='same'
        )(x)
        x = Dropout(self.dropout)(x)

        # Global pooling
        x = GlobalAveragePooling1D()(x)

        # Dense layers
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)

        # Create model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model

    def create_sequences(self, X):
        """Create sequences for CNN input."""
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
        """Train the CNN model."""
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
                    patience=3,
                    restore_best_weights=True
                )
            ]

            # Train
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
            print(f"Error in CNN fit: {str(e)}")
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
