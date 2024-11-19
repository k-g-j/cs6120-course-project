import logging

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class SVRRegressor(BaseEstimator, RegressorMixin):
    """SVR model wrapper with preprocessing."""

    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale', max_samples=10000):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.max_samples = max_samples
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """Fit SVR model with scaled features and subsampling for large datasets."""
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        # Subsample if dataset is too large
        if len(X) > self.max_samples:
            logging.info(
                f"Subsampling SVR training data from {len(X)} to {self.max_samples} samples")
            indices = np.random.choice(len(X), self.max_samples, replace=False)
            X_sampled = X[indices]
            y_sampled = y[indices]
        else:
            X_sampled = X
            y_sampled = y

        # Scale features
        X_scaled = self.scaler.fit_transform(X_sampled)

        # Initialize and train model
        self.model = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma
        )

        self.model.fit(X_scaled, y_sampled)
        return self

    def predict(self, X):
        """Generate predictions with scaled features."""
        if hasattr(X, 'values'):
            X = X.values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
