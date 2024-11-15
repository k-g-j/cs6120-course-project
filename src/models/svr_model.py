from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class SVRRegressor(BaseEstimator, RegressorMixin):
    """SVR model wrapper with preprocessing."""

    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """Fit SVR model with scaled features."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Initialize and train model
        self.model = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma
        )

        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        """Generate predictions with scaled features."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)