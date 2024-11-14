import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit


class StackedEnsembleRegressor(BaseEstimator, RegressorMixin):
    """Stacked ensemble combining multiple models with meta-learner."""

    def __init__(self, n_folds=5):
        self.n_folds = n_folds

        # Base models
        self.rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        self.gb = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

        # Meta-learner
        self.meta_learner = LassoCV(
            cv=3,
            random_state=42,
            max_iter=1000
        )

        self.base_models = [self.rf, self.gb]

    def fit(self, X, y):
        """Fit ensemble using time series cross-validation."""
        # Initialize arrays for meta-features
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_folds)

        # Generate predictions from base models
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            for i, model in enumerate(self.base_models):
                model.fit(X_train, y_train)
                meta_features[val_idx, i] = model.predict(X_val)

        # Train base models on full dataset
        for i, model in enumerate(self.base_models):
            model.fit(X, y)

        # Train meta-learner
        self.meta_learner.fit(meta_features, y)

        return self

    def predict(self, X):
        """Generate predictions using the stacked ensemble."""
        meta_features = np.column_stack([
            model.predict(X) for model in self.base_models
        ])

        return self.meta_learner.predict(meta_features)
