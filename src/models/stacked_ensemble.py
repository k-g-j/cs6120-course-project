import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


class EnhancedStackedEnsemble(BaseEstimator, RegressorMixin):
    """Enhanced stacked ensemble combining multiple models with meta-learner."""

    def __init__(self, n_folds=5, use_predictions_as_features=True):
        self.n_folds = n_folds
        self.use_predictions_as_features = use_predictions_as_features

        # Initialize base models
        self.base_models = {
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }

        # Meta-learner with built-in feature selection
        self.meta_learner = LassoCV(
            cv=3,
            random_state=42,
            max_iter=2000,
            selection='random'
        )

        self.feature_importances_ = None
        self.model_weights_ = None

    def _generate_meta_features(self, X, y=None, is_train=True):
        """Generate meta-features using time series cross-validation."""
        n_samples = X.shape[0]
        n_models = len(self.base_models)

        # Initialize meta-features array
        meta_features = np.zeros((n_samples, n_models))

        if is_train:
            # Use time series cross-validation for training
            tscv = TimeSeriesSplit(n_splits=self.n_folds)

            for train_idx, val_idx in tscv.split(X):
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train = y.iloc[train_idx] if y is not None else None

                # Train each base model and generate predictions
                for i, (name, model) in enumerate(self.base_models.items()):
                    if is_train:
                        model.fit(X_fold_train, y_fold_train)
                    meta_features[val_idx, i] = model.predict(X_fold_val)

        else:
            # For test data, use fully trained models
            for i, (name, model) in enumerate(self.base_models.items()):
                meta_features[:, i] = model.predict(X)

        if self.use_predictions_as_features:
            # Add original features alongside model predictions
            meta_features = np.column_stack([meta_features, X])

        return meta_features

    def fit(self, X, y):
        """Fit ensemble using time series cross-validation."""
        # Generate meta-features for training
        meta_features = self._generate_meta_features(X, y, is_train=True)

        # Train meta-learner
        self.meta_learner.fit(meta_features, y)

        # Train base models on full dataset
        for name, model in self.base_models.items():
            model.fit(X, y)

        # Calculate feature importances and model weights
        self._calculate_importance_weights(meta_features)

        return self

    def predict(self, X):
        """Generate predictions using the stacked ensemble."""
        # Generate meta-features for prediction
        meta_features = self._generate_meta_features(X, is_train=False)
        return self.meta_learner.predict(meta_features)

    def _calculate_importance_weights(self, meta_features):
        """Calculate feature importances and model weights."""
        # Get coefficients from meta-learner
        coef = self.meta_learner.coef_

        # Calculate normalized weights for base models
        n_models = len(self.base_models)
        model_weights = np.abs(coef[:n_models])
        self.model_weights_ = model_weights / np.sum(model_weights)

        # Calculate feature importances if using original features
        if self.use_predictions_as_features:
            feature_importances = np.abs(coef[n_models:])
            self.feature_importances_ = feature_importances / np.sum(feature_importances)

    def get_model_weights(self):
        """Return the importance weight of each base model."""
        if self.model_weights_ is None:
            raise ValueError("Model has not been fitted yet")

        return dict(zip(self.base_models.keys(), self.model_weights_))

    def score(self, X, y):
        """Calculate R² score for the ensemble."""
        predictions = self.predict(X)
        return r2_score(y, predictions)

    def get_metrics(self, X, y):
        """Calculate comprehensive metrics for the ensemble."""
        predictions = self.predict(X)

        return {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions),
            'model_weights': self.get_model_weights()
        }
