import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from .cnn_model import CNNRegressor
from .lstm_model import LSTMRegressor
from .svr_model import SVRRegressor


def _ensure_datetime_index(df):
    """Ensure the dataframe has a datetime index."""
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'datetime' in df.columns:
            df.set_index('datetime', inplace=True)
        elif 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'], format='%Y/%m/%d %I:%M:%S %p')
            df.set_index('datetime', inplace=True)
            df.drop('date', axis=1, errors='ignore', inplace=True)


class AdvancedModels:
    def __init__(self, train_data, test_data, target_col='kWh'):
        self.train_data = train_data.copy()
        self.test_data = test_data.copy()
        self.target_col = target_col
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.scaler = StandardScaler()
        self.feature_cols = None

        # Ensure datetime index
        _ensure_datetime_index(self.train_data)
        _ensure_datetime_index(self.test_data)

        # Initialize models
        self._initialize_models()

        # Set memory-efficient parameters
        if 'random_forest' in self.models:
            self.models['random_forest'].set_params(n_jobs=2)

    def prepare_data(self, feature_columns=None):
        """Prepare features and target variables with scaling."""
        # Get numeric columns only
        numeric_cols = self.train_data.select_dtypes(include=['float64', 'int64']).columns

        # Use provided feature columns or default to numeric columns except target
        if feature_columns is not None:
            # Ensure all feature columns are numeric
            feature_columns = [col for col in feature_columns if col in numeric_cols]
        else:
            feature_columns = [col for col in numeric_cols if col != self.target_col]

        self.feature_cols = feature_columns

        logging.info(f"Using numeric features: {self.feature_cols}")

        # Scale features
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.train_data[self.feature_cols]),
            columns=self.feature_cols,
            index=self.train_data.index
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.test_data[self.feature_cols]),
            columns=self.feature_cols,
            index=self.test_data.index
        )

        self.y_train = self.train_data[self.target_col]
        self.y_test = self.test_data[self.target_col]

        logging.info(f"Training features shape: {self.X_train.shape}")
        logging.info(f"Training target shape: {self.y_train.shape}")

        # Log feature correlation with target
        self._log_feature_correlations()

    def _initialize_models(self):
        """Initialize all advanced models."""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=5,
                learning_rate=0.1,
                random_state=42
            ),
            'linear_sgd': SGDRegressor(
                loss='squared_error',
                penalty='l2',
                alpha=0.0001,
                max_iter=1000,
                tol=1e-3,
                random_state=42
            ),
            'lstm': LSTMRegressor(
                units=64,
                dropout=0.2,
                learning_rate=0.001,
                batch_size=64,
                epochs=20,
                sequence_length=24
            ),
            'cnn': CNNRegressor(
                filters=64,
                kernel_size=3,
                dropout=0.2,
                learning_rate=0.001,
                batch_size=64,
                epochs=20,
                sequence_length=24
            ),
            'svr': SVRRegressor(
                kernel='rbf',
                C=1.0,
                epsilon=0.1,
                gamma='scale'
            )
        }

    def train_models(self):
        """Train all advanced models with memory management."""
        model_metrics = {}
        import gc

        for name, model in self.models.items():
            # Clear memory before training each model
            gc.collect()

            logging.info(f"\nTraining {name}...")

            try:
                if name in ['lstm', 'cnn']:
                    # Set smaller batch size for deep learning models
                    model.batch_size = 32
                    model.epochs = 10  # Reduce epochs
                    model.fit(self.X_train, self.y_train)
                elif name == 'linear_sgd':
                    X_train_array = self.X_train.values
                    y_train_array = self.y_train.values
                    model.fit(X_train_array, y_train_array)
                else:
                    model.fit(self.X_train, self.y_train)

                # Make predictions and get metrics
                predictions = self.evaluate_model(name)

                if predictions is not None:
                    metrics = self.get_metrics(name)
                    model_metrics[name] = metrics

                    # Log metrics
                    logging.info(f"\nMetrics for {name}:")
                    for metric, value in metrics.items():
                        if metric not in ['model_name', 'model_type']:
                            logging.info(f"{metric.upper()}: {value:.4f}")

                    # Log feature importances
                    if hasattr(model, 'feature_importances_'):
                        self._log_feature_importances(model, name)
                    elif name == 'linear_sgd':
                        self._log_sgd_coefficients(model)

                    # Clear memory
                    del predictions
                    gc.collect()

            except Exception as e:
                logging.error(f"Error training {name}: {str(e)}")
                continue

        return model_metrics

    def evaluate_model(self, model_name):
        """Evaluate a specific model."""
        try:
            model = self.models[model_name]
            if model_name == 'linear_sgd':
                predictions = model.predict(self.X_test.values)
            else:
                predictions = model.predict(self.X_test)
            return predictions
        except Exception as e:
            logging.error(f"Error evaluating {model_name}: {str(e)}")
            return None

    def _log_feature_correlations(self):
        """Log correlation between features and target variable."""
        train_data = self.X_train.copy()
        train_data['target'] = self.y_train
        correlations = train_data.corr()['target'].drop('target').abs().sort_values(ascending=False)
        logging.info("\nFeature correlations with target:")
        logging.info(correlations)

    def _log_sgd_coefficients(self, model):
        """Log feature coefficients for SGD model."""
        coef_df = pd.DataFrame({
            'feature': self.feature_cols,
            'coefficient': np.abs(model.coef_)
        }).sort_values('coefficient', ascending=False)

        logging.info("\nFeature coefficients:")
        logging.info(coef_df)

    def _log_feature_importances(self, model, model_name):
        """Log feature importances for the model."""
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logging.info("\nFeature importances:")
        logging.info(feature_importance)

    def get_metrics(self, model_name):
        """Calculate metrics for a model."""

        predictions = self.evaluate_model(model_name)
        if predictions is None:
            return None

        metrics = {
            'model_name': model_name,
            'model_type': 'advanced',
            'rmse': np.sqrt(mean_squared_error(self.y_test, predictions)),
            'mae': mean_absolute_error(self.y_test, predictions),
            'r2': r2_score(self.y_test, predictions)
        }

        # Calculate MAPE
        non_zero_mask = self.y_test != 0
        if np.any(non_zero_mask):
            metrics['mape'] = np.mean(
                np.abs((self.y_test[non_zero_mask] - predictions[non_zero_mask]) /
                       self.y_test[non_zero_mask])) * 100
        else:
            metrics['mape'] = np.nan

        return metrics


class PreprocessedData:
    """Helper class to ensure data is properly preprocessed for models."""

    def __init__(self, data):
        self.data = data.copy()
        self._preprocess()

    def _preprocess(self):
        if not isinstance(self.data.index, pd.DatetimeIndex):
            if 'date' in self.data.columns:
                self.data['datetime'] = pd.to_datetime(
                    self.data['date'],
                    format='%Y/%m/%d %I:%M:%S %p'
                )
                self.data.set_index('datetime', inplace=True)
                self.data.drop('date', axis=1, errors='ignore', inplace=True)

        # Sort index
        self.data.sort_index(inplace=True)

    def get_data(self):
        return self.data.copy()
