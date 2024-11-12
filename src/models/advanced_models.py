import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def _ensure_datetime_index(df):
    """Ensure the dataframe has a datetime index."""
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'datetime' in df.columns:
            df.set_index('datetime', inplace=True)
        elif 'date' in df.columns:
            # Parse with explicit format to avoid warnings
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

        # Ensure datetime index
        _ensure_datetime_index(self.train_data)
        _ensure_datetime_index(self.test_data)

    def create_advanced_features(self, df):
        """Create enhanced features including lagged and rolling features."""
        df = df.copy()

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            _ensure_datetime_index(df)

        # Basic time components
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofweek'] = df.index.dayofweek
        df['dayofyear'] = df.index.dayofyear

        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)

        # Time of day features
        df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
        df['is_peak_sun'] = ((df['hour'] >= 10) & (df['hour'] <= 14)).astype(int)
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

        # Add lagged features
        df['kWh_lag_1h'] = df[self.target_col].shift(1)
        df['kWh_lag_24h'] = df[self.target_col].shift(24)
        df['kWh_lag_168h'] = df[self.target_col].shift(168)  # 1 week

        # Add rolling statistics
        df['kWh_rolling_mean_24h'] = df[self.target_col].rolling(window=24, min_periods=1).mean()
        df['kWh_rolling_std_24h'] = df[self.target_col].rolling(window=24, min_periods=1).std()
        df['kWh_rolling_max_24h'] = df[self.target_col].rolling(window=24, min_periods=1).max()

        # Handle NaN values
        df = df.bfill().ffill()  # Use bfill followed by ffill to handle any remaining NaNs

        return df

    def prepare_data(self):
        """Prepare features and target variables with scaling."""
        # Create advanced features
        train_with_features = self.create_advanced_features(self.train_data)
        test_with_features = self.create_advanced_features(self.test_data)

        # Select features for training
        self.feature_cols = [
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'day_sin', 'day_cos', 'is_weekend', 'is_daytime',
            'is_peak_sun', 'kWh_lag_1h', 'kWh_lag_24h',
            'kWh_lag_168h', 'kWh_rolling_mean_24h',
            'kWh_rolling_std_24h', 'kWh_rolling_max_24h'
        ]

        # Scale features
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(train_with_features[self.feature_cols]),
            columns=self.feature_cols,
            index=train_with_features.index
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(test_with_features[self.feature_cols]),
            columns=self.feature_cols,
            index=test_with_features.index
        )

        self.y_train = train_with_features[self.target_col]
        self.y_test = test_with_features[self.target_col]

        logging.info(f"Training features shape: {self.X_train.shape}")
        logging.info(f"Training target shape: {self.y_train.shape}")

        # Log feature correlation with target
        self._log_feature_correlations()

    def _log_feature_correlations(self):
        """Log correlation between features and target variable."""
        # Combine features and target for correlation calculation
        train_data = self.X_train.copy()
        train_data['target'] = self.y_train

        # Calculate correlations with target
        correlations = train_data.corr()['target'].drop('target').abs().sort_values(ascending=False)

        logging.info("\nFeature correlations with target:")
        logging.info(correlations)

    def train_models(self):
        """Train and evaluate advanced models with optimized parameters."""
        models = {
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
            'linear_sgd': SGDRegressor(  # Replace SVR with SGDRegressor
                loss='squared_error',
                penalty='l2',
                alpha=0.0001,
                max_iter=1000,
                tol=1e-3,
                random_state=42
            )
        }

        for name, model in models.items():
            logging.info(f"\nTraining {name}...")

            try:
                if name == 'linear_sgd':
                    # For SGD, standardize the data and convert to dense arrays
                    X_train = self.X_train.values
                    y_train = self.y_train.values
                    X_test = self.X_test.values
                else:
                    X_train = self.X_train
                    y_train = self.y_train
                    X_test = self.X_test

                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                # Store model and predictions
                self.models[name] = model
                self.predictions[name] = {
                    'train': train_pred,
                    'test': test_pred
                }

                # Calculate metrics
                self.metrics[name] = self._calculate_metrics(
                    self.y_test, test_pred, name
                )

                # Log feature importances for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self._log_feature_importances(model, name)
                elif name == 'linear_sgd':
                    self._log_sgd_coefficients(model)

            except Exception as e:
                logging.error(f"Error in {name} training: {str(e)}")
                continue

        return self.metrics

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

    def _calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate comprehensive performance metrics."""
        metrics = {
            'model_name': model_name,
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

        # Calculate MAPE only for non-zero values
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            metrics['mape'] = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) /
                                             y_true[non_zero_mask])) * 100

        logging.info(f"\nMetrics for {model_name}:")
        for metric, value in metrics.items():
            if metric != 'model_name':
                logging.info(f"{metric.upper()}: {value:.4f}")

        return metrics

    def get_all_metrics(self):
        """Return metrics for all trained models."""
        return pd.DataFrame([metrics for metrics in self.metrics.values()])


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
