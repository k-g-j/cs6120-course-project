import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


class BaselineModels:
    def __init__(self, train_data, test_data, target_col='kWh'):
        self.train_data = train_data.copy()
        self.test_data = test_data.copy()
        self.target_col = target_col
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.scaler = StandardScaler()

        # Ensure datetime index
        self._ensure_datetime_index(self.train_data)
        self._ensure_datetime_index(self.test_data)

    def _ensure_datetime_index(self, df):
        """Ensure the dataframe has a datetime index."""
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'datetime' in df.columns:
                df.set_index('datetime', inplace=True)
            elif 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'])
                df.set_index('datetime', inplace=True)
                df.drop('date', axis=1, errors='ignore', inplace=True)
            else:
                raise ValueError("No datetime column found in the data")

    def create_time_features(self, df):
        """Create enhanced time-based features from datetime index."""
        df = df.copy()

        # Verify datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            self._ensure_datetime_index(df)

        # Basic time components
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['dayofyear'] = df.index.dayofyear
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

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

        # Season features
        seasons = pd.cut(df['month'],
                         bins=[0, 3, 6, 9, 12],
                         labels=['winter', 'spring', 'summer', 'fall'])
        season_dummies = pd.get_dummies(seasons, prefix='season')
        df = pd.concat([df, season_dummies], axis=1)

        return df

    def prepare_data(self):
        """Prepare features and target variables with scaling."""
        try:
            # Create time-based features
            train_with_features = self.create_time_features(self.train_data)
            test_with_features = self.create_time_features(self.test_data)

            # Select features for training
            feature_cols = [
                'hour_sin', 'hour_cos',
                'month_sin', 'month_cos',
                'day_sin', 'day_cos',
                'is_weekend', 'is_daytime', 'is_peak_sun',
                'season_winter', 'season_spring',
                'season_summer', 'season_fall'
            ]

            # Scale features
            self.X_train = pd.DataFrame(
                self.scaler.fit_transform(train_with_features[feature_cols]),
                columns=feature_cols,
                index=train_with_features.index
            )
            self.X_test = pd.DataFrame(
                self.scaler.transform(test_with_features[feature_cols]),
                columns=feature_cols,
                index=test_with_features.index
            )

            self.y_train = train_with_features[self.target_col]
            self.y_test = test_with_features[self.target_col]

            logging.info(f"Training features shape: {self.X_train.shape}")
            logging.info(f"Training target shape: {self.y_train.shape}")

        except Exception as e:
            logging.error(f"Error in data preparation: {str(e)}")
            logging.debug("Train data columns: %s", self.train_data.columns.tolist())
            logging.debug("Train data index type: %s", type(self.train_data.index))
            raise

    def train_linear_models(self):
        """Train and evaluate multiple linear models."""
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0)
        }

        for name, model in models.items():
            logging.info(f"\nTraining {name}...")

            try:
                # Train model
                model.fit(self.X_train, self.y_train)

                # Make predictions
                train_pred = model.predict(self.X_train)
                test_pred = model.predict(self.X_test)

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

                # Log feature importances
                if hasattr(model, 'coef_'):
                    feature_importance = pd.DataFrame({
                        'feature': self.X_train.columns,
                        'importance': np.abs(model.coef_)
                    })
                    logging.info("\nFeature importances:")
                    logging.info(feature_importance.sort_values('importance',
                                                                ascending=False))

            except Exception as e:
                logging.error(f"Error in {name} training: {str(e)}")
                continue

        return self.metrics

    def _calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate comprehensive performance metrics."""
        metrics = {
            'model_name': model_name,
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'adjusted_r2': 1 - (1 - r2_score(y_true, y_pred)) * (len(y_true) - 1) / (
                    len(y_true) - self.X_train.shape[1] - 1)
        }

        # Calculate MAPE only if there are no zero values in y_true
        if not np.any(y_true == 0):
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        logging.info(f"\nMetrics for {model_name}:")
        for metric, value in metrics.items():
            if metric != 'model_name':
                logging.info(f"{metric.upper()}: {value:.4f}")

        return metrics

    def get_all_metrics(self):
        """Return metrics for all trained models."""
        return pd.DataFrame([metrics for metrics in self.metrics.values()])
