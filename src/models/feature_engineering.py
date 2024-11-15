import logging

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Centralized feature engineering for all models."""

    def _ensure_datetime_index(self, df):
        """Ensure the DataFrame has a datetime index."""
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'datetime' in df.columns:
                df.set_index('datetime', inplace=True)
            elif 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'])
                df.set_index('datetime', inplace=True)
                df.drop('date', axis=1, errors='ignore', inplace=True)
            else:
                raise ValueError("No datetime or date column found in the data")
        return df

    def create_all_features(self, df):
        """Create all features needed for any model."""
        # Create a copy and ensure datetime index
        features = df.copy()
        features = self._ensure_datetime_index(features)

        logging.info(f"Creating features. Data shape: {features.shape}")
        logging.info(f"Index type: {type(features.index)}")

        # Time-based features
        features['hour'] = features.index.hour
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['is_daytime'] = ((features['hour'] >= 6) &
                                  (features['hour'] <= 18)).astype(int)
        features['is_weekend'] = features.index.dayofweek.isin([5, 6]).astype(int)

        # Lag features
        if 'kWh' in features.columns:
            features['kWh_lag_1h'] = features['kWh'].shift(1)
            features['kWh_lag_24h'] = features['kWh'].shift(24)
            features['kWh_lag_168h'] = features['kWh'].shift(168)  # 1 week

            # Rolling statistics
            features['kWh_rolling_mean_24h'] = features['kWh'].rolling(
                window=24, min_periods=1).mean()
            features['kWh_rolling_std_24h'] = features['kWh'].rolling(
                window=24, min_periods=1).std()
            features['kWh_rolling_max_24h'] = features['kWh'].rolling(
                window=24, min_periods=1).max()

        # Fill NaN values
        features = features.ffill().bfill()

        return features

    @staticmethod
    def get_feature_sets():
        """Define feature sets for different models."""
        base_features = [
            'hour_sin', 'hour_cos', 'is_daytime', 'is_weekend'
        ]

        lag_features = [
            'kWh_lag_1h', 'kWh_lag_24h', 'kWh_lag_168h'
        ]

        rolling_features = [
            'kWh_rolling_mean_24h', 'kWh_rolling_std_24h', 'kWh_rolling_max_24h'
        ]

        # Define feature groups
        feature_sets = {
            'base': base_features,
            'lag': lag_features,
            'rolling': rolling_features,
            'all': base_features + lag_features + rolling_features
        }

        return feature_sets
