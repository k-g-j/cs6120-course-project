import numpy as np


class FeatureEngineer:
    """Centralized feature engineering for all models."""

    @staticmethod
    def create_all_features(df):
        """Create all features needed for any model."""
        features = df.copy()

        # Time-based features
        features['hour'] = features.index.hour
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['is_daytime'] = ((features['hour'] >= 6) &
                                  (features['hour'] <= 18)).astype(int)
        features['is_weekend'] = features.index.dayofweek.isin([5, 6]).astype(int)

        # Lag features
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
            'hour_sin', 'hour_cos', 'is_daytime', 'is_weekend',
            'kWh_lag_1h', 'kWh_lag_24h', 'kWh_lag_168h',
            'kWh_rolling_mean_24h', 'kWh_rolling_std_24h', 'kWh_rolling_max_24h'
        ]

        deep_features = [
            'kWh_rolling_mean_24h', 'hour_cos',
            'kWh_rolling_std_24h', 'kWh_lag_1h',
            'kWh_lag_24h', 'is_daytime', 'is_weekend'
        ]

        return {
            'base': base_features,
            'deep': deep_features,
            'all': list(set(base_features + deep_features))
        }
