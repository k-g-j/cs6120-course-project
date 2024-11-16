import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import TimeSeriesSplit

from src.models.advanced_models import AdvancedModels
from src.models.cnn_model import CNNRegressor
from src.models.feature_engineering import FeatureEngineer
from src.models.lstm_model import LSTMRegressor
from src.models.svr_model import SVRRegressor
from src.visualization.model_evaluation import create_visualizations


class AblationStudy:
    """Conduct comprehensive ablation studies on the solar prediction models."""

    def __init__(self, data, config):
        """Initialize ablation study with data and configuration."""
        self.data = data.copy()
        self.config = config
        self.feature_engineer = FeatureEngineer()
        self.results = []

        # Ensure datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            if 'datetime' in self.data.columns:
                self.data.set_index('datetime', inplace=True)
            elif 'date' in self.data.columns:
                self.data['datetime'] = pd.to_datetime(self.data['date'])
                self.data.set_index('datetime', inplace=True)
                self.data.drop('date', axis=1, errors='ignore', inplace=True)

        # Pre-filter non-numeric columns
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data = self.data[list(numeric_cols)]

        logging.info(f"Using numeric columns: {list(numeric_cols)}")

    def _initialize_models(self):
        """Initialize all models."""
        # Initialize models based on computational efficiency
        sample_size = len(self.data)
        logging.info(f"Initializing models for dataset size: {sample_size}")

        # For datasets larger than 50k samples, use efficient models only
        if sample_size > 50000:
            logging.info("Using efficient models only due to large dataset size")
            self.models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=50,  # Reduced estimators
                    max_depth=8,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=5,
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
                    units=32,
                    dropout=0.2,
                    learning_rate=0.001,
                    batch_size=128,
                    epochs=10,
                    sequence_length=24
                ),
                'cnn': CNNRegressor(
                    filters=32,
                    kernel_size=3,
                    dropout=0.2,
                    learning_rate=0.001,
                    batch_size=128,
                    epochs=10,
                    sequence_length=24
                )
            }
        else:
            # For smaller datasets, include SVR
            logging.info("Including all models including SVR for smaller dataset")
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
                'svr': SVRRegressor(
                    kernel='rbf',
                    C=1.0,
                    epsilon=0.1,
                    gamma='scale'
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
                )
            }

    def run_input_dimension_ablation(self):
        """Study impact of different input features."""
        logging.info("Running input dimension ablation study...")

        # First ensure data has datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            logging.error("Data index is not datetime. Current index type: %s",
                          type(self.data.index))
            logging.info("Available columns: %s", self.data.columns.tolist())
            raise ValueError("Data must have datetime index")

        # Get all features
        processed_data = self.feature_engineer.create_all_features(self.data)
        feature_sets = self.feature_engineer.get_feature_sets()

        # Test different feature combinations
        feature_groups = {
            'all': feature_sets['all'],
            'time_only': ['hour_sin', 'hour_cos', 'is_daytime', 'is_weekend'],
            'lag_only': ['kWh_lag_1h', 'kWh_lag_24h', 'kWh_lag_168h'],
            'rolling_only': ['kWh_rolling_mean_24h', 'kWh_rolling_std_24h', 'kWh_rolling_max_24h']
        }

        results = []
        for group_name, features in feature_groups.items():
            metrics = self._evaluate_feature_set(processed_data, features)
            metrics['feature_group'] = group_name
            results.append(metrics)

        return pd.DataFrame(results)

    def _preprocess_data(self, processed_data):
        """Preprocess data by removing non-numeric columns and handling missing values."""
        # Get only numeric columns
        numeric_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
        data = processed_data[numeric_cols].copy()

        # Handle missing values
        data = data.fillna(data.mean())

        return data

    def run_preprocessing_ablation(self):
        """Study impact of different preprocessing steps."""
        logging.info("Running preprocessing ablation study...")

        results = []

        # Test with different preprocessing configurations
        configs = {
            'full': {'scale': True, 'handle_missing': True, 'engineer_features': True},
            'no_scaling': {'scale': False, 'handle_missing': True, 'engineer_features': True},
            'no_missing_handling': {'scale': True, 'handle_missing': False,
                                    'engineer_features': True},
            'minimal': {'scale': False, 'handle_missing': False, 'engineer_features': False}
        }

        for config_name, settings in configs.items():
            metrics = self._evaluate_preprocessing(settings)
            metrics['preprocessing_config'] = config_name
            results.append(metrics)

        return pd.DataFrame(results)

    def run_temporal_resolution_ablation(self):
        """Study model performance at different forecast horizons."""
        logging.info("Running temporal resolution ablation study...")

        horizons = [1, 3, 6, 12, 24]  # hours
        results = []

        for horizon in horizons:
            metrics = self._evaluate_forecast_horizon(horizon)
            metrics['forecast_horizon'] = horizon
            results.append(metrics)

        return pd.DataFrame(results)

    def run_data_volume_ablation(self):
        """Study how model performance scales with training data volume."""
        logging.info("Running data volume ablation study...")

        data_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
        results = []

        for fraction in data_fractions:
            metrics = self._evaluate_data_fraction(fraction)
            metrics['data_fraction'] = fraction
            results.append(metrics)

        return pd.DataFrame(results)

    def _evaluate_feature_set(self, data, features):
        """Evaluate model performance with given feature set."""
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, test_idx in tscv.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            model = AdvancedModels(train_data, test_data, target_col='kWh')
            model.prepare_data(feature_columns=features)

            metrics = model.train_models()
            scores.append(metrics)

        return self._aggregate_scores(scores)

    def _evaluate_preprocessing(self, settings):
        """Evaluate model performance with different preprocessing settings."""
        processed_data = self.data.copy()

        # Always filter out non-numeric columns first
        processed_data = self._preprocess_data(processed_data)

        if settings['engineer_features']:
            processed_data = self.feature_engineer.create_all_features(processed_data)
            feature_cols = self.feature_engineer.get_feature_sets()['all']
        else:
            # Use basic numeric columns if not engineering features
            feature_cols = list(processed_data.columns)
            if 'kWh' in feature_cols:  # Remove target column from features
                feature_cols.remove('kWh')

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, test_idx in tscv.split(processed_data):
            train_data = processed_data.iloc[train_idx]
            test_data = processed_data.iloc[test_idx]

            model = AdvancedModels(train_data, test_data, target_col='kWh')
            model.feature_cols = feature_cols  # Set feature columns explicitly

            if not settings['scale']:
                # Handle case where no scaling is needed
                model.scaler = None
                # Prepare data without scaling
                model.X_train = model.train_data[model.feature_cols].values
                model.X_test = model.test_data[model.feature_cols].values
                model.y_train = model.train_data[model.target_col].values
                model.y_test = model.test_data[model.target_col].values
            else:
                # Use normal data preparation with scaling
                model.prepare_data()

            metrics = model.train_models()
            scores.append(metrics)

        return self._aggregate_scores(scores)

    def _evaluate_forecast_horizon(self, horizon):
        """Evaluate model performance for different forecast horizons."""
        processed_data = self.feature_engineer.create_all_features(self.data)

        # Shift target variable for different horizons
        processed_data[f'target_h{horizon}'] = processed_data['kWh'].shift(-horizon)
        processed_data = processed_data.dropna()

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, test_idx in tscv.split(processed_data):
            train_data = processed_data.iloc[train_idx]
            test_data = processed_data.iloc[test_idx]

            model = AdvancedModels(train_data, test_data, target_col=f'target_h{horizon}')
            model.prepare_data()

            metrics = model.train_models()
            scores.append(metrics)

        return self._aggregate_scores(scores)

    def _evaluate_data_fraction(self, fraction):
        """Evaluate model performance with different amounts of training data."""
        # For very large datasets, take a smaller sample first
        max_sample_size = 50000  # Reduced from 100000

        if len(self.data) > max_sample_size:
            sample_size = int(max_sample_size * fraction)
            sampled_data = self.data.sample(n=sample_size, random_state=42)
            processed_data = self.feature_engineer.create_all_features(sampled_data)
        else:
            processed_data = self.feature_engineer.create_all_features(self.data)

        # Calculate split point
        split_idx = int(len(processed_data) * 0.8)  # 80% train, 20% test
        train_data = processed_data.iloc[:split_idx]
        test_data = processed_data.iloc[split_idx:]

        # Take fraction of training data
        train_size = int(len(train_data) * fraction)
        train_data = train_data.iloc[-train_size:]

        model = AdvancedModels(train_data, test_data, target_col='kWh')
        model.prepare_data()

        metrics = model.train_models()
        return metrics

    def _aggregate_scores(self, scores):
        """Aggregate scores across cross-validation folds."""
        if not scores:
            return {}

        agg_scores = {}
        for metric in ['rmse', 'r2', 'mae']:
            values = [score.get(metric, 0) for score in scores if metric in score]
            if values:
                agg_scores[metric] = np.mean(values)
                agg_scores[f'{metric}_std'] = np.std(values)

        return agg_scores

    def run_all_studies(self):
        """Run all ablation studies and generate comprehensive report."""
        studies = {
            'input_dimension': self.run_input_dimension_ablation(),
            'preprocessing': self.run_preprocessing_ablation(),
            'temporal_resolution': self.run_temporal_resolution_ablation(),
            'data_volume': self.run_data_volume_ablation()
        }

        # Save results
        results_dir = Path(self.config.RESULTS_DIR) / 'ablation_studies'
        results_dir.mkdir(parents=True, exist_ok=True)

        for study_name, results in studies.items():
            results.to_csv(results_dir / f'{study_name}_results.csv', index=False)

        # Generate visualizations
        self._generate_ablation_visualizations(studies)

        return studies

    def _generate_ablation_visualizations(self, studies):
        """Generate visualizations for ablation study results."""
        viz_dir = Path(self.config.VISUALIZATIONS_DIR) / 'ablation_studies'
        viz_dir.mkdir(parents=True, exist_ok=True)

        for study_name, results in studies.items():
            create_visualizations(
                results,
                None,  # No predictions needed for ablation studies
                None,  # No feature importance needed
                output_dir=str(viz_dir / study_name)
            )
