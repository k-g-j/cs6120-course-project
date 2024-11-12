import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SolarDataPreprocessor:
    def __init__(self, output_dir='processed_data'):
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        os.makedirs(output_dir, exist_ok=True)

    def load_solar_production_data(self, filepath):
        """Load and preprocess the Solar Energy Production dataset."""
        df = pd.read_csv(filepath)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        return df

    def load_solar_plant_data(self, filepath):
        """Load and preprocess the Solar Power Plant dataset."""
        return pd.read_csv(filepath)

    def load_renewable_energy_data(self, base_path):
        """Load relevant solar-related datasets from Renewable Energy World Wide."""
        solar_files = {
            'solar_consumption': '12 solar-energy-consumption.csv',
            'solar_capacity': '13 installed-solar-PV-capacity.csv',
            'solar_share': '14 solar-share-energy.csv',
            'solar_elec': '15 share-electricity-solar.csv'
        }

        solar_data = {}
        for key, filename in solar_files.items():
            filepath = os.path.join(base_path, filename)
            solar_data[key] = pd.read_csv(filepath)

        return solar_data

    def engineer_time_features(self, df):
        """Add time-based features to the dataset."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        df = df.copy()
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['dayofyear'] = df.index.dayofyear

        # Add cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def process_weather_features(self, df):
        """Process weather-related features if they exist."""
        weather_cols = ['temperature', 'humidity', 'cloud_cover', 'pressure', 'wind_speed']
        existing_cols = [col for col in weather_cols if col in df.columns]

        if existing_cols:
            for col in existing_cols:
                df[f'{col}_rolling_6h'] = df[col].rolling(window=6, min_periods=1).mean()
                df[f'{col}_rolling_24h'] = df[col].rolling(window=24, min_periods=1).mean()
                df[f'{col}_daily_range'] = df[col].rolling(window=24, min_periods=1).max() - \
                                           df[col].rolling(window=24, min_periods=1).min()

        return df

    def scale_numerical_features(self, df, exclude_cols=None):
        """Scale numerical features while excluding specified columns."""
        if exclude_cols is None:
            exclude_cols = []

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]

        if cols_to_scale:
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])

        return df

    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        df = df.copy()

        # For numerical columns, use interpolation
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear')

        # For categorical columns, use forward fill (updated to use ffill())
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].ffill()

        # Fill any remaining NaN values with backward fill
        df = df.bfill()

        return df

    def process_all_datasets(self, config):
        """Process all datasets according to the provided configuration."""
        try:
            # Load Solar Energy Production data
            solar_prod = self.load_solar_production_data(config['solar_production_path'])
            solar_prod = self.engineer_time_features(solar_prod)
            solar_prod = self.process_weather_features(solar_prod)
            solar_prod = self.handle_missing_values(solar_prod)
            solar_prod = self.scale_numerical_features(solar_prod,
                                                       exclude_cols=['hour', 'day', 'month',
                                                                     'year'])

            # Load Solar Power Plant data
            plant_data = self.load_solar_plant_data(config['solar_plant_path'])
            plant_data = self.handle_missing_values(plant_data)
            plant_data = self.scale_numerical_features(plant_data)

            # Load Renewable Energy data
            renewable_data = self.load_renewable_energy_data(config['renewable_energy_path'])

            # Process each renewable dataset
            processed_renewable = {}
            for key, df in renewable_data.items():
                processed_df = self.handle_missing_values(df)
                processed_df = self.scale_numerical_features(processed_df)
                processed_renewable[key] = processed_df

            # Save processed datasets
            solar_prod.to_csv(os.path.join(self.output_dir, 'processed_solar_production.csv'))
            plant_data.to_csv(os.path.join(self.output_dir, 'processed_power_plant.csv'))

            for key, df in processed_renewable.items():
                output_name = f'processed_{key}.csv'
                df.to_csv(os.path.join(self.output_dir, output_name))

            return {
                'solar_production': solar_prod,
                'power_plant': plant_data,
                'renewable': processed_renewable
            }

        except Exception as e:
            print(f"Error processing datasets: {str(e)}")
            raise
