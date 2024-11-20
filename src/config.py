from pathlib import Path

# Base directories
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = BASE_DIR / 'processed_data'
MODEL_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
REPORTS_DIR = BASE_DIR / 'reports'
VISUALIZATIONS_DIR = BASE_DIR / 'visualizations'
LOGS_DIR = BASE_DIR / 'logs'

# Configuration dictionary
CONFIG = {
    # Directory paths
    'data_dir': str(DATA_DIR),
    'processed_dir': str(PROCESSED_DIR),
    'model_dir': str(MODEL_DIR),
    'results_dir': str(RESULTS_DIR),
    'reports_dir': str(REPORTS_DIR),
    'visualizations_dir': str(VISUALIZATIONS_DIR),
    'logs_dir': str(LOGS_DIR),

    # Data paths
    'solar_production_path': str(DATA_DIR / 'solar_data' / 'Solar_Energy_Production.csv'),
    'solar_plant_path': str(DATA_DIR / 'solar_data' / 'Solar_Power_Plant_Data.csv'),
    'renewable_energy_path': str(DATA_DIR / 'Renewable Energy World Wide 1965-2022'),

    # Output paths
    'processed_solar_production': str(PROCESSED_DIR / 'processed_solar_production.csv'),
    'processed_power_plant': str(PROCESSED_DIR / 'processed_power_plant.csv'),
    'model_metrics': str(RESULTS_DIR / 'model_metrics.csv'),
    'final_metrics': str(RESULTS_DIR / 'final_model_metrics.csv'),
    'ablation_results': str(RESULTS_DIR / 'ablation_studies'),
    'ensemble_results': str(RESULTS_DIR / 'ensemble'),
    'baseline_models': str(MODEL_DIR / 'baseline'),
    'advanced_models': str(MODEL_DIR / 'ensemble'),
    'checkpoints': str(MODEL_DIR / 'checkpoints'),

    # Random seed for reproducibility
    'random_state': 42,

    # Model parameters
    'test_size': 0.2,
    'cv_folds': 5,
    'batch_size': 64,
    'epochs': 20,
    'learning_rate': 0.001,
    'early_stopping_patience': 5,

    # Feature engineering parameters
    'sequence_length': 24,
    'rolling_window_sizes': [24, 168, 720],  # 1 day, 1 week, 1 month

    # Logging settings
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(levelname)s - %(message)s'
}
