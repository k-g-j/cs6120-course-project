import os

DATA_DIR = 'data'
RENEWABLE_DIR = os.path.join(DATA_DIR, 'Renewable Energy World Wide 1965-2022')

CONFIG = {
    'solar_production_path': os.path.join(DATA_DIR, 'Solar_Energy_Production.csv'),
    'solar_plant_path': os.path.join(DATA_DIR, 'Solar Power Plant Data.csv'),
    'renewable_energy_path': RENEWABLE_DIR,
    'output_dir': 'processed_data',
    'random_state': 42
}
