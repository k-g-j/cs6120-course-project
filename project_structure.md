# Project Structure

```
cs6120-course-project/
├── data/                             # Raw data directory
│   └── solar_data/                   # Solar production data
├── docs/                             # Documentation
├── literature/                       # Research papers and references
├── logs/                            # Log files
│   ├── advanced_pipeline_*.log      # Advanced pipeline logs
│   └── pipeline_*.log               # Basic pipeline logs
├── results/                         # Model results and metrics
│   ├── hyperparameter_tuning/       # Hyperparameter tuning results
│   │   ├── ensemble_tuning.csv      # Ensemble model tuning results
│   │   └── cv_results.csv           # Cross-validation results
│   └── final_model_metrics.csv      # Combined final model metrics
├── models/                          # Saved model files
│   ├── deep_learning/              # Deep learning model files
│   │   ├── lstm_*.h5               # LSTM model files
│   │   └── cnn_*.h5                # CNN model files
│   ├── ensemble/                   # Ensemble model files
│   │   └── stacked_ensemble_*.joblib # Stacked ensemble models
│   ├── checkpoints/                # Model checkpoints
│   └── baseline/                   # Baseline model files
│       ├── linear_regression_*.joblib
│       ├── ridge_*.joblib
│       └── lasso_*.joblib
├── processed_data/                  # Preprocessed datasets
│   └── solar_production/           # Processed solar data
├── reports/                         # Generated reports
│   └── model_evaluation_report.md   # Model evaluation report
├── src/                            # Source code
│   ├── models/                     # Model implementations
│   │   ├── __init__.py
│   │   ├── advanced_ensemble.py    # Stacked ensemble implementation
│   │   ├── advanced_models.py      # Advanced ML models
│   │   ├── cnn_model.py           # CNN model implementation
│   │   ├── lstm_model.py          # LSTM model implementation
│   │   ├── svr_model.py           # SVR model implementation
│   │   ├── feature_engineering.py  # Feature engineering utilities
│   │   └── hyperparameter_tuning.py # Hyperparameter tuning utilities
│   ├── visualization/              # Visualization utilities
│   │   ├── __init__.py
│   │   └── model_evaluation.py     # Model evaluation plots
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data preprocessing pipeline
│   ├── train_advanced_models.py    # Advanced model training
│   ├── train_ensemble.py          # Ensemble model training
│   ├── train_lstm.py             # LSTM model training
│   └── train_models.py           # Baseline model training
├── visualizations/                  # Generated plots and figures
│   ├── model_comparison/           # Model comparison plots
│   │   ├── performance_comparison.png
│   │   └── feature_importance.png
│   ├── error_analysis/            # Error analysis plots
│   │   ├── error_distribution.png
│   │   └── residual_plots.png
│   └── time_series/               # Time series plots
│       ├── actual_vs_predicted.png
│       └── forecast_plots.png
├── .gitignore                      # Git ignore file
├── config.py                       # Configuration settings
├── pipeline_runner.py              # Basic pipeline runner
├── advanced_pipeline_runner.py     # Advanced pipeline runner
└── project_structure.md            # This file
```

## Directory Descriptions

### Core Directories

- **src/**: Source code containing all model implementations and utilities
    - **models/**: Model implementations including advanced, ensemble, and deep learning
    - **visualization/**: Visualization utilities for model evaluation
- **models/**: Saved model files organized by model type
    - **deep_learning/**: LSTM and CNN models
    - **ensemble/**: Stacked ensemble models
    - **baseline/**: Basic regression models
- **results/**: Model performance metrics and tuning results
- **logs/**: Pipeline execution logs with timestamps

### Documentation and References

- **docs/**: Project documentation
- **reports/**: Generated analysis reports
- **literature/**: Research papers and references

### Data Management

- **data/**: Raw data storage
- **processed_data/**: Preprocessed and cleaned datasets
- **results/**: Model performance metrics and analysis
    - **hyperparameter_tuning/**: Tuning results for different models
    - **final_model_metrics.csv**: Combined performance metrics

### Visualization Outputs

- **visualizations/**: Generated plots and figures
    - **model_comparison/**: Performance comparison visualizations
    - **error_analysis/**: Error distribution and residual plots
    - **time_series/**: Time series specific visualizations

### Pipeline and Configuration

- **config.py**: Global configuration settings
- **pipeline_runner.py**: Basic pipeline orchestration
- **advanced_pipeline_runner.py**: Advanced model pipeline orchestration
- **.gitignore**: Git ignore patterns

## Project Components

### Model Types

1. **Baseline Models**
    - Linear Regression
    - Ridge Regression
    - Lasso Regression

2. **Advanced Models**
    - Random Forest
    - Gradient Boosting
    - SGD Regressor
    - Support Vector Regression (SVR)

3. **Deep Learning Models**
    - LSTM (Long Short-Term Memory)
    - CNN (Convolutional Neural Network)

4. **Ensemble Models**
    - Stacked Ensemble

### Pipeline Stages

1. Data Preprocessing
2. Feature Engineering
3. Model Training
4. Hyperparameter Tuning
5. Model Evaluation
6. Results Visualization

The structure supports a comprehensive machine learning pipeline with:

- Multiple model types (baseline, advanced, ensemble, deep learning)
- Separate training scripts for different model types
- Organized storage of model artifacts and results
- Clear separation of concerns between different components
- Support for both basic and advanced pipelines
- Comprehensive visualization capabilities
- Systematic evaluation and reporting

This organization ensures reproducibility, maintainability, and clear documentation of the entire machine learning workflow.