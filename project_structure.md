]# Project Structure

```
cs6120-course-project/
├── docs/                              # Documentation
├── literature/                        # Research papers and references
├── logs/                             # Log files
│   ├── advanced_model_training_*.log  # Advanced model training logs
│   ├── model_training_*.log          # Basic model training logs
│   ├── pipeline_*.log                # Pipeline execution logs
│   └── preprocessing_*.log           # Data preprocessing logs
├── model_results/                    # Model performance metrics
│   ├── advanced_metrics.csv          # Advanced models metrics
│   ├── baseline_metrics.csv          # Baseline models metrics
│   └── model_metrics.csv             # Combined model metrics
├── models/                           # Saved model files
│   ├── gradient_boosting_fold_*.joblib    # Gradient boosting models
│   ├── gradient_boosting_fold_*_metrics.json
│   ├── linear_sgd_fold_*.joblib          # SGD models
│   ├── linear_sgd_fold_*_metrics.json
│   ├── random_forest_fold_*.joblib        # Random forest models
│   └── random_forest_fold_*_metrics.json
├── processed_data/                   # Preprocessed datasets
├── reports/                          # Generated reports
│   └── model_evaluation_report.md    # Model evaluation report
├── src/                             # Source code
│   ├── models/                      # Model implementations
│   │   ├── __init__.py
│   │   ├── advanced_models.py       # Advanced ML models
│   │   └── baseline_models.py       # Baseline models
│   ├── visualization/               # Visualization utilities
│   │   ├── __init__.py
│   │   └── model_evaluation.py      # Model evaluation plots
│   ├── __init__.py
│   ├── data_preprocessing.py        # Data preprocessing pipeline
│   └── train_advanced_models.py     # Advanced model training
├── visualizations/                   # Generated plots and figures
│   ├── actual_vs_predicted.png      # Actual vs predicted plot
│   ├── error_distribution.png       # Error distribution plot
│   ├── feature_importance.png       # Feature importance plot
│   ├── model_comparison.png         # Model comparison plot
│   ├── performance_mae_by_fold.png  # MAE by fold plot
│   ├── performance_mape_by_fold.png # MAPE by fold plot
│   ├── performance_r2_by_fold.png   # R² by fold plot
│   └── performance_rmse_by_fold.png # RMSE by fold plot
├── .gitignore                       # Git ignore file
├── analysis_results.xlsx            # Analysis results spreadsheet
├── config.py                        # Configuration settings
├── group-johnson-project-proposal.md # Project proposal
├── group-johnson-project-proposal.pdf # Project proposal PDF
├── main.py                          # Main script
├── pipeline_runner.py               # Pipeline orchestration
└── project_structure.md             # This file

```

## Directory Descriptions

### Core Directories

- **src/**: Source code for all model implementations and utilities
- **models/**: Saved model files and their metrics
- **model_results/**: CSV files containing model performance metrics
- **logs/**: Timestamped log files from different pipeline stages

### Documentation

- **docs/**: Project documentation
- **reports/**: Generated analysis reports
- **literature/**: Research papers and references

### Data

- **processed_data/**: Preprocessed and cleaned datasets
- **model_results/**: Model performance metrics and analysis

### Outputs

- **visualizations/**: Generated plots and performance visualizations
- **reports/**: Detailed model evaluation reports
- **models/**: Saved model states for each fold

### Configuration

- **config.py**: Global configuration settings
- **.gitignore**: Git ignore patterns
- **pipeline_runner.py**: Main pipeline orchestration script

Each directory serves a specific purpose in the ML pipeline, from data preprocessing through model training to evaluation and visualization. The
structure supports reproducible research and clear organization of all project components.