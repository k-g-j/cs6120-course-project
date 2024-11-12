# Project Structure

```
cs6120-course-project/
├── .idea/                              # IDE settings
├── assignment-instructions/            # Course assignment details
├── data/                              # Raw data directory
│   ├── Renewable Energy World Wide 1965-2022/
│   ├── Solar Power Plant Data.csv
│   └── Solar_Energy_Production.csv
├── docs/                              # Documentation
│   └── model_analysis.md              # Model analysis documentation
├── literature/                        # Research papers and references
├── logs/                              # Log files
├── model_results/                     # Model performance metrics
├── models/                            # Saved model files
├── processed_data/                    # Preprocessed datasets
│   ├── processed_solar_production.csv
│   ├── processed_solar_capacity.csv
│   ├── processed_solar_consumption.csv
│   ├── processed_solar_elec.csv
│   └── processed_solar_share.csv
├── reports/                           # Generated reports
├── src/                              # Source code
│   ├── models/                       # Model implementations
│   │   ├── __init__.py
│   │   ├── advanced_models.py        # Advanced ML models
│   │   └── baseline_models.py        # Baseline models
│   ├── visualization/                # Visualization utilities
│   │   ├── __init__.py
│   │   └── model_evaluation.py       # Model evaluation plots
│   ├── data_preprocessing.py         # Data preprocessing pipeline
│   ├── train_advanced_models.py      # Advanced model training
│   └── train_models.py               # Model training utilities
├── visualizations/                    # Generated plots and figures
├── .gitignore                        # Git ignore file
├── analysis_results.xlsx             # Analysis results
├── config.py                         # Configuration settings
├── group-johnson-project-proposal.md  # Project proposal
├── group-johnson-project-proposal.pdf # Project proposal PDF
├── main.py                           # Main script
└── pipeline_runner.py                # Pipeline orchestration

```

## Directory Descriptions

### Core Directories
- **src/**: Contains all source code
    - **models/**: Model implementations
    - **visualization/**: Visualization tools
- **data/**: Raw data storage
- **processed_data/**: Preprocessed datasets
- **models/**: Saved trained models
- **reports/**: Generated analysis reports

### Documentation
- **docs/**: Project documentation
- **literature/**: Research papers and references
- **assignment-instructions/**: Course-related materials

### Output Directories
- **logs/**: Application logs
- **model_results/**: Model metrics and evaluations
- **visualizations/**: Generated figures and plots

### Configuration Files
- **config.py**: Global configuration settings
- **.gitignore**: Git ignore patterns
- **pipeline_runner.py**: Pipeline orchestration script

### Project Documentation
- **group-johnson-project-proposal.md**: Project proposal (Markdown)
- **group-johnson-project-proposal.pdf**: Project proposal (PDF)
- **analysis_results.xlsx**: Analysis results spreadsheet