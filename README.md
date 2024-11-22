# Solar Energy Production Prediction

This project implements a machine learning pipeline for predicting solar energy production using various models including Random Forest, Gradient
Boosting, LSTM networks, and an ensemble approach. The system achieves a 153% improvement over baseline models with an R² score of 0.6964.

## Project Structure

```
cs6120-course-project/
├── data/                              # Raw data directory
│   ├── solar_data/                    # Solar production data
│   └── Renewable Energy World Wide/   # Historical energy data
├── docs/                              # Documentation
├── literature/                        # Research papers and references
├── logs/                             # Log files
├── model_results/                    # Detailed model outputs
├── models/                          # Saved model files
├── processed_data/                  # Preprocessed datasets
├── reports/                         # Generated reports
├── results/                         # Model results and metrics
├── src/                            # Source code
└── visualizations/                  # Generated plots and figures
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:

```bash
git clone https://github.com/k-g-j/cs6120-course-project.git
cd cs6120-course-project
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Make bash scripts executable:

```bash
chmod +x *.sh
```

### Data Setup

1. Download the required datasets and place them in the appropriate directories:
    - Place solar production data in `data/solar_data/`
    - Place renewable energy data in `data/Renewable Energy World Wide 1965-2022/`

### Running the Pipeline

1. Initialize the project structure:

```bash
python setup_project.py
```

2. Run the complete pipeline:

```bash
./run_pipeline.sh
```

This will execute:

- Data preprocessing
- Model training
- Ablation studies
- Ensemble evaluation
- Final analysis

### Individual Components

You can also run individual components:

1. Data Preprocessing:

```bash
python -m src.data_preprocessing
```

2. Model Training:

```bash
python -m src.train_models
```

3. Advanced Models:

```bash
python -m src.advanced_pipeline_runner
```

4. Ablation Studies:

```bash
python -m src.run_all_ablation_studies
```

5. Ensemble Evaluation:

```bash
python -m src.run_ensemble_evaluation
```

6. Final Analysis:

```bash
python -m src.run_final_analysis
```

## Project Components

### Data Processing

- Handles missing values
- Implements feature engineering
- Performs data validation

### Models

- Baseline Models (Linear, Ridge, Lasso)
- Advanced Models (Random Forest, Gradient Boosting, LSTM)
- Ensemble Model

### Evaluation

- Cross-validation
- Ablation studies
- Performance metrics
- Visualization generation

## Results

The project achieves:

- Ensemble model R² score: 0.6964
- RMSE: 0.5625 (31% reduction in error)
- Sub-second inference time (78.3ms)
- 45% reduction in computational requirements

## Project Report and Presentation

- Full project report: `final_project_report/final_project_report.pdf`
- Presentation slides: `presentation/Presentation.pdf`
- Presentation video: `presentation/Presentation.mp4`

## Author

Kate Johnson

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Solar Energy Production dataset by Ivan Lee on Kaggle
- Solar Power Generation Data by Afroz on Kaggle
- Renewable Energy World Wide dataset by Belayet HossainDS on Kaggle

For more detailed information about the implementation and results, please refer to the project report and documentation in the `docs` directory.
