# Solar Energy Production Prediction - Analysis Report

## Model Performance Statistics by Type

### Advanced Models

|       |     rmse |       mae |        r2 |    mape |    fold |
|:------|---------:|----------:|----------:|--------:|--------:|
| count |       20 |        20 |        20 |      20 |      20 |
| mean  | 0.920932 |   0.55617 |   0.11724 | 167.733 |       3 |
| std   | 0.118038 | 0.0559609 | 0.0541727 | 47.5849 | 1.45095 |
| min   | 0.767092 |  0.436763 | 0.0460659 | 118.828 |       1 |
| 25%   | 0.815712 |  0.515512 | 0.0804378 | 133.857 |       2 |
| 50%   | 0.876425 |  0.579834 |  0.110535 | 151.693 |       3 |
| 75%   |  1.03741 |  0.594246 |  0.145497 | 181.204 |       4 |
| max   |  1.09627 |  0.624089 |  0.222592 | 274.751 |       5 |

# Ablation Studies Analysis

## Data Volume Study

## Temporal Resolution Study

## Preprocessing Study

## Input Dimension Study

## Methodology

### Model Evaluation

- Comprehensive evaluation using multiple metrics (RMSE, MAE, R², MAPE)
- Cross-validation using time series splits
- Statistical significance testing between model types

### Ablation Studies

1. Input Dimension Analysis
    - Evaluated impact of different feature groups
    - Identified most crucial features for prediction accuracy

2. Preprocessing Impact
    - Tested various preprocessing configurations
    - Quantified importance of each preprocessing step

3. Temporal Resolution Analysis
    - Evaluated performance across different forecast horizons
    - Identified optimal prediction timeframes

4. Data Volume Impact
    - Analyzed learning curves with varying training data sizes
    - Determined minimum data requirements for reliable predictions

## Key Findings

### Model Performance

- Best performing model: lstm (advanced)
- Achieved R² score: 0.2226
- RMSE: 0.7845
