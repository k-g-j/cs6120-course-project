# Solar Energy Production Prediction - Comprehensive Report

## Executive Summary

### Model Performance

- Best performing model: random_forest
- Best R² score: 0.3275
- Best RMSE: 0.7354

### Ensemble Model Performance

- Average R² score: 0.6964
- Average RMSE: 0.5625

### Key Findings from Ablation Studies



## Model Performance Analysis

### Performance by Model Type

| model_type   |   ('r2', 'mean') |   ('r2', 'std') |   ('rmse', 'mean') |   ('rmse', 'std') |
|:-------------|-----------------:|----------------:|-------------------:|------------------:|
| advanced     |           0.2745 |          0.0505 |             0.874  |            0.1226 |
| baseline     |           0.1054 |          0.0805 |             0.9681 |            0.1215 |

![Model Type Performance](model_type_performance.png)


### Ensemble Model Analysis

- Consistent performance across folds (R² std: 0.0353)
- Stable predictions (RMSE std: 0.0890)


## Feature Importance Analysis



## Detailed Results Analysis

### Ablation Studies Results


### Model Stability Analysis

Standard deviation across folds:

| model_name        |     r2 |   rmse |
|:------------------|-------:|-------:|
| gradient_boosting | 0.0817 | 0.1517 |
| lasso             | 0.0044 | 0.125  |
| linear_regression | 0.0183 | 0.1223 |
| linear_sgd        | 0.0205 | 0.1156 |
| random_forest     | 0.0382 | 0.1269 |
| ridge             | 0.0183 | 0.1223 |