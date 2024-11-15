# Solar Energy Production Prediction - Model Evaluation Report

## Model Performance Summary

| model_name        | model_type |     rmse |      mae |          r2 |    mape |
|:------------------|:-----------|---------:|---------:|------------:|--------:|
| gradient_boosting | advanced   | 0.866636 | 0.511658 |    0.285358 | 189.889 |
| lasso             | baseline   |  1.02543 | 0.650101 | -0.00281764 | 103.719 |
| linear_regression | baseline   | 0.939479 | 0.578909 |    0.159507 |  167.17 |
| linear_sgd        | advanced   | 0.875468 | 0.539177 |    0.270519 | 188.806 |
| lstm              | advanced   | 0.840765 | 0.530787 |    0.307787 | 180.039 |
| random_forest     | advanced   | 0.867734 | 0.512241 |    0.284693 | 190.445 |
| ridge             | baseline   | 0.939437 | 0.578854 |    0.159581 | 167.182 |
| stacked_ensemble  | ensemble   |  0.80451 |      nan |    0.352764 |     nan |

## Detailed Metrics by Fold

### Baseline Models

| model_name        |     rmse |      mae |           r2 |  adjusted_r2 |    mape | fold | model_type |
|:------------------|---------:|---------:|-------------:|-------------:|--------:|-----:|:-----------|
| linear_regression |  1.09394 | 0.621777 |     0.143916 |     0.143657 | 148.126 |    1 | baseline   |
| ridge             |  1.09387 | 0.621608 |     0.144029 |      0.14377 | 147.889 |    1 | baseline   |
| lasso             |  1.18854 | 0.671604 |   -0.0105614 |   -0.0108665 | 108.175 |    1 | baseline   |
| linear_regression |  1.02907 | 0.619796 |     0.135906 |     0.135645 | 170.326 |    2 | baseline   |
| ridge             |  1.02906 | 0.619791 |     0.135918 |     0.135657 |  170.33 |    2 | baseline   |
| lasso             |   1.1076 |  0.67673 |  -0.00101222 |  -0.00131445 | 99.7745 |    2 | baseline   |
| linear_regression |  0.93114 | 0.579561 |     0.168438 |     0.168187 | 182.416 |    3 | baseline   |
| ridge             |  0.93102 | 0.579568 |     0.168653 |     0.168402 |  182.85 |    3 | baseline   |
| lasso             |  1.02113 |  0.65655 | -6.65673e-05 |  -0.00036852 | 103.641 |    3 | baseline   |
| linear_regression | 0.815682 | 0.544026 |     0.172585 |     0.172336 | 172.176 |    4 | baseline   |
| ridge             | 0.815668 | 0.543917 |     0.172615 |     0.172365 | 172.033 |    4 | baseline   |
| lasso             | 0.897035 | 0.626854 | -0.000690594 | -0.000992734 | 104.328 |    4 | baseline   |
| linear_regression | 0.827569 | 0.529383 |      0.17669 |     0.176442 | 162.803 |    5 | baseline   |
| ridge             |  0.82757 | 0.529387 |     0.176689 |      0.17644 | 162.808 |    5 | baseline   |
| lasso             |  0.91286 | 0.618764 |  -0.00175744 |  -0.00205991 | 102.678 |    5 | baseline   |

### Advanced Models

| model_name        |     rmse |      mae |       r2 | adjusted_r2 |    mape | fold | model_type |
|:------------------|---------:|---------:|---------:|------------:|--------:|-----:|:-----------|
| random_forest     |  1.03797 |  0.61429 | 0.229265 |         nan | 260.313 |    1 | advanced   |
| gradient_boosting |  1.01442 | 0.588033 | 0.263849 |         nan | 243.726 |    1 | advanced   |
| linear_sgd        |  1.02101 | 0.608727 | 0.254251 |         nan | 227.587 |    1 | advanced   |
| lstm              |  1.08087 | 0.548096 | 0.164246 |         nan | 155.328 |    1 | advanced   |
| random_forest     | 0.948238 | 0.552899 | 0.266316 |         nan | 184.885 |    2 | advanced   |
| gradient_boosting |  0.95143 | 0.554735 | 0.261369 |         nan | 182.417 |    2 | advanced   |
| linear_sgd        | 0.962118 | 0.588728 | 0.244681 |         nan |  192.12 |    2 | advanced   |
| lstm              | 0.982492 | 0.580771 | 0.212352 |         nan | 185.055 |    2 | advanced   |
| random_forest     | 0.853375 |  0.49773 | 0.301536 |         nan | 181.178 |    3 | advanced   |
| gradient_boosting | 0.858637 |  0.50547 | 0.292895 |         nan | 185.591 |    3 | advanced   |
| linear_sgd        |  0.86847 | 0.528889 | 0.276608 |         nan | 186.967 |    3 | advanced   |
| lstm              |  0.88457 | 0.529156 | 0.249538 |         nan | 186.278 |    3 | advanced   |
| random_forest     | 0.737409 | 0.453918 | 0.323764 |         nan | 172.135 |    4 | advanced   |
| gradient_boosting | 0.741885 | 0.459975 | 0.315531 |         nan | 172.198 |    4 | advanced   |
| linear_sgd        | 0.750583 | 0.491154 | 0.299386 |         nan | 180.017 |    4 | advanced   |
| lstm              | 0.770655 | 0.493152 | 0.261413 |         nan | 189.106 |    4 | advanced   |
| random_forest     | 0.761673 | 0.442369 | 0.302585 |         nan | 153.716 |    5 | advanced   |
| gradient_boosting |  0.76681 | 0.450079 | 0.293146 |         nan | 165.511 |    5 | advanced   |
| linear_sgd        | 0.775159 | 0.478389 | 0.277669 |         nan | 157.336 |    5 | advanced   |
| lstm              |  0.79392 |  0.50276 | 0.242281 |         nan | 184.426 |    5 | advanced   |

### Ensemble Models

| model_name       |    rmse | mae |       r2 | adjusted_r2 | mape | fold | model_type |
|:-----------------|--------:|----:|---------:|------------:|-----:|:-----|:-----------|
| stacked_ensemble | 0.80451 | nan | 0.352764 |         nan |  nan | all  | ensemble   |

### Deep_Learning Models

| model_name |     rmse | mae |       r2 | adjusted_r2 | mape | fold | model_type    |
|:-----------|---------:|----:|---------:|------------:|-----:|:-----|:--------------|
| lstm       | 0.532079 | nan | 0.716892 |         nan |  nan | all  | deep_learning |