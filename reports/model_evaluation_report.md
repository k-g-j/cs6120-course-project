# Solar Energy Production Prediction - Model Evaluation Report

## Model Performance Summary

| model_name        | model_type |     rmse |      mae |          r2 |    mape |
|:------------------|:-----------|---------:|---------:|------------:|--------:|
| gradient_boosting | advanced   |  0.88124 | 0.517455 |    0.264116 | 197.569 |
| lasso             | baseline   |  1.02543 | 0.650101 | -0.00281764 | 103.719 |
| linear_regression | baseline   | 0.939479 | 0.578909 |    0.159507 |  167.17 |
| linear_sgd        | advanced   | 0.874764 | 0.536543 |    0.271552 | 183.221 |
| random_forest     | advanced   | 0.865934 | 0.508624 |    0.287757 | 194.849 |
| ridge             | baseline   | 0.939437 | 0.578854 |    0.159581 | 167.182 |

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
| random_forest     |  1.03663 | 0.610313 | 0.231256 |         nan | 285.084 |    1 | advanced   |
| gradient_boosting |  1.10666 | 0.647007 | 0.123881 |         nan |  297.81 |    1 | advanced   |
| linear_sgd        |  1.01858 | 0.601271 | 0.257799 |         nan | 214.861 |    1 | advanced   |
| random_forest     | 0.947313 |  0.54845 | 0.267748 |         nan | 181.522 |    2 | advanced   |
| gradient_boosting | 0.950322 | 0.549839 | 0.263088 |         nan |  181.42 |    2 | advanced   |
| linear_sgd        |  0.96148 | 0.585761 | 0.245682 |         nan | 187.895 |    2 | advanced   |
| random_forest     | 0.851173 | 0.494124 | 0.305136 |         nan | 184.166 |    3 | advanced   |
| gradient_boosting | 0.852293 | 0.496576 | 0.303306 |         nan | 187.578 |    3 | advanced   |
| linear_sgd        | 0.867346 | 0.525998 | 0.278479 |         nan | 187.332 |    3 | advanced   |
| random_forest     | 0.735361 | 0.451367 | 0.327516 |         nan | 171.299 |    4 | advanced   |
| gradient_boosting | 0.735506 | 0.452464 | 0.327251 |         nan | 166.445 |    4 | advanced   |
| linear_sgd        | 0.750931 | 0.489545 | 0.298736 |         nan | 173.422 |    4 | advanced   |
| random_forest     | 0.759188 | 0.438866 | 0.307127 |         nan | 152.176 |    5 | advanced   |
| gradient_boosting | 0.761415 | 0.441391 | 0.303056 |         nan |  154.59 |    5 | advanced   |
| linear_sgd        | 0.775484 | 0.480141 | 0.277064 |         nan | 152.594 |    5 | advanced   |