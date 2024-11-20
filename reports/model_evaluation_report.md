# Solar Energy Production Prediction - Model Evaluation Report

## Model Performance Summary

|                                   |     rmse |      mae |        r2 |    mape |
|:----------------------------------|---------:|---------:|----------:|--------:|
| ('advanced', 'gradient_boosting') | 0.933188 | 0.563457 | 0.0939207 |  171.05 |
| ('advanced', 'linear_sgd')        | 0.936068 | 0.579528 | 0.0881489 | 159.737 |
| ('advanced', 'lstm')              | 0.881019 | 0.518063 |  0.193533 | 168.446 |
| ('advanced', 'random_forest')     | 0.933452 | 0.563633 | 0.0933565 | 171.698 |

## Detailed Metrics by Fold

### Advanced Models

| model_name        | model_type |     rmse |      mae |        r2 |    mape | fold |
|:------------------|:-----------|---------:|---------:|----------:|--------:|-----:|
| random_forest     | advanced   | 0.792401 | 0.473631 |  0.138723 | 124.433 |    1 |
| gradient_boosting | advanced   | 0.791468 | 0.472729 |  0.140751 | 123.728 |    1 |
| linear_sgd        | advanced   | 0.796403 | 0.487642 |  0.130003 | 118.828 |    1 |
| lstm              | advanced   | 0.767092 | 0.482997 |  0.192863 | 159.058 |    1 |
| random_forest     | advanced   |  1.09328 | 0.600981 | 0.0804188 | 180.875 |    2 |
| gradient_boosting | advanced   |  1.09326 | 0.601099 | 0.0804441 | 180.285 |    2 |
| linear_sgd        | advanced   |  1.09627 | 0.624089 | 0.0753707 | 170.917 |    2 |
| lstm              | advanced   |  1.03755 | 0.541956 |  0.171774 | 180.332 |    2 |
| random_forest     | advanced   |  1.03736 | 0.592001 | 0.0843975 | 134.105 |    3 |
| gradient_boosting | advanced   |  1.03725 | 0.591746 | 0.0845928 | 133.155 |    3 |
| linear_sgd        | advanced   |  1.03837 | 0.614436 | 0.0826038 | 134.091 |    3 |
| lstm              | advanced   | 0.993763 | 0.603797 |  0.159735 | 196.264 |    3 |
| random_forest     | advanced   | 0.876444 | 0.571793 |  0.114369 | 274.751 |    4 |
| gradient_boosting | advanced   | 0.876406 | 0.571802 |  0.114446 | 273.984 |    4 |
| linear_sgd        | advanced   |  0.88023 | 0.582401 |  0.106701 | 238.708 |    4 |
| lstm              | advanced   | 0.822148 | 0.524802 |    0.2207 | 182.191 |    4 |
| random_forest     | advanced   | 0.867781 | 0.579762 | 0.0488735 | 144.328 |    5 |
| gradient_boosting | advanced   | 0.867555 | 0.579906 | 0.0493693 | 144.098 |    5 |
| linear_sgd        | advanced   | 0.869061 | 0.589074 | 0.0460659 |  136.14 |    5 |
| lstm              | advanced   | 0.784541 | 0.436763 |  0.222592 | 124.382 |    5 |