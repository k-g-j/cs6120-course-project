# Solar Energy Forecasting Model Analysis

## Model Performance Summary

The solar energy forecasting pipeline trained and evaluated several baseline and advanced models using 5-fold cross-validation. The baseline models
included Linear Regression, Ridge Regression, and Lasso Regression. The advanced models were Random Forest, Gradient Boosting, and Linear SGD.

### Baseline Model Performance

| Model             | R²      | RMSE   | MAE    | MAPE     |
|-------------------|---------|--------|--------|----------|
| Linear Regression | 0.1726  | 0.8157 | 0.5440 | 172.1763 | 
| Ridge Regression  | 0.1726  | 0.8157 | 0.5439 | 172.0325 |
| Lasso Regression  | -0.0007 | 0.8970 | 0.6269 | 104.3280 |

Among the baseline models, Linear Regression and Ridge Regression performed similarly, achieving an R² of 0.1726 and RMSE around 0.8157. Lasso
Regression performed poorly with a negative R², indicating it failed to capture the patterns in the data.

### Advanced Model Performance

| Model             | R²     | RMSE   | MAE    | MAPE     |
|-------------------|--------|--------|--------|----------|
| Random Forest     | 0.3071 | 0.7592 | 0.4389 | 152.1765 |
| Gradient Boosting | 0.3031 | 0.7614 | 0.4414 | 154.5903 | 
| Linear SGD        | 0.2771 | 0.7755 | 0.4801 | 152.5943 |

The advanced models outperformed the baselines, with Random Forest achieving the highest R² of 0.3071 and lowest RMSE of 0.7592. Gradient Boosting had
similar performance to Random Forest. Linear SGD was slightly behind the tree-based models but still an improvement over the linear baselines.

## Best Model

Based on the 5-fold cross-validation results, the best performing model was:

**Random Forest**

- R²: 0.3071
- RMSE: 0.7592
- MAE: 0.4389
- MAPE: 152.1765

The Random Forest model was able to capture more complex relationships between the input features and target variable compared to the linear models.
It leveraged an ensemble of decision trees to make robust predictions.

## Feature Importance

The Random Forest model identified the following features as most important for predicting solar energy production:

1. kWh_rolling_mean_24h (62.51%)
2. kWh_rolling_std_24h (9.75%)
3. hour_sin (7.18%)
4. kWh_lag_1h (5.70%)
5. hour_cos (4.62%)

The 24-hour rolling mean of energy production was by far the most influential predictor, followed by the rolling standard deviation. Sine and cosine
transformations of the hour and the 1-hour lagged production were also important.

## Areas for Improvement

While the Random Forest model outperformed the baselines, there is still significant room for improvement with an R² of 0.3071. Some areas to explore:

- Feature engineering: Create additional relevant features, such as weather conditions, solar radiation measurements, or holiday indicators.
- Hyperparameter tuning: Perform an extensive search to find optimal hyperparameters for the models.
- Ensembling: Combine predictions from multiple models to improve robustness.
- Deep learning: Experiment with neural network architectures like LSTMs that can capture complex temporal patterns.

By iterating on feature engineering and model architectures, it should be possible to further improve the accuracy of solar energy forecasting. The
current pipeline provides a solid foundation to build upon.