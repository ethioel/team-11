# D Model Evaluation

## Validation Benchmark

The validation benchmark below compares the strongest candidate models on the same 80/20 split of the train set.

| Model | RMSE | MAE | R² |
| --- | ---: | ---: | ---: |
| Stacking Ensemble | 0.4599 | 0.3368 | 0.8940 |
| XGBoost | 0.4693 | 0.3441 | 0.8897 |
| Hist Gradient Boosting | 0.4719 | 0.3429 | 0.8885 |
| LightGBM | 0.4770 | 0.3478 | 0.8860 |
| MLP Neural Net | 0.4844 | 0.3536 | 0.8825 |
| Gradient Boosting | 0.4962 | 0.3619 | 0.8767 |
| Extra Trees | 0.5303 | 0.3882 | 0.8591 |
| Random Forest | 0.5370 | 0.3948 | 0.8556 |
| SVR | 0.7299 | 0.5199 | 0.7331 |
| Decision Tree | 0.7452 | 0.5526 | 0.7219 |
| Linear Regression | 0.8896 | 0.6650 | 0.6036 |
| Ridge | 0.8896 | 0.6650 | 0.6036 |
| SGD | 0.8932 | 0.6688 | 0.6004 |
| Elastic Net | 0.8942 | 0.6651 | 0.5995 |
| Lasso | 0.8955 | 0.6654 | 0.5983 |
| KNN | 0.9337 | 0.6875 | 0.5633 |
| Mean Baseline | 1.4131 | 1.1042 | -0.0001 |

## Selected Model

The **Stacking Ensemble** was selected as the final model. It combines the strengths of Random Forest, HistGradientBoosting, Extra Trees, and optionally XGBoost/LightGBM with a Ridge meta-learner.

### Performance Summary

| Metric | Value |
| --- | ---: |
| Validation RMSE | 0.4599 |
| Validation MAE | 0.3368 |
| Validation R² | 0.8940 |
| 5-fold CV RMSE (mean) | 0.5576 |
| 5-fold CV RMSE (std) | 0.0046 |

### Key Observations

1. **Ensemble superiority** — The Stacking Ensemble outperforms all individual models, demonstrating the value of combining diverse learners.
2. **Gradient boosting models** — XGBoost, HistGradientBoosting, and LightGBM all perform strongly, confirming the effectiveness of boosting for this task.
3. **Low variance** — The 5-fold CV RMSE standard deviation (0.0046) indicates stable performance across data splits.
4. **Linear models** — Linear regression and Ridge perform similarly, suggesting that the relationship between features and yield is non-linear.
5. **Tree-based models** — Random Forest and Extra Trees provide a solid baseline, but are outperformed by boosting and stacking approaches.