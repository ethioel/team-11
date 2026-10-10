## fig01_missingness.png
This figure shows missingness percentages across the raw inputs. The largest gaps are in rainfall, fertilizer, and soil depth measures, which guide the cleaning decisions for the modeling dataset.

## fig02_before_after_cleaning.png
This comparison shows how the farm-size distribution shifts after standardization and imputation, confirming the cleaning step did not distort the central shape of the data.

## fig03_yield_distribution.png
The yield distribution is right-skewed, suggesting that the model should be evaluated with robust error metrics and careful handling of heteroskedasticity.

## fig04_region_crop_heatmap.png
This heatmap highlights where yield is strongest and weakest by region-crop combination, helping guide the agronomic interpretation of the final model.

## fig05_correlation_heatmap.png
This heatmap shows how the final engineered features relate to yield and to each other, identifying weather and agronomic variables worth keeping in the model.

## fig06_climate_by_region.png
The monthly climate trend shows how temperature patterns vary by region and should be interpreted with the growing-season timing in mind.

## fig07_yield_vs_season_temp.png
There is a crop-specific relationship between season temperature and yield, which is one reason the weather join materially improves model structure.

## fig08_price_trends.png
This trend plot shows how crop prices have moved over time and explains why price was kept separate from yield modeling.

## fig09_revenue_by_crop_region.png
Revenue per hectare varies sharply across region and crop, revealing economically important differences that are distinct from production efficiency.

## fig10_model_comparison.png
This comparison shows validation RMSE across all 18 candidate models. The Stacking Ensemble achieves the lowest RMSE (0.4599), followed by XGBoost (0.4693) and HistGradientBoosting (0.4719).

## fig11_predicted_vs_actual_residuals.png
The plot helps reveal systematic residual patterns and shows whether prediction errors are concentrated in particular yield ranges. The residuals are generally well-distributed around zero.

## fig12_feature_importance.png
This importance chart shows that weather-derived features (season_mean_temp, season_extreme_heat_days) are among the top predictors, validating the seasonal climate join. Crop type and altitude are also highly important.

