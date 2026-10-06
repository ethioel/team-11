## fig01_missingness.png
This figure shows missingness percentages across the raw inputs. The largest gaps are in rainfall, fertilizer, and soil-related measurements, guiding the cleaning strategy.

## fig02_before_after_cleaning.png
This comparison highlights how the farm-size distribution shifts after cleaning and imputation, confirming the cleaning step preserved the main data shape while reducing anomalies.

## fig03_yield_distribution.png
The yield distribution is right-skewed, which supports the use of robust validation and careful interpretation of outliers.

## fig04_region_crop_heatmap.png
This heatmap shows mean yield by region and crop, revealing strong regional differences in agricultural performance.

## fig05_correlation_heatmap.png
This correlation view summarizes the relationships among the engineered features and yield, helping identify stronger agronomic and climate predictors.

## fig06_climate_by_region.png
This monthly climate series shows how temperature patterns differ across the major agricultural regions.

## fig07_yield_vs_season_temp.png
The crop-specific relationship between season temperature and yield shows why weather-based features materially improve the model.

## fig08_price_trends.png
This price trend visualization highlights the inflation and seasonality in crop prices over time and explains why price was retained for revenue analysis rather than model training.

## fig10_model_comparison.png
This comparison shows model validation RMSE across candidate estimators and supports the final RandomForest choice.

## fig11_predicted_vs_actual_residuals.png
This plot helps assess whether prediction errors are concentrated at certain yield levels or under specific conditions.

## fig12_feature_importance.png
This importance chart shows which engineered and source features most strongly drive the final model’s predictions.
