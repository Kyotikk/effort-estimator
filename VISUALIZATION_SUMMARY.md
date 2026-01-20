# HRV Recovery Model Visualizations

## Overview
7 comprehensive visualizations have been generated for the HRV recovery effort estimation model. All plots are saved in `output/` directory.

## Generated Plots

### 1. **Feature Importance** (`plot_01_feature_importance.png`)
- **Content**: Top 15 most important features from XGBoost model
- **Key Features**:
  - EDA features dominate importance scores
  - Accelerometer metrics contribute significantly
  - PPG metrics present in top features
  - RMSSD during effort is ranked

### 2. **ElasticNet Predictions** (`plot_02_predictions_elasticnet.png`)
- **Content**: Actual vs Predicted scatter plots for train and test sets
- **Train Set**: 19 samples, Shows model fit quality during training
- **Test Set**: 5 samples, Shows generalization performance
- **Perfect fit line**: Red dashed diagonal line (ideal predictions)

### 3. **XGBoost Predictions** (`plot_03_predictions_xgboost.png`)
- **Content**: Actual vs Predicted scatter plots for train and test sets
- **Comparison**: Side-by-side comparison with ElasticNet
- **Overfitting indicator**: Shows XGBoost overfitting on train data

### 4. **Residuals Analysis** (`plot_04_residuals.png`)
- **4 subplots**:
  1. ElasticNet test residuals vs predictions
  2. XGBoost test residuals vs predictions
  3. ElasticNet residuals distribution
  4. XGBoost residuals distribution
- **Key metric**: Red dashed line at y=0 shows perfect residuals

### 5. **Error Metrics Comparison** (`plot_05_error_metrics.png`)
- **3 metrics panels**:
  - **MAE (Mean Absolute Error)**: Average prediction error magnitude
  - **RMSE (Root Mean Squared Error)**: Penalizes larger errors
  - **R² Score**: Proportion of variance explained
- **Color coding**:
  - Blue: Train set performance
  - Red: Test set performance

### 6. **Model Summary** (`plot_06_summary.png`)
- **Metrics Table**: Complete performance metrics for all models
- **Target Distribution**: Histogram of Δ RMSSD values
- **Subject Distribution**: Sample count per subject
- **Dataset Info**: Key statistics and findings

### 7. **Distribution Comparison** (`plot_07_distribution_comparison.png`)
- **4 subplots**:
  1. ElasticNet train predictions distribution
  2. ElasticNet test predictions distribution
  3. XGBoost train predictions distribution
  4. XGBoost test predictions distribution
- **Colors**:
  - Blue/Green: Predicted values
  - Coral: Actual values

## Key Metrics Summary

### Dataset
- **Total Samples**: 24 (after NaN removal)
- **Train**: 19 samples (79%)
- **Test**: 5 samples (21%)
- **Features**: 15 selected from 252 original

### Model Performance

#### ElasticNet (Best Generalization)
| Metric | Train | Test |
|--------|-------|------|
| MAE | 0.0465 | 0.0944 |
| RMSE | 0.0657 | 0.1356 |
| R² | 0.6188 | -0.4200 |

#### XGBoost (Overfitted)
| Metric | Train | Test |
|--------|-------|------|
| MAE | 0.0006 | 0.0958 |
| RMSE | 0.0008 | 0.1398 |
| R² | 0.9999 | -0.5087 |

## Interpretation

### Strong Points
✓ ElasticNet shows reasonable generalization on train set (R²=0.62)  
✓ Feature selection reduced feature space by 94% (252→15)  
✓ Low MAE (~0.09) indicates manageable prediction error  
✓ Distribution of predictions reasonably aligned with actuals  

### Challenges
⚠ XGBoost severely overfits (train R²=1.0, test R²=-0.51)  
⚠ Very small test set (n=5) limits generalization assessment  
⚠ Negative R² on test indicates poor out-of-sample performance  
⚠ Some residuals show systematic bias  

### Top Features Selected
1. EDA conductivity metrics (std, range, IQR, MAD)
2. Accelerometer z-dynamics (quantiles, cardinality)
3. PPG metrics (cross-rate, zero-crossing)
4. RMSSD during effort

## Recommendations

1. **Increase Training Data**: Current n=24 is too small for robust models
2. **Use ElasticNet**: Regularization helps with small samples
3. **Feature Engineering**: Focus on the top 5-7 most important features
4. **Cross-Validation**: Use k-fold instead of single train/test split
5. **Ensemble Methods**: Combine multiple weak learners rather than single strong model

## File Sizes
- plot_01_feature_importance.png: 166 KB
- plot_02_predictions_elasticnet.png: 240 KB
- plot_03_predictions_xgboost.png: 235 KB
- plot_04_residuals.png: 165 KB
- plot_05_error_metrics.png: 320 KB
- plot_06_summary.png: 225 KB
- plot_07_distribution_comparison.png: 230 KB

**Total**: ~1.6 MB
