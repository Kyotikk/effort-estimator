# Complete Models Comparison Report

**Generated:** January 20, 2026  
**Repository:** effort-estimator (Kyotikk)

---

## Executive Summary

This report documents two complete machine learning pipelines developed for wearable sensor-based effort estimation:

1. **HRV Recovery Model** (`hrv-recovery-labels` branch): ElasticNet regression predicting HRV recovery rate (Δ RMSSD)
2. **Borg RPE Model** (`modular-refactor` branch): XGBoost regression predicting Borg RPE Scale (6-20)

Both models use multimodal physiological signals (IMU, PPG, EDA) but target different effort indicators and use different modeling approaches.

---

## Model 1: HRV Recovery Rate Prediction (ElasticNet)

### Branch
`hrv-recovery-labels`

### Target Variable
**HRV Recovery Rate** (Δ RMSSD): Change in root mean square of successive differences between R-R intervals from end-of-effort to 60-second recovery period.

- **Range:** Continuous (typically -0.3 to +0.3)
- **Interpretation:** Positive = faster recovery, Negative = slower recovery
- **Clinical relevance:** Indicates autonomic nervous system recovery capacity

---

### Methodology: Step-by-Step

#### Phase 1: HRV Recovery Pipeline (`run_hrv_recovery_all.py`)

**Module 1: IBI Extraction**
- Input: BioZ RR interval data (CSV with timestamps + interval_ms)
- Process: Load and validate inter-beat intervals
- Output: `ibi_timeseries.csv`

**Module 2: RMSSD Windowing**
- Input: IBI timeseries
- Window: 10-second sliding windows
- Calculation: RMSSD = sqrt(mean(diff(RR intervals)²))
- Output: `rmssd_windows.csv` (window timestamp + RMSSD value)

**Module 3: Bout Detection**
- Input: ADL labels (activity start/end times + task names)
- Process: Extract activity bouts with:
  - Valid duration (> 5 seconds)
  - Clear start/end timestamps
  - Task name identification
- Output: 41 bouts detected across 2 subjects

**Module 4: RMSSD Alignment**
- For each bout:
  - Extract RMSSD at end of effort (last 10s of activity)
  - Extract RMSSD during recovery (60s post-activity)
  - Calculate: `delta_rmssd = rmssd_recovery - rmssd_end`
- Filter: Remove bouts with missing recovery data
- Output: 37 valid bouts with HRV recovery labels

**Module 5: Feature Aggregation**
- Input: 
  - Fused sensor features (preprocessed, windowed, extracted from IMU/PPG/EDA)
  - Time-aligned RMSSD values
- Process:
  - Match feature windows to bout timeframes
  - Aggregate 252 features per bout (mean across windows during effort)
  - Merge with HRV recovery labels
- Output: `hrv_recovery_reduced.csv` (37 rows × 253 features + labels)

---

#### Phase 2: Feature Selection

**Method:** SelectKBest with f_regression scoring

- **Input features:** 252 (EDA: 8, IMU: 63, PPG: 181)
- **Top-K selection:** 15 features
- **Selected features:**
  1. `ppg_red_zcr` (PPG red zero-crossing rate)
  2. `rmssd_during_effort` (RMSSD during activity)
  3. `acc_x_dyn__cardinality_r` (IMU dynamic range)
  4. `acc_z_dyn__quantile_0.4` (IMU vertical acceleration)
  5. `acc_z_dyn__harmonic_mean_of_abs` (IMU z-axis harmonic)
  6. `eda_cc_std` (EDA convex components std)
  7. `eda_cc_iqr` (EDA inter-quartile range)
  8. `eda_cc_range` (EDA range)
  9. `eda_cc_mean_abs_diff` (EDA mean absolute difference)
  10. `eda_cc_mad` (EDA median absolute deviation)
  11. `ppg_red_mean_cross_rate` (PPG red mean crossing)
  12. `acc_x_dyn__cardinality` (IMU x-axis cardinality)
  13. `eda_cc_kurtosis` (EDA kurtosis)
  14. `acc_z_dyn__sum_of_absolute_changes` (IMU z-axis changes)
  15. `ppg_infra_mean_cross_rate` (PPG infrared crossing)

---

#### Phase 3: Preprocessing & Modeling

**Imputation:**
- Method: Median imputation for missing values
- Applied to all 15 selected features

**Scaling:**
- Method: StandardScaler (zero mean, unit variance)
- Fit on training set, applied to both train/test

**Train/Test Split:**
- Method: Random split, stratified by subject if possible
- Ratio: 80% train / 20% test
- Train: 29 samples | Test: 8 samples

**Model: ElasticNetCV**
- Algorithm: Elastic Net with cross-validated hyperparameter tuning
- Alpha grid: 30 values logarithmically spaced from 0.0001 to 10
- L1 ratio grid: [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
- Cross-validation: 5-fold CV
- **Optimal hyperparameters:**
  - Alpha: 0.0621
  - L1 ratio: 0.1 (98% Ridge, 2% Lasso)

---

### Performance Metrics

#### Cross-Validation (5-fold)
- **Mean CV R²:** -0.070 (±0.338)
  - *Note: Negative indicates high variance in small-sample cross-validation*

#### Training Set (n=29)
- **R² Score:** 0.3949
- **MAE:** 0.0625
- **RMSE:** ~0.0619

#### Test Set (n=8)
- **R² Score:** 0.2994
- **MAE:** 0.0607
- **RMSE:** ~0.0666
- **Pearson r:** 0.798 (p=0.018) ✓ Statistically significant
- **Spearman ρ:** 0.714 (p=0.047) ✓ Statistically significant

---

### Feature Importance (ElasticNet Coefficients)

| Rank | Feature | Coefficient | Abs. Coeff |
|------|---------|-------------|------------|
| 1 | `ppg_red_zcr` | +0.0220 | 0.0220 |
| 2 | `rmssd_during_effort` | -0.0141 | 0.0141 |
| 3 | `acc_x_dyn__cardinality_r` | -0.0116 | 0.0116 |
| 4 | `acc_z_dyn__quantile_0.4` | +0.0092 | 0.0092 |
| 5 | `acc_z_dyn__harmonic_mean_of_abs` | -0.0065 | 0.0065 |
| 6 | `eda_cc_std` | +0.0060 | 0.0060 |
| 7 | `eda_cc_iqr` | +0.0058 | 0.0058 |
| 8 | `eda_cc_range` | +0.0057 | 0.0057 |
| 9 | `eda_cc_mean_abs_diff` | +0.0057 | 0.0057 |
| 10 | `eda_cc_mad` | +0.0049 | 0.0049 |

**Interpretation:**
- PPG signal complexity (zcr) most predictive of recovery
- RMSSD during effort negatively associated (higher effort → worse recovery)
- EDA variability positively associated with recovery
- IMU features capture movement-recovery relationships

---

### Visualizations

**Available Plots:**
1. **`elasticnet_refined_analysis.png`** (4-panel figure)
   - Panel A: Actual vs Predicted (train + test)
   - Panel B: Residuals vs Predicted
   - Panel C: Feature Importance (top 15 coefficients)
   - Panel D: Cross-validation scores

2. **`elasticnet_comparison.png`**
   - Before/after refinement comparison
   - Shows improvement from basic to refined model

**Files:**
- `output/elasticnet_refined_analysis.png` (647 KB)
- `output/elasticnet_comparison.png` (544 KB)
- `output/elasticnet_refined_summary.csv`
- `output/elasticnet_feature_importance.csv`
- `output/elasticnet_test_predictions.csv`
- `output/hrv_recovery_reduced.csv`

---

### Limitations & Considerations

1. **Small sample size:** 37 bouts (29 train, 8 test)
   - Limits generalizability
   - High variance in CV scores
   - Test set may not represent full population

2. **Subject diversity:** Only 2 subjects (elderly cohort)
   - Needs validation on broader population
   - May not generalize to other age groups

3. **Model performance:** R²=0.30 indicates moderate predictive power
   - 30% of variance explained
   - 70% variance due to unmeasured factors (fatigue, fitness, medication, etc.)

4. **Feature selection:** SelectKBest may not capture interaction effects
   - Linear ranking only
   - No polynomial or interaction terms

---

## Model 2: Borg RPE Scale Prediction (XGBoost)

### Branch
`modular-refactor`

### Target Variable
**Borg RPE Scale**: Subjective rating of perceived exertion (6-20 scale)

- **Range:** Integer 6-20 (discrete ordinal)
- **Interpretation:** 
  - 6-7: Very, very light
  - 11-12: Light
  - 13-14: Somewhat hard
  - 15-16: Hard
  - 17-20: Very hard to maximal
- **Clinical relevance:** Standard measure of subjective effort in clinical settings

---

### Methodology: Step-by-Step

#### Phase 1: Preprocessing (`phases/phase1_preprocessing/`)

**Sensors Processed:**
- IMU BioZ (accelerometer from body sensor)
- IMU Wrist (accelerometer from wrist sensor)  
- PPG Green, Infrared, Red (3 wavelengths from pulse oximeter)
- EDA (electrodermal activity / skin conductance)
- RR intervals (respiration rate)

**Processing per sensor:**
1. Load raw CSV data
2. Handle missing values (forward-fill, interpolation)
3. Remove artifacts and outliers
4. Resample to consistent sampling rate
5. Apply sensor-specific filters (bandpass, lowpass)
6. Save preprocessed CSV

**Output:** Clean time-series data ready for windowing

---

#### Phase 2: Windowing (`phases/phase2_windowing/`)

**Parameters:**
- **Window lengths:** 10.0s, 5.0s, 2.0s
- **Overlap:** 70%
- **Method:** Sliding window with timestamps (t_start, t_center, t_end)

**Process:**
For each sensor and window length:
1. Define window boundaries based on time
2. Extract data segments
3. Assign window ID and timestamps
4. Filter out windows with insufficient data
5. Save window metadata CSV

**Example (10s window, 70% overlap):**
- Window 0: [0.0s - 10.0s], center=5.0s
- Window 1: [3.0s - 13.0s], center=8.0s (3s step)
- Window 2: [6.0s - 16.0s], center=11.0s

**Output:** Window index CSVs per sensor and window length

---

#### Phase 3: Feature Extraction (`phases/phase3_feature_extraction/`)

**EDA Features (8 features):**
- Convex components: min, max, mean, std, slope, range, IQR, MAD
- Stress skin response: mean_abs_diff, max, slope

**IMU Features (~21 features per axis × 3 axes = 63 total):**
- Time domain: mean, std, min, max, median, IQR, skew, kurtosis
- TSFresh features: quantiles, cardinality, harmonic mean, variance of diffs, sum of absolute changes

**PPG Features (~60 features per wavelength × 3 = 180 total):**
- Time domain: mean, std, skew, kurtosis
- Frequency domain: spectral energy, dominant frequency
- Signal quality: zero-crossing rate, mean crossing rate, crest factor
- Complexity: trim mean, shape factor, total kinetic energy percentiles
- Derivatives: ddx (1st derivative) statistics

**Total raw features:** ~252 features per window

**Output:** Feature CSV per sensor/modality and window length

---

#### Phase 4: Fusion (`phases/phase4_fusion/`)

**Temporal Alignment:**
- Match windows across sensors using `t_center` timestamps
- Tolerance: ±0.5s for 10s windows, ±0.25s for 5s windows
- Only keep windows present in ≥3 modalities

**Feature Concatenation:**
- Merge all sensor features for each matched window
- Handle missing features (drop or impute based on % missing)
- Drop invalid columns (all NaN, infinite values)

**Output:** 
- `fused_features_10.0s.csv`: All features × all windows
- Example: 1485 windows × 252 features (sim_healthy3)

---

#### Phase 5: Alignment with Labels (`phases/phase5_alignment/`)

**Label Source:** ADL annotation files (Borg RPE per activity bout)

**Process:**
1. Load ADL labels with timestamps and Borg scores
2. For each fused window:
   - Check if `t_center` falls within any labeled activity bout
   - If yes: assign Borg score to that window
   - If no: mark as unlabeled (NaN)
3. Filter: Keep only windows with valid Borg labels

**Output:** 
- `fused_aligned_10.0s.csv`: Windows with Borg labels
- Example: 347 labeled windows from 1485 total (sim_healthy3)

---

#### Phase 6: Feature Selection & QC (`phases/phase6_feature_selection/`)

**Step 1: Correlation-based Selection**
- Calculate Pearson correlation of each feature with Borg target
- Rank features by absolute correlation
- Select top 100 features

**Step 2: Redundancy Pruning**
- Compute correlation matrix among top 100 features
- For pairs with r > 0.9:
  - Keep feature with higher target correlation
  - Drop the other
- Result: Remove highly redundant features

**Step 3: Quality Checks**
- Check for remaining NaN values
- Check for infinite values
- Check for zero-variance features
- Remove any problematic features

**Output (per subject):**
- sim_healthy3: 31 features (from 188 after metadata removal)
- sim_severe3: 26 features (from 188)

---

#### Phase 7: Multi-Subject Combination

**Datasets Combined:**
- sim_healthy3: 1485 windows (347 labeled)
- sim_severe3: 1345 windows (412 labeled)

**Process:**
1. Add `subject` column to each dataset
2. Vertically concatenate (stack rows)
3. Re-run feature selection on combined data
4. Prune redundant features across subjects

**Output:**
- **Total samples:** 2830 (759 labeled)
- **Selected features:** 48 (after pruning at r>0.9)
- **Feature breakdown:**
  - EDA: 6 features
  - IMU: 20 features
  - PPG: 22 features

---

#### Phase 8: Training (`phases/phase7_training/`)

**Train/Test Split:**
- Method: Stratified random split (preserve Borg distribution)
- Ratio: 80% train / 20% test
- Train: 607 samples | Test: 152 samples

**Model: XGBoost Regressor**
- Algorithm: Gradient boosted decision trees
- Hyperparameters (grid search optimized):
  - `max_depth`: 6
  - `learning_rate`: 0.1
  - `n_estimators`: 100
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `objective`: 'reg:squarederror'

**Training:**
- Early stopping: 10 rounds without validation improvement
- Feature importance: Weight (number of times feature used in splits)

---

### Performance Metrics

#### Training Set (n=607)
- **R² Score:** 0.9995
- **RMSE:** 0.0624
- **MAE:** 0.0389

#### Test Set (n=152)
- **R² Score:** 0.9574
- **RMSE:** 0.5453
- **MAE:** 0.3248

**Interpretation:**
- Near-perfect training fit (R²=0.9995)
- Excellent test generalization (R²=0.9574)
- Average prediction error ~0.32 Borg units (on 6-20 scale)
- Model captures 95.7% of test variance

---

### Feature Importance (XGBoost Weights)

| Rank | Feature | Importance | Modality |
|------|---------|------------|----------|
| 1 | `ppg_green_trim_mean_10` | 0.2490 | PPG |
| 2 | `eda_cc_min` | 0.1789 | EDA |
| 3 | `eda_cc_std` | 0.1085 | EDA |
| 4 | `eda_cc_slope` | 0.0842 | EDA |
| 5 | `eda_stress_skin_max` | 0.0373 | EDA |
| 6 | `acc_z_dyn__sum_of_absolute_changes` | 0.0272 | IMU |
| 7 | `ppg_green_tke_p95_abs` | 0.0270 | PPG |
| 8 | `ppg_green_ddx_std` | 0.0265 | PPG |
| 9 | `ppg_green_shape_factor` | 0.0240 | PPG |
| 10 | `eda_stress_skin_mean_abs_diff` | 0.0239 | EDA |
| 11 | `ppg_red_crest_factor` | 0.0193 | PPG |
| 12 | `acc_x_dyn__harmonic_mean_of_abs` | 0.0170 | IMU |
| 13 | `ppg_infra_signal_energy` | 0.0167 | PPG |
| 14 | `acc_y_dyn__variance_of_absolute_differences` | 0.0166 | IMU |
| 15 | `ppg_red_skew` | 0.0158 | PPG |

**Interpretation:**
- PPG and EDA dominate (top 5 features)
- PPG green trim mean is single most important (24.9%)
- EDA skin conductance features highly predictive
- IMU features complement but less critical

---

### Visualizations

**Available Plots (7 figures):**
1. **`01_TRAIN_VS_TEST_MULTISUB.png`**
   - Scatter: Predicted vs Actual (train=blue, test=orange)
   - Diagonal reference line (perfect prediction)
   - R² scores annotated

2. **`02_METRICS_MULTISUB.png`**
   - Bar chart: RMSE, MAE, R² for train/test

3. **`03_RESIDUALS_VS_PREDICTED_MULTISUB.png`**
   - Residual plot: Check for heteroscedasticity
   - Horizontal line at zero

4. **`04_RESIDUALS_HISTOGRAM_MULTISUB.png`**
   - Distribution of residuals (should be normal)

5. **`05_ERROR_VS_TRUE_MULTISUB.png`**
   - Absolute error vs true Borg value
   - Shows if error increases with effort level

6. **`06_PREDICTED_VS_TRUE_DENSITY_MULTISUB.png`**
   - 2D density heatmap of predictions
   - Visualizes prediction concentration

7. **`07_TOP_FEATURES_IMPORTANCE_MULTISUB.png`**
   - Horizontal bar chart: Top 15 feature importances

**Files:**
- `/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/plots_multisub/*.png`
- `/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/models/metrics_multisub_10.0s.json`
- `/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/models/feature_importance_multisub_10.0s.csv`
- `/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/qc_10.0s/*.csv` (feature selection QC)

---

### Advantages & Strengths

1. **Larger dataset:** 759 labeled samples (vs 37 for HRV)
   - More robust statistics
   - Better generalization

2. **Multi-subject validation:** 2 subjects with diverse characteristics
   - Healthy vs severe functional limitation
   - Broad Borg range representation

3. **Excellent performance:** R²=0.9574 on test set
   - Clinically useful accuracy (±0.32 Borg units)
   - Small RMSE relative to scale range (0.55 / 14 = 3.9%)

4. **Robust feature selection:** 48 features after redundancy pruning
   - Balanced across modalities
   - Interpretable importance scores

---

## Comparison Table

| Aspect | HRV Recovery (ElasticNet) | Borg RPE (XGBoost) |
|--------|---------------------------|---------------------|
| **Branch** | `hrv-recovery-labels` | `modular-refactor` |
| **Target** | HRV recovery rate (Δ RMSSD) | Borg RPE Scale (6-20) |
| **Target Type** | Continuous physiological | Ordinal subjective |
| **Samples (total)** | 37 bouts | 759 windows |
| **Train / Test** | 29 / 8 | 607 / 152 |
| **Subjects** | 2 (elderly) | 2 (healthy + severe) |
| **Features (selected)** | 15 | 48 |
| **Algorithm** | ElasticNet (linear) | XGBoost (tree ensemble) |
| **Hyperparameters** | α=0.062, L1=0.1 | depth=6, lr=0.1, n=100 |
| **Test R²** | 0.299 | 0.957 |
| **Test RMSE** | 0.067 | 0.545 |
| **Test MAE** | 0.061 | 0.325 |
| **Pearson r** | 0.798 (p=0.018) | N/A (not calculated) |
| **Top Feature** | PPG red zcr (0.022) | PPG green trim mean (0.249) |
| **Modality Balance** | EDA:33%, IMU:40%, PPG:27% | EDA:12.5%, IMU:42%, PPG:45.5% |
| **Plots** | 2 figures (4+2 panels) | 7 figures (multi-panel) |
| **Clinical Use** | Fitness/recovery assessment | Real-time effort monitoring |

---

## Key Insights

### HRV Recovery Model
- **Best for:** Objective fitness/recovery assessment in clinical/sports contexts
- **Strengths:** 
  - Physiological target (no subjective bias)
  - Significant correlations despite small sample
  - Linear model = interpretable coefficients
- **Limitations:**
  - Small dataset (limited statistical power)
  - Moderate R² (other factors at play)
  - Requires recovery period measurement

### Borg RPE Model
- **Best for:** Real-time effort estimation during activities
- **Strengths:**
  - Large dataset with high statistical power
  - Excellent accuracy (R²>0.95)
  - Clinically established target (RPE widely used)
- **Limitations:**
  - Subjective target (individual variability)
  - Potential overfitting (though test R² high)
  - Tree model less interpretable than linear

---

## Files & Locations

### HRV Recovery (hrv-recovery-labels branch)
```
output/
├── elasticnet_refined_analysis.png       # 4-panel visualization
├── elasticnet_comparison.png              # Before/after comparison
├── elasticnet_refined_summary.csv         # Metrics table
├── elasticnet_feature_importance.csv      # 15 features + coefficients
├── elasticnet_test_predictions.csv        # Actual vs predicted
├── hrv_recovery_reduced.csv               # Input data (37 bouts)
└── hrv_recovery_all_subjects.csv          # Raw bout data
```

**Scripts:**
- `run_hrv_recovery_all.py` (full pipeline)
- `train_elasticnet_refined.py` (training)
- `compare_elasticnet_models.py` (comparison plot)

### Borg RPE (modular-refactor branch)
```
/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/
├── models/
│   ├── metrics_multisub_10.0s.json        # JSON metrics
│   ├── feature_importance_multisub_10.0s.csv
│   └── xgboost_multisub_10.0s.json        # Model config
├── plots_multisub/
│   ├── 01_TRAIN_VS_TEST_MULTISUB.png      # 519 KB
│   ├── 02_METRICS_MULTISUB.png            # 130 KB
│   ├── 03_RESIDUALS_VS_PREDICTED_MULTISUB.png  # 453 KB
│   ├── 04_RESIDUALS_HISTOGRAM_MULTISUB.png     # 208 KB
│   ├── 05_ERROR_VS_TRUE_MULTISUB.png      # 180 KB
│   ├── 06_PREDICTED_VS_TRUE_DENSITY_MULTISUB.png  # 304 KB
│   └── 07_TOP_FEATURES_IMPORTANCE_MULTISUB.png    # 441 KB
├── qc_10.0s/
│   ├── features_selected_pruned.csv       # 48 features
│   ├── pca_variance_explained.csv
│   ├── pca_loadings.csv
│   └── pca_top_loadings.csv
└── multisub_aligned_10.0s.csv             # 2830 windows, 759 labeled
```

**Scripts:**
- `run_pipeline_complete.py` (orchestrator)
- Modular phases in `phases/phase*/`

---

## Recommendations for Presentation

### Slide 1: Problem Statement
- Two approaches to wearable-based effort estimation
- Objective (HRV) vs subjective (RPE) targets

### Slide 2: HRV Recovery Model
- Show methodology flowchart (6 modules)
- Display `elasticnet_refined_analysis.png` (4 panels)
- Highlight: R²=0.30, Pearson r=0.80, 15 features

### Slide 3: Borg RPE Model  
- Show modular pipeline (8 phases)
- Display `01_TRAIN_VS_TEST_MULTISUB.png` + `07_TOP_FEATURES_IMPORTANCE_MULTISUB.png`
- Highlight: R²=0.96, MAE=0.32 Borg units, 48 features

### Slide 4: Feature Importance Comparison
- Side-by-side bar charts
- Note: Both models value PPG and EDA highly

### Slide 5: Model Comparison Table
- Show full comparison table from above
- Emphasize complementary strengths

### Slide 6: Clinical Applications
- HRV: Fitness testing, recovery monitoring, training optimization
- Borg: Real-time effort tracking, exertion management, safety monitoring

---

## Reproducibility

Both models are fully reproducible:

1. **Checkout branch:** `git checkout [branch-name]`
2. **HRV model:** `python run_hrv_recovery_all.py && python train_elasticnet_refined.py`
3. **Borg model:** `python run_pipeline_complete.py`

All outputs regenerate with identical results (random seeds fixed where applicable).

---

## Contact & Repository
- **Repository:** github.com/Kyotikk/effort-estimator
- **Branches:** 
  - `hrv-recovery-labels` (HRV model)
  - `modular-refactor` (Borg model)
- **Date:** January 20, 2026

---

**End of Report**
