# Effort Estimation Pipeline - Technical Documentation
## For Machine Learning Expert Presentation

---

## 1. Executive Summary

**Objective:** Predict perceived exertion (Borg scale 0-20) from wearable sensor data in elderly patients during activities of daily living (ADL).

**Dataset:** 3 elderly patients (sim_elderly3, sim_elderly4, sim_elderly5)

**Best Result:** Pearson r = **0.644** (Ridge regression, 5s windows)

---

## 2. Data Sources & Preprocessing

### 2.1 Sensor Modalities (6 total)

| Modality | Sensor Location | Sampling Rate | Preprocessing |
|----------|-----------------|---------------|---------------|
| **IMU (BioZ)** | Chest | 32 Hz | Gravity removal (HPF 0.3 Hz), noise filter (LPF 5 Hz) |
| **IMU (Wrist)** | Wrist | 32 Hz | Same as above |
| **PPG Green** | Wrist | 32 Hz | No HPF (strong signal, 8614 units baseline) |
| **PPG Infrared** | Wrist | 32 Hz | HPF 0.5 Hz (baseline drift removal) |
| **PPG Red** | Wrist | 32 Hz | HPF 0.5 Hz (weak signal, 2731 units baseline) |
| **EDA** | Wrist | 32 Hz | Tonic/phasic decomposition |

### 2.2 Target Variable

- **Borg RPE Scale:** 0-20 (Rate of Perceived Exertion)
- **Collection method:** ADL annotations with timestamps
- **Range observed:** 0-17 (no extreme exertion in elderly cohort)

---

## 3. Windowing Strategy

### 3.1 Window Configuration

| Parameter | 5s Windows (Best) | 10s Windows |
|-----------|-------------------|-------------|
| **Window length** | 5.0 seconds | 10.0 seconds |
| **Overlap** | 10% (0.5s stride) | 10% (1.0s stride) |
| **Fusion tolerance** | 2.0 seconds | 2.0 seconds |
| **Label alignment** | Window center time | Window center time |

### 3.2 Window Size Comparison Results

| Window | N Samples | Features Selected | XGBoost r | Ridge r | Best MAE |
|--------|-----------|-------------------|-----------|---------|----------|
| **5s** | **855** | **48** | **0.626** | **0.644** | **1.17** |
| 10s | 424 | 51 | 0.548 | 0.567 | 1.30 |

**Conclusion:** 5s windows optimal due to:
1. 2× more training samples (855 vs 424)
2. Better temporal resolution for effort changes
3. More stable cross-validation

---

## 4. Feature Extraction

### 4.1 Raw Feature Count by Modality

| Modality | Features Extracted |
|----------|-------------------|
| IMU (BioZ + Wrist) | ~90 features (tsfresh-based) |
| PPG Green | ~44 features + HRV metrics |
| PPG Infrared | ~44 features + HRV metrics |
| PPG Red | ~44 features + HRV metrics |
| EDA | ~40 features (basic + advanced SCR/SCL) |
| **Total Raw** | **~260 features** |

### 4.2 Feature Categories

**Time-domain statistics:**
- Mean, std, min, max, range, median, IQR, MAD
- Quantiles (0.1, 0.3, 0.4, 0.9)
- Skewness, kurtosis

**Frequency-domain:**
- Spectral energy, dominant frequency
- Zero-crossing rate (zcr), mean-cross rate

**Signal morphology:**
- Crest factor, shape factor, clearance factor
- Peak-to-peak, signal energy
- First/second derivative statistics (dx, ddx)

**HRV metrics (PPG):**
- HR mean, std, min, max, range
- RMSSD, SDNN, pNN20, pNN50
- Mean IBI, CV of IBI

**EDA-specific:**
- Skin conductance level (SCL): min, slope
- Skin conductance response (SCR): count, amplitude sum
- Phasic energy, stress skin metrics

---

## 5. Feature Selection Pipeline

### 5.1 Two-Stage Selection Process

**Stage 1: Correlation-based selection**
```python
# Select top 100 features by absolute Pearson correlation with Borg target
correlations = X.corrwith(y).abs()
top_100 = correlations.nlargest(100).index.tolist()
```

**Stage 2: Redundancy pruning**
```python
# Remove highly correlated feature pairs (r > 0.90)
# Keep the one with higher target correlation
for each pair (i, j) in top_100:
    if corr(feature_i, feature_j) > 0.90:
        drop the one with lower |corr(feature, target)|
```

### 5.2 Selection Results (5s Windows)

| Stage | EDA | IMU | PPG | Total |
|-------|-----|-----|-----|-------|
| Before pruning | 26 | 15 | 59 | 100 |
| After pruning | 6 | 14 | 28 | **48** |

**Dimensionality reduction:** 260 → 48 features (81.5% reduction)

### 5.3 PCA Quality Check

| Metric | Value |
|--------|-------|
| PCs for 90% variance | 24 |
| PCs for 95% variance | 30 |
| PCs for 99% variance | 41 |

---

## 6. Model Training

### 6.1 Cross-Validation Strategy: GroupKFold

**Why GroupKFold instead of standard KFold?**

Standard KFold would leak information because consecutive windows from the same activity share temporal context. GroupKFold ensures:

1. **Groups = Activity IDs** (65 unique activities across 3 patients)
2. **5 folds**, each fold tests on ~13 held-out activities
3. **No data leakage:** All windows from one activity stay together
4. **Realistic evaluation:** Simulates predicting effort for unseen activities

```python
from sklearn.model_selection import GroupKFold

groups = df['activity_id'].values  # 65 unique activities
gkf = GroupKFold(n_splits=5)

for train_idx, test_idx in gkf.split(X, y, groups):
    # Train on ~52 activities, test on ~13 activities
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
```

### 6.2 XGBoost Configuration

```python
XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### 6.3 Ridge Regression Configuration

```python
Ridge(alpha=1.0)  # Standard L2 regularization
```

**Note:** Features standardized (StandardScaler) before Ridge training.

---

## 7. Results

### 7.1 Performance Metrics (5s Windows, 855 samples)

| Model | Pearson r | p-value | RMSE | MAE |
|-------|-----------|---------|------|-----|
| **Ridge** | **0.644** | 1.79e-101 | 1.52 | **1.17** |
| XGBoost | 0.626 | 4.29e-94 | 1.52 | 1.22 |

### 7.2 Top 10 Features by XGBoost Importance (Gain)

| Rank | Feature | Importance | Modality | Interpretation |
|------|---------|------------|----------|----------------|
| 1 | `ppg_green_p95` | 0.2461 | PPG | 95th percentile of pulse amplitude |
| 2 | `ppg_infra_n_peaks` | 0.0772 | PPG | Number of detected heartbeats (HR proxy) |
| 3 | `eda_phasic_energy` | 0.0701 | EDA | Sympathetic arousal energy |
| 4 | `acc_x_dyn__quantile_0.9` | 0.0594 | IMU | High movement intensity (x-axis) |
| 5 | `acc_z_dyn__variance_of_absolute_differences` | 0.0483 | IMU | Vertical movement variability |
| 6 | `eda_stress_skin_max` | 0.0472 | EDA | Peak stress response |
| 7 | `eda_stress_skin_iqr` | 0.0404 | EDA | Stress response variability |
| 8 | `acc_z_dyn__sum_of_absolute_changes` | 0.0393 | IMU | Total vertical movement |
| 9 | `acc_y_dyn__quantile_0.3` | 0.0384 | IMU | Lateral movement (lower quantile) |
| 10 | `eda_cc_min` | 0.0303 | EDA | Minimum skin conductance |

### 7.3 Top 10 Features by Ridge |Coefficient|

| Rank | Feature | Coefficient | Direction |
|------|---------|-------------|-----------|
| 1 | `ppg_green_p95` | -0.8873 | Higher pulse → Lower predicted effort* |
| 2 | `ppg_green_range` | -0.6438 | Higher range → Lower predicted effort* |
| 3 | `ppg_green_diff_rms` | +0.5837 | Higher derivative RMS → Higher effort |
| 4 | `acc_x_dyn__cardinality` | +0.4418 | More unique values → Higher effort |
| 5 | `eda_cc_min` | +0.3708 | Higher baseline conductance → Higher effort |
| 6 | `ppg_green_diff_mean_abs` | -0.3447 | Complex relationship |
| 7 | `eda_stress_skin_max` | -0.3158 | Complex relationship |
| 8 | `acc_x_dyn__quantile_0.9` | -0.2965 | Complex relationship |
| 9 | `ppg_green_p90_p10` | +0.2823 | Higher percentile range → Higher effort |
| 10 | `acc_z_dyn__quantile_0.4` | +0.2111 | Vertical movement → Higher effort |

*Negative coefficients indicate inverse relationship after standardization; physiological interpretation requires domain context.

---

## 8. Key Findings

### 8.1 Most Predictive Signal: PPG Green

**`ppg_green_p95` alone accounts for 24.6% of XGBoost's predictive gain.**

Why? The 95th percentile of PPG amplitude captures:
- Cardiac output changes during exertion
- Pulse pressure variations
- Blood volume shifts

### 8.2 EDA as Stress/Effort Indicator

EDA features (`eda_phasic_energy`, `eda_stress_skin_max`, `eda_stress_skin_iqr`) capture:
- Sympathetic nervous system activation
- Sweat gland activity during effort
- Arousal level independent of movement

### 8.3 IMU Movement Features

Movement features (`acc_*_dyn__*`) provide:
- Direct measure of physical activity intensity
- Complementary to physiological signals
- Robust even when PPG signal quality degrades

### 8.4 Ridge vs XGBoost

Ridge outperforms XGBoost (r=0.644 vs 0.626) likely because:
- Dataset size (855 samples) favors simpler models
- Linear relationships dominate in this feature space
- Less prone to overfitting with 48 features

---

## 9. Per-Subject Breakdown

| Subject | Total Windows | Labeled Windows | % Labeled |
|---------|---------------|-----------------|-----------|
| sim_elderly3 | 438 | 287 | 65.5% |
| sim_elderly4 | 386 | 299 | 77.5% |
| sim_elderly5 | 336 | 269 | 80.1% |
| **Combined** | **1160** | **855** | **73.7%** |

---

## 10. Limitations & Future Work

### 10.1 Current Limitations

1. **Within-subject validation only:** GroupKFold groups by activity, not by subject
2. **Small cohort:** 3 elderly patients, limited generalization
3. **Elderly-only:** Results may not transfer to younger/healthier populations
4. **Correlation ≠ Causation:** Feature importance reflects association, not mechanism

### 10.2 Recommended Next Steps

1. **Leave-One-Subject-Out (LOSO)** validation for true cross-subject generalization
2. **Expand to 8-10 subjects** across age/health groups
3. **Online prediction** for real-time effort monitoring
4. **Calibration period** for personalized model fine-tuning

---

## 11. Reproducibility

### 11.1 Commands to Reproduce

```bash
# Run full 5s pipeline
python run_elderly_pipeline.py

# Run specific window size
python run_elderly_10s_30s.py --window 5.0
python run_elderly_10s_30s.py --window 10.0
```

### 11.2 Output Locations

```
/Users/pascalschlegel/data/interim/elderly_combined/
├── elderly_aligned_5.0s.csv          # 855 labeled samples
├── elderly_aligned_10.0s.csv         # 424 labeled samples
├── qc_5.0s/
│   ├── features_selected_pruned.csv  # 48 selected features
│   ├── pca_variance_explained.csv
│   └── pca_loadings.csv
├── xgboost_results_5.0s/
│   ├── summary.yaml
│   ├── feature_importance.csv
│   └── predictions.csv
└── ridge_results_5.0s/
    ├── summary.yaml
    ├── coefficients.csv
    └── predictions.csv
```

---

## 12. Summary for Presentation

| Aspect | Value |
|--------|-------|
| **Patients** | 3 elderly |
| **Best window** | 5 seconds |
| **Samples** | 855 |
| **Raw features** | 260 |
| **Selected features** | 48 |
| **Best model** | Ridge regression |
| **Pearson r** | **0.644** |
| **MAE** | **1.17 Borg points** |
| **CV method** | GroupKFold (5 folds, 65 activity groups) |
| **Top feature** | `ppg_green_p95` (24.6% importance) |

---

**Key takeaway:** Using multi-modal wearable sensors (PPG, EDA, IMU), we achieve r=0.644 correlation with perceived exertion in elderly patients, with PPG amplitude and EDA stress features being the most predictive.
