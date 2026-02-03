# Stage 5: Feature Selection

## Overview

**Purpose:** Reduce feature count from 270+ → 48-51 features to prevent overfitting and improve model generalization.

**Method:** Two-stage selection:
1. **Correlation-based selection** - Keep top 100 features correlated with Borg
2. **Redundancy pruning** - Remove features with r > 0.90 correlation with each other

---

## Current Results by Window Size

| Window | Raw Features | After Selection | Method |
|--------|--------------|-----------------|--------|
| **5s** | 270+ | **48** | Corr + Prune (r=0.90) |
| **10s** | 270+ | **51** | Corr + Prune (r=0.90) |
| **30s** | 270+ | **20** | Corr + Prune (r=0.90) |

---

## The Problem: Why More Features Hurt

### Curse of Dimensionality

With too many features relative to samples, models overfit:

| Window | Samples | Raw Features | Ratio | Status |
|--------|---------|--------------|-------|--------|
| 5s | 855 | 270+ | 3.2 | Too low |
| 5s | 855 | 48 | 17.8 | ✅ Good |
| 10s | 424 | 270+ | 1.6 | Very low |
| 10s | 424 | 51 | 8.3 | Acceptable |
| 30s | 100 | 270+ | 0.4 | Critical |
| 30s | 100 | 20 | 5.0 | Marginal |

**Rule of thumb:** Need 10-20 samples per feature for stable ML models

---

## Two-Stage Feature Selection

### Stage 1: Correlation-Based Selection

**SelectKBest with f_regression:**

```python
from sklearn.feature_selection import SelectKBest, f_regression

# Score each feature's correlation with target (Borg)
selector = SelectKBest(f_regression, k=100)
X_selected = selector.fit_transform(X, y)

# Get feature scores (regression F-statistic)
scores = selector.scores_
```

**F-Regression Scoring:**

For each feature, compute the F-statistic:
$$F = \frac{\text{Explained Variance}}{\text{Unexplained Variance}}$$

**High F-score:** Feature correlates strongly with target (effort)  
**Low F-score:** Feature is noise or independent of effort

### Stage 2: Redundancy Pruning

**Problem:** Many features are highly correlated with each other (redundant).

**Solution:** Remove features with Pearson r > 0.90 correlation:

```python
def prune_correlated_features(X, threshold=0.90):
    """Remove features with > threshold correlation"""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = []
    for column in upper.columns:
        if any(upper[column] > threshold):
            to_drop.append(column)
    
    return X.drop(columns=to_drop)
```

**Example pruning (5s windows):**
```
Before pruning: 100 features
After pruning:  48 features
Removed: 52 redundant features
```

---

## Top Features by Modality (5s Windows)

### Feature Distribution After Selection

| Modality | Selected | % of Total |
|----------|----------|------------|
| **EDA** | 18 | 37.5% |
| **PPG Green** | 12 | 25.0% |
| **PPG Infra** | 8 | 16.7% |
| **PPG Red** | 4 | 8.3% |
| **IMU** | 6 | 12.5% |
| **TOTAL** | **48** | **100%** |

### Key Insight: EDA Dominates

**EDA features account for 37.5% of selected features**, revealing that **electrodermal activity (stress/arousal)** is the strongest predictor of perceived effort.

### Top 15 Selected Features (5s)

| Rank | Feature | Modality | Interpretation |
|------|---------|----------|----------------|
| 1 | eda_cc_mean_abs_diff | EDA | Stress conductance changes |
| 2 | eda_cc_range | EDA | Stress level variability |
| 3 | eda_cc_std | EDA | Conductance deviation |
| 4 | eda_cc_slope | EDA | Stress trend |
| 5 | eda_cc_iqr | EDA | Stress spread |
| 6 | eda_cc_mad | EDA | Median absolute deviation |
| 7 | ppg_infra_zcr | PPG IR | Zero-crossing rate |
| 8 | ppg_infra_mean_cross_rate | PPG IR | Mean crossing rate |
| 9 | ppg_red_zcr | PPG Red | Zero-crossing rate |
| 10 | ppg_green_p95_p5 | PPG Green | Percentile range |
| 11 | imu_magnitude_mean | IMU | Movement intensity |
| 12 | imu_acc_energy | IMU | Kinetic energy |
| 13 | ppg_green_hr_std | PPG Green | HR variability |
| 14 | ppg_infra_max | PPG IR | Max signal |
| 15 | imu_jerk_magnitude | IMU | Movement abruptness |

---

## Window Size Comparison

### Why 30s Has Fewer Features (20)

With only 100 samples for 30s windows:
- Correlation estimates become unstable
- Many features appear uncorrelated by chance
- Aggressive pruning needed to avoid overfitting
- Result: Only 20 robust features survive

### Feature Overlap Across Windows

Most selected features are consistent across window sizes:
- **EDA features:** Selected in all window sizes
- **PPG Zero-crossing:** Selected in all window sizes
- **IMU energy/magnitude:** Selected in 5s and 10s
- **Spectral features:** More selected in 5s (higher temporal resolution)

---

## Quality Control Output

Feature selection generates QC files:

```
/Users/pascalschlegel/data/interim/elderly_combined/qc_5.0s/
├── features_selected_pruned.csv   # Final 48 feature names
├── feature_correlations.csv       # Correlation matrix
├── feature_scores.csv             # SelectKBest F-scores
├── pca_variance.csv               # PCA variance explained
└── pca_loadings.csv               # PCA component loadings
```

### PCA Quality Check

After selection, PCA is run to verify feature quality:

```
PCA on 48 selected features (5s windows):
  PC1: 28.4% variance (general activity level)
  PC2: 18.2% variance (EDA vs IMU contrast)
  PC3: 12.1% variance (PPG wavelength differences)
  ...
  Cumulative variance (10 PCs): 78.5%
```

---

## Implementation

**Full selection pipeline (`ml/feature_selection_and_qc.py`):**

```python
def select_and_prune_features(X, y, k_select=100, corr_threshold=0.90):
    """
    Two-stage feature selection:
    1. SelectKBest (k=100)
    2. Prune r > 0.90 correlated
    """
    # Stage 1: Correlation selection
    selector = SelectKBest(f_regression, k=min(k_select, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_names = X.columns[selector.get_support()].tolist()
    X_df = pd.DataFrame(X_selected, columns=selected_names)
    
    # Stage 2: Redundancy pruning
    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = []
    for column in upper.columns:
        highly_correlated = upper[column] > corr_threshold
        if highly_correlated.any():
            to_drop.append(column)
    
    X_pruned = X_df.drop(columns=to_drop, errors='ignore')
    
    return X_pruned, X_pruned.columns.tolist()
```

---

## Why This Approach Works

### Benefits of Two-Stage Selection

1. **SelectKBest:** Fast, univariate scoring that keeps meaningful features
2. **Correlation pruning:** Removes redundant information
3. **Combined:** Get diverse, informative feature set

### Comparison to Other Methods

| Method | Pros | Cons | Used? |
|--------|------|------|-------|
| **SelectKBest + Prune** | Fast, interpretable | Ignores interactions | ✅ Yes |
| **RFE** | Considers interactions | Very slow | No |
| **L1/Lasso** | Automatic, embedded | Linear only | No |
| **Tree importance** | Nonlinear | Circular logic | No |

---

## Impact on Model Performance

### 5s Windows

| Metric | Before Selection | After Selection |
|--------|-----------------|-----------------|
| Features | 270+ | 48 |
| Samples/feature | 3.2 | 17.8 |
| XGBoost r | Overfit | **0.626** |
| Ridge r | Overfit | **0.644** |
| CV stable | No | Yes |

### 10s Windows

| Metric | Before Selection | After Selection |
|--------|-----------------|-----------------|
| Features | 270+ | 51 |
| Samples/feature | 1.6 | 8.3 |
| XGBoost r | Overfit | **0.548** |
| Ridge r | Overfit | **0.567** |

---

## Future Improvements

With more data, could:

1. **Increase k:** More samples = more features feasible
2. **Relax pruning:** r > 0.95 threshold instead of 0.90
3. **Add domain features:** HR/RR/BP interactions
4. **Embedded selection:** Use model-based importance

---

## Summary

| Aspect | 5s Windows | 10s Windows | 30s Windows |
|--------|------------|-------------|-------------|
| **Input features** | 270+ | 270+ | 270+ |
| **Output features** | 48 | 51 | 20 |
| **Method** | Corr + Prune | Corr + Prune | Corr + Prune |
| **Top modality** | EDA (37.5%) | EDA (~35%) | EDA (~40%) |
| **Samples/feature** | 17.8 | 8.3 | 5.0 |
| **Status** | ✅ Good | ✅ OK | ⚠️ Marginal |

---

## Next Step: Training

The selected features are now ready for training with GroupKFold cross-validation. See [06_TRAINING.md](06_TRAINING.md).
