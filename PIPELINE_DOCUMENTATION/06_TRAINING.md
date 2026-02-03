# Stage 6: Training & Model Development

## Overview

This stage trains XGBoost and Ridge regression models to predict Borg effort ratings from selected features using GroupKFold cross-validation.

**Key Results:**
- **5s Windows (Best):** XGBoost r=0.626, Ridge r=0.644
- **10s Windows:** XGBoost r=0.548, Ridge r=0.567
- **30s Windows:** XGBoost r=0.364, Ridge r=0.184

---

## Data Preparation

### Input

**File:** `elderly_aligned_5.0s.csv` (855 labeled windows)

```
Shape: [855 rows × 280+ columns]
Columns: window_id, t_start, t_center, t_end, subject, borg, + 270+ features
```

### Data Cleaning

```python
# Load combined aligned dataset
df = pd.read_csv("elderly_aligned_5.0s.csv")

# Remove rows with missing Borg labels
df_labeled = df.dropna(subset=['borg'])
print(f"Labeled windows: {len(df_labeled)}")  # → 855

# Identify feature columns (exclude metadata)
skip_cols = {
    'window_id', 'start_idx', 'end_idx', 'valid',
    't_start', 't_center', 't_end', 'n_samples', 'win_sec',
    'modality', 'subject', 'label', 'borg',
}

def is_metadata(col):
    if col in skip_cols:
        return True
    if col.endswith("_r"):  # Redundant columns
        return True
    return False

feature_cols = [col for col in df.columns if not is_metadata(col)]
print(f"Feature columns: {len(feature_cols)}")  # → 270+

# Extract features and target
X = df_labeled[feature_cols].values
y = df_labeled['borg'].values

print(f"X shape: {X.shape}")  # → (855, 270+)
print(f"y stats: mean={y.mean():.2f}, std={y.std():.2f}")
```

### Input Statistics (5s Windows)

```
Total samples:       855
Feature count:       270+ (before selection)
                     48 (after selection)

Target (Borg) statistics:
  Mean:   4.8
  Std:    2.4
  Min:    0 (rest)
  Max:    8 (hard effort)
  Median: 5.0

Per-subject breakdown:
  sim_elderly3: ~280 samples
  sim_elderly4: ~290 samples
  sim_elderly5: ~285 samples
```

---

## Feature Selection

### Method: Correlation-Based Selection with Pruning

```python
from ml.feature_selection_and_qc import select_and_prune_features

# Select top 100 by correlation with Borg
# Then prune redundant features (r > 0.90)
pruned_indices, pruned_cols = select_and_prune_features(
    X, y, feature_cols,
    corr_threshold=0.90,  # Remove if r > 0.90 with another feature
    top_n=100             # Start with top 100 by correlation
)

X_selected = X[:, pruned_indices]
print(f"Selected {len(pruned_cols)} features from {len(feature_cols)}")
# → 48 from 270+
```

### Selection Results (5s Windows)

```
Before pruning:
  EDA: 28 features
  IMU: 19 features
  PPG: 53 features
  Total: 100

After pruning (r > 0.90 threshold):
  EDA: 8 features
  IMU: 19 features
  PPG: 19 features
  HRV: 2 features
  Total: 48 features
```

---

## Cross-Validation Strategy

### GroupKFold Cross-Validation

**Why GroupKFold instead of standard KFold?**

Standard KFold problem:
```
Window 1: Activity A, Borg=6 → Fold 1 (train)
Window 2: Activity A, Borg=6 → Fold 2 (test)
Window 3: Activity A, Borg=6 → Fold 1 (train)

Result: Model sees similar windows in train and test
        → Over-optimistic performance estimates
```

GroupKFold solution:
```
Group all windows from same activity together:
  Activity A (windows 1-5): All in Fold 1
  Activity B (windows 6-10): All in Fold 2
  ...

Result: No leakage between related windows
```

### Activity Group Creation

```python
# Create activity groups from subject + borg changes
activity_ids = []
current_id = 0
prev_subject = None
prev_borg = None

for i, (subj, borg) in enumerate(zip(df_labeled["subject"], y)):
    # New activity when subject or Borg changes
    if subj != prev_subject or borg != prev_borg:
        current_id += 1
    activity_ids.append(current_id)
    prev_subject = subj
    prev_borg = borg

groups = np.array(activity_ids)
n_activities = len(np.unique(groups))
print(f"Activities detected: {n_activities}")  # → 65 (5s windows)
```

### Fold Distribution

```
5s Windows:
  Total activities: 65
  Fold 1: ~13 activities, ~170 samples
  Fold 2: ~13 activities, ~170 samples
  Fold 3: ~13 activities, ~170 samples
  Fold 4: ~13 activities, ~170 samples
  Fold 5: ~13 activities, ~175 samples
```

---

## Model Training

### XGBoost Configuration

```python
import xgboost as xgb
from sklearn.model_selection import GroupKFold, cross_val_predict

model = xgb.XGBRegressor(
    n_estimators=100,      # Number of trees
    max_depth=4,           # Tree depth (regularization)
    learning_rate=0.1,     # Step size shrinkage
    random_state=42,       # Reproducibility
    n_jobs=-1,             # Use all CPUs
)

# GroupKFold CV
n_splits = min(5, n_activities)
cv = GroupKFold(n_splits=n_splits)

# Get cross-validated predictions
y_pred = cross_val_predict(model, X_selected, y, groups=groups, cv=cv)
```

### Ridge Regression Configuration

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Standardize features (required for Ridge)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Ridge regression with L2 regularization
model = Ridge(alpha=1.0)

# GroupKFold CV
y_pred = cross_val_predict(model, X_scaled, y, groups=groups, cv=cv)
```

---

## Evaluation Metrics

### Primary Metrics

```python
from scipy.stats import pearsonr
import numpy as np

# Pearson correlation
r, p = pearsonr(y, y_pred)

# Root Mean Square Error
rmse = np.sqrt(np.mean((y - y_pred) ** 2))

# Mean Absolute Error
mae = np.mean(np.abs(y - y_pred))

print(f"Pearson r: {r:.3f} (p={p:.2e})")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
```

### Results Summary

**5s Windows:**
```
XGBoost:
  Pearson r: 0.626 (p=2.8e-90)
  RMSE: 1.52
  MAE: 1.22

Ridge:
  Pearson r: 0.644 (p=1.4e-97)
  RMSE: 1.48
  MAE: 1.17
```

**10s Windows:**
```
XGBoost:
  Pearson r: 0.548 (p=1.2e-34)
  RMSE: 1.65
  MAE: 1.36

Ridge:
  Pearson r: 0.567 (p=2.0e-37)
  RMSE: 1.68
  MAE: 1.30
```

---

## Feature Importance

### XGBoost Feature Importance

```python
# Fit model on all data for feature importance
model.fit(X_selected, y)

importance = pd.DataFrame({
    "feature": pruned_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("Top 10 features by importance:")
for i, row in importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")
```

**Top 10 XGBoost Features (5s):**
```
1. ppg_green_range:             0.1824
2. ppg_green_p95:               0.0949
3. acc_x_dyn__cardinality:      0.0642
4. eda_stress_skin_max:         0.0456
5. ppg_green_trim_mean_10:      0.0398
6. acc_y_dyn__harmonic_mean:    0.0374
7. ppg_infra_p95:               0.0321
8. eda_phasic_energy:           0.0298
9. acc_z_dyn__lower_moment:     0.0287
10. ppg_green_ddx_kurtosis:     0.0256
```

### Ridge Coefficients

```python
# Fit model on all data for coefficients
model.fit(X_scaled, y)

coefficients = pd.DataFrame({
    "feature": pruned_cols,
    "coefficient": model.coef_,
    "abs_coefficient": np.abs(model.coef_)
}).sort_values("abs_coefficient", ascending=False)

print("Top 10 features by |coefficient|:")
for i, row in coefficients.head(10).iterrows():
    print(f"  {row['feature']}: {row['coefficient']:.4f}")
```

**Top 10 Ridge Features (5s):**
```
1. ppg_green_p95:               -0.8487 (negative: higher PPG = lower effort)
2. acc_x_dyn__cardinality:      +0.6033 (positive: more movement = higher effort)
3. ppg_red_signal_energy:       +0.3918
4. ppg_infra_n_peaks:           +0.3738
5. acc_x_dyn__quantile_0.9:     -0.3500
6. ppg_infra_shape_factor:      +0.3451
7. eda_cc_min:                  +0.3200
8. eda_stress_skin_max:         -0.3120
9. acc_y_dyn__harmonic_mean:    -0.2254
10. ppg_infra_ddx_kurtosis:     -0.2251
```

---

## Hyperparameter Choices

### XGBoost

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_estimators | 100 | Sufficient for 855 samples |
| max_depth | 4 | Prevents overfitting |
| learning_rate | 0.1 | Standard value |
| n_jobs | -1 | Use all CPUs |

### Ridge

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| alpha | 1.0 | Standard L2 regularization |
| StandardScaler | Yes | Required for meaningful coefficients |

### GroupKFold

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_splits | 5 | Standard CV, enough folds |
| groups | activity_ids | Prevent leakage |

---

## Training Time

| Model | 5s (855 samples) | 10s (424 samples) |
|-------|------------------|-------------------|
| XGBoost | ~15 seconds | ~8 seconds |
| Ridge | ~2 seconds | ~1 second |
| Total | ~20 seconds | ~10 seconds |

---

## Saving Results

### Model Artifacts

```python
import yaml

# Save XGBoost results
results_dir = output_path / "xgboost_results"
results_dir.mkdir(parents=True, exist_ok=True)

# Feature importance
importance.to_csv(results_dir / "feature_importance.csv", index=False)

# Predictions
results_df = pd.DataFrame({
    "y_true": y,
    "y_pred": y_pred,
    "subject": df_labeled["subject"].values,
    "activity_id": groups,
})
results_df.to_csv(results_dir / "predictions.csv", index=False)

# Summary metrics
summary = {
    "model": "XGBoost",
    "cv_method": f"GroupKFold ({n_splits} folds)",
    "n_samples": len(y),
    "n_activities": n_activities,
    "n_features": len(pruned_cols),
    "pearson_r": float(r),
    "p_value": float(p),
    "rmse": float(rmse),
    "mae": float(mae),
}
with open(results_dir / "summary.yaml", "w") as f:
    yaml.dump(summary, f)
```

### Output Structure

```
/Users/pascalschlegel/data/interim/elderly_combined/
├── xgboost_results/           # 5s XGBoost
│   ├── summary.yaml
│   ├── feature_importance.csv
│   └── predictions.csv
├── linear_results/            # 5s Ridge
│   ├── summary.yaml
│   ├── coefficients.csv
│   └── predictions.csv
├── xgboost_results_10.0s/     # 10s XGBoost
├── ridge_results_10.0s/       # 10s Ridge
├── xgboost_results_30.0s/     # 30s XGBoost
└── ridge_results_30.0s/       # 30s Ridge
```

---

## Why Ridge Outperforms XGBoost Here

**Observation:** Ridge (r=0.644) slightly beats XGBoost (r=0.626) on 5s windows.

**Possible reasons:**
1. **Limited samples (855):** Linear models can outperform trees on small datasets
2. **Selected features already correlated:** Top correlation-selected features have linear relationship with target
3. **Feature standardization:** Ridge benefits from scaled features
4. **Regularization:** L2 penalty works well with correlated features

**Recommendation:** Use Ridge for interpretability, XGBoost for non-linear patterns.

---

## Common Issues & Solutions

### Issue: RuntimeWarning in Ridge

```
RuntimeWarning: divide by zero encountered in matmul
```

**Cause:** Some features have zero variance or inf values

**Solution:**
```python
# Remove constant features before training
X_var = X_selected.var(axis=0)
valid_features = X_var > 0
X_filtered = X_selected[:, valid_features]
```

### Issue: GroupKFold fails

```
ValueError: n_splits cannot be greater than number of groups
```

**Cause:** Too few activities (groups) for requested folds

**Solution:**
```python
n_splits = min(5, n_activities)  # Adapt to available groups
```

### Issue: Poor CV scores

**Possible causes:**
1. Data leakage (use GroupKFold, not KFold)
2. Too few samples per fold
3. Noisy features included

---

## Summary

| Aspect | 5s Windows | 10s Windows |
|--------|------------|-------------|
| N Samples | 855 | 424 |
| N Activities | 65 | 61 |
| N Features | 48 | 51 |
| XGBoost r | 0.626 | 0.548 |
| Ridge r | 0.644 | 0.567 |
| Best Model | Ridge | Ridge |
| CV Method | GroupKFold (5) | GroupKFold (5) |

**Conclusion:** 5s windows with Ridge regression provide best performance (r=0.644, MAE=1.17).
