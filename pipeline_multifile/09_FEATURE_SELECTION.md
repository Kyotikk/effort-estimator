# Stage 9: Feature Selection

## Purpose

Reduce feature dimensionality from 257 → 100 features per condition using variance-based ranking, removing non-predictive columns.

---

## 9.1 Feature Selection Method

**File**: `train_condition_specific_xgboost.py` - `get_drop_columns()` function

### Step 1: Identify Columns to Drop

```python
def get_drop_columns(df):
    """
    Identify all non-predictive/metadata columns to drop.
    """
    cols_to_drop = []
    
    # Always drop target
    if 'borg' in df.columns:
        cols_to_drop.append('borg')
    
    # Drop subject/metadata
    for col in ['subject', 'modality']:
        if col in df.columns:
            cols_to_drop.append(col)
    
    # Drop timing columns
    for col in df.columns:
        if col.startswith('t_') or 'time' in col.lower():
            cols_to_drop.append(col)
        elif 'window' in col.lower() or 'idx' in col.lower():
            cols_to_drop.append(col)
        elif col in ['n_samples', 'win_sec', 'valid', 'valid_r']:
            cols_to_drop.append(col)
        elif col.endswith('_r') or '_r.' in col:
            # Duplicate columns
            cols_to_drop.append(col)
    
    return list(set(cols_to_drop))  # Remove duplicates
```

### Columns Dropped (69 total)

```
Metadata/Timing (25):
  start_idx, end_idx, window_id, t_start, t_center, t_end
  subject, modality, valid, valid_r, n_samples, win_sec
  (+ 13 other timing columns)

Target (1):
  borg

Duplicates with _r suffix (43):
  (all features ending in _r or containing _r.)
```

### Step 2: Calculate Feature Variance

```python
# Select only condition data
cond_df = combined_df[combined_df['subject'] == condition].copy()
cond_df = cond_df.dropna(subset=['borg'])

# Drop non-feature columns
cols_drop = get_drop_columns(cond_df)
X = cond_df.drop(columns=cols_drop)

# Calculate variance per feature
variance = X.var()
variance_sorted = variance.sort_values(ascending=False)

print(f"Feature variance (top 20):")
print(variance_sorted.head(20))
```

### Step 3: Select Top 100 Features

```python
# Select top 100 by variance
top_100_features = variance_sorted.head(100).index.tolist()

print(f"Variance threshold: {variance_sorted.iloc[99]:.6f}")
print(f"Selected {len(top_100_features)} features")

# Save feature list
with open(f"{condition}_features.json", 'w') as f:
    json.dump(top_100_features, f, indent=2)
```

---

## 9.2 Example Feature Ranking

### elderly3 Top 20 Features by Variance

```
Rank | Feature Name                    | Variance   | Modality
──────┼─────────────────────────────────┼────────────┼──────────
1    | std_accel_y                     | 0.0892     | IMU Bioz
2    | rms_accel_z                     | 0.0734     | IMU Bioz
3    | heart_rate_mean                 | 0.0651     | PPG Green
4    | peak_count_x                    | 0.0512     | IMU Wrist
5    | entropy_accel_z                 | 0.0456     | IMU Bioz
6    | mean_accel_magnitude            | 0.0398     | IMU Wrist
7    | ppg_amplitude_red               | 0.0365     | PPG Red
8    | skewness_accel_y                | 0.0312     | IMU Bioz
9    | hrv_rmssd_green                 | 0.0287     | PPG Green
10   | zero_cross_accel_x              | 0.0251     | IMU Wrist
11   | scl_level_std                   | 0.0198     | EDA
12   | iqr_accel_y                     | 0.0167     | IMU Bioz
13   | max_accel_z                     | 0.0145     | IMU Bioz
14   | heart_rate_std                  | 0.0134     | PPG Infra
15   | pnn50_red                       | 0.0125     | PPG Red
16   | mean_accel_x                    | 0.0098     | IMU Bioz
17   | kurtosis_accel_z                | 0.0089     | IMU Wrist
18   | signal_quality_green            | 0.0067     | PPG Green
19   | rr_mean                         | 0.0056     | RR
20   | ppg_dc_shift                    | 0.0045     | PPG Green
...
100  | kurtosis_eda                    | 0.0001     | EDA
```

---

## 9.3 Feature Selection Characteristics

### By Modality

```
Features selected (top 100):
  IMU Bioz:      32 features
  IMU Wrist:     24 features
  PPG Green:     15 features
  PPG Infra:     12 features
  PPG Red:       10 features
  EDA:            5 features
  RR:             2 features
```

### Why This Works

1. **IMU dominates** (56/100): Raw acceleration has high variance
2. **PPG secondary** (37/100): Heart rate metrics informative
3. **EDA/RR minimal** (7/100): Lower variance, but still useful
4. **Removes noise**: Low-variance features often are artifacts

---

## 9.4 Per-Condition Selection

**Important**: Feature selection is performed **per condition**

```python
for condition in CONDITIONS:  # elderly3, healthy3, severe3
    # Each condition gets its own top 100 features
    condition_df = combined_df[combined_df['subject'] == condition]
    
    # Calculate variance for THIS condition only
    variance = condition_df.var()
    top_100 = variance.nlargest(100).index.tolist()
    
    # Save condition-specific feature list
    with open(f"{condition}_features.json", 'w') as f:
        json.dump(top_100, f)
    
    print(f"{condition}: selected {len(top_100)} features")
```

### Why Per-Condition?

Different conditions have different feature importance:
- **elderly3**: Emphasizes activity-related features (IMU)
- **healthy3**: May emphasize subtle variations (heart rate)
- **severe3**: May emphasize extreme accelerations

---

## 9.5 Output Artifacts

### Files Saved

```
/multisub_combined/models/
├── sim_elderly3_features.json       ← Top 100 feature names
├── sim_elderly3_feature_selector.pkl ← Scikit-learn ColumnTransformer
├── sim_healthy3_features.json
├── sim_healthy3_feature_selector.pkl
├── sim_severe3_features.json
└── sim_severe3_feature_selector.pkl
```

### Content of _features.json

```json
[
  "std_accel_y",
  "rms_accel_z",
  "heart_rate_mean",
  "peak_count_x",
  "entropy_accel_z",
  ...
  "kurtosis_eda"
]
```

---

## 9.6 Feature Selection in Training

```python
def train_condition_model(condition, df):
    # Filter to condition
    cond_data = df[df["subject"] == condition].copy()
    cond_data = cond_data.dropna(subset=["borg"])
    
    # Get feature selector
    feature_selector = joblib.load(f"{condition}_feature_selector.pkl")
    
    # Apply feature selection
    X = cond_data.drop(columns=get_drop_columns(df))
    X_selected = feature_selector.transform(X)  # Keep only top 100
    
    y = cond_data["borg"]
    
    # Now train on 100 features (not 257)
    model = xgb.XGBRegressor(...)
    model.fit(X_selected, y)
```

---

## Summary

- **Purpose**: Reduce from 257 → 100 features per condition
- **Method**: Variance-based ranking
- **Benefit**: Faster training, reduced overfitting, simpler models
- **Per-condition**: Each condition gets its own top-100 list
- **Output**: Feature lists saved as JSON for inference
- **Next**: Model training with selected features
