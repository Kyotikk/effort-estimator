# Stage 5: Feature Fusion

## Purpose

Merge features from all 7 modalities into a single master feature matrix, time-aligned at window centers.

---

## 5.1 Fusion Process

**File**: `ml/run_fusion.py`
**Function**: `main(modality_list, window_length)`

### Input

- 7 modality feature files (from Stage 4)
  - imu_bioz_features_10.0s.csv (69 features)
  - imu_wrist_features_10.0s.csv (69 features)
  - ppg_green_features_10.0s.csv (17 features)
  - ppg_infra_features_10.0s.csv (17 features)
  - ppg_red_features_10.0s.csv (17 features)
  - eda_features_10.0s.csv (7 features)
  - rr_features_10.0s.csv (5 features)

### Algorithm

**Step 1: Load All Modality Files**

```python
modalities = {}
for modality_name in modality_list:
    df = pd.read_csv(f"{modality_name}_features_{window_length}s.csv")
    modalities[modality_name] = df
    print(f"Loaded {modality_name}: {df.shape}")
```

**Step 2: Ensure Alignment Columns**

```python
# Each modality must have window_id and t_center
for name, df in modalities.items():
    if 'window_id' not in df.columns:
        df['window_id'] = [f"w_{i:05d}" for i in range(len(df))]
    
    if 't_center' not in df.columns:
        # Compute from t_start and t_end if available
        df['t_center'] = (df['t_start'] + df['t_end']) / 2
    
    print(f"{name} has columns: {df.columns.tolist()}")
```

**Step 3: Merge on (window_id, t_center)**

```python
# Start with first modality
fused = modalities[list(modalities.keys())[0]][['window_id', 't_center']].copy()

# Merge all others
for name in list(modalities.keys())[1:]:
    df_mod = modalities[name]
    
    # Select non-alignment columns
    feature_cols = [c for c in df_mod.columns 
                    if c not in ['window_id', 't_center']]
    
    # Merge
    fused = fused.merge(
        df_mod[['window_id', 't_center'] + feature_cols],
        on=['window_id', 't_center'],
        how='outer'
    )
    
    print(f"After {name}: {fused.shape}")
```

**Step 4: Forward-Fill Missing Values**

```python
# Handle missing values from different sampling rates
# Forward-fill up to 1 window gap
fused = fused.fillna(method='ffill', limit=1)

# Mark rows with remaining NaN
n_nan_before = fused.isna().sum().sum()
print(f"NaN values before removal: {n_nan_before}")

# Remove rows with ANY NaN
fused = fused.dropna(axis=0, how='any')
n_after = len(fused)
print(f"Rows after NaN removal: {n_after}")
```

**Step 5: Add Metadata Columns**

```python
# Add modality tracking
fused['subject'] = subject_name
fused['modality'] = 'fused'
fused['valid'] = 1  # Will be updated by quality checks
```

**Step 6: Save Fused Matrix**

```python
output_file = f"{output_dir}/fused_{window_length}s.csv"
fused.to_csv(output_file, index=False)

print(f"✓ Fused matrix saved: {output_file}")
print(f"  Shape: {fused.shape}")
print(f"  Columns: {len(fused.columns)}")
```

---

## 5.2 Temporal Alignment

### Alignment Strategy

All modalities are aligned to **window center times** (`t_center`):

**Scenario 1: Exact Match**
```
Window 42:
  IMU captured:        t_center = 1700000005.2
  PPG captured:        t_center = 1700000005.2
  Both same → FUSED ✓
```

**Scenario 2: Slight Offset**
```
Window 42:
  IMU sampled:         t_center = 1700000005.2
  PPG sampled:         t_center = 1700000005.3 (±0.1s)
  Offset < 0.5s → Matched to same window ✓
```

**Scenario 3: No Match**
```
Window 42:
  IMU sampled:         t_center = 1700000005.2
  PPG sampled:         t_center = 1700000010.5 (large gap)
  No match within tolerance → Row removed (NaN)
```

### Example Fusion

**Input Files**:

```
imu_bioz_features_10.0s.csv:
  window_id | t_center    | mean_x | std_x | ... (69 IMU features)
  w_0042    | 1700000005  | 0.012  | 0.023 | ...
  w_0043    | 1700000008  | 0.015  | 0.025 | ...

ppg_green_features_10.0s.csv:
  window_id | t_center    | heart_rate | hrv | ... (17 PPG features)
  w_0042    | 1700000005  | 72         | 45  | ...
  w_0043    | 1700000008  | 71         | 48  | ...

eda_features_10.0s.csv:
  window_id | t_center    | scl_level | ... (7 EDA features)
  w_0042    | 1700000005  | 2.5       | ...
  w_0043    | 1700000008  | 2.8       | ...
```

**Fusion Process**:

```
Step 1: Start with IMU
  window_id | t_center | [69 IMU features]

Step 2: Merge PPG Green
  window_id | t_center | [69 IMU] | [17 PPG Green]

Step 3: Merge PPG Infra
  window_id | t_center | [69 IMU] | [17 PPG Green] | [17 PPG Infra]

Step 4: Merge PPG Red
  window_id | t_center | [69 IMU] | [17 PPG Green] | [17 PPG Infra] | [17 PPG Red]

Step 5: Merge EDA
  window_id | t_center | [69 IMU] | [51 PPG] | [7 EDA]

Step 6: Merge RR
  window_id | t_center | [69 IMU] | [51 PPG] | [7 EDA] | [5 RR]
```

**Output**:

```
fused_10.0s.csv:
  window_id | t_center | [257 total features]
  w_0042    | 1700000005 | 0.012 | 0.023 | ... | 72 | 45 | ... | 2.5 | ... (257 columns)
  w_0043    | 1700000008 | 0.015 | 0.025 | ... | 71 | 48 | ... | 2.8 | ... (257 columns)
```

---

## 5.3 Handling Missing Modalities

**Forward-Fill Strategy**:

If a modality is missing for a window, use previous value:

```python
fused['ppg_green_heart_rate'].fillna(method='ffill', limit=1)
```

**Removal Strategy**:

If forward-fill can't fill (start of data or gap > 1 window), remove row:

```python
# Remove rows where ANY feature is NaN
fused = fused.dropna(axis=0, how='any')
```

---

## 5.4 Output Structure

**File**: `{subject}/effort_estimation_output/fused_{window_length}s.csv`

### Columns

```
Core Columns:
  window_id        | w_00000, w_00001, ...
  t_start          | Start timestamp
  t_center         | Center timestamp (alignment key)
  t_end            | End timestamp
  subject          | Subject name
  modality         | 'fused' (indicates all modalities combined)
  valid            | 1/0 (validity from source)

Feature Columns (257 total):
  [69 IMU bioz features]
  [69 IMU wrist features]
  [17 PPG green features]
  [17 PPG infra features]
  [17 PPG red features]
  [7 EDA features]
  [5 RR features]
```

### Example Row

```
window_id: w_00042
t_start:   1700000000
t_center:  1700000005
t_end:     1700000010
subject:   sim_elderly3
modality:  fused
valid:     1
mean_imu_bioz_x: 0.0125
std_imu_bioz_x:  0.0234
...
heart_rate_green: 72
hrv_green: 45
...
scl_level: 2.5
...
rr_mean:   750
```

---

## Summary

- **Input**: 7 separate modality feature files (257 features total)
- **Process**: Merge on window_id and t_center
- **Alignment**: All modalities aligned to window center times
- **Missing Data**: Forward-fill up to 1 gap, then remove rows with NaN
- **Output**: Single fused feature matrix with 257 columns
- **Next**: Quality checks and ADL label attachment
