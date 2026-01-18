# Stage 4: Alignment, Labeling & Fusion

## Overview

This stage:
1. **Aligns** feature windows with Borg effort labels from ADL annotations
2. **Labels** each window with corresponding effort rating
3. **Fuses** all modality features into a single unified table
4. Produces final training-ready dataset

---

## Stage 4A: Alignment & Labeling

### Input: Feature Tables + ADL Labels

**Feature tables:** One per modality/window-length
- `imu_features_10.0s.csv`
- `ppg_green_features_10.0s.csv`
- `eda_features_10.0s.csv`
- etc.

**ADL (Activity of Daily Living) file:** Contains effort labels
- Path: `/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/scai_app/ADLs_1.csv`
- Format:

```
start_timestamp,end_timestamp,borg_rating,activity_type
2025-12-04 10:00:00,2025-12-04 10:02:30,2,rest
2025-12-04 10:02:30,2025-12-04 10:04:45,4,light_walk
2025-12-04 10:04:45,2025-12-04 10:08:15,8,moderate_exercise
2025-12-04 10:08:15,2025-12-04 10:10:00,12,heavy_exercise
...
```

### Alignment Process

**Algorithm (`ml/targets/run_target_alignment.py`):**

```python
def align_features_to_labels(features_csv, adl_csv, window_length):
    """
    Align each feature window with corresponding Borg label
    """
    # Load feature windows
    features_df = pd.read_csv(features_csv)  # [N rows, features + window metadata]
    adl_df = pd.read_csv(adl_csv)            # [M rows, time ranges + labels]
    
    # Convert ADL timestamps to seconds relative to session start
    adl_df['t_start_sec'] = to_seconds(adl_df['start_timestamp'])
    adl_df['t_end_sec'] = to_seconds(adl_df['end_timestamp'])
    
    # For each feature window, find matching ADL label
    for idx, row in features_df.iterrows():
        t_start = row['t_start']
        t_end = row['t_end']
        
        # Window center time (or use start time)
        t_center = (t_start + t_end) / 2
        
        # Find ADL segment that contains this window center
        matching_adl = adl_df[
            (adl_df['t_start_sec'] <= t_center) & 
            (t_center < adl_df['t_end_sec'])
        ]
        
        if len(matching_adl) == 1:
            features_df.loc[idx, 'borg'] = matching_adl.iloc[0]['borg_rating']
            features_df.loc[idx, 'activity'] = matching_adl.iloc[0]['activity_type']
        elif len(matching_adl) == 0:
            features_df.loc[idx, 'borg'] = NaN
        else:
            # Multiple matches - shouldn't happen with proper ADL definition
            features_df.loc[idx, 'borg'] = matching_adl.iloc[0]['borg_rating']  # Take first
    
    # Drop rows without labels
    features_aligned = features_df.dropna(subset=['borg'])
    
    return features_aligned
```

### Alignment Output

**imu_aligned_10.0s.csv** (example):
```
window_id,t_start,t_end,imu_acc_x_mean,imu_acc_x_std,...,borg,activity
0,0.0,10.0,0.042,0.089,...,2,rest
1,3.0,13.0,-0.031,0.095,...,4,light_walk
2,6.0,16.0,0.115,0.142,...,4,light_walk
...
429,415.2,425.2,0.487,0.201,...,12,heavy_exercise
```

**Each row now has:**
- ✓ Window metadata (id, times)
- ✓ Modality features (30-44 features)
- ✓ Borg label (0-20 scale)
- ✓ Activity type (metadata)

### Labeling Statistics (10s windows)

From the alignment step on current data:

```
Total feature windows generated:    ~2000
Windows successfully labeled:        429 (21%)
Windows with missing labels:         ~1571 (79%)

Borg Rating Distribution (429 labeled):
  Borg 0 (rest):           45 windows (10.5%)
  Borg 2 (very light):    102 windows (23.8%)
  Borg 4 (light):          89 windows (20.8%)
  Borg 6 (slightly hard):  78 windows (18.2%)
  Borg 8 (hard):           65 windows (15.2%)
  Borg 10 (very hard):     31 windows (7.2%)
  Borg 12+ (extreme):      19 windows (4.4%)
```

**Note:** Many windows dropped because they don't align with ADL segments (e.g., overlap activity boundaries or occur between marked activities).

---

## Stage 4B: Fusion

### Purpose

Combine all modality features into single table for training:

```
Features from multiple tables:
  - imu_aligned_10.0s.csv          [30 features]
  - ppg_green_aligned_10.0s.csv    [44 features]
  - ppg_infra_aligned_10.0s.csv    [44 features]
  - ppg_red_aligned_10.0s.csv      [44 features]
  - eda_aligned_10.0s.csv          [26 features]
           ↓
        FUSION
           ↓
  - fused_aligned_10.0s.csv        [188 features + borg label]
```

### Fusion Algorithm

**Algorithm (`ml/run_fusion.py`):**

```python
def fuse_modalities(aligned_tables, window_length):
    """
    Fuse features from multiple modalities using time-based merge
    """
    # Start with IMU as reference
    fused = pd.read_csv(f'imu_aligned_{window_length}.csv')
    
    # List of other modalities to merge
    modalities = ['ppg_green', 'ppg_infra', 'ppg_red', 'eda']
    
    for modality in modalities:
        other = pd.read_csv(f'{modality}_aligned_{window_length}.csv')
        
        # Merge based on time windows (t_start, t_end must match)
        # Use merge_asof with tolerance to account for slight time differences
        fused = pd.merge_asof(
            fused.sort_values('t_start'),
            other.sort_values('t_start'),
            on='t_start',
            direction='nearest',
            tolerance=2.0  # 2 second tolerance
        )
    
    return fused
```

### Fusion Configuration

**Tolerance settings (pipeline.yaml):**
```yaml
fusion:
  tolerance_s:
    "2.0": 2.0
    "5.0": 2.0
    "10.0": 2.0
```

**Rationale:** 
- 2.0s tolerance allows for minor timing skew between sensors
- Larger than window stride (0.3-1.5s) to ensure matches
- Smaller than window length to avoid cross-window merging

### Fusion Output

**fused_aligned_10.0s.csv:**
```
[429 rows × 195 columns]

Columns:
- window_id_r               (reference IMU window ID)
- t_start_r, t_end_r       (reference window times)
- borg                      (Borg effort label - TARGET)
- activity                  (activity type)

Feature columns (188 total):
- imu_acc_x_mean, imu_acc_x_std, ..., imu_entropy
- ppg_green_hr_mean, ppg_green_std, ..., ppg_green_entropy
- ppg_infra_hr_mean, ppg_infra_std, ..., ppg_infra_entropy
- ppg_red_hr_mean, ppg_red_std, ..., ppg_red_entropy
- eda_cc_mean, eda_cc_std, ..., eda_stress_skin_entropy
```

**Example row:**
```
window_id_r=142, t_start_r=14.2, t_end_r=24.2, borg=8,
imu_acc_x_mean=0.15, imu_acc_x_std=0.23, ...
ppg_green_hr_mean=92, ppg_green_hrv_rmssd=45, ...
eda_cc_mean=2.8, eda_cc_range=0.95, ...
```

### Fusion Statistics

**Before fusion:**
```
IMU features:        429 windows × 30 features
PPG Green features:  429 windows × 44 features  
PPG Infra features:  429 windows × 44 features
PPG Red features:    429 windows × 44 features
EDA features:        429 windows × 26 features
```

**After fusion:**
```
Fused features:      429 windows × 195 columns
                     = 429 rows × (31 + 45 + 45 + 45 + 27 + borg)
                     = 429 samples × 188 feature columns + 1 target
```

---

## Detailed Flow Diagram

```
RAW DATA STREAM
    ↓
PREPROCESSING
    ↓ [6 preprocessed files]
WINDOWING
    ↓ [6 window definition files]
FEATURE EXTRACTION
    ↓ [6 feature files: imu, ppg_green, ppg_infra, ppg_red, eda, rr]
    ↓
ALIGNMENT
    ├─ imu_features_10.0s.csv + ADL labels → imu_aligned_10.0s.csv
    ├─ ppg_green_features_10.0s.csv + ADL labels → ppg_green_aligned_10.0s.csv
    ├─ ppg_infra_features_10.0s.csv + ADL labels → ppg_infra_aligned_10.0s.csv
    ├─ ppg_red_features_10.0s.csv + ADL labels → ppg_red_aligned_10.0s.csv
    └─ eda_features_10.0s.csv + ADL labels → eda_aligned_10.0s.csv
    ↓ [5 aligned files, each with borg label]
FUSION
    ↓ [merge on (t_start, borg)]
fused_aligned_10.0s.csv
    ↓
READY FOR TRAINING
```

---

## Data Quality During Alignment

**Checks performed:**

1. ✓ Window times are reasonable (t_end > t_start)
2. ✓ Borg labels valid (0-20 range, int type)
3. ✓ All modalities have data for same time windows
4. ✓ No duplicate windows
5. ✓ Time progression monotonic

**Example issues detected:**
- ⚠ PPG window missing (tolerance exceeded): Dropped
- ⚠ ADL label extends beyond data: Window discarded
- ⚠ Multiple ADL segments in one window: Take earliest

---

## Alignment & Fusion Configuration

From `config/pipeline.yaml`:

```yaml
targets:
  imu:
    adl_path: /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/scai_app/ADLs_1.csv

fusion:
  output_dir: /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3
  window_lengths_sec: [10.0, 5.0, 2.0]
  
  tolerance_s:
    "2.0": 2.0
    "5.0": 2.0
    "10.0": 2.0
  
  modalities:
    imu: .../imu_bioz/imu_features_{window_length}.csv
    ppg_green: .../ppg_green/ppg_green_features_{window_length}.csv
    ppg_infra: .../ppg_infra/ppg_infra_features_{window_length}.csv
    ppg_red: .../ppg_red/ppg_red_features_{window_length}.csv
    eda: .../eda/eda_features_{window_length}.csv
```

---

## Edge Cases & Handling

| Case | Handling |
|------|----------|
| **Window at activity boundary** | Use center time; include if center in activity |
| **Window spans 2 activities** | Drop (ambiguous label) |
| **No ADL label for window** | Drop from training set |
| **Modality data missing** | Drop (incomplete feature vector) |
| **NaN values in features** | Fill with column median after fusion |

---

## Summary Statistics

**Dataset after alignment & fusion (10s windows):**

| Metric | Value |
|--------|-------|
| Total windows created | ~2,000 |
| Windows labeled (with ADL match) | 429 |
| Windows used in training | 429 |
| Features per sample | 188 |
| Target range (Borg) | 0-20 |
| Target mean | 6.4 ± 3.2 |
| Train/test split | 80/20 |
| Training samples | 343 |
| Test samples | 86 |

---

## Next Step: Feature Selection & Training

The fused aligned table is now ready for:
1. **Feature selection** - Reduce 188 → 100 features
2. **Training** - Train XGBoost model

See [05_FEATURE_SELECTION.md](05_FEATURE_SELECTION.md) and [06_TRAINING.md](06_TRAINING.md).

