# Stage 4: Alignment, Labeling & Fusion

## Overview

This stage:
1. **Aligns** feature windows with Borg effort labels from ADL annotations
2. **Labels** each window with corresponding effort rating
3. **Fuses** all modality features into a single unified table
4. Produces final training-ready dataset

---

## Current Configuration (3 Elderly Patients)

| Setting | Value |
|---------|-------|
| **Subjects** | sim_elderly3, sim_elderly4, sim_elderly5 |
| **Best Window Size** | 5.0 seconds |
| **Comparison Window** | 10.0 seconds |
| **Overlap** | 10% |
| **Fusion Tolerance** | 2s (5s windows), 5s (10s windows) |

---

## Stage 4A: Alignment & Labeling

### Input: Feature Tables + ADL Labels

**Feature tables:** One per modality/window-length
- `imu_features_5.0s.csv`
- `ppg_green_features_5.0s.csv`
- `eda_features_5.0s.csv`
- etc.

**ADL (Activity of Daily Living) file:** Contains effort labels
- Path pattern: `/Users/pascalschlegel/data/interim/{subject_path}/scai_app/ADLs_1.csv`
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
def align_features_to_labels(features_csv, adl_csv, window_length, tolerance):
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
        
        # Window center time
        t_center = (t_start + t_end) / 2
        
        # Find ADL segment within tolerance of window center
        matching_adl = adl_df[
            (adl_df['t_start_sec'] <= t_center + tolerance) & 
            (t_center - tolerance < adl_df['t_end_sec'])
        ]
        
        if len(matching_adl) >= 1:
            features_df.loc[idx, 'borg'] = matching_adl.iloc[0]['borg_rating']
            features_df.loc[idx, 'activity'] = matching_adl.iloc[0]['activity_type']
        else:
            features_df.loc[idx, 'borg'] = NaN
    
    # Drop rows without labels
    features_aligned = features_df.dropna(subset=['borg'])
    
    return features_aligned
```

### Tolerance Settings by Window Size

**Critical insight:** Different window sizes require different alignment tolerances.

| Window | Tolerance | Rationale |
|--------|-----------|-----------|
| **5s** | 2.0s | Window is short, tight alignment needed |
| **10s** | 5.0s | Longer window needs looser tolerance |
| **30s** | 15.0s | Very long window, widest tolerance |

**Implemented in `run_elderly_10s_30s.py`:**
```python
TOLERANCE_MAP = {
    5.0: 2.0,
    10.0: 5.0,
    30.0: 15.0,
}
```

### Labeling Statistics by Window Size

**5s Windows (Best):**
```
Total feature windows generated:    ~4000
Windows successfully labeled:        855 (21%)
Windows with missing labels:         ~3145 (79%)
Unique activities (groups):          65

Borg Rating Distribution (855 labeled):
  Borg 0-2 (rest/light):   ~170 windows (20%)
  Borg 3-5 (light):        ~220 windows (26%)
  Borg 6-8 (moderate):     ~250 windows (29%)
  Borg 9-12 (hard):        ~180 windows (21%)
  Borg 13+ (very hard):    ~35 windows (4%)
```

**10s Windows (Comparison):**
```
Total feature windows generated:    ~2000
Windows successfully labeled:        424 (21%)
Windows with missing labels:         ~1576 (79%)

Borg rating distribution similar to 5s
```

**30s Windows (Poor):**
```
Total feature windows generated:    ~700
Windows successfully labeled:        100 (14%)
Windows with missing labels:         ~600 (86%)

⚠️ sim_elderly5 lost ALL labels (0 windows aligned)
```

---

## Stage 4B: Fusion

### Purpose

Combine all modality features into single table for training:

```
Features from multiple tables (5s example):
  - imu_aligned_5.0s.csv           [~30 features]
  - ppg_green_aligned_5.0s.csv     [~44 features]
  - ppg_infra_aligned_5.0s.csv     [~44 features]
  - ppg_red_aligned_5.0s.csv       [~44 features]
  - eda_aligned_5.0s.csv           [~26 features]
           ↓
        FUSION
           ↓
  - fused_aligned_5.0s.csv         [270+ features + borg label]
```

### Fusion Algorithm

**Algorithm (`ml/fusion/fuse_windows.py`):**

```python
def fuse_modalities(aligned_tables, window_length, tolerance):
    """
    Fuse features from multiple modalities using time-based merge
    """
    # Start with IMU as reference
    fused = pd.read_csv(f'imu_aligned_{window_length}.csv')
    
    # List of other modalities to merge
    modalities = ['ppg_green', 'ppg_infra', 'ppg_red', 'eda']
    
    for modality in modalities:
        other = pd.read_csv(f'{modality}_aligned_{window_length}.csv')
        
        # Merge based on time windows (t_center must match within tolerance)
        fused = pd.merge_asof(
            fused.sort_values('t_center'),
            other.sort_values('t_center'),
            on='t_center',
            direction='nearest',
            tolerance=tolerance
        )
    
    return fused
```

### Fusion Output by Window Size

**5s Windows (elderly_aligned_5.0s.csv):**
```
[855 rows × 270+ columns]

Columns:
- subject                   (sim_elderly3, sim_elderly4, sim_elderly5)
- t_start, t_center, t_end (window times)
- borg                      (Borg effort label - TARGET)
- activity_id               (activity ID for grouping)
- 270+ feature columns

After feature selection: 48 features retained
```

**10s Windows (elderly_aligned_10.0s.csv):**
```
[424 rows × 270+ columns]

After feature selection: 51 features retained
```

### Multi-Subject Fusion

**Cross-subject combination:**
```python
# Per-subject data
subjects = ['sim_elderly3', 'sim_elderly4', 'sim_elderly5']
all_data = []

for subject in subjects:
    subject_df = load_aligned_data(subject, window_length)
    subject_df['subject'] = subject
    all_data.append(subject_df)

# Concatenate all subjects
combined = pd.concat(all_data, ignore_index=True)
combined.to_csv(f'elderly_aligned_{window_length}.csv')
```

---

## Detailed Flow Diagram

```
RAW DATA STREAM (3 subjects)
    ↓
PREPROCESSING
    ↓ [6 preprocessed files × 3 subjects]
WINDOWING
    ↓ [6 window definition files × 3 subjects]
FEATURE EXTRACTION
    ↓ [6 feature files per subject: imu, ppg_green, ppg_infra, ppg_red, eda, rr]
    ↓
ALIGNMENT (per subject)
    ├─ imu_features_5.0s.csv + ADL labels → imu_aligned_5.0s.csv
    ├─ ppg_green_features_5.0s.csv + ADL labels → ppg_green_aligned_5.0s.csv
    ├─ ppg_infra_features_5.0s.csv + ADL labels → ppg_infra_aligned_5.0s.csv
    ├─ ppg_red_features_5.0s.csv + ADL labels → ppg_red_aligned_5.0s.csv
    └─ eda_features_5.0s.csv + ADL labels → eda_aligned_5.0s.csv
    ↓ [5 aligned files per subject, each with borg label]
FUSION (per subject)
    ↓ [merge on t_center with tolerance]
subject_fused_aligned_5.0s.csv
    ↓
MULTI-SUBJECT CONCATENATION
    ↓
elderly_aligned_5.0s.csv (855 samples)
    ↓
FEATURE SELECTION (correlation + pruning)
    ↓
elderly_aligned_5.0s.csv (48 features)
    ↓
READY FOR TRAINING
```

---

## Data Quality During Alignment

**Checks performed:**

1. ✓ Window times are reasonable (t_end > t_start)
2. ✓ Borg labels valid (0-20 range, numeric type)
3. ✓ All modalities have data for same time windows
4. ✓ No duplicate windows
5. ✓ Time progression monotonic
6. ✓ Subject column populated correctly

**Example issues detected:**
- ⚠ PPG window missing (tolerance exceeded): Dropped
- ⚠ ADL label extends beyond data: Window discarded
- ⚠ Multiple ADL segments in one window: Take earliest
- ⚠ sim_elderly5 with 30s windows: 0 aligned (alignment issue)

---

## Output Files Location

**Combined multi-subject output:**
```
/Users/pascalschlegel/data/interim/elderly_combined/
├── elderly_aligned_5.0s.csv      # 855 samples, best
├── elderly_aligned_10.0s.csv     # 424 samples, comparison
├── elderly_aligned_30.0s.csv     # 100 samples, poor
│
├── qc_5.0s/                       # Feature selection QC
│   ├── features_selected_pruned.csv
│   └── pca_*.csv
│
└── xgboost_results/               # Model outputs
    ├── summary.yaml
    └── predictions.csv
```

---

## Edge Cases & Handling

| Case | Handling |
|------|----------|
| **Window at activity boundary** | Use center time; include if center within tolerance |
| **Window spans 2 activities** | Drop (ambiguous label) |
| **No ADL label for window** | Drop from training set |
| **Modality data missing** | Drop (incomplete feature vector) |
| **NaN values in features** | Fill with column median after fusion |
| **Subject has no aligned windows** | Warning logged, excluded from combined |

---

## Summary Statistics by Window Size

| Metric | 5s Windows | 10s Windows | 30s Windows |
|--------|------------|-------------|-------------|
| **Total windows labeled** | 855 | 424 | 100 |
| **Raw features** | 270+ | 270+ | 270+ |
| **Selected features** | 48 | 51 | 20 |
| **Unique activities (groups)** | 65 | ~35 | ~12 |
| **Subjects contributing** | 3 | 3 | 2* |
| **Borg range** | 0-17 | 0-17 | 0-15 |
| **Borg mean ± std** | 5.8 ± 3.5 | 5.9 ± 3.4 | 5.5 ± 3.2 |

*sim_elderly5 lost all labels with 30s windows due to alignment tolerance issues

---

## Why 5s Windows Are Best

| Factor | 5s | 10s | 30s |
|--------|----|----|-----|
| **N samples** | 855 | 424 | 100 |
| **Statistical power** | High | Medium | Low |
| **Temporal resolution** | Best | Good | Poor |
| **Label alignment** | Most precise | Good | Problematic |
| **CV stability** | Best | Good | Unstable |

---

## Next Step: Feature Selection & Training

The fused aligned table is now ready for:
1. **Feature selection** - Reduce 270+ → 48 features (correlation + pruning)
2. **Training** - Train XGBoost and Ridge models with GroupKFold CV

See [05_FEATURE_SELECTION.md](05_FEATURE_SELECTION.md) and [06_TRAINING.md](06_TRAINING.md).

