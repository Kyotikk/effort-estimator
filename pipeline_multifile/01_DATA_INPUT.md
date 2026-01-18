# Stage 1: Data Input & Organization

## 1.1 Input Data Sources

All data stored in: `/Users/pascalschlegel/data/interim/parsingsim3/{subject}/`

### Raw Sensor Data

**IMU Bioz (Chest)**: `corsano_bioz_acc/`
- 3-axis accelerometer (X, Y, Z)
- Sampling rate: 32 Hz
- File: `corsano_bioz_acc*.csv` or `.csv.gz`
- Columns: `time`, `x`, `y`, `z`

**IMU Wrist**: `corsano_wrist_acc/`
- 3-axis accelerometer (X, Y, Z)
- Sampling rate: 32 Hz
- File: `corsano_wrist_acc*.csv` or `.csv.gz`
- Columns: `time`, `x`, `y`, `z`

**PPG Green**: `corsano_wrist_ppg2_green_*/`
- Green LED photoplethysmogram
- Sampling rate: 32 Hz
- File: `.csv` or `.csv.gz`
- Columns: `time`, `value`

**PPG Infrared**: `corsano_wrist_ppg2_infra_*/`
- Infrared LED photoplethysmogram
- Sampling rate: 32 Hz
- File: `.csv` or `.csv.gz`
- Columns: `time`, `value`

**PPG Red**: `corsano_wrist_ppg2_red_*/`
- Red LED photoplethysmogram
- Sampling rate: 32 Hz
- File: `.csv` or `.csv.gz`
- Columns: `time`, `value`

**EDA**: `corsano_bioz_emography/`
- Electrodermal activity (skin conductance)
- Sampling rate: 32 Hz
- File: `corsano_bioz_emography*.csv` or `.csv.gz`
- Columns: `time`, `value`

**RR (Heart Rate)**: `corsano_bioz_rr_interval/`
- RR interval (respiration/heart rate variability)
- Variable sampling rate (typically 0.5-2 Hz)
- File: `corsano_bioz_rr_interval*.csv` or `.csv.gz`
- Columns: `time`, `rr` (interval in ms)

### Effort Labels (ADL)

**Location**: `scai_app/ADLs_*.csv`
- Activity of Daily Living records from SCAI mobile app
- User-labeled with Borg effort scale (0-10)

**Format 1 (Primary)**:
```csv
Time,Activities,Borg RPE,Additional
2024-10-15 09:15:30.000,Walking,3,
2024-10-15 09:25:45.000,Sitting,1,
2024-10-15 09:45:20.000,Climbing stairs,6,
```

**Format 2 (Alternative)**:
```csv
Start,Stop,Activities,Borg
1700000100,1700000160,Walking,3
1700000200,1700000270,Sitting,1
1700000500,1700000620,Climbing stairs,6
```

---

## 1.2 Subject Conditions

### sim_elderly3 (Elderly Population)

- **Total Samples**: 450 window pairs
- **Labeled Samples**: 429 (95.3%)
- **Borg Range**: 0.5 - 6.0
- **Mean Borg**: 3.30 ± 1.88
- **Distribution**: Well-distributed across range
- **Use Case**: Aging adults, daily activities

### sim_healthy3 (Healthy/Low Effort)

- **Total Samples**: 380 window pairs
- **Labeled Samples**: 347 (91.3%)
- **Borg Range**: 0.0 - 1.5
- **Mean Borg**: 0.28 ± 0.32
- **Distribution**: 93.7% at 0-1 Borg (very light activities)
- **Use Case**: Young, healthy, light-intensity activities

### sim_severe3 (High Intensity/Severe)

- **Total Samples**: 450 window pairs
- **Labeled Samples**: 412 (91.6%)
- **Borg Range**: 1.5 - 8.0
- **Mean Borg**: 4.71 ± 2.06
- **Distribution**: 50% at extreme (5-8 Borg)
- **Use Case**: High-intensity effort, severe conditions

---

## 1.3 Combined Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Subjects** | 3 |
| **Total Windows** | 1,280 |
| **Labeled Windows** | 1,188 (92.8%) |
| **Unlabeled Windows** | 92 (7.2%) |
| **Borg Range (Overall)** | 0.0 - 8.0 |
| **Mean Borg (Overall)** | 2.76 |
| **Feature Columns** | 257 |

---

## 1.4 Directory Structure

```
/Users/pascalschlegel/data/interim/parsingsim3/
├── sim_elderly3/
│   ├── corsano_bioz_acc/
│   │   └── *.csv or *.csv.gz
│   ├── corsano_wrist_acc/
│   ├── corsano_wrist_ppg2_green_*/
│   ├── corsano_wrist_ppg2_infra_*/
│   ├── corsano_wrist_ppg2_red_*/
│   ├── corsano_bioz_emography/
│   ├── corsano_bioz_rr_interval/
│   ├── scai_app/
│   │   └── ADLs_1-2.csv (or ADLs_*.csv.gz)
│   └── effort_estimation_output/
│       ├── imu_bioz_preprocessed.csv
│       ├── imu_wrist_preprocessed.csv
│       ├── ppg_green_preprocessed.csv
│       ├── ...
│       ├── fused_10.0s.csv
│       ├── aligned_10.0s.csv
│       └── ...
├── sim_healthy3/
│   └── (same structure as elderly3)
├── sim_severe3/
│   └── (same structure as elderly3)
└── multisub_combined/
    ├── multisub_aligned_10.0s.csv (combined labeled dataset)
    └── models/
        ├── sim_elderly3_model.json
        ├── sim_elderly3_scaler.pkl
        ├── sim_elderly3_features.json
        ├── sim_healthy3_model.json
        ├── sim_healthy3_scaler.pkl
        ├── sim_healthy3_features.json
        ├── sim_severe3_model.json
        ├── sim_severe3_scaler.pkl
        └── sim_severe3_features.json
```

---

## 1.5 File Discovery Strategy

**Script**: `run_multisub_pipeline.py` - `find_file()` function

**Key Challenge**: Multiple sensor files with similar names

**Solution**:
- Pattern matching on glob patterns
- Prioritize uncompressed `.csv` over `.csv.gz`
- Select most recently modified file on multiple matches
- Explicit `exclude_gz=True` for ADL files (to get latest labeled data)

**Example**:
```python
def find_file(subject_path, pattern_parts, exclude_gz=False):
    """
    Find a file matching pattern parts.
    
    For ADL files: exclude_gz=True to get latest uncompressed CSV
    For sensor files: allows both .csv and .csv.gz
    """
```

---

## 1.6 Data Loading Workflow

```python
# In run_multisub_pipeline.py

for subject in SUBJECTS:  # ['sim_elderly3', 'sim_healthy3', 'sim_severe3']
    subject_path = Path(DATA_ROOT) / subject
    
    # Find all required files
    imu_bioz_path = find_file(subject_path, ["corsano_bioz_acc"])
    imu_wrist_path = find_file(subject_path, ["corsano_wrist_acc"])
    ppg_green_path = find_file(subject_path, ["corsano_wrist_ppg2_green"])
    # ... etc for all 7 modalities
    
    adl_path = find_file(subject_path, ["scai_app", "ADLs"], exclude_gz=True)
    
    # Generate config for this subject
    config = generate_config(subject)
    
    # Run single-subject pipeline
    run_subject_pipeline(subject, config)
```

---

## Key Points

✅ **All data files exist** in `/data/interim/parsingsim3/`
✅ **3 conditions** with different populations and effort ranges
✅ **File discovery** handles multiple formats (compressed/uncompressed)
✅ **Labels ready** in ADL CSV files with Borg scale
⚠️ **Time synchronization** handled in Stage 5 (ADL Alignment)
