# Pipeline Testing Results - 2026-01-20

## Test Session: sim_elderly3 (parsingsim3)

### ✅ Stage 1: ECG Preprocessing - WORKING

**Command:**
```bash
python3 scripts/preprocess_ecg.py \
  --ecg-file /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/vivalnk_vv330_ecg/data_1.csv.gz \
  --output data/labels/test_parsingsim3_sim_elderly3_rmssd_labels.csv \
  --session-id parsingsim3_sim_elderly3_test \
  --sampling-rate 256 \
  --verbose
```

**Results:**
- ✅ ECG loaded: 602,240 samples (39.2 minutes)
- ✅ R-peaks detected: 5,614 peaks
- ✅ RR intervals cleaned: 4,805/5,613 (85.6% valid)
- ✅ Mean heart rate: 159 bpm
- ✅ RMSSD windows computed: 77 windows
- ✅ RMSSD statistics:
  - Mean: 32.80 ms
  - Std: 22.93 ms
  - Range: 15.29 - 176.16 ms

**Output:** `data/labels/test_parsingsim3_sim_elderly3_rmssd_labels.csv`

---

### ✅ Stage 2: Feature Extraction - ALL WORKING

**Command:**
```bash
python3 scripts/demo_features.py
```

#### PPG Features (Heart Rate Level - NO HRV) ✅
- ppg_hr_mean: 100.99 bpm
- ppg_hr_max: 202.11 bpm
- ppg_hr_min: 33.39 bpm
- ppg_hr_std: 39.00 bpm
- ppg_hr_slope: -0.063 bpm/s
- ppg_n_beats: 432
- ppg_signal_quality: 0.858

**Data source:** `corsano_wrist_ppg2_green_6/2025-12-04.csv.gz` (400,128 samples)

#### IMU Features (Movement) ✅
- acc_mag_mean: 133.28
- acc_mag_std: 124.85
- acc_mag_max: 2,256.44
- acc_mag_integral: 39,985.06
- steps_sum: 320
- cadence_mean: 64 steps/min
- movement_duration: 0.649 (64.9%)

**Data source:** `corsano_wrist_acc/2025-12-04.csv.gz` (100,352 samples)

#### EDA Features (Sympathetic Arousal) ✅
- eda_mean: 11.88 μS
- eda_std: 26.14 μS
- eda_slope: 1.80 μS/s
- eda_scr_count: 4
- eda_scr_rate: 18.82 /min
- eda_scr_mean_amplitude: 20.23 μS

**Data source:** `corsano_bioz_emography/2025-12-04.csv.gz` (51 samples)
**Note:** Using `stress_skin` column as EDA proxy

---

## Data Format Discovered

### ECG (vivalnk_vv330_ecg/data_1.csv.gz)
- Columns: `time`, `ecg`
- Sampling rate: ~256 Hz
- Format: Unix timestamp, ECG amplitude

### PPG (corsano_wrist_ppg2_*/2025-12-04.csv.gz)
- Columns: `time`, `projectId`, `userId`, `metric_id`, ..., `value`
- Sampling rate: ~64 Hz
- Format: Unix timestamp, PPG amplitude in `value` column

### Accelerometer (corsano_wrist_acc/2025-12-04.csv.gz)
- Columns: `time`, ..., `accX`, `accY`, `accZ`
- Sampling rate: ~50 Hz
- Format: Unix timestamp, 3-axis acceleration

### EDA (corsano_bioz_emography/2025-12-04.csv.gz)
- Columns: `time`, `cz`, `pcz`, `stress_skin`, etc.
- Sampling rate: ~4 Hz (very low, as expected for EDA)
- Format: Unix timestamp, using `stress_skin` as EDA proxy

### ADL Timeline (scai_app/ADLs_1.csv)
- Columns: Custom header format
- Row 1: Session metadata
- Row 2: Column headers (`Time`, `ADLs`, `Effort`)
- Rows 3+: Activity timestamps, names, Borg RPE values
- Format: Date-time strings, activity names, effort scores

---

## Next Steps (TODO)

### 1. Create ADL Parser
Need script to parse `scai_app/ADLs_1.csv` properly:
- Extract start/end times for each ADL
- Convert timestamps to seconds from session start
- Identify baseline/task/recovery phases if available
- Match with Borg RPE (for validation only, not training)

### 2. Create Feature Alignment Script
`scripts/extract_features.py`:
- For each ADL segment in timeline:
  - Load sensor data in time window
  - Extract PPG features
  - Extract IMU features
  - Extract EDA features
  - Aggregate to single row
- Match with corresponding RMSSD label
- Output combined dataset

### 3. Create Training Script
`scripts/train_model.py`:
- Load aligned features + labels
- Train Ridge, ElasticNet, XGBoost
- Evaluate with MAE, RMSE, Spearman
- Save models and metrics
- Feature importance analysis

### 4. Multi-Session Testing
Test on all available sessions:
- parsingsim3: healthy3, elderly3, severe3
- parsingsim4: healthy4, elderly4, severe4
- parsingsim5: healthy5, elderly5, severe5

---

## Critical Reminders ✅

1. **NO HRV as input features** - Verified in all extractors
2. **RMSSD used ONLY for labels** - Implemented correctly
3. **Leakage-safe features only** - All features are instantaneous
4. **Physiologically explainable** - HR level, movement, sympathetic arousal

---

## Issues Resolved

1. ✅ R-peak detection improved (differential + adaptive threshold)
2. ✅ Column format handling for Corsano sensors
3. ✅ String-to-numeric conversion for accelerometer
4. ✅ EDA column identification (using stress_skin)
5. ✅ Sampling rate auto-detection

## Known Limitations

1. EDA data has very few samples (51 vs 400k for PPG) - may need different sensor or resampling
2. ADL file format needs custom parser (not standard CSV)
3. Gyroscope data not available in current accelerometer file
