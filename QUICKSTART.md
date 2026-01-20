# Quick Start Guide - Effort Estimator Pipeline

## Architecture Overview

**Goal**: Estimate physiological effort during ADLs from wearable sensors

**Critical Design Rule**: NO HRV features as inputs
- RMSSD and all HRV metrics → LABELS ONLY (from ECG)
- Input features → HR level (PPG), movement (IMU), arousal (EDA)

## Pipeline Stages

### Stage 1: ECG → RMSSD Labels ✓ IMPLEMENTED

**Purpose**: Compute effort ground truth from ECG-derived RMSSD

```bash
python scripts/preprocess_ecg.py \
  --ecg-file /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/vivalnk_vv330_ecg/data_1.csv.gz \
  --output data/labels/parsingsim3_sim_elderly3_rmssd_labels.csv \
  --session-id parsingsim3_sim_elderly3 \
  --adl-file /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/scai_app/ADLs_1.csv \
  --verbose
```

**Output**: CSV with RMSSD per ADL segment
- Columns: session_id, adl_id, rmssd, ln_rmssd, n_beats, mean_rr, start_time, end_time

**What it does**:
1. Load ECG → detect R-peaks
2. Compute RR intervals → clean artifacts
3. Calculate RMSSD per ADL window
4. Save as effort labels

---

### Stage 2: Sensor → Features ✓ MODULES READY

**Purpose**: Extract leakage-safe features from sensors

**Modules implemented**:
- [features/ppg_features.py](features/ppg_features.py) - Heart rate level
- [features/imu_features.py](features/imu_features.py) - Movement/acceleration
- [features/eda_features.py](features/eda_features.py) - Sympathetic arousal

**Demo**:
```bash
python scripts/demo_features.py
```

**Feature list** (NO HRV):

**PPG** (heart rate level):
- ppg_hr_mean, ppg_hr_max, ppg_hr_min
- ppg_hr_std, ppg_hr_slope, ppg_hr_range
- ppg_signal_quality, ppg_n_beats

**IMU** (movement):
- acc_mag_mean, acc_mag_std, acc_mag_max
- acc_mag_integral, movement_duration
- steps_sum, cadence_mean
- acc_x/y/z_mean, acc_x/y/z_std
- gyro_mag_mean/std/max (if available)

**EDA** (sympathetic):
- eda_mean, eda_std, eda_min, eda_max
- eda_slope, eda_scl_mean
- eda_scr_count, eda_scr_rate
- eda_scr_mean_amplitude, eda_scr_max_amplitude

---

### Stage 3: Alignment → Dataset (TODO)

**Purpose**: Align features and labels to ADL segments

**To implement**:
```python
# For each ADL:
#   1. Load sensor data in time window
#   2. Extract features using modules
#   3. Match with RMSSD label
#   4. Aggregate to one row per ADL
```

**Output**: CSV with one row per (session, ADL)
- Columns: [all features], effort_label

---

### Stage 4: Train Models (TODO)

**Purpose**: Regression to predict effort from features

**Models**:
- Ridge (baseline)
- ElasticNet
- XGBoost

**Metrics**:
- MAE, RMSE
- Spearman correlation (predicted vs true effort)

---

## Data Paths

**Data roots**:
```
/Users/pascalschlegel/data/interim/parsingsim3/
/Users/pascalschlegel/data/interim/parsingsim4/
/Users/pascalschlegel/data/interim/parsingsim5/
```

**Each sim has 3 cohorts**: healthy, elderly, severe

**Example (sim_elderly3)**:
```
parsingsim3/sim_elderly3/
├── vivalnk_vv330_ecg/data_1.csv.gz          # ECG → labels
├── corsano_wrist_ppg2_green_6/2025-12-04.csv.gz    # PPG → HR features
├── corsano_wrist_acc/2025-12-04.csv.gz              # IMU → movement
├── corsano_bioz_emography/2025-12-04.csv.gz        # EDA → arousal
├── vivalnk_vv330_heart_rate/data_1.csv.gz         # HR trace (optional)
└── scai_app/ADLs_1.csv                              # Activity timeline
```

---

## Configuration

Edit [configs/pipeline_sim_elderly3.yaml](configs/pipeline_sim_elderly3.yaml) to:
- Set data paths
- Choose RMSSD method (delta_ln_rmssd vs recovery_slope)
- Enable/disable sensors
- Configure model hyperparameters

---

## Critical Rules (Non-Negotiable)

1. **NO HRV as input features**
   - Forbidden: RMSSD, SDNN, pNN50, NN50, LF, HF, LF/HF, PRV
   - ECG/RR intervals → labels ONLY

2. **Leakage-safe features only**
   - No future knowledge
   - No label-derived features

3. **Physiologically explainable**
   - HR level (cardiovascular load)
   - Movement (external mechanical load)
   - EDA (sympathetic arousal)

---

## Next Steps (To Implement)

1. **Feature extraction CLI**
   - Script: `scripts/extract_features.py`
   - Load all sensors for session
   - Extract features per ADL using existing modules
   - Save to CSV

2. **Alignment script**
   - Script: `scripts/align_labels_features.py`
   - Match features with RMSSD labels
   - Handle missing data
   - Create train-ready dataset

3. **Training script**
   - Script: `scripts/train_model.py`
   - Load aligned dataset
   - Train Ridge, ElasticNet, XGBoost
   - Save models and metrics

4. **Validation**
   - Optional Borg correlation
   - Cross-session validation
   - Feature importance analysis

---

## File Format Questions

Before implementing alignment, please confirm:

1. **ADLs_1.csv structure**:
   - What are the exact column names?
   - How are timestamps formatted?
   - Is there a phase column (baseline/task/recovery)?

2. **Sensor CSV structure**:
   - Are column names consistent across sensors?
   - Is there a timestamp column in each?
   - What are the actual value column names?

Ask if anything is unclear - the pipeline is designed to be adapted to your exact data format.
