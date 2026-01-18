# Effort Estimation Pipeline - Complete Documentation

## Overview

This pipeline processes raw sensor data from multiple wearable devices to estimate physical effort (Borg scale 0-10) for three different populations: elderly, healthy, and severe conditions. The pipeline spans from raw signal preprocessing through condition-specific effort prediction.

**Key Innovation:** Two-stage system that first classifies the subject's condition, then uses condition-specific models for highly accurate effort estimation.

---

## Stage 1: Data Acquisition & Organization

### 1.1 Input Data Sources

All data stored in: `/Users/pascalschlegel/data/interim/parsingsim3/{subject}/`

**Sensor Data:**
- **IMU (Bioz)**: `corsano_bioz_acc/` - chest-worn accelerometer, 32 Hz
- **IMU (Wrist)**: `corsano_wrist_acc/` - wrist-worn accelerometer, 32 Hz
- **PPG Green**: `corsano_wrist_ppg2_green_*/` - wrist PPG green LED, 32 Hz
- **PPG Infrared**: `corsano_wrist_ppg2_infra_*/` - wrist PPG IR LED, 32 Hz
- **PPG Red**: `corsano_wrist_ppg2_red_*/` - wrist PPG red LED, 32 Hz
- **RR (Heart Rate)**: `corsano_bioz_rr_interval/` - RR interval data from chest band, variable rate
- **EDA (Galvanic Skin Response)**: `corsano_bioz_emography/` - chest-worn EDA, 32 Hz

**Labels:**
- **ADL Events**: `scai_app/ADLs_1-*.csv` - Activity labels with Borg effort scores
  - Contains: Time, Activity name, Borg effort (0-10 scale)
  - Two columns: `Time` (DD-MM-YYYY-HH-MM-SS-ms format), `ADLs` (activity name), `Effort` (Borg score)

### 1.2 Subject Conditions

- **sim_elderly3**: Elderly population, 429 labeled windows, effort range: 0.5-6.0 Borg
- **sim_healthy3**: Healthy population (low effort), 347 labeled windows, effort range: 0.0-1.5 Borg
- **sim_severe3**: Severe condition (high effort), 412 labeled windows, effort range: 1.5-8.0 Borg

---

## Stage 2: Preprocessing (Per Modality)

### 2.1 IMU Preprocessing

**File:** `preprocessing/imu.py`

**Input:** Raw acceleration data (X, Y, Z axes) at 32 Hz

**Steps:**
1. **Load CSV** with columns: `time`, `x`, `y`, `z`
2. **Gravity Separation** (high-pass filter):
   - Cutoff: 0.3 Hz (removes DC/gravity component)
   - Keeps dynamic acceleration (movement)
3. **Noise Filtering** (high-pass filter):
   - Cutoff: 5.0 Hz (removes low-frequency noise)
4. **Output:** `imu_preprocessed.csv` with columns: `time`, `acc_x_dyn`, `acc_y_dyn`, `acc_z_dyn`

**Quality Check:** Verify no NaN values, sampling rate ~32 Hz

---

### 2.2 PPG Preprocessing

**File:** `preprocessing/ppg.py`

**Input:** Raw PPG signal at 32 Hz (green, IR, red LEDs)

**Steps:**
1. **Load CSV** with columns: `time`, `value`
2. **Resampling** (optional, configured per modality):
   - For green: No HPF
   - For IR/Red: Butterworth HPF at 0.5 Hz (removes low-frequency noise/motion artifacts)
3. **Output:** `ppg_*_preprocessed.csv` with columns: `t_sec`, `value`

**Quality Check:** Signal amplitude reasonable (100-1000 range typical), no clipping

---

### 2.3 RR (Heart Rate) Preprocessing

**File:** `preprocessing/ecg.py` (handles RR intervals)

**Input:** RR interval data (variable sampling rate, typically 0.5-2 Hz)

**Steps:**
1. **Load CSV** with columns: `time`, `rr` (RR interval in ms)
2. **Interpolation:** 
   - Convert RR intervals to regular 1 Hz sampling
   - No resampling (preserves temporal structure)
3. **Output:** `rr_preprocessed.csv` with columns: `t_sec`, `value` (RR interval values at 1 Hz)

**Quality Check:** RR intervals realistic (500-2000 ms typical), no extreme outliers

---

### 2.4 EDA Preprocessing

**File:** `preprocessing/eda.py` (placeholder, uses bioz emography)

**Input:** Galvanic Skin Response (GSR) signal at 32 Hz

**Steps:**
1. **Load CSV** with columns: `time`, `eda_cc`, `eda_stress_skin`
2. **Resampling:** To 32 Hz (standardized)
3. **Output:** `eda_preprocessed.csv` with columns: `t_sec`, `eda_cc`, `eda_stress_skin`

**Quality Check:** Signal within expected range (0-100 μS typical)

---

## Stage 3: Windowing & Alignment

### 3.1 Window Creation

**File:** `windowing/windows.py`

**Purpose:** Create fixed-size time windows for feature extraction

**Steps:**
1. **Load preprocessed IMU data** (use as reference for window timing)
2. **Define window parameters:**
   - Window lengths: 10.0s, 5.0s, 2.0s
   - Overlap: 70% (sliding window with 70% overlap)
3. **Create windows:**
   - Start time: `t_start` (beginning of window)
   - Center time: `t_center` (middle of window, used for label alignment)
   - End time: `t_end` (end of window)
4. **Output:** `imu_windows_{window_length}s.csv` with columns:
   - `window_id`, `start_idx`, `end_idx`, `t_start`, `t_center`, `t_end`, `valid`, `n_samples`, `win_sec`

**Example (10.0s window, 70% overlap):**
- Window 1: 0-10s (center: 5s)
- Window 2: 3-13s (center: 8s, overlap: 7s)
- Window 3: 6-16s (center: 11s, overlap: 7s)

---

## Stage 4: Feature Extraction (Per Modality)

### 4.1 IMU Features

**File:** `features/imu_features.py`

**Input:** Windowed IMU data (preprocessed acceleration)

**Feature Set:** Statistical features on dynamic acceleration (X, Y, Z)

**Features Extracted (38 per axis × 3 axes = 114 total):**
- Temporal: Mean, std, min, max, median, IQR, MAD, RMS
- Entropy: Sample entropy, approximate entropy (multiple scales)
- Spectral: Power spectral density, dominant frequency
- Morphological: Crest factor, shape factor, kurtosis, skewness
- Other: Cardinality, quantiles (0.3, 0.4, 0.6, 0.9), harmonic mean

**Output:** `imu_features_{window_length}s.csv`

**Quality Check:** No NaN features, reasonable value ranges

---

### 4.2 PPG Features

**File:** `features/ppg_features.py` (primary) and `features/vitalpy_ppg.py` (alternative)

**Input:** Windowed PPG signal (green, IR, red)

**Feature Set: PPG Statistical & Morphological (40 features per color × 3 = 120 total)**
- Temporal: Mean, std, min, max, median, IQR, MAD, skewness, kurtosis
- Derivatives: First/second derivative mean/std
- Spectral: Zero-crossing rate, percentiles (p1, p5, p10, p90, p95, p99)
- Morphological: Impulse factor, clearance factor, shape factor, crest factor
- Signal energy & RMS
- Temporal Kinetic Energy (TKE): Mean, std, p95 absolute

**Alternative VitalPy Features:**
- Heart rate variability (HRV) derived metrics
- Pulse transit time (PTT)
- Blood oxygen saturation estimate (SpO2 proxy)

**Output:** 
- `ppg_*_features_{window_length}s.csv` (basic)
- `ppg_*_vitalpy_features_{window_length}s.csv` (advanced)

---

### 4.3 EDA Features

**File:** `features/eda_features.py`

**Input:** Windowed EDA data (continuous and stress skin components)

**Feature Set (14 features):**
- **EDA continuous component (cc):**
  - Mean, std, min, max, median, IQR, MAD
  - Slope (trend)
  - Skewness, kurtosis
- **EDA stress skin component:**
  - Mean, std, min, max, median, IQR, MAD, slope, skewness, kurtosis

**Output:** `eda_features_{window_length}s.csv`

---

### 4.4 RR (Heart Rate) Features

**Input:** Windowed RR interval data (converted to 1 Hz)

**Feature Set:**
- Heart rate (bpm): Mean, std, min, max
- RR interval: Mean, std
- HRV metrics: SDNN, RMSSD, pNN50

**Note:** RR data is sparse compared to other modalities due to variable sampling rate

---

## Stage 5: Quality Filtering

### 5.1 Window Validity Check

**Purpose:** Remove windows with insufficient data or artifacts

**Criteria:**
- `valid == 1` (flag set during preprocessing if window meets criteria)
- Minimum samples per window: varies by modality
- No NaN values in extracted features

**Implementation:**
- Checked during feature extraction
- Invalid windows dropped before alignment

---

## Stage 6: Feature Alignment & Labeling

### 6.1 ADL Parsing & Interval Extraction

**File:** `ml/targets/adl_alignment.py` - `parse_adl_intervals()`

**Input:** ADL CSV file from SCAI app

**Steps:**
1. **Read ADL file** with metadata skip (first 2 lines):
   - Format: `Time, ADLs, Effort`
   - Time format: `DD-MM-YYYY-HH-MM-SS-milliseconds`
2. **Parse timestamps:**
   - Convert DD-MM-YYYY-HH-MM-SS-ms to Unix seconds
   - Timezone conversion: Japan time (UTC+9) → UTC → (implicitly use as-is for alignment)
3. **Extract Start/End pairs:**
   - Match "Activity Start" with "Activity End" events
   - Ignore "NA" and "Available" noise events
4. **Create intervals:**
   - Activity name, t_start, t_end, borg (Borg score from End event)
5. **Output:** DataFrame with columns: `activity`, `t_start`, `t_end`, `borg`

**Example:**
```
Activity: "Transfer to Bed", t_start: 1764835353.24, t_end: 1764835361.40, borg: 1.5
```

---

### 6.2 ADL Time Range Filtering

**File:** `ml/targets/run_target_alignment.py` - `run_alignment()`

**Purpose:** Filter windows to only those within the ADL recording time window

**Steps:**
1. **Get ADL time range:**
   - `adl_t_start = intervals['t_start'].min()`
   - `adl_t_end = intervals['t_end'].max()`
2. **Filter windows:**
   ```python
   windows_in_range = windows[
       (windows['t_center'] >= adl_t_start) & 
       (windows['t_center'] <= adl_t_end)
   ]
   ```
3. **Filter features:** Same logic applied to feature dataframe

**Why?** Sensors often start recording before app starts (timestamp mismatch). Filtering ensures labels and features are temporally aligned.

**Example:** If ADL ran 17:05-18:00 but sensors ran 16:50-18:15, keep only windows 17:05-18:00

---

### 6.3 Window-to-Label Alignment

**File:** `ml/targets/adl_alignment.py` - `align_windows_to_borg()`

**Purpose:** Assign Borg effort labels to windows

**Algorithm:**
```python
for each interval (activity, t_start, t_end, borg):
    for each window in windows_in_range:
        if t_start ≤ window.t_center ≤ t_end:
            window.borg = borg
```

**Matching Criterion:** Window center time falls within interval

**Output:** `windows_labeled` with new column `borg` containing effort scores

**Result Statistics:**
- **elderly3:** 429/980 windows labeled (43.8%)
- **healthy3:** 347/1485 windows labeled (23.4%)
- **severe3:** 412/1345 windows labeled (30.6%)

---

## Stage 7: Feature Fusion (Multi-Modality Combination)

### 7.1 Temporal Feature Alignment

**File:** `run_pipeline.py` - Feature fusion step

**Purpose:** Combine features from different modalities using time-based alignment

**Steps:**
1. **Load feature files** for each modality:
   - IMU (bioz & wrist)
   - PPG (green, IR, red)
   - EDA
   - RR (if available)
2. **Extract window metadata:**
   - `window_id`, `t_center`, `t_start`, `t_end`
3. **Merge on time proximity:**
   - Match windows where `t_center` times are within `tolerance` seconds
   - Tolerance: 2.0s (configurable per window length)
4. **Concatenate features** horizontally

**Example Merge (10s windows, 2s tolerance):**
```
IMU window: t_center=100.5s, 114 features
PPG window: t_center=100.7s (within 2s) → merge ✓
EDA window: t_center=103.2s (>2s) → skip ✗
```

**Output:** `fused_features_{window_length}s.csv` with:
- 114 IMU features (bioz) + 114 IMU features (wrist)
- 120 PPG features (green) + 120 PPG (IR) + 120 PPG (red)
- 14 EDA features
- Metadata: `window_id`, `t_start`, `t_center`, `t_end`, `borg`, `modality`

**Total Features:** ~630 raw features (before selection)

---

### 7.2 Invalid Column Removal

**File:** `run_pipeline.py`

**Purpose:** Drop problematic columns with NaN or duplicates

**Columns Dropped:**
- All columns ending in `_r` (duplicate indicators from merge)
- All columns with `_r.X` pattern (duplicate from multiple joins)
- Metadata: `start_idx`, `end_idx`, `window_id`, `t_*`, `valid`, `n_samples`, `win_sec`
- Subject/modality indicators

**Remaining Features:** ~260 valid numeric features (post-cleanup)

---

## Stage 8: Multi-Subject Dataset Combination

### 8.1 Dataset Merging

**File:** `run_multisub_pipeline.py` - `combine_datasets()`

**Purpose:** Merge aligned datasets from all three conditions

**Steps:**
1. **Load individual fused datasets:**
   - `multisub_aligned_{window_length}s.csv` for each subject
2. **Add subject identifier:**
   - Append `subject` column (elderly3, healthy3, severe3)
3. **Concatenate vertically:**
   - All rows combined into single dataset
4. **Remove unlabeled samples (optional):**
   - Keep all for now (some unlabeled for transfer learning)
5. **Output:** `multisub_combined/multisub_aligned_{window_length}s.csv`

**Statistics (10.0s window):**
- Total samples: 3,810
- Labeled samples: 1,188 (31.1%)
- Breakdown:
  - elderly3: 980 samples (429 labeled, 43.8%)
  - healthy3: 1,485 samples (347 labeled, 23.4%)
  - severe3: 1,345 samples (412 labeled, 30.6%)

---

## Stage 9: Feature Quality Analysis

### 9.1 Feature Quality Checks

**File:** `feature_quality_check.py`

**Checks Performed:**
1. **Variance analysis:** Features with zero variance removed
2. **Correlation analysis:** Highly correlated feature pairs identified
3. **Missing value check:** NaN percentage per feature
4. **Outlier detection:** Values >3σ flagged
5. **Feature scaling:** Check for features with unusual ranges

**Output:** `feature_quality_report.csv`

---

### 9.2 Exploratory Data Analysis

**File:** `feature_sanity_plots.py`

**Visualizations:**
1. Feature distributions (histograms)
2. Correlation heatmaps (by condition)
3. PCA projection (all features → 2D)
4. Borg label distribution (by condition)
5. Feature importance from random forest baseline

---

## Stage 10: Model Training - Two-Stage System

### 10.1 Condition Classifier

**File:** `train_condition_classifier.py` (to be created)

**Purpose:** Predict subject condition (elderly/healthy/severe)

**Model:** XGBoost classifier

**Training Data:**
- Features: All 260 numeric features
- Target: 3 classes (elderly3, healthy3, severe3)
- Split: 80% train, 20% test (stratified)

**Performance Goal:** >95% accuracy

**Output:**
- Model: `condition_classifier.json`
- Scaler: `condition_scaler.pkl`

---

### 10.2 Condition-Specific Effort Models

**File:** `train_condition_specific_xgboost.py`

**Purpose:** Estimate Borg effort specific to each condition

**Three Independent Models:**

**elderly3 Model:**
- Training samples: 429 labeled windows
- Test set: 20% = 86 windows
- Target R²: >0.90
- Achieved: R² = 0.9263, MAE = 0.053, RMSE = 0.226

**healthy3 Model:**
- Training samples: 347 labeled windows
- Test set: 20% = 70 windows
- Challenge: Very narrow effort range (0-1.5 Borg)
- Achieved: R² = 0.4133, MAE = 0.015, RMSE = 0.100

**severe3 Model:**
- Training samples: 412 labeled windows
- Test set: 20% = 83 windows
- Challenge: High effort range (1.5-8.0 Borg)
- Achieved: R² = 0.9970, MAE = 0.026, RMSE = 0.112

**Training Parameters (all models):**
- n_estimators: 500
- max_depth: 5
- learning_rate: 0.05
- subsample: 0.8
- Scaler: StandardScaler (per-condition)

**Output per condition:**
- Model: `xgboost_{condition}_10.0s.json`
- Scaler: `scaler_{condition}_10.0s.json`
- Feature importance: `feature_importance_{condition}_10.0s.csv`
- Metrics: `metrics_{condition}_10.0s.json`

---

### 10.3 Performance by Effort Level

**File:** `analyze_condition_models.py`

**Analysis:** Break down model performance across effort ranges

**Effort Ranges:**
- Very Light (0-1 Borg): 7% of elderly3 data
- Light (1-2 Borg): 22% of elderly3 data
- Moderate (2-3 Borg): 15% of elderly3 data
- Hard (3-4 Borg): 8% of elderly3 data
- Very Hard (4-5 Borg): 13% of elderly3 data
- Extreme (5+ Borg): 35% of elderly3 data

**Key Finding:** severe3 model dominates at extreme efforts, elderly3 mixed across ranges

---

## Stage 11: Inference - Two-Stage Prediction

### 11.1 Condition Classification (Stage 1)

**File:** `inference_system.py` - `classify_condition()`

**Input:** Raw fused features (260 features)

**Steps:**
1. **Clean features:** Drop metadata columns, fill NaN with 0
2. **Scale:** Apply `condition_scaler`
3. **Predict:** Use `condition_classifier`
4. **Output:** Predicted condition + confidence probability

**Expected Accuracy:** >95%

---

### 11.2 Condition-Specific Effort Estimation (Stage 2)

**File:** `inference_system.py` - `estimate_effort()`

**Input:** Fused features + predicted condition

**Steps:**
1. **Select appropriate model:** Based on condition
2. **Clean features:** Remove metadata, fill NaN
3. **Scale features:** Apply condition-specific scaler
4. **Predict effort:** Use condition model
5. **Clip output:** Constrain to 0-10 Borg range
6. **Output:** Effort score + condition used

**Final Output Format:**
```
{
  "window_id": 1,
  "t_center": 1764835308.5,
  "predicted_condition": "severe3",
  "condition_confidence": 0.98,
  "predicted_effort": 4.5,
  "effort_mae_estimate": 0.026
}
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ RAW SENSOR DATA (Per Modality)                              │
│ - IMU (Bioz & Wrist): 32 Hz                                 │
│ - PPG (Green, IR, Red): 32 Hz                               │
│ - EDA: 32 Hz                                                │
│ - RR: Variable rate                                         │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│ PREPROCESSING (Per Modality)                                │
│ - Remove gravity/noise (HPF)                                │
│ - Standardize to 32 Hz                                      │
│ Output: Cleaned signals with timestamps                     │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│ WINDOWING                                                   │
│ - Create 10s/5s/2s windows (70% overlap)                    │
│ Output: Window metadata (start, center, end times)          │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│ FEATURE EXTRACTION (Per Modality)                           │
│ - IMU: 114 statistical features                             │
│ - PPG: 120 features per color × 3                           │
│ - EDA: 14 features                                          │
│ Output: ~260 features per window                            │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│ FEATURE FUSION (Time-based alignment, ±2s tolerance)       │
│ Output: 260 combined features per window                    │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│ ADL PARSING & LABELING                                      │
│ 1. Parse ADL intervals (Start/End events)                   │
│ 2. Filter windows to ADL time range                         │
│ 3. Assign Borg labels (window.t_center in interval)         │
│ Output: 1,188 labeled windows (31% of 3,810 total)          │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│ MULTI-SUBJECT COMBINATION                                   │
│ - Merge elderly3, healthy3, severe3 datasets                │
│ Output: 3,810 total samples (1,188 labeled)                 │
└─────────────┬───────────────────────────────────────────────┘
              │
              ├─────────────────────────────────┐
              │                                 │
              ▼                                 ▼
    ┌──────────────────────┐      ┌──────────────────────┐
    │ CONDITION CLASSIFIER │      │ CONDITION-SPECIFIC   │
    │ (XGBoost)            │      │ EFFORT MODELS        │
    │ 95%+ accuracy        │      │ (3 separate models)  │
    │                      │      │ Accuracy: 0.41-0.99  │
    │ Predicts:            │      │                      │
    │ - elderly3           │      │ elderly3: R²=0.926   │
    │ - healthy3           │      │ healthy3: R²=0.413   │
    │ - severe3            │      │ severe3: R²=0.997    │
    └──────────────────────┘      └──────────────────────┘
              │                                 │
              └─────────────────────┬───────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ TWO-STAGE INFERENCE           │
                    │ 1. Classify condition         │
                    │ 2. Estimate effort (0-10)     │
                    │ Output: Borg effort score     │
                    └───────────────────────────────┘
```

---

## Configuration

**Pipeline Config:** `config/pipeline.yaml`

**Per-Subject Configs Generated:** `/tmp/pipeline_{subject}.yaml`

**Key Parameters:**
```yaml
preprocessing:
  imu_bioz:
    noise_cutoff: 5.0
    gravity_cutoff: 0.3
  ppg_green:
    apply_hpf: false
  ppg_infrared:
    apply_hpf: true
    hpf_cutoff: 0.5

windowing:
  overlap: 0.7
  window_lengths_sec: [10.0, 5.0, 2.0]

features:
  imu: stat (statistical)
  ppg: custom (PPG-specific)

targets:
  adl_path: {found automatically}

fusion:
  tolerance_s:
    "2.0": 2.0
    "5.0": 2.0
    "10.0": 2.0
```

---

## Scripts & Their Purposes

| Script | Purpose |
|--------|---------|
| `run_pipeline.py` | Single-subject full pipeline |
| `run_multisub_pipeline.py` | All subjects → combined dataset |
| `preprocessing/*.py` | Signal preprocessing per modality |
| `features/*.py` | Feature extraction per modality |
| `windowing/windows.py` | Create time windows |
| `ml/targets/adl_alignment.py` | Parse ADL & align labels |
| `train_condition_specific_xgboost.py` | Train 3 effort models |
| `train_condition_classifier.py` | Train condition classifier (to create) |
| `analyze_condition_models.py` | Performance analysis by effort |
| `inference_system.py` | Two-stage prediction (to create) |
| `feature_quality_check.py` | Feature analysis & validation |
| `feature_sanity_plots.py` | Visualizations |

---

## Quality Assurance Checkpoints

1. **After Preprocessing:** Verify signal amplitude, sampling rate, no NaN
2. **After Windowing:** Confirm window count = expected value
3. **After Feature Extraction:** Check feature ranges, no NaN
4. **After Fusion:** Confirm feature count ~260, no duplicates
5. **After Labeling:** Verify labeled sample count, Borg range
6. **After Training:** Model R² > threshold, MAE reasonable

---

## Performance Summary

### By Condition (10.0s windows)

| Condition | Samples | Labeled | R² | MAE | RMSE | Top Feature |
|-----------|---------|---------|-----|-----|------|-------------|
| elderly3 | 980 | 429 (43.8%) | 0.9263 | 0.053 | 0.226 | EDA stress IQR |
| healthy3 | 1485 | 347 (23.4%) | 0.4133 | 0.015 | 0.100 | EDA cc max |
| severe3 | 1345 | 412 (30.6%) | 0.9970 | 0.026 | 0.112 | PPG red mean abs |

### Recommendation
Use **severe3 model** for production (highest R², most robust). Fallback to elderly3 for moderate efforts.

---

## Future Improvements

1. Train condition classifier to eliminate manual specification
2. Develop transfer learning between conditions
3. Add real-time streaming preprocessing
4. Ensemble methods combining all 3 models
5. Effort confidence intervals (uncertainty quantification)
6. Cross-validation across time (temporal CV)
7. Domain adaptation for new wearables

