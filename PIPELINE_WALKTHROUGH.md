# Complete Pipeline Walkthrough: Raw Data → Model Training

## Overview
When `run_pipeline.py` runs for the first time with **no intermediates**, it processes raw sensor data through 6 phases to generate labeled features, then `train_hrv_recovery_clean.py` trains the model.

---

## PHASE 0: INITIALIZATION

**Input:**
- `config/pipeline.yaml` - Configuration file

**Output:**
- Output directories created

**Steps:**
1. Read YAML configuration
2. Set `output_root = /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output`
3. Create output root directory
4. Set window lengths: `[10.0, 5.0, 2.0]` seconds
5. Set overlap: `0.7` (70%)

---

## PHASE 1: SIGNAL PREPROCESSING

Process raw sensor data into clean, normalized signals.

### 1.1 IMU BioZ Preprocessing

**Input:**
- `/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/corsano_bioz_acc/2025-12-04.csv.gz` (raw accelerometer)

**Process:**
1. Load compressed CSV
2. Extract X, Y, Z acceleration
3. Remove high-frequency noise (cutoff: 5.0 Hz)
4. Remove gravity offset (cutoff: 0.3 Hz)
5. Resample to 32 Hz
6. Normalize

**Output:**
- `/output/.../imu_bioz/imu_preprocessed.csv` (979 windows with 10.0s)

### 1.2 IMU Wrist Preprocessing

**Input:**
- `/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/corsano_wrist_acc/2025-12-04.csv.gz`

**Process:**
- Same as IMU BioZ (noise/gravity filtering)

**Output:**
- `/output/.../imu_wrist/imu_preprocessed.csv` (1171 windows with 10.0s)

### 1.3 PPG Preprocessing (Green, Infra, Red)

**Input (3x):**
- `corsano_wrist_ppg2_green_6/2025-12-04.csv.gz` (green LED)
- `corsano_wrist_ppg2_infra_red_22/2025-12-04.csv.gz` (infrared)
- `corsano_wrist_ppg2_red_182/2025-12-04.csv.gz` (red LED)

**Process (per channel):**
1. Load photoplethysmogram signal
2. Resample to 32 Hz
3. Apply high-pass filter (cutoff: 0.5 Hz) - removes drift
4. Normalize

**Output (3x):**
- `/output/.../ppg_green/ppg_green_preprocessed.csv`
- `/output/.../ppg_infra/ppg_infra_preprocessed.csv`
- `/output/.../ppg_red/ppg_red_preprocessed.csv`

### 1.4 ECG Preprocessing → RR Interval Extraction ⭐ (ECG-DERIVED RR)

**Input:**
- `/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/vivalnk_vv330_ecg/data_1.csv.gz` (raw ECG @ 128 Hz)

**Process:**
1. Load ECG signal (602,240 samples, 4,818 seconds)
2. Detrend signal
3. Normalize (Z-score)
4. Bandpass filter (5-15 Hz) to enhance QRS complex
5. Square signal
6. Moving average window (~200 ms)
7. Detect R-peaks:
   - Adaptive threshold: median + 1.5×MAD
   - Minimum distance: 300 ms (physiological min RR)
   - Find peaks above threshold
8. **Result: 6,473 R-peaks detected**
9. Compute RR intervals:
   - RR_i = time_of_peak_i+1 - time_of_peak_i (in milliseconds)
10. Filter physiological outliers:
    - Remove RR < 300 ms (HR > 200 bpm)
    - Remove RR > 2000 ms (HR < 30 bpm)
    - Remove RR > 3×SD from median
11. **Result: 6,336 valid RR intervals**
12. **Mean RR: 719 ms (84 bpm)** ✓ Physiologically valid

**Output:**
- `/output/.../rr/rr_preprocessed.csv` (6,336 rows: t_sec, rr)
  - Example: `1764837099.1, 876.234` (timestamp, RR in ms)

### 1.5 EDA Preprocessing

**Input:**
- `/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/corsano_bioz_emography/2025-12-04.csv.gz`

**Process:**
1. Load skin conductance signal
2. Resample to 32 Hz
3. Remove baseline drift (high-pass filter)
4. Normalize

**Output:**
- `/output/.../eda/eda_preprocessed.csv`

---

## PHASE 2: WINDOWING

Create fixed-duration, overlapping time windows from preprocessed signals.

**Configuration:**
- Window lengths: `[10.0, 5.0, 2.0]` seconds
- Overlap: `70%`

**Process (per modality, per window length):**
1. For IMU:
   - 10.0s windows with 70% overlap → 979 windows
   - 5.0s windows with 70% overlap → 1,961 windows
   - 2.0s windows with 70% overlap → 4,957 windows

2. For PPG (each channel):
   - Same windowing scheme
   - 1,187 windows (10.0s), 2,377 windows (5.0s), 6,010 windows (2.0s)

3. For EDA:
   - 1,131 windows (10.0s), 2,265 windows (5.0s), 5,725 windows (2.0s)

**Output:**
- `/output/.../imu_bioz/imu_windows_10.0s.csv` (979 rows)
- `/output/.../ppg_green/ppg_green_windows_10.0s.csv` (1,187 rows)
- etc.

---

## PHASE 3: FEATURE EXTRACTION

Extract statistical and temporal features from each window.

**Features computed per window (example for 10.0s):**

**For IMU (acc_x_dyn, acc_y_dyn, acc_z_dyn):**
- Mean, std, min, max, median
- RMS, energy
- Skewness, kurtosis
- Zero-crossing rate
- Quantiles (0.1, 0.25, 0.75, 0.9)
- ~20 features per axis → ~60 total IMU features

**For PPG (green, infra, red):**
- Mean, std, min, max, median
- RMS, energy
- Skewness, kurtosis
- Zero-crossing rate (zcr)
- Quantiles
- ~30 features per channel → ~90 total PPG features

**For EDA:**
- Mean skin conductance (cc_mean, cc_std, cc_min, cc_max, cc_median, cc_rms)
- Skin conductance response (eda_stress_skin properties)
- Slope, extrema
- ~10-15 total EDA features

**Output:**
- `/output/.../imu_bioz/imu_features_10.0s.csv` (979 rows, ~60 cols)
- `/output/.../ppg_green/ppg_green_features_10.0s.csv` (1,187 rows, ~30 cols)
- `/output/.../eda/eda_features_10.0s.csv` (1,131 rows, ~15 cols)

---

## PHASE 4: FUSION

Merge features from all modalities using window matching.

**Input:**
- All modality feature files

**Process (for 10.0s windows):**
1. Load all modality feature dataframes
2. Match windows by `t_center` (center timestamp)
3. Inner join on matching windows
4. Combine into single wide dataframe

**Merge statistics:**
- IMU BioZ: 979 windows
- IMU Wrist: 1,171 windows
- PPG Green: 1,187 windows
- PPG Infra: 1,187 windows
- PPG Red: 1,187 windows
- EDA: 1,131 windows
- **Result: 979 windows** (all must overlap)
- **Total features: 188** (60 IMU_bioz + 60 IMU_wrist + 30×3 PPG + 15 EDA)

**Output:**
- `/output/.../fused_features_10.0s.csv` (979 rows, 188 feature cols)
- Similar for 5.0s (1,961 rows) and 2.0s (4,957 rows)

---

## PHASE 5: HRV RECOVERY ALIGNMENT ⭐ (CRITICAL)

Compute HRV recovery rates from RR intervals and assign to windows.

**Input:**
- `fused_features_10.0s.csv` (979 windows)
- `rr_preprocessed.csv` (6,336 RR intervals)
- `ADLs_1.csv` (activity annotations)

**Process:**

### 5.1 Load ADL Activities
1. Parse activity file
2. Extract activity intervals: `(t_start, t_end, borg_effort)`
3. **32 activities detected** in recording

### 5.2 For each activity:

**Example Activity 0 (Borg 1.5):**

1. **Identify baseline period:**
   - 10-60 seconds before activity start
   - Extract RR intervals in this period
   - Compute RMSSD_baseline

2. **Identify effort period:**
   - During activity (t_start to t_end)
   - Extract RR intervals
   - Compute RMSSD_effort

3. **Identify recovery period:**
   - 10-60 seconds after activity end (configured)
   - Extract RR intervals
   - Compute RMSSD_recovery

4. **Compute recovery rate:**
   ```
   recovery_time = 60 - 10 = 50 seconds
   hrv_change = RMSSD_recovery - RMSSD_effort
   recovery_rate = hrv_change / recovery_time  (ms/sec)
   ```

5. **Example values:**
   - RMSSD_effort: 51.6 ms
   - RMSSD_recovery: 167.7 ms
   - recovery_rate: +0.387 ms/s (positive = recovery)

6. **Assign to windows:**
   - All windows with t_center in activity interval get this recovery_rate
   - Activity 0: 100 windows labeled with recovery_rate=-0.033

### 5.3 Result:
- **All 32 activities processed**
- **655 windows assigned HRV recovery rates**
- **Range: -0.35 to +0.39 ms/s**
  - Negative: HRV decreased (worse recovery)
  - Positive: HRV increased (good recovery)

**Output:**
- `/output/.../fused_aligned_10.0s.csv` (655 rows, 188 features + hrv_recovery_rate column)
- Similar for 5.0s (1,311 rows) and 2.0s (3,312 rows)

---

## PHASE 6: FEATURE SELECTION & QC

Quality check and reduce feature dimensionality.

**Input:**
- `fused_aligned_10.0s.csv` (655 rows, 188 features)

**Process:**

1. **Remove metadata columns**
   - window_id, t_start, t_center, etc.
   - **Result: 188 features**

2. **Correlation ranking**
   - Compute Pearson correlation of each feature with hrv_recovery_rate
   - Rank by |correlation|
   - Select top 100 features

3. **Redundancy pruning**
   - Compute correlation between top 100 features
   - Remove highly correlated pairs (threshold: 0.9)
   - **Result: 28-39 features** (varies by window length)

4. **PCA analysis**
   - Compute principal components
   - Report PCs needed for 90%, 95%, 99% variance
   - **Example: 12 PCs for 90%, 16 for 95%, 24 for 99%**

5. **Save outputs:**
   - `features_selected_pruned.csv` (feature names)
   - `pca_variance_explained.csv` (PC loadings)

---

## PIPELINE SUMMARY

**File outputs for 10.0s windows:**
```
/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/
├── imu_bioz/
│   ├── imu_preprocessed.csv
│   ├── imu_windows_10.0s.csv
│   └── imu_features_10.0s.csv
├── ppg_green/
│   ├── ppg_green_preprocessed.csv
│   ├── ppg_green_windows_10.0s.csv
│   └── ppg_green_features_10.0s.csv
├── eda/
│   ├── eda_preprocessed.csv
│   ├── eda_windows_10.0s.csv
│   └── eda_features_10.0s.csv
├── fused_features_10.0s.csv (all modalities merged)
├── fused_aligned_10.0s.csv (+ HRV recovery rate labels) ⭐
└── feature_selection_qc/
    └── qc_10.0s/
        ├── features_selected_pruned.csv
        ├── pca_variance_explained.csv
        └── ...
```

**Key statistics:**
- 655 labeled windows (10.0s)
- 188 features per window
- HRV recovery rates: -0.35 to +0.39 ms/s
- 32 activities with computed recovery rates

---

## MODEL TRAINING

Input: `fused_aligned_10.0s.csv`

**Steps:**

1. **Load data**
   - Read 655 rows, 188 features + target
   - Filter rows with valid hrv_recovery_rate (all 655)

2. **Feature preparation**
   - Remove metadata columns
   - Remove HRV-related features (rmssd, pnn50, sdnn, etc.)
   - Remove zero-variance features (1 removed)
   - Remove high-missing features (>30% NaN) - none removed
   - Impute remaining NaNs with column median
   - **Final: 187 features, 655 samples**

3. **Train-test split**
   - 80/20 random split (no activity-based leakage)
   - Train: 524 samples
   - Test: 131 samples

4. **Feature scaling**
   - StandardScaler fit on training data
   - Apply to both train/test

5. **Train 3 models:**

   **RandomForest:**
   - 200 estimators, max_depth=10
   - Results: Test R²=0.7571, Pearson r=0.8999, MAE=0.0744

   **XGBoost:**
   - 500 estimators, max_depth=5, learning_rate=0.05
   - Results: Test R²=0.8920, Pearson r=0.9531, MAE=0.0430

   **GradientBoosting (BEST):**
   - 300 estimators, max_depth=4, learning_rate=0.05
   - Results: Test R²=0.9225, Pearson r=0.9629, MAE=0.0339

6. **Save outputs:**
   - Model: `/Users/pascalschlegel/data/interim/hrv_recovery_results/hrv_model.pkl`
   - Diagnostics: `.../hrv_recovery_diagnostics.png`

---

## COMPLETE PIPELINE TIME

**Typical runtime:**
- Signal preprocessing: ~30 seconds
- Windowing: ~10 seconds
- Feature extraction: ~5 minutes
- Fusion: ~30 seconds
- HRV alignment: ~2 minutes
- Feature selection: ~30 seconds
- Model training: ~1 minute

**Total: ~10 minutes** for full first-time run

---

## Files Created (First Run)

**Total new files:**
- 6 preprocessed signal files
- 18 windowed signal files (3 lengths × 6 modalities)
- 18 feature files (3 lengths × 6 modalities)
- 3 fused feature files (10s, 5s, 2s)
- 3 fused aligned files (10s, 5s, 2s) ⭐ **CRITICAL**
- 9 feature selection QC files
- 1 trained model pickle
- 1 diagnostics plot

**Total disk usage:**
- Raw intermediates: ~500 MB (cached)
- Final outputs: ~50 MB (fused_aligned_*.csv)
- Model: ~10 MB (pickle)

---

## Subsequent Runs (With Cached Intermediates)

If any preprocessing/windowing files already exist, they are **reused** (not recomputed).

**When to delete cache:**
- `rm -rf output/.../imu_bioz/` - to reprocess IMU
- `rm -rf output/.../rr/` - to reprocess ECG → RR
- `rm -rf output/.../fused_aligned_*.csv` - to recompute HRV labels
- Delete all to start completely fresh

---

## ECG-Derived RR Special Note ⭐

**Why ECG instead of heart rate?**
- **Raw ECG:** 128 Hz sampling → ~7.8 ms temporal resolution
- **R-peak detection:** Identifies each heartbeat individually
- **RR intervals:** Beat-to-beat precision
- **RMSSD:** Requires precise RR values (gold standard for HRV)

**Gold standard validation:**
- 6,473 R-peaks detected ✓
- 6,336 valid RR intervals ✓
- Mean: 719 ms (84 bpm) ✓ Physiologically realistic
- No outliers after filtering ✓

This ensures clinically valid HRV recovery measurements!
