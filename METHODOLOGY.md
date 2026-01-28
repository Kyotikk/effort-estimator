# Borg CR10 Effort Estimation Pipeline - Complete Methodology

## Overview

**Objective:** Predict perceived exertion (Borg CR10 scale, 0-10) from wearable physiological signals during Activities of Daily Living (ADLs).

**Data Sources:**
- Empatica E4 wristband: PPG (3 wavelengths), EDA, Temperature
- Custom chest sensor: IMU (accelerometer + gyroscope)
- Ground truth: Borg CR10 ratings collected during ADL tasks

---

## 1. RAW DATA ACQUISITION

### 1.1 Sensor Signals Collected

| Signal | Sensor | Sampling Rate | Unit |
|--------|--------|---------------|------|
| PPG Green | Empatica E4 | 64 Hz | arbitrary |
| PPG Infrared | Empatica E4 | 64 Hz | arbitrary |
| PPG Red | Empatica E4 | 64 Hz | arbitrary |
| EDA (Electrodermal Activity) | Empatica E4 | 4 Hz | µS |
| Skin Temperature | Empatica E4 | 4 Hz | °C |
| Accelerometer (X, Y, Z) | Chest IMU | 100 Hz | g |
| Gyroscope (X, Y, Z) | Chest IMU | 100 Hz | °/s |

### 1.2 Ground Truth Labels

- **Borg CR10 Scale:** 0 (nothing at all) to 10 (extremely strong)
- **Collection method:** Subject verbally reports effort rating at end of each ADL task
- **ADL tasks:** Walking, stair climbing, sitting, standing, household activities, etc.
- **Label format:** CSV with columns `[t_start, t_end, activity, borg]`

---

## 2. PREPROCESSING

Each modality undergoes signal-specific preprocessing before feature extraction.

### 2.1 PPG Preprocessing (`preprocessing/ppg.py`)

```
Raw PPG (64 Hz) 
    │
    ▼
[1] Bandpass Filter: 0.5 - 4.0 Hz (Butterworth, order=3)
    - Removes DC offset and high-frequency noise
    - Preserves cardiac frequency range (30-240 BPM)
    │
    ▼
[2] Normalization: z-score per recording session
    - Mean = 0, Std = 1
    - Enables cross-session comparison
    │
    ▼
Preprocessed PPG
```

### 2.2 EDA Preprocessing (`preprocessing/eda.py`)

```
Raw EDA (4 Hz)
    │
    ▼
[1] Lowpass Filter: 1.0 Hz (Butterworth, order=4)
    - Removes motion artifacts
    - Preserves slow skin conductance changes
    │
    ▼
[2] Decomposition into Tonic + Phasic components
    - Tonic: baseline skin conductance level (SCL)
    - Phasic: skin conductance responses (SCR)
    │
    ▼
Preprocessed EDA (tonic + phasic)
```

### 2.3 IMU Preprocessing (`preprocessing/imu.py`)

```
Raw Accelerometer/Gyroscope (100 Hz)
    │
    ▼
[1] Gravity Separation (Accelerometer only)
    - Lowpass filter (0.3 Hz) → Static component (gravity)
    - Highpass filter (0.3 Hz) → Dynamic component (movement)
    │
    ▼
[2] Magnitude Calculation
    - acc_mag = sqrt(x² + y² + z²)
    - gyro_mag = sqrt(x² + y² + z²)
    │
    ▼
[3] Optional: Orientation estimation via complementary filter
    │
    ▼
Preprocessed IMU (static, dynamic, magnitude)
```

---

## 3. WINDOWING

Continuous signals are segmented into fixed-length, non-overlapping windows.

### 3.1 Window Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Window size | 10 seconds | Balances temporal resolution with feature stability |
| Overlap | 0% | **Prevents temporal leakage** between train/test |
| Minimum samples | 80% of expected | Windows with >20% missing data are discarded |

### 3.2 Windowing Process (`windowing/windows.py`)

```
Preprocessed Signal (continuous)
    │
    ▼
[1] Segment into 10-second windows
    - Window timestamps: [t_start, t_end, t_center]
    - t_center used for label alignment
    │
    ▼
[2] Quality Check
    - Reject if >20% samples missing
    - Reject if signal clipping detected
    - Reject if excessive motion artifacts (PPG)
    │
    ▼
[3] Output: List of valid windows with timestamps
```

### 3.3 Why 0% Overlap?

With 70% overlap (previous setting), consecutive windows share 7 seconds of data:
- Window 1: t=0-10s
- Window 2: t=3-13s (7s shared with Window 1!)

If Window 1 is in training and Window 2 is in test → **temporal leakage**.

With 0% overlap:
- Window 1: t=0-10s
- Window 2: t=10-20s (0s shared)

No information leak between windows.

---

## 4. FEATURE EXTRACTION

Each window produces a feature vector. Features are extracted per modality.

### 4.1 PPG Features (`features/ppg_features.py`)

**A. Time-Domain Features (per wavelength: green, infrared, red)**

| Feature | Formula | Physiological Meaning |
|---------|---------|----------------------|
| mean | $\bar{x} = \frac{1}{N}\sum x_i$ | Average blood volume |
| std | $\sigma = \sqrt{\frac{1}{N}\sum(x_i - \bar{x})^2}$ | Signal variability |
| min, max | $\min(x), \max(x)$ | Amplitude range |
| p1, p5, p10, p25, p50, p75, p90, p95, p99 | Percentiles | Distribution shape |
| skewness | $\frac{E[(x-\mu)^3]}{\sigma^3}$ | Asymmetry |
| kurtosis | $\frac{E[(x-\mu)^4]}{\sigma^4}$ | Peak sharpness |
| iqr | $Q3 - Q1$ | Robust spread |
| range | $\max - \min$ | Full amplitude |
| p99_p1 | $P_{99} - P_{1}$ | Robust range |
| trim_mean_10 | Trimmed mean (10%) | Outlier-robust mean |
| rms | $\sqrt{\frac{1}{N}\sum x_i^2}$ | Signal power |

**B. Derivative Features**

| Feature | Description |
|---------|-------------|
| diff_mean, diff_std | First derivative statistics |
| diff_mean_abs | Mean absolute change |
| ddx_std | Second derivative std (acceleration) |

**C. Energy Features**

| Feature | Formula |
|---------|---------|
| tke_mean | $\frac{1}{N}\sum x_i^2$ (Teager-Kaiser proxy) |
| tke_std | Std of squared signal |

### 4.2 HRV Features (`features/ppg_features.py` → peak detection)

From PPG, we detect heartbeats and compute Heart Rate Variability:

```
PPG Window
    │
    ▼
[1] Peak Detection
    - Find systolic peaks using scipy.signal.find_peaks
    - Minimum distance: 0.4s (150 BPM max)
    - Prominence threshold: adaptive
    │
    ▼
[2] RR/IBI Intervals
    - IBI = Inter-Beat Interval (time between peaks)
    - RR intervals in milliseconds
    │
    ▼
[3] HRV Metrics (if ≥5 beats in window)
```

| Feature | Formula | Meaning |
|---------|---------|---------|
| hr_mean | $\frac{60000}{\bar{IBI}}$ | Mean heart rate (BPM) |
| hr_min, hr_max | Min/max instantaneous HR | HR range |
| mean_ibi | $\bar{IBI}$ (ms) | Average beat interval |
| sdnn | $\sigma_{IBI}$ | Overall HRV |
| rmssd | $\sqrt{\frac{1}{N-1}\sum(\Delta IBI_i)^2}$ | Short-term HRV (parasympathetic) |

**Expected correlations with effort:**
- HR ↑ as effort ↑ (positive correlation) ✓
- IBI ↓ as effort ↑ (negative correlation) ✓
- RMSSD ↓ as effort ↑ (negative correlation) ✓

### 4.3 EDA Features (`features/eda_features.py`)

| Feature | Description |
|---------|-------------|
| eda_mean, eda_std | Basic statistics |
| eda_min, eda_max | Range |
| eda_slope | Linear trend (sympathetic arousal) |
| scr_count | Number of skin conductance responses |
| scr_amplitude_mean | Mean SCR amplitude |
| eda_phasic_mean | Mean phasic component |
| eda_tonic_mean | Mean tonic (baseline) level |

### 4.4 IMU Features (`features/manual_features_imu.py`)

**Computed separately for:** acc_x, acc_y, acc_z, acc_x_dyn, acc_y_dyn, acc_z_dyn, gyro_x, gyro_y, gyro_z, acc_mag, gyro_mag

| Feature | Description |
|---------|-------------|
| mean, std, var | Basic statistics |
| min, max, range | Amplitude |
| energy | $\sum x_i^2$ |
| entropy | Signal complexity |
| zero_crossings | Activity indicator |
| peak_count | Movement events |

**TSFRESH features** (automated extraction):
- `variance_of_absolute_differences`
- `sample_entropy`
- `lower_complete_moment`
- And many more...

---

## 5. FEATURE FUSION

Features from all modalities are combined into a single feature matrix.

### 5.1 Fusion Process (`ml/fusion/fuse_windows.py`)

```
PPG Features (per window)     ─┐
EDA Features (per window)     ─┼─► Merge on t_center ─► Combined Features
IMU Features (per window)     ─┤
HRV Features (per window)     ─┘
```

**Merge key:** `t_center` (window center timestamp)
- Each modality has windows with t_center
- Inner join ensures all modalities present

### 5.2 Output Columns

```
[t_center, subject, activity, modality,
 ppg_green_mean, ppg_green_std, ..., ppg_green_hr_mean, ppg_green_rmssd, ...,
 ppg_infra_mean, ..., ppg_infra_rmssd, ...,
 ppg_red_mean, ..., ppg_red_rmssd, ...,
 eda_mean, eda_std, ...,
 acc_x_mean, acc_x_dyn_std, ...,
 gyro_x_mean, ...]
```

---

## 6. LABEL ALIGNMENT

Borg ratings are aligned to feature windows based on ADL task intervals.

### 6.1 ADL Interval Matching (`ml/targets/adl_alignment.py`)

```
ADL Labels: [t_start, t_end, activity, borg]
Feature Windows: [t_center, features...]
    │
    ▼
For each feature window:
    Find ADL interval where: t_start ≤ t_center ≤ t_end
    │
    ▼
If found: assign borg label
If not found: borg = NaN (unlabeled window)
```

### 6.2 Time Tolerance

- Using `merge_asof` with 5-second tolerance
- Handles minor timestamp misalignments between sensors

### 6.3 Alignment Result

| Metric | Value |
|--------|-------|
| Total windows | ~3000+ |
| Labeled windows | 1199 |
| Unlabeled windows | ~1800+ (rest periods, transitions) |

---

## 7. FEATURE SANITIZATION

Remove metadata columns and handle missing values.

### 7.1 Metadata Removal (`ml/features/sanitise.py`)

**Excluded patterns:**
- `t_start`, `t_end` (temporal information → leakage)
- `window_id`, `*_idx` (identifiers → leakage)
- `subject` (used only for grouping, not as feature)
- `activity`, `modality` (categorical metadata)

**Kept:**
- `t_center` (for alignment only, not used as feature)
- All numeric feature columns

### 7.2 NaN Handling

```
For each feature column:
    If NaN% > 50%: DROP column (too sparse)
    If NaN% ≤ 50%: IMPUTE with median
```

- HRV features have ~16% NaN (windows with <5 detected beats)
- Median imputation preserves distribution

---

## 8. MULTI-SUBJECT COMBINATION

Data from multiple subjects is combined for training.

### 8.1 Subject Data

| Subject ID | Description | Labeled Windows |
|------------|-------------|-----------------|
| sim_elderly3 | Elderly participant | 429 |
| sim_healthy3 | Healthy participant | 358 |
| sim_severe3 | Participant with condition | 412 |
| **Total** | | **1199** |

### 8.2 Combination Process

```
Subject 1 aligned data ─┐
Subject 2 aligned data ─┼─► pd.concat() ─► multisub_aligned_10.0s.csv
Subject 3 aligned data ─┘
```

Each row retains `subject` column for leave-one-subject-out validation.

---

## 9. FEATURE MATRIX SUMMARY

### 9.1 Final Feature Counts

| Category | Features | Example |
|----------|----------|---------|
| PPG Green | ~40 | ppg_green_mean, ppg_green_rmssd |
| PPG Infrared | ~40 | ppg_infra_mean, ppg_infra_hr_mean |
| PPG Red | ~40 | ppg_red_std, ppg_red_sdnn |
| EDA | ~20 | eda_mean, eda_slope, scr_count |
| IMU (per axis) | ~150 | acc_x_dyn_entropy, gyro_z_std |
| **Total** | **~300** | |

### 9.2 HRV Features Available

| Feature | Wavelength | Description |
|---------|------------|-------------|
| hr_mean | green, infra, red | Mean heart rate |
| hr_min | green, infra, red | Minimum heart rate |
| hr_max | green, infra, red | Maximum heart rate |
| mean_ibi | green, infra, red | Mean inter-beat interval |
| sdnn | green, infra, red | HRV: std of IBI |
| rmssd | green, infra, red | HRV: root mean square of successive differences |

---

## 10. VALIDATION APPROACH

### 10.1 Leave-One-Subject-Out Cross-Validation (LOSO CV)

```
For each subject s in {elderly3, healthy3, severe3}:
    Train on: all subjects except s
    Test on: subject s only
    Record: R², MAE
    
Report: Mean ± Std across folds
```

**Why LOSO?**
- Tests generalization to unseen individuals
- Prevents subject-specific pattern leakage
- More realistic for deployment scenario

### 10.2 Correlation Validation

Before training, verify physiological plausibility:

| Feature | Expected | Observed | Status |
|---------|----------|----------|--------|
| HR vs Borg | Positive | r = +0.43 | ✓ |
| IBI vs Borg | Negative | r = -0.46 | ✓ |
| RMSSD vs Borg | Negative | r = -0.23 | ✓ |
| EDA vs Borg | Positive | r = +0.15 | ✓ |
| ACC magnitude vs Borg | Positive | r = +0.38 | ✓ |

---

## 11. PIPELINE FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAW DATA                                     │
│  PPG (64Hz) │ EDA (4Hz) │ IMU (100Hz) │ Borg Labels (per ADL)       │
└──────┬──────────┬───────────┬──────────────┬────────────────────────┘
       │          │           │              │
       ▼          ▼           ▼              │
┌──────────────────────────────────────┐     │
│           PREPROCESSING              │     │
│  • Bandpass filtering                │     │
│  • Normalization                     │     │
│  • Gravity separation (IMU)          │     │
└──────────────────┬───────────────────┘     │
                   │                         │
                   ▼                         │
┌──────────────────────────────────────┐     │
│            WINDOWING                 │     │
│  • 10-second windows                 │     │
│  • 0% overlap (no leakage)           │     │
│  • Quality filtering                 │     │
└──────────────────┬───────────────────┘     │
                   │                         │
                   ▼                         │
┌──────────────────────────────────────┐     │
│        FEATURE EXTRACTION            │     │
│  • Time-domain statistics            │     │
│  • HRV metrics (HR, IBI, RMSSD)      │     │
│  • EDA decomposition                 │     │
│  • IMU energy/entropy                │     │
└──────────────────┬───────────────────┘     │
                   │                         │
                   ▼                         │
┌──────────────────────────────────────┐     │
│          FEATURE FUSION              │     │
│  • Merge modalities on t_center      │     │
│  • ~300 features per window          │     │
└──────────────────┬───────────────────┘     │
                   │                         │
                   ▼                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LABEL ALIGNMENT                                 │
│  • Match window t_center to ADL interval [t_start, t_end]           │
│  • Assign Borg rating to each window                                │
│  • 1199 labeled windows from 3 subjects                             │
└──────────────────────────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│        FEATURE SANITIZATION          │
│  • Remove metadata (time, IDs)       │
│  • Impute NaN (median)               │
│  • Drop columns with >50% NaN        │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│              OUTPUT                  │
│  X: [1199 × 299] feature matrix      │
│  y: [1199] Borg CR10 labels          │
│  groups: [1199] subject IDs          │
└──────────────────────────────────────┘
```

---

## 12. KEY DESIGN DECISIONS

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Window size | 10 seconds | Standard in HRV literature; captures multiple heartbeats |
| Window overlap | 0% | Prevents temporal leakage between train/test |
| NaN threshold | 50% | Balances data retention vs. feature quality |
| NaN imputation | Median | Robust to outliers, preserves distribution |
| Label alignment | t_center ± 5s | Accounts for sensor synchronization errors |
| Validation | LOSO CV | Tests cross-subject generalization |

---

## 13. FILE OUTPUTS

| File | Location | Contents |
|------|----------|----------|
| Raw features (PPG) | `data/interim/{subject}/effort_estimation_output/ppg_*_features_10.0s.csv` | PPG features per window |
| Raw features (HRV) | `data/interim/{subject}/effort_estimation_output/ppg_*_hrv_features_10.0s.csv` | HRV features per window |
| Raw features (EDA) | `data/interim/{subject}/effort_estimation_output/eda_*_features_10.0s.csv` | EDA features per window |
| Raw features (IMU) | `data/interim/{subject}/effort_estimation_output/imu_features_10.0s.csv` | IMU features per window |
| Fused features | `data/interim/{subject}/effort_estimation_output/fused_features_10.0s.csv` | All modalities merged |
| Aligned features | `data/interim/{subject}/effort_estimation_output/aligned_features_10.0s.csv` | With Borg labels |
| Combined dataset | `data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv` | All subjects combined |

---

## 14. REPRODUCIBILITY

### Configuration Files

**`config/pipeline.yaml`:**
```yaml
windowing:
  window_size_sec: 10.0
  overlap: 0.0  # 0% overlap

fusion:
  modalities:
    - ppg_green
    - ppg_infra
    - ppg_red
    - ppg_green_hrv
    - ppg_infra_hrv
    - ppg_red_hrv
    - eda
    - eda_advanced
    - imu
```

### Running the Pipeline

```bash
# Full pipeline for multiple subjects
python run_multisub_pipeline.py --subjects sim_elderly3 sim_healthy3 sim_severe3

# Individual steps
python run_multisub_pipeline.py --steps preprocess window extract fuse align combine
```

---

## 15. LIMITATIONS

1. **Sample size:** Only 3 subjects → model overfits to individual patterns
2. **Label granularity:** One Borg rating per ADL task (not continuous)
3. **HRV coverage:** ~16% of windows lack HRV (insufficient detected beats)
4. **Sensor synchronization:** ±5s tolerance needed for alignment

---

## 16. SUMMARY TABLE

| Pipeline Stage | Input | Process | Output |
|----------------|-------|---------|--------|
| 1. Acquisition | Sensors | Record at specified Hz | Raw CSV files |
| 2. Preprocessing | Raw signals | Filter, normalize | Cleaned signals |
| 3. Windowing | Cleaned signals | Segment 10s, 0% overlap | Window list |
| 4. Feature Extraction | Windows | Statistical/HRV/energy features | Feature vectors |
| 5. Fusion | Per-modality features | Merge on t_center | Combined features |
| 6. Alignment | Combined + Borg labels | Match to ADL intervals | Labeled dataset |
| 7. Sanitization | Labeled dataset | Remove metadata, impute NaN | Clean feature matrix |
| 8. Combination | Per-subject data | Concatenate | Multi-subject dataset |
| **Final** | | | **1199 × 299 matrix** |
