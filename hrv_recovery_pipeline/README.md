# HRV Recovery Estimation Pipeline

Clean, modular approach to effort estimation using **HRV (Heart Rate Variability) recovery** as the training target instead of Borg ratings.

## Key Philosophy

- **Target**: RMSSD recovery after effort (physiologically grounded)
- **Borg**: Secondary validation only, NOT the training target
- **Data**: One row per effort bout with features X from IMU/EDA/PPG during effort and label y = HRV recovery post-effort
- **Focus**: Correctness first, minimal feature engineering, interpretable models

## Architecture: 6 Modules

### 1. **Module 1: IBI Extraction** (`module1_ibi.py`)
- Input: Green PPG signal (preprocessed)
- Detect heartbeat peaks robustly (scipy.signal.find_peaks)
- Compute inter-beat intervals (IBIs = beat-to-beat timings)
- Apply validity filters: reject implausible IBIs (<0.3s or >2.0s), erratic jumps (ratio > 1.5)
- Output: `ibi_timeseries.csv` [t, ibi_sec]

### 2. **Module 2: Windowed RMSSD** (`module2_rmssd.py`)
- Input: IBI timeseries
- Sliding windows (default 60s, step 10s)
- For each window: compute RMSSD = √(mean(Δ IBI²))
- Require ≥10 IBIs per window; otherwise NaN
- Output: `rmssd_windows.csv` [t_start, t_center, t_end, rmssd]

### 3. **Module 3: Effort Bouts** (`module3_bouts.py`)
- **Prefer ADL intervals** (from task/activity logs)
- **Fallback**: detect from IMU intensity thresholding
  - Compute intensity metric (e.g., acc_mag_rms)
  - Threshold at 60th percentile
  - Merge contiguous active windows with gap tolerance
  - Enforce ≥20s minimum duration
- Output: `effort_bouts.csv` [bout_id, t_start, t_end, task_name]

### 4. **Module 4: HRV Recovery Labels** (`module4_labels.py`)
- Input: RMSSD windows, effort bouts
- For each bout:
  - **RMSSD_end**: mean RMSSD in [t_end - 30s, t_end]
  - **RMSSD_60**: mean RMSSD in [t_end + 30s, t_end + 90s]
  - **Label** (option A, default):
    - `delta_rmssd = RMSSD_60 - RMSSD_end` (higher = better recovery)
  - **Label** (option B):
    - `recovery_slope`: linear fit of RMSSD over recovery interval
- QC: require ≥2 valid RMSSD windows in recovery; otherwise label = NaN
- Output: `bout_labels.csv` [bout_id, rmssd_end, rmssd_60, delta_rmssd, recovery_slope, qc_ok, note]

### 5. **Module 5: Feature Extraction** (`module5_features.py`)
- Aggregate per-window features over effort bout intervals
- **IMU features**: acc_mag_mean, acc_mag_rms, acc_mag_std, smoothness, etc.
- **EDA features**: scr_count, scr_rate, scl_mean, scl_slope (from preprocessed windows)
- **PPG/IBI features**:
  - `hr_mean`, `hr_std` (from IBI mean/std)
  - `rmssd_during_effort`: RMSSD during the effort interval
- Output: `model_table.csv` [bout_id, session_id, task_name, imu_*, eda_*, ppg_*, delta_rmssd, recovery_slope, qc_ok]

### 6. **Module 6: Training & Evaluation** (`module6_training.py`)
- Split data by participant/session (train/test)
- **Baseline models**:
  - ElasticNet (linear)
  - XGBoost (gradient boosting)
- **Metrics**:
  - MAE (mean absolute error)
  - R² (coefficient of determination)
  - Pearson r (correlation)
- **Output**: model artifacts, metrics report
- Borg correlation computed for secondary validation only

## Quick Start

### 1. Prepare Config

Edit `config/hrv_pipeline_example.yaml`:
- Set `output_dir`, `session_id`
- Point `input_paths` to your preprocessed data:
  - `ppg_green`: preprocessed green PPG CSV
  - `adl`: ADL intervals CSV (columns: t_start, t_end, task_name, optional borg)
  - `imu_features`: (optional) per-window IMU features
  - `eda_features`: (optional) per-window EDA features

### 2. Run Pipeline

```bash
cd /Users/pascalschlegel/effort-estimator
.venv/bin/python run_hrv_pipeline.py config/hrv_pipeline_example.yaml --output-dir ./output/elderly3
```

### 3. Check Outputs

In output directory:
- `ibi_timeseries.csv`: Extracted IBIs
- `rmssd_windows.csv`: Windowed RMSSD
- `effort_bouts.csv`: Detected/parsed effort bouts
- `bout_labels.csv`: HRV recovery labels per bout (with QC)
- `model_table.csv`: Complete feature + label table (ready for ML)
- `training_summary_delta_rmssd.txt`: Model metrics

## Configuration Parameters

### IBI Extraction
- `fs`: Sampling frequency of PPG signal (Hz)
- `distance_ms`: Minimum distance between peaks (ms) - adjust for expected HR
- `min_ibi_sec`, `max_ibi_sec`: Plausible IBI range (seconds)
- `max_ibi_ratio`: Reject IBIs that jump >1.5x from neighbors

### RMSSD Windowing
- `window_len_sec`: Window length (60s default, try 30s for faster recovery)
- `step_sec`: Overlap (10s step → 50s overlap with 60s window)
- `min_beats`: Minimum IBIs per window (10 default)

### Effort Bouts
- ADL parsing (if CSV exists): automatic
- IMU fallback: threshold at `threshold_percentile` on `intensity_col`
- `min_duration_sec`: Discard bouts <20s
- `merge_gap_sec`: Merge active windows separated by <5s

### HRV Recovery Labels
- `label_method`: "delta" (Δ RMSSD, default) or "slope" (recovery slope over time)
- `recovery_start_sec`, `recovery_end_sec`: Recovery observation window (30–90s post-effort)
- `min_recovery_windows`: ≥2 valid RMSSD windows in recovery (QC threshold)

## Expected Data Flow

```
PPG → [Module 1] → IBI timeseries
                    ↓
                 [Module 2] → RMSSD windows
                    ↓
ADL/IMU ← [Module 3] → Effort bouts
            ↓
         [Module 4] → HRV recovery labels
            ↓
IMU/EDA features ← [Module 5] → Model table
            ↓
         [Module 6] → Trained model + metrics
```

## Quality Checks

- **IBI validity**: Range filter + ratio filter for sudden jumps
- **RMSSD per window**: Require ≥10 IBIs; else NaN
- **HRV recovery QC**: Require ≥2 valid RMSSD windows during recovery; drop bout if not met
- **Feature completeness**: Log % NaN per feature; fill with median if needed

## Next Steps

1. Test on one participant (elderly3) to validate pipeline
2. Extend to multi-participant (parallel processing per subject)
3. Compare HRV recovery labels vs Borg: correlation analysis
4. Feature selection: stability, SHAP importance
5. Cross-validation and generalization metrics

---

**Contact**: Pascal Schlegel  
**Branch**: `hrv-recovery-labels`
