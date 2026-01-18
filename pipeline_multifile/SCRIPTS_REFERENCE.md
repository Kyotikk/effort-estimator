# Scripts Reference

## Main Orchestrators

### run_pipeline.py
**Purpose**: Single-subject end-to-end pipeline
**Usage**: 
```bash
python run_pipeline.py
```
**What it does**:
1. Loads configuration
2. Preprocesses all modalities
3. Creates windows with overlap
4. Extracts features per modality
5. Fuses all modalities
6. Quality checks
7. Aligns with ADL labels
8. Outputs aligned dataset

**Key Functions**:
- `run_pipeline(config_path)` - Main orchestrator
- Handles multi-IMU loop (bioz + wrist)
- Calls alignment at end

**Output**: `{subject}/effort_estimation_output/aligned_*.csv`

---

### run_multisub_pipeline.py
**Purpose**: Process all 3 subjects and combine datasets
**Usage**:
```bash
python run_multisub_pipeline.py
```
**What it does**:
1. Finds all files for each subject
2. Generates per-subject configs
3. Runs full pipeline for each
4. Combines aligned datasets
5. Outputs merged labeled dataset

**Key Functions**:
- `find_file(subject_path, pattern_parts, exclude_gz)` - Locate data files
- `generate_config(subject)` - Create YAML config per subject
- `run_subject_pipeline(subject)` - Execute pipeline for one subject
- `combine_datasets(subjects, window_length)` - Merge all subjects

**Output**: `/multisub_combined/multisub_aligned_10.0s.csv` (1,188 labeled samples)

---

### train_condition_specific_xgboost.py
**Purpose**: Train separate effort models for each condition
**Usage**:
```bash
python train_condition_specific_xgboost.py
```
**What it does**:
1. Loads combined multi-subject dataset
2. Filters by condition (elderly3, healthy3, severe3)
3. Selects top 100 features per condition
4. Standardizes with condition scaler
5. Trains XGBoost regressor (80-20 split)
6. Evaluates and saves model

**Key Functions**:
- `train_condition_model(condition, df)` - Train one condition model
- `get_drop_columns(df)` - Identify metadata to drop

**Output**: Per condition (3 total):
- `{condition}_model.json` - XGBoost model
- `{condition}_scaler.pkl` - StandardScaler
- `{condition}_features.json` - Top 100 feature names
- `{condition}_metrics.json` - R², MAE, RMSE

**Results**:
- elderly3: R²=0.926, MAE=0.053
- healthy3: R²=0.405, MAE=0.015
- severe3: R²=0.997, MAE=0.026

---

### analyze_condition_models.py
**Purpose**: Performance breakdown by effort level
**Usage**:
```bash
python analyze_condition_models.py
```
**What it does**:
1. Loads trained models
2. Generates predictions on test sets
3. Bins effort into 6 ranges (Very Light to Extreme)
4. Calculates R², MAE, RMSE per bin
5. Shows sample counts per bin
6. Creates visualizations

**Key Functions**:
- `analyze_condition_model(condition)` - Analyze one condition

**Output**: Performance tables and plots showing where models excel/struggle

---

## Preprocessing Modules

### preprocessing/imu.py
**Function**: `preprocess_imu(df, cfg)`
**Inputs**: Raw acceleration data
**Outputs**: Gravity-removed, normalized acceleration
**Key Steps**:
- Gravity removal (HPF 0.3-0.5 Hz)
- Normalization to 'g' units
- Timestamp conversion to Unix seconds

---

### preprocessing/ppg.py
**Function**: `preprocess_ppg(df, cfg)`
**Inputs**: Raw PPG signal
**Outputs**: Filtered PPG signal
**Key Steps**:
- High-pass filter (some channels only)
- Resampling to 8 Hz
- Motion artifact handling

---

### preprocessing/eda.py
**Function**: `preprocess_eda(df, cfg)`
**Inputs**: Raw electrodermal activity
**Outputs**: Smoothed EDA signal
**Key Steps**:
- Savitzky-Golay smoothing
- Outlier detection
- Resampling to 2 Hz

---

### preprocessing/rr.py
**Function**: `preprocess_rr(df, cfg)`
**Inputs**: RR interval data
**Outputs**: Regularized RR intervals
**Key Steps**:
- Physiological validation (300-2000 ms)
- Interpolation
- Resampling to 1 Hz

---

## Feature Extraction

### features/manual_features_imu.py
**Function**: `compute_top_imu_features_from_windows(windows_df, cfg)`
**Output**: 138 IMU features (69 per IMU type)
**Features per axis**: mean, std, RMS, entropy, kurtosis, zero-crossings, etc.

---

### features/ppg_features.py
**Function**: `compute_ppg_features(...)`
**Output**: 50 PPG features
**Metrics**: Heart rate, HRV, signal quality, amplitude

---

## Windowing & Fusion

### windowing/windows.py
**Function**: `create_windows(time_series, window_length, overlap, fs)`
**Purpose**: Segment time series into fixed windows with overlap
**Output**: Window metadata + aggregated measurements

---

### ml/run_fusion.py
**Function**: `main(modality_files, output_path)`
**Purpose**: Merge all modalities time-aligned
**Output**: Fused feature matrix (257 features)

---

## Quality & Alignment

### windowing/feature_quality_check_any.py
**Purpose**: Validate feature coverage and quality
**Output**: Quality reports + visualization plots

---

### ml/targets/run_target_alignment.py
**Function**: `run_alignment()`
**Purpose**: Main alignment orchestrator
**Key Feature**: Window time range filtering
**Output**: Labeled windows with Borg column

---

### ml/targets/adl_alignment.py
**Functions**:
- `parse_adl_intervals(adl_csv)` - Parse ADL events
- `align_windows_to_borg(windows, intervals)` - Match windows to intervals

**Key Feature**: Automatic timestamp format detection + time zone handling

---

## Configuration Files

### config/pipeline.yaml
Master configuration file with:
- Data paths
- Preprocessing parameters
- Windowing configuration
- Feature extraction settings
- Quality check thresholds

---

## Utility Scripts

### feature_sanity_plots.py
**Purpose**: Visualize features for quality assurance

### feature_quality_check.py
**Purpose**: Statistical validation of features

---

## Analysis Scripts

### manual_selected_features.py
**Purpose**: Analyze and validate feature selection

---

## Execution Order

**Full Pipeline**:
1. `run_multisub_pipeline.py` - Process all subjects
2. `train_condition_specific_xgboost.py` - Train models
3. `analyze_condition_models.py` - Evaluate performance

**Single Subject**:
1. `run_pipeline.py` - Full pipeline (or call directly in Python)

**Inference Only**:
1. Load models (see `11_INFERENCE.md`)
2. Call `estimate_effort(features, condition)`

---

## Summary

| Category | Scripts | Purpose |
|----------|---------|---------|
| **Orchestration** | run_*.py | End-to-end pipeline execution |
| **Preprocessing** | preprocessing/*.py | Signal cleaning per modality |
| **Features** | features/*.py | Feature extraction |
| **Windowing** | windowing/*.py | Time segmentation |
| **Fusion** | ml/run_fusion.py | Modality combination |
| **Alignment** | ml/targets/*.py | Label attachment |
| **Training** | train_*.py | Model training |
| **Analysis** | analyze_*.py, *_plots.py | Performance evaluation |
