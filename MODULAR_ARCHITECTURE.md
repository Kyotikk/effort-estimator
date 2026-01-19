# Modular Pipeline Architecture

## Overview

The effort estimation pipeline is now organized into clean, modular components. Each pipeline step is separated into its own callable function module for better code organization and reusability.

## Pipeline Structure

```
run_multisub_pipeline.py
  └── Orchestrates pipeline steps
      ├── Step 1: Run subject pipelines (run_pipeline.py)
      ├── Step 2: Combine datasets (ml.combine)
      ├── Step 3: Quality check (ml.quality_check)
      └── Step 4: Feature selection (ml.feature_selection)
```

## Modular Components

### 1. **ml/fusion.py** - Feature Fusion
Combines features from multiple modalities into a single dataframe.

**Key Functions:**
```python
fuse_modalities(feature_paths, window_length, tolerance_s=2.0)
    # Args: dict of {modality: path}, window_length, alignment tolerance
    # Returns: Fused DataFrame with all modalities
    
save_fused_data(df, output_path, window_length)
    # Saves fused data to CSV
```

**Usage:**
```python
from ml.fusion import fuse_modalities

feature_paths = {
    'imu_bioz': 'path/to/imu_bioz_features.csv',
    'ppg_green': 'path/to/ppg_green_features.csv',
}
fused = fuse_modalities(feature_paths, window_length=10.0)
```

---

### 2. **ml/alignment.py** - Temporal Alignment
Aligns fused features with target labels using time-based matching.

**Key Functions:**
```python
align_fused_data(fused_csv, target_csv, tolerance_s=2.0)
    # Args: path to fused features, path to targets, time tolerance
    # Returns: Aligned DataFrame with matching labels
    
save_aligned_data(df, output_path, window_length)
    # Saves aligned data to CSV
```

**Usage:**
```python
from ml.alignment import align_fused_data

aligned = align_fused_data(
    'fused_10.0s.csv',
    'targets.csv',
    tolerance_s=2.0
)
```

---

### 3. **ml/quality_check.py** - Data Validation
Performs comprehensive quality checks on data integrity.

**Key Functions:**
```python
check_data_quality(df, features_only=False)
    # Returns: dict with QC results (missing values, zero variance, correlations)
    
print_qc_results(qc_results)
    # Pretty-prints QC results
```

**Usage:**
```python
from ml.quality_check import check_data_quality, print_qc_results

qc = check_data_quality(df, features_only=True)
print_qc_results(qc)
```

---

### 4. **ml/feature_extraction.py** - Feature Extraction Orchestration
Orchestrates feature extraction for all modalities (placeholder for consistency).

**Key Functions:**
```python
extract_all_features(config, output_dir)
    # Verifies feature extraction is complete
    
verify_features_extracted(output_dir, modalities, window_length)
    # Returns: dict of {modality: exists (bool)}
```

---

### 5. **ml/feature_selection.py** - Feature Selection & Pruning
Selects optimal features using correlation-based pruning.

**Key Functions:**
```python
select_features(df, target_col='borg', corr_threshold=0.90, top_n=100)
    # Args: DataFrame, target column name, correlation threshold, top N features
    # Returns: (selected_cols, X_selected, y)
    
save_feature_selection_outputs(output_dir, df, selected_cols, window_length)
    # Saves selected features and PCA analysis
```

**Usage:**
```python
from ml.feature_selection import select_features

cols, X, y = select_features(
    df_combined,
    target_col='borg',
    corr_threshold=0.90,
    top_n=100
)
```

---

## Main Pipeline Script

**run_multisub_pipeline.py** - Clean orchestration of all steps

```python
# Step 1: Run individual subject pipelines
# (calls run_pipeline.py for each subject)

# Step 2: Combine datasets from all subjects
combined = combine_datasets(subjects, window_length)

# Step 3: Quality check
qc_results = check_data_quality(combined, features_only=True)

# Step 4: Feature selection
pruned_cols, X, y = select_features(combined, ...)
save_feature_selection_outputs(output_path, combined, pruned_cols, window_length)

# Step 5: Summary statistics
```

## Complete Pipeline Flow

```
┌─────────────────────────────────────────┐
│ run_multisub_pipeline.py                │
│ (Main orchestrator)                     │
└────────────┬────────────────────────────┘
             │
     ┌───────┴──────────┬──────────────────┐
     │                  │                  │
     ▼                  ▼                  ▼
 sim_elderly3      sim_healthy3      sim_severe3
 run_pipeline.py   run_pipeline.py   run_pipeline.py
 (each subject)    (each subject)    (each subject)
     │                  │                  │
     └───────┬──────────┴──────────────────┘
             │
             ▼
      combine_datasets()
      (concatenate all subjects)
             │
             ▼
      check_data_quality()  ◄─── ml.quality_check
      (validate data)
             │
             ▼
      select_features()     ◄─── ml.feature_selection
      (correlation pruning)
             │
             ▼
      Save outputs:
      - multisub_aligned_10.0s.csv
      - features_selected_pruned.csv
      - qc_10.0s/ (QC analysis)
             │
             ▼
      train_multisub_xgboost.py
      (train final model)
```

## Running the Pipeline

### Full pipeline (preprocessing → training):
```bash
source .venv/bin/activate
python run_multisub_pipeline.py
python train_multisub_xgboost.py
```

### Skip subject pipelines (use cached results):
```bash
python run_multisub_pipeline.py --skip-pipeline
```

### Specific subjects only:
```bash
python run_multisub_pipeline.py --subjects sim_elderly3 sim_healthy3
```

## Data Flow

```
Raw Data (per subject)
  ↓
[run_pipeline.py] - Preprocessing, windowing, feature extraction
  ↓
Features per modality (imu_bioz, imu_wrist, ppg_green, ppg_infra, ppg_red, eda, rr)
  ↓
[ml.fusion.fuse_modalities] - Combine all modalities
  ↓
Fused features (fused_10.0s.csv)
  ↓
[ml.alignment.align_fused_data] - Align with targets
  ↓
Aligned features with labels (fused_aligned_10.0s.csv)
  ↓
[combine_datasets] - Stack all subjects
  ↓
Combined multi-subject data (multisub_aligned_10.0s.csv)
  ↓
[ml.quality_check.check_data_quality] - Validate
  ↓
[ml.feature_selection.select_features] - Correlation-based pruning
  ↓
Selected features (50 features from 188)
  ↓
[train_multisub_xgboost.py] - Train model + plots
```

## Quality Assurance

Each modular component:
- ✓ Has clear input/output contracts
- ✓ Includes docstrings with usage examples
- ✓ Returns structured data (dict/DataFrame)
- ✓ Can be tested independently
- ✓ Follows consistent naming conventions

## Key Features

1. **Modular**: Each step is isolated and callable
2. **Readable**: Main script is clean and easy to follow
3. **Reusable**: Functions can be imported and used elsewhere
4. **Documented**: Clear docstrings and usage examples
5. **Tested**: All imports and functions verified working
6. **Efficient**: No redundant code or data processing

## File Locations

```
ml/
  ├── alignment.py              # Temporal alignment
  ├── fusion.py                 # Feature fusion
  ├── quality_check.py          # Data validation
  ├── feature_extraction.py     # Feature extraction orchestration
  ├── feature_selection.py      # Feature selection
  ├── feature_selection_and_qc.py  # (existing QC functions)
  ├── train_and_save_all.py     # Single-subject training
  ├── targets/                  # (existing target utilities)
  ├── fusion/                   # (existing fusion utilities)
  ├── features/                 # (existing feature utilities)
  └── [other modules]

run_multisub_pipeline.py         # Main multi-subject orchestrator
train_multisub_xgboost.py        # Multi-subject trainer + plots
run_pipeline.py                  # Single-subject pipeline
```

## Next Steps

To train the final model after running the pipeline:

```bash
python train_multisub_xgboost.py
```

This will:
1. Load the selected 50 features
2. Train XGBoost with regularization
3. Perform random 80/20 train-test split
4. Generate 7 diagnostic plots
5. Save model and metrics
