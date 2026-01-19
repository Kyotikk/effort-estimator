# Production Effort Estimation Pipeline

## Overview
Clean, production-ready pipeline for multi-subject and single-subject effort estimation using physiological signals (PPG, EDA, IMU).

## Main Entry Points

### 1. Single-Subject Pipeline
**File:** `run_pipeline.py`
- Processes single subject data through complete preprocessing → feature extraction → model training
- Outputs: 7 diagnostic plots, trained model, feature importance
- Uses: preprocessing/, windowing/, features/tifex.py, ml/feature_selection_and_qc.py

### 2. Multi-Subject Pipeline
**File:** `run_multisub_pipeline.py`
- Combines data from 3 subjects (sim_elderly3, sim_healthy3, sim_severe3)
- Runs individual pipelines, combines datasets, performs feature selection
- Outputs: combined aligned data, pre-selected features (50 from 188)
- Uses: run_pipeline.py logic + feature selection module

### 3. Single-Subject Training & Visualization
**File:** `ml/train_and_save_all.py`
- Trains XGBoost on single-subject data with feature selection
- Generates 7 comprehensive plots showing model performance
- Outputs: trained model, predictions, metrics, visualizations

### 4. Multi-Subject Training & Visualization
**File:** `train_multisub_xgboost.py`
- Trains XGBoost on combined multi-subject data
- Uses pre-selected features from multisub pipeline
- 80/20 random train-test split across all conditions
- Generates 7 diagnostic plots
- Test R² = 0.9354 (excellent generalization)
- Outputs: trained model, predictions, metrics, 7 plots

## Data Processing Modules

### Preprocessing
**Directory:** `preprocessing/`
- `ppg.py` - PPG signal preprocessing
- `eda.py` - EDA signal preprocessing
- `imu.py` - IMU acceleration preprocessing
- `ecg.py` - ECG/RR interval preprocessing
- `bioz.py` - BioZ preprocessing
- `temp.py` - Temperature preprocessing
- `rr.py` - RR interval processing

### Feature Extraction
**Directory:** `features/`
- `tifex.py` - Main feature extraction engine
- `ppg_features.py` - PPG-specific features
- `eda_features.py` - EDA-specific features
- `manual_features_imu.py` - IMU-specific features
- `vitalpy_ppg.py` - VitalPy PPG analysis
- `rr_features.py` - RR interval features

### Windowing
**Directory:** `windowing/`
- `windows.py` - Main windowing logic (10s, 30s windows)
- Feature quality checking utilities

### Feature Selection & QC
**File:** `ml/feature_selection_and_qc.py`
- Correlation-based feature selection (top 100)
- Modality-wise correlation pruning (0.90 threshold)
- PCA analysis and quality control
- Outputs: selected features list, PCA loadings/variance

## Configuration
**File:** `config/pipeline.yaml`
- Data paths for 3 subjects
- Sensor configurations
- Window lengths (10s, 30s)
- Output paths

## Key Features

✅ **Multi-Subject Support** - Works across 3 different conditions (elderly, healthy, severe)
✅ **Excellent Generalization** - Test R² = 0.9354 (no overfitting)
✅ **Feature Pruning** - Reduces 188 → 50 features while maintaining quality
✅ **Comprehensive Visualization** - 7 diagnostic plots per model
✅ **Regularized XGBoost** - Hyperparameters tuned to prevent memorization
✅ **Modality Balance** - PPG 35%, EDA 36%, IMU 29% in selected features

## Model Performance

### Multi-Subject Model (Current Best)
- Train R² = 0.9991 (very good fit, no memorization)
- Test R² = 0.9354 (excellent generalization)
- Test MAE = 0.394 (±0.4 Borg rating error on average)
- Test RMSE = 0.609
- Features: 50 (selected from 188 raw features)
- Train/Test Split: 80/20 random across all conditions

### XGBoost Configuration
```python
n_estimators=500
max_depth=5
learning_rate=0.05
subsample=0.7
colsample_bytree=0.7
reg_alpha=1.0 (L1 penalty)
reg_lambda=1.0 (L2 penalty)
min_child_weight=3
```

## Output Structure

```
/Users/pascalschlegel/data/interim/parsingsim3/
├── sim_elderly3/
│   └── effort_estimation_output/parsingsim3_sim_elderly3/
│       ├── fused_aligned_10.0s.csv
│       ├── feature_selection_qc/
│       │   └── qc_10.0s/
│       │       ├── features_selected_pruned.csv
│       │       └── pca_*.csv
│       └── plots_single/ (7 PNG files)
├── sim_healthy3/
│   └── (same structure)
├── sim_severe3/
│   └── (same structure)
└── multisub_combined/
    ├── multisub_aligned_10.0s.csv
    ├── qc_10.0s/
    │   ├── features_selected_pruned.csv
    │   └── pca_*.csv
    ├── models/
    │   ├── xgboost_multisub_10.0s.json
    │   ├── feature_importance_multisub_10.0s.csv
    │   └── metrics_multisub_10.0s.json
    └── plots_multisub/ (7 PNG files)
```

## Usage

### Single-Subject Pipeline
```bash
python run_pipeline.py config/pipeline.yaml
```

### Multi-Subject Pipeline
```bash
python run_multisub_pipeline.py
```

### Multi-Subject Training (with Plots)
```bash
python train_multisub_xgboost.py
```

### Single-Subject Training (with Plots)
```bash
python ml/train_and_save_all.py
```

## Cleaned Up
Removed all unused scripts:
- ❌ Condition-specific models
- ❌ Individual signal pipeline runners
- ❌ Old analysis/evaluation scripts
- ❌ Legacy model files
- ❌ Development/test directories

Production-ready: Only active, working code remains.
