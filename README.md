# Effort Estimator

Research-grade ML pipeline for physiological effort estimation in neuropatients during Activities of Daily Living (ADLs).

## Overview

This pipeline estimates **physiological effort** (autonomic load) using wearable sensor data. The effort labels are derived from ECG-based RMSSD (parasympathetic marker), while input features come from PPG (heart rate level), IMU (movement), and EDA (sympathetic arousal).

**Key principle**: HRV metrics (RMSSD, SDNN, etc.) are ONLY used to compute effort labels, never as input features.

## Project Structure

```
effort-estimator/
├── ecg/                      # ECG preprocessing for label computation
│   ├── preprocessing.py      # R-peak detection, RR intervals
│   └── __init__.py
├── features/                 # Feature extraction (NO HRV)
│   ├── ppg_features.py       # Heart rate level features
│   ├── imu_features.py       # Movement/acceleration features
│   ├── eda_features.py       # Electrodermal activity features
│   └── __init__.py
├── ml/                       # Machine learning modules
│   ├── labels/               # Label computation
│   │   ├── rmssd_label.py    # RMSSD-based effort labels
│   │   └── __init__.py
│   ├── models/               # Training scripts (to be implemented)
│   └── __init__.py
├── scripts/                  # CLI tools
│   └── preprocess_ecg.py     # ECG preprocessing CLI
├── configs/                  # Pipeline configurations
│   └── pipeline_sim_elderly3.yaml
├── data/                     # Output data
│   ├── labels/               # RMSSD labels per session
│   ├── features/             # Extracted features
│   └── logs/                 # Processing logs
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Data Structure

The pipeline expects data organized by simulation and cohort:

```
/Users/pascalschlegel/data/interim/parsingsimX/
└── sim_<cohort>X/            # cohort: healthy, elderly, severe
    ├── vivalnk_vv330_ecg/    # ECG for labels ONLY
    │   └── data_1.csv.gz
    ├── corsano_wrist_ppg2_green_6/
    │   └── 2025-12-04.csv.gz
    ├── corsano_wrist_acc/
    │   └── 2025-12-04.csv.gz
    ├── corsano_bioz_emography/
    │   └── 2025-12-04.csv.gz
    └── scai_app/
        └── ADLs_1.csv        # Activity timeline
```

## Usage

### Stage 1: ECG Preprocessing and Label Generation

Process ECG to generate RMSSD-based effort labels:

```bash
python scripts/preprocess_ecg.py \
  --ecg-file /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/vivalnk_vv330_ecg/data_1.csv.gz \
  --output data/labels/parsingsim3_sim_elderly3_rmssd_labels.csv \
  --session-id parsingsim3_sim_elderly3 \
  --adl-file /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/scai_app/ADLs_1.csv \
  --verbose
```

**Output**: CSV with columns:
- `session_id`, `adl_id`
- `rmssd`, `ln_rmssd` (effort markers)
- `n_beats`, `mean_rr`, `std_rr`
- `start_time`, `end_time`, `duration_sec`

### Stage 2: Feature Extraction (Coming Soon)

Extract features from PPG, IMU, and EDA sensors (NO HRV features):

```bash
python scripts/extract_features.py \
  --config configs/pipeline_sim_elderly3.yaml \
  --output data/features/parsingsim3_sim_elderly3_features.csv
```

**Features extracted:**
- **PPG**: hr_mean, hr_max, hr_min, hr_std, hr_slope
- **IMU**: acc_mag_mean, acc_mag_std, steps_sum, cadence_mean, movement_duration
- **EDA**: eda_mean, eda_std, eda_slope, scr_count, scr_mean_amplitude

### Stage 3: Model Training (Coming Soon)

Train regression models to predict effort:

```bash
python scripts/train_model.py \
  --config configs/pipeline_sim_elderly3.yaml \
  --labels data/labels/parsingsim3_sim_elderly3_rmssd_labels.csv \
  --features data/features/parsingsim3_sim_elderly3_features.csv \
  --output data/models/parsingsim3_sim_elderly3/
```

**Models:**
- Ridge regression (baseline)
- ElasticNet
- XGBoost

**Metrics:**
- MAE, RMSE
- Spearman correlation (predicted vs true effort)

## Configuration

Each session has a YAML config file (see `configs/pipeline_sim_elderly3.yaml`):

- **Paths**: Data roots, sensor files, output locations
- **Labels**: RMSSD computation method (delta_ln_rmssd, recovery_slope)
- **Features**: Which sensors and features to extract
- **Model**: Training parameters, models, metrics

## Critical Design Principles

1. **NO HRV as input features**: RMSSD, SDNN, pNN50, LF/HF, and all RR-variability metrics are FORBIDDEN as model inputs. They are ONLY used to compute the effort label.

2. **Leakage-safe features**: All input features must be computable without knowledge of future RR intervals or HRV.

3. **Physiologically interpretable**: Features reflect cardiovascular load (HR level), movement (IMU), and sympathetic arousal (EDA).

4. **Robust for neuropatients**: ADL data with irregular movement, artifacts, and comorbidities.

## Example Workflow

```bash
# 1. Process ECG → RMSSD labels
python scripts/preprocess_ecg.py \
  --ecg-file /path/to/ecg/data_1.csv.gz \
  --output data/labels/session_labels.csv \
  --session-id my_session \
  --adl-file /path/to/ADLs_1.csv

# 2. Extract features (when implemented)
python scripts/extract_features.py \
  --config configs/my_config.yaml

# 3. Train model (when implemented)
python scripts/train_model.py \
  --config configs/my_config.yaml
```

## Questions?

If any file formats or column layouts are unclear, please ask before proceeding. The pipeline assumes specific data structures that may need adaptation.

## License

Research use only.
