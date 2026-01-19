#!/usr/bin/env python3
"""
MODULAR PIPELINE ARCHITECTURE - QUICK START
============================================

This file explains the clean 7-phase modular pipeline structure.
Read this first before diving into the code.

DIRECTORY LAYOUT (modular-refactor branch)
==========================================

pipeline/
├── 01_preprocessing/         ← Phase 1: Load & clean raw signals
│   ├── __init__.py           ← Exports: preprocess_imu, preprocess_ppg, preprocess_eda, preprocess_rr
│   └── preprocessing.py      ← Implementation
│
├── 02_windowing/             ← Phase 2: Create windows + QC
│   ├── __init__.py           ← Exports: create_windows, quality_check_windows
│   └── windowing.py          ← Implementation (quality_check_windows moved here from old windowing/)
│
├── 03_features/              ← Phase 3: Extract features (per modality)
│   ├── __init__.py
│   ├── imu_features.py       ← extract_imu_features()
│   ├── ppg_features.py       ← extract_ppg_features()
│   ├── rr_features.py        ← extract_rr_features()
│   └── eda_features.py       ← extract_eda_features()
│
├── 04_fusion/                ← Phase 4: Combine modalities
│   ├── __init__.py           ← Exports: fuse_modalities
│   └── fusion.py             ← Implementation
│
├── 05_alignment/             ← Phase 5: Add target labels
│   ├── __init__.py           ← Exports: align_with_targets
│   └── alignment.py          ← Implementation
│
├── 06_selection/             ← Phase 6: Select top features
│   ├── __init__.py           ← Exports: select_features
│   └── selection.py          ← Implementation (PCA-based ranking)
│
├── 07_training/              ← Phase 7: Train model
│   ├── __init__.py           ← Exports: train_model, evaluate_model
│   └── training.py           ← Implementation (XGBoost)
│
├── run_clean_pipeline.py     ← ORCHESTRATOR: Runs all 7 phases
└── README.md                 ← Detailed documentation


WHAT CHANGED FROM OLD STRUCTURE
================================

OLD (scattered, hard to follow):
  preprocessing/      → isolated modules
  windowing/          → mixed with feature quality check (WRONG!)
  features/           → complex manual_features_imu.py
  ml/                 → fusion.py, alignment.py, selection.py scattered
  No clear orchestration

NEW (clean, modular):
  ✓ Numbered phases (01-07) show chronological order
  ✓ Each phase in its own directory
  ✓ Each phase has __init__.py that exports functions
  ✓ Feature quality check moved to 02_windowing (correct place!)
  ✓ Each modality feature extractor separated (imu, ppg, rr, eda)
  ✓ Single orchestrator shows complete flow
  ✓ All functions are CALLABLE (not scripts with argparse)


CALLABLE FUNCTIONS (KEY DIFFERENCE)
===================================

OLD: Scripts with command-line args
  python windowing/windows.py --input data.csv --out windows.csv --fs 125 ...

NEW: Import and call functions
  from pipeline.02_windowing import create_windows
  windows_df = create_windows(df, fs=125, win_sec=10, overlap=0.5)

This makes it:
  - Easier to chain phases together
  - Easier to test individual components
  - Easier to build on top of
  - No scattered subprocess calls


EXAMPLE: Process Single Modality (Old Way vs New Way)
====================================================

OLD (scattered calls):
  subprocess.run(["python", "preprocessing/imu.py", ...])
  subprocess.run(["python", "windowing/windows.py", ...])
  subprocess.run(["python", "features/manual_features_imu.py", ...])

NEW (clean functions):
  from pipeline.01_preprocessing import preprocess_imu
  from pipeline.02_windowing import create_windows
  from pipeline.03_features import extract_imu_features
  
  imu_df = preprocess_imu("path/to/raw_imu.csv.gz")
  windows_df = create_windows(imu_df, fs=125.0, win_sec=10.0, overlap=0.5)
  features_df = extract_imu_features(imu_df, windows_df)


RUNNING THE PIPELINE
====================

Option 1: Run everything (all 7 phases)
  python pipeline/run_clean_pipeline.py --config config/pipeline.yaml

Option 2: Run specific subject
  python pipeline/run_clean_pipeline.py --config config/pipeline.yaml --subject sim_elderly3

Option 3: Use phases in your own script
  from pipeline.01_preprocessing import preprocess_imu, preprocess_ppg
  from pipeline.02_windowing import create_windows
  from pipeline.03_features import extract_imu_features
  
  # Your custom pipeline logic here
  imu_df = preprocess_imu("...")
  windows = create_windows(imu_df, ...)
  features = extract_imu_features(imu_df, windows)


FEATURES OF EACH PHASE
=======================

Phase 1: PREPROCESSING (01_preprocessing/)
  Functions: preprocess_imu, preprocess_ppg, preprocess_eda, preprocess_rr
  Input:  Raw CSV/CSV.gz files (time, accX, accY, accZ) or (time, value)
  Output: DataFrame with t_unix, t_sec, cleaned signal columns
  What:   Loads, normalizes time, applies filters, computes dynamic components

Phase 2: WINDOWING (02_windowing/)
  Functions: create_windows, quality_check_windows
  Input:  Preprocessed signal DataFrame
  Output: Windows metadata (start_idx, end_idx, t_start, t_center, t_end)
  What:   Creates sliding windows + performs QC (removes high-NaN/low-var features)

Phase 3: FEATURES (03_features/)
  Functions: extract_imu_features, extract_ppg_features, extract_rr_features, extract_eda_features
  Input:  Preprocessed signal + windows
  Output: One row per window with ~20-40 features per modality
  What:   Computes IMU entropy/fractal/complexity features; basic stats for PPG/RR/EDA

Phase 4: FUSION (04_fusion/)
  Function: fuse_modalities
  Input:  Feature DataFrames from 7 modalities
  Output: Single DataFrame with all features aligned by time window
  What:   Merges all modality features on t_start column

Phase 5: ALIGNMENT (05_alignment/)
  Function: align_with_targets
  Input:  Fused features + ADL ground truth labels
  Output: Same features + effort column with labels
  What:   Matches windows to ADL time ranges, assigns Borg effort labels

Phase 6: SELECTION (06_selection/)
  Function: select_features
  Input:  All fused features + labels
  Output: Top 50 selected features (+ PCA analysis)
  What:   Removes constant/correlated features, ranks by PCA energy, selects top N

Phase 7: TRAINING (07_training/)
  Functions: train_model, evaluate_model
  Input:  Selected features + effort labels
  Output: Trained XGBoost model + metrics (R², RMSE, MAE)
  What:   Splits data, standardizes, trains model, evaluates on test set


TESTING INDIVIDUAL PHASES
=========================

import pandas as pd
from pipeline.01_preprocessing import preprocess_imu
from pipeline.02_windowing import create_windows
from pipeline.03_features import extract_imu_features

# Load and preprocess IMU
imu_df = preprocess_imu("path/to/imu.csv.gz")
print(f"Preprocessed: {imu_df.shape}")  # Should see t_unix, acc_x, etc.

# Create windows
windows_df = create_windows(imu_df, fs=125, win_sec=10, overlap=0.5)
print(f"Windows: {windows_df.shape}")  # Should see start_idx, end_idx, t_start, etc.

# Extract features
features_df = extract_imu_features(imu_df, windows_df)
print(f"Features: {features_df.shape}")  # Should see IMU feature columns


MIGRATION NOTES
===============

Old code still exists in:
  - preprocessing/ (original scripts)
  - windowing/ (original scripts + quality_check)
  - features/ (original manual_features_imu.py, etc.)
  - ml/ (original fusion.py, alignment.py, selection.py)

But pipeline/ is the NEW source of truth.
Once tested and validated, old files can be archived/deleted.

Current status on modular-refactor branch:
  ✓ Clean pipeline structure created
  ✓ All 7 phases implemented as callable functions
  ✓ Orchestrator (run_clean_pipeline.py) ready to test
  ⏳ Next: Run and validate outputs match original pipeline


GIT BRANCHES
============

pascal_update          → Production (untouched, working)
modular-refactor       → Experimental (you are here)
pipeline-backup-v1     → Backup
effort-estimator-ORIGINAL-WORKING → Full filesystem backup


YOUR NEXT STEPS
===============

1. Test the orchestrator:
   python pipeline/run_clean_pipeline.py --config config/pipeline.yaml --subject sim_elderly3

2. Validate outputs:
   - Check if feature counts match
   - Check if model metrics (R²) match original

3. Once validated, extend phases:
   - Add more feature types as needed
   - Refactor other utilities into pipeline

4. When satisfied:
   - Merge modular-refactor → pascal_update
   - Archive old preprocessing/, windowing/, ml/ structure

"""
print(__doc__)
