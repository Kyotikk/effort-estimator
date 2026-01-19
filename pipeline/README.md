"""
Clean Pipeline Structure Documentation
======================================

The pipeline is now organized into 7 chronological phases, each with:
- Clear callable functions
- Proper directory structure
- No mixed concerns

DIRECTORY STRUCTURE:

pipeline/
├── 01_preprocessing/
│   ├── __init__.py (exports functions)
│   └── preprocessing.py (preprocess_imu, preprocess_ppg, preprocess_eda, preprocess_rr)
│
├── 02_windowing/
│   ├── __init__.py (exports functions)
│   └── windowing.py (create_windows, quality_check_windows)
│
├── 03_features/
│   ├── __init__.py (exports functions)
│   ├── imu_features.py (extract_imu_features)
│   ├── ppg_features.py (extract_ppg_features)
│   ├── rr_features.py (extract_rr_features)
│   └── eda_features.py (extract_eda_features)
│
├── 04_fusion/
│   ├── __init__.py (exports functions)
│   └── fusion.py (fuse_modalities)
│
├── 05_alignment/
│   ├── __init__.py (exports functions)
│   └── alignment.py (align_with_targets)
│
├── 06_selection/
│   ├── __init__.py (exports functions)
│   └── selection.py (select_features)
│
├── 07_training/
│   ├── __init__.py (exports functions)
│   └── training.py (train_model, evaluate_model)
│
└── run_clean_pipeline.py (orchestrator - runs all 7 phases)


FLOW:

Phase 1: PREPROCESSING
  Input:  Raw CSV/CSV.gz files (7 modalities)
  Output: Cleaned DataFrames with normalized time columns
  
Phase 2: WINDOWING
  Input:  Preprocessed signals
  Output: Window definitions (start_idx, end_idx, t_start, etc.)
  Note:   Also includes quality checks on features
  
Phase 3: FEATURE EXTRACTION
  Input:  Preprocessed signals + windows
  Output: Feature DataFrames per modality
  Note:   IMU, PPG, RR, EDA handled separately
  
Phase 4: FUSION
  Input:  Feature DataFrames from all modalities
  Output: Single fused feature matrix (all modalities combined)
  
Phase 5: ALIGNMENT
  Input:  Fused features + ground-truth ADL labels
  Output: Features + effort labels
  
Phase 6: SELECTION
  Input:  All features + labels
  Output: Top 50 selected features + PCA analysis
  
Phase 7: TRAINING
  Input:  Selected features + labels
  Output: Trained XGBoost model + evaluation metrics


USAGE:

Run entire pipeline:
  python pipeline/run_clean_pipeline.py --config config/pipeline.yaml

Process single subject:
  python pipeline/run_clean_pipeline.py --config config/pipeline.yaml --subject sim_elderly3

Use individual phases in code:
  from pipeline.01_preprocessing import preprocess_imu
  from pipeline.02_windowing import create_windows
  from pipeline.03_features import extract_imu_features
  
  imu_df = preprocess_imu("path/to/imu.csv.gz")
  windows_df = create_windows(imu_df, fs=125.0, win_sec=10.0, overlap=0.5)
  features_df = extract_imu_features(imu_df, windows_df)


COMPARISON:

OLD (scattered):
  - preprocessing/ had isolated scripts
  - windowing/ mixed feature quality check (wrong place)
  - features/ had manual_features_imu.py with long complex logic
  - ml/ had fusion.py, alignment.py, selection.py (hard to track)
  - No clear entry point or orchestration

NEW (clean):
  ✓ Each phase in numbered directory (clear order)
  ✓ Each phase has clear __init__.py exports
  ✓ Feature quality check moved to proper phase (02_windowing)
  ✓ Each modality has dedicated feature extractor
  ✓ Single orchestrator that runs all phases
  ✓ Callable functions with clear inputs/outputs
  ✓ No mixed concerns or scattered logic


MIGRATION:

Old files still exist in:
  - preprocessing/
  - windowing/
  - features/
  - ml/

But new pipeline/ directory is the source of truth.
When confident, old directories can be removed.


NEXT STEPS:

1. Test that run_clean_pipeline.py executes successfully
2. Validate outputs match original pipeline
3. Extend each phase with additional logic as needed
4. Remove old files once fully migrated
"""
