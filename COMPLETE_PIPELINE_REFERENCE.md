# Multi-Subject Pipeline - Complete Data Flow & Module Reference

**Last Updated:** 2026-01-19

---

## 📊 Complete Pipeline Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MULTI-SUBJECT EFFORT ESTIMATION                      │
│                       Full End-to-End Pipeline                          │
└─────────────────────────────────────────────────────────────────────────┘

STAGE 1: DATA PREPROCESSING (Per Subject: 3 subjects × 7 modalities)
═════════════════════════════════════════════════════════════════════════

Raw Data Files (3 subjects × 7 modalities = 21 files)
    ├── sim_elderly3/
    │   ├── corsano_bioz_acc/          → preprocessing.imu  → imu_preprocessed.csv
    │   ├── corsano_wrist_acc/         → preprocessing.imu  → imu_preprocessed.csv
    │   ├── corsano_wrist_ppg2_green/  → preprocessing.ppg  → ppg_green_preprocessed.csv
    │   ├── corsano_wrist_ppg2_infra/  → preprocessing.ppg  → ppg_infra_preprocessed.csv
    │   ├── corsano_wrist_ppg2_red/    → preprocessing.ppg  → ppg_red_preprocessed.csv
    │   ├── corsano_bioz_emography/    → preprocessing.eda  → eda_preprocessed.csv
    │   └── corsano_bioz_rr_interval/  → preprocessing.rr   → rr_preprocessed.csv
    │
    ├── sim_healthy3/     [same structure]
    └── sim_severe3/      [same structure]

STAGE 2: WINDOWING (Per Subject: 7 modalities × 3 window lengths)
═════════════════════════════════════════════════════════════════════════

imu_preprocessed.csv ──→ windowing.windows.create_windows() ──→ imu_windows_10.0s.csv
ppg_green_preprocessed.csv ──→ windowing.windows.create_windows() ──→ ppg_green_windows_10.0s.csv
... (repeat for all modalities and window lengths: 10.0s, 5.0s, 2.0s)

STAGE 3: FEATURE EXTRACTION (Per Subject: 7 modalities)
═════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│ IMU FEATURES                                                    │
├─────────────────────────────────────────────────────────────────┤
│ imu_windows_10.0s.csv + imu_preprocessed.csv                  │
│    ↓                                                            │
│ features.manual_features_imu.compute_top_imu_features()       │
│    ↓                                                            │
│ imu_features_10.0s.csv (window_id, acc_x_mean, acc_y_mean, ...) │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ PPG FEATURES (Green, Infra, Red)                               │
├─────────────────────────────────────────────────────────────────┤
│ ppg_windows_10.0s.csv + preprocessing.ppg output               │
│    ↓                                                            │
│ [Built-in features: heart_rate, hrv, morphology, etc.]        │
│    ↓                                                            │
│ ppg_green_features_10.0s.csv (window_id, ppg_green_hr, ...)   │
│ ppg_infra_features_10.0s.csv                                   │
│ ppg_red_features_10.0s.csv                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ OTHER MODALITY FEATURES (EDA, RR)                              │
├─────────────────────────────────────────────────────────────────┤
│ [Extracted using similar windowing approach]                   │
│    ↓                                                            │
│ eda_features_10.0s.csv (window_id, eda_stat, ...)            │
│ rr_features_10.0s.csv (window_id, rr_mean, ...)              │
└─────────────────────────────────────────────────────────────────┘

STAGE 4: QUALITY CHECK (Per Subject)
═════════════════════════════════════════════════════════════════════════

imu_features_10.0s.csv ──→ windowing.feature_quality_check_any.py (subprocess)
ppg_green_features_10.0s.csv ──→ [subprocess QC]
... (repeat for all modalities)

Output: data/feature_extraction/analysis/quality_imu_10.0s_70ol/ (analysis files)

STAGE 5: TARGET ALIGNMENT (Per Subject)
═════════════════════════════════════════════════════════════════════════

imu_features_10.0s.csv + scai_app/ADLs/*.csv
    ↓
ml.targets.run_target_alignment.run_alignment()
    ├─→ ml.targets.adl_alignment.align_targets()
    └─→ Add 'borg' label column based on time alignment

Output: imu_aligned_10.0s.csv (with borg labels)

STAGE 6: FEATURE FUSION (Per Subject)
═════════════════════════════════════════════════════════════════════════

imu_aligned_10.0s.csv
ppg_green_aligned_10.0s.csv
ppg_infra_aligned_10.0s.csv
ppg_red_aligned_10.0s.csv
eda_aligned_10.0s.csv
rr_aligned_10.0s.csv
    ↓ (all fed to fusion)
ml.run_fusion.main()
    ├─→ ml.fusion.fuse_modalities()
    │   └─→ ml.fusion.fuse_windows.fuse_windows()
    │       (Combines all 6 modalities into single feature vector)
    ├─→ ml.features.sanitise.sanitise_columns()
    │   (Cleans column names, removes metadata)
    └─→ ml.fusion.save_fused_data()

Output: fused_10.0s.csv (188 features, no labels yet)

STAGE 7: ALIGNMENT WITH TARGETS (Per Subject)
═════════════════════════════════════════════════════════════════════════

fused_10.0s.csv + ADL targets
    ↓
ml.alignment.align_fused_data()
    └─→ Add 'borg' column from aligned ADL data

Output: fused_aligned_10.0s.csv (188 features + borg label)

STAGE 8: COMBINE SUBJECTS (Multi-Subject)
═════════════════════════════════════════════════════════════════════════

fused_aligned_10.0s.csv (sim_elderly3)   + subject='sim_elderly3'
fused_aligned_10.0s.csv (sim_healthy3)   + subject='sim_healthy3'
fused_aligned_10.0s.csv (sim_severe3)    + subject='sim_severe3'
    ↓ (pd.concat)
run_multisub_pipeline.combine_datasets()

Output: multisub_aligned_10.0s.csv (1,188 samples, 188 features, borg labels)

STAGE 9: QUALITY CHECK (Combined Data)
═════════════════════════════════════════════════════════════════════════

multisub_aligned_10.0s.csv
    ↓
ml.quality_check.check_data_quality()
    ├─→ Check missing values
    ├─→ Check feature distributions
    └─→ ml.quality_check.print_qc_results()

Output: Console report + QC statistics

STAGE 10: FEATURE SELECTION
═════════════════════════════════════════════════════════════════════════

multisub_aligned_10.0s.csv (188 features)
    ↓
ml.feature_selection.select_features()
    └─→ ml.feature_selection_and_qc.select_and_prune_features()
        ├─→ Drop low-variance features
        ├─→ Drop highly correlated features (threshold: 0.90)
        ├─→ Keep top 100 features
        └─→ Final: ~50 features selected

Output: features_selected_pruned.csv (feature names, 1 per line)

STAGE 11: TRAINING
═════════════════════════════════════════════════════════════════════════

multisub_aligned_10.0s.csv + features_selected_pruned.csv
    ↓
train_multisub_xgboost.py
    ├─→ Load data
    ├─→ Load pre-selected features
    ├─→ 80/20 train-test split
    ├─→ StandardScaler normalization
    ├─→ XGBoost training (n_estimators=500, max_depth=5)
    ├─→ Model evaluation
    │   └─→ Train R² = 0.96, Test R² = 0.94
    ├─→ Feature importance analysis
    └─→ Generate 7 diagnostic plots

Output:
  ├─ xgboost_multisub_10.0s.json (trained model)
  ├─ feature_importance_multisub_10.0s.csv
  ├─ predictions_train.csv
  ├─ predictions_test.csv
  └─ plots_multisub/
      ├─ 01_train_vs_test_scatter.png
      ├─ 02_residuals_histogram.png
      ├─ 03_residuals_vs_predicted.png
      ├─ 04_feature_importance_top15.png
      ├─ 05_feature_importance_cumsum.png
      ├─ 06_model_performance_metrics.png
      └─ 07_subject_distribution.png
```

---

## 📚 Module Reference

### PREPROCESSING MODULES
```
preprocessing/
├── imu.py
│   └── preprocess_imu(path, fs_out, noise_cutoff, gravity_cutoff)
│       Output: DataFrame with columns [time, acc_x, acc_y, acc_z, acc_x_dyn, ...]
│
├── ppg.py
│   └── preprocess_ppg(in_path, out_path, fs, time_col, metric_id, ...)
│       Output: CSV file with columns [t_sec, value]
│
├── eda.py
│   └── preprocess_eda(in_path, out_path, fs, time_col, ...)
│       Output: CSV file with columns [t_sec, eda_cc, eda_stress_skin]
│
└── rr.py
    └── preprocess_rr(in_path, out_path, fs, time_col, rr_col)
        Output: CSV file with columns [t_sec, value]
```

### WINDOWING MODULES
```
windowing/
├── windows.py
│   └── create_windows(df, fs, win_sec, overlap)
│       Input: Continuous time-series DataFrame
│       Output: DataFrame with columns [window_id, start_idx, end_idx, ..., t_start, t_end]
│
└── feature_quality_check_any.py
    └── Main subprocess that generates QC analysis plots
```

### FEATURE EXTRACTION
```
features/
└── manual_features_imu.py
    └── compute_top_imu_features_from_windows(data, windows, signal_cols)
        Input: Raw signal data + window definitions
        Output: DataFrame with calculated features for each window
```

### ML PIPELINE MODULES
```
ml/
├── alignment.py
│   ├── align_fused_data(fused_df, targets_df, time_col, ...)
│   │   Input: Fused features + Target labels (aligned by time)
│   │   Output: Same features with 'borg' column added
│   │
│   └── save_aligned_data(df, output_path)
│       Output: CSV file
│
├── fusion.py
│   ├── fuse_modalities(config)
│   │   Orchestrates multi-modality fusion
│   │
│   └── save_fused_data(df, output_path)
│       Output: CSV with all modalities combined
│
├── quality_check.py
│   ├── check_data_quality(df, features_only=True)
│   │   Output: Dict with QC statistics
│   │
│   └── print_qc_results(qc_results)
│       Console output
│
├── feature_selection.py
│   ├── select_features(df, target_col, corr_threshold, top_n)
│   │   Output: List of selected feature names
│   │
│   └── save_feature_selection_outputs(output_path, df, features, window_length)
│       Output: CSV files with selected features
│
├── feature_selection_and_qc.py
│   ├── select_and_prune_features(X, y, corr_threshold, top_n)
│   │   Output: List of selected feature indices
│   │
│   └── perform_pca_analysis(X_selected, y)
│       Output: PCA statistics
│
├── run_fusion.py
│   └── main(config)
│       Orchestrator for entire fusion pipeline
│
├── targets/
│   ├── run_target_alignment.py
│   │   └── run_alignment(features_path, windows_path, adl_path, out_path)
│   │       Aligns features with ADL-based target labels
│   │
│   └── adl_alignment.py
│       └── align_targets(features_df, adl_df)
│           Internal alignment logic
│
├── fusion/
│   └── fuse_windows.py
│       └── fuse_windows(feature_dfs, modality_times)
│           Combines multiple modality windows
│
├── features/
│   └── sanitise.py
│       └── sanitise_columns(df)
│           Cleans column names, removes metadata
│
├── scalers/
│   └── imu_scaler.py
│       IMU-specific scaling utilities
│
└── time/
    └── ensure_unix.py
        Time conversion utilities
```

---

## 🔄 Function Call Sequence

### run_multisub_pipeline.py
```python
main()
├─ For each subject in [sim_elderly3, sim_healthy3, sim_severe3]:
│  └─ run_subject_pipeline(subject)
│     └─ subprocess.run("python run_pipeline.py config.yaml")
│        └─ [See run_pipeline.py sequence below]
│
├─ combine_datasets(succeeded, WINDOW_LENGTH)
│  └─ pd.concat([fused_aligned_*.csv for each subject])
│
├─ check_data_quality(combined, features_only=True)
│  └─ ml.quality_check.check_data_quality()
│
├─ select_features(combined, target_col='borg', ...)
│  └─ ml.feature_selection.select_features()
│     └─ ml.feature_selection_and_qc.select_and_prune_features()
│
└─ save_feature_selection_outputs()
   └─ Save: features_selected_pruned.csv
```

### run_pipeline.py (called once per subject)
```python
run_pipeline(config_path)
├─ For imu_bioz, imu_wrist:
│  ├─ preprocessing.imu.preprocess_imu() → imu_preprocessed.csv
│  ├─ windowing.windows.create_windows() → imu_windows_10.0s.csv
│  ├─ features.manual_features_imu.compute_top_imu_features_from_windows()
│  │  → imu_features_10.0s.csv
│  ├─ windowing.feature_quality_check_any.py (subprocess)
│  └─ ml.targets.run_target_alignment.run_alignment()
│     → imu_aligned_10.0s.csv
│
├─ For ppg_green, ppg_infra, ppg_red:
│  ├─ preprocessing.ppg.preprocess_ppg() → ppg_*_preprocessed.csv
│  ├─ windowing.windows.create_windows() → ppg_*_windows_10.0s.csv
│  ├─ [Feature extraction] → ppg_*_features_10.0s.csv
│  ├─ windowing.feature_quality_check_any.py (subprocess)
│  └─ ml.targets.run_target_alignment.run_alignment()
│     → ppg_*_aligned_10.0s.csv
│
├─ For eda, rr:
│  └─ [Same sequence as above]
│
└─ ml.run_fusion.main(config)
   ├─ ml.fusion.fuse_modalities()
   │  └─ ml.fusion.fuse_windows.fuse_windows()
   ├─ ml.features.sanitise.sanitise_columns()
   └─ Save: fused_aligned_10.0s.csv
```

### train_multisub_xgboost.py
```python
main()
├─ Load multisub_aligned_10.0s.csv
├─ Load features_selected_pruned.csv (optional)
├─ prepare_features(df, pre_selected_features)
│  └─ Extract X, y, feature_cols
├─ train_multisub_model(X, y, groups, feature_cols)
│  ├─ 80/20 train-test split
│  ├─ StandardScaler.fit_transform()
│  ├─ XGBoost.fit()
│  └─ Evaluate and print metrics
├─ generate_plots(y_train, y_test, y_train_pred, ...)
│  └─ Create 7 diagnostic PNG files
└─ Save outputs (model, predictions, metrics)
```

---

## 💾 Output Directory Structure

```
/Users/pascalschlegel/data/interim/parsingsim3/

├── sim_elderly3/
│   └── effort_estimation_output/
│       └── parsingsim3_sim_elderly3/
│           ├── imu_bioz/
│           │   ├── imu_preprocessed.csv
│           │   ├── imu_windows_10.0s.csv
│           │   └── imu_features_10.0s.csv
│           ├── [imu_wrist, ppg_green, ppg_infra, ppg_red, eda, rr]/ [same]
│           ├── fused_10.0s.csv
│           └── fused_aligned_10.0s.csv        ← Used by multisub pipeline
│
├── sim_healthy3/ [same structure]
├── sim_severe3/  [same structure]
│
└── multisub_combined/
    ├── multisub_aligned_10.0s.csv            ← Combined from 3 subjects
    ├── qc_10.0s/
    │   └── features_selected_pruned.csv      ← Pre-selected features
    └── models/
        ├── xgboost_multisub_10.0s.json       ← Trained model
        ├── feature_importance_multisub_10.0s.csv
        ├── predictions_train.csv
        ├── predictions_test.csv
        └── plots_multisub/
            ├── 01_train_vs_test_scatter.png
            ├── 02_residuals_histogram.png
            ├── 03_residuals_vs_predicted.png
            ├── 04_feature_importance_top15.png
            ├── 05_feature_importance_cumsum.png
            ├── 06_model_performance_metrics.png
            └── 07_subject_distribution.png
```

---

## 🎯 Summary Table

| Step | Module | Input | Output | Purpose |
|------|--------|-------|--------|---------|
| 1 | preprocessing/* | Raw signals | Preprocessed CSV | Clean, resample signals |
| 2 | windowing.windows | Preprocessed signals | Windows CSV | Create fixed-length windows |
| 3 | features/* | Signals + windows | Features CSV | Extract statistical features |
| 4 | windowing.feature_quality_check | Features | QC plots | Validate feature quality |
| 5 | ml.targets.run_target_alignment | Features + ADL | Features w/ labels | Add Borg labels |
| 6 | ml.run_fusion | Multi-modality | Fused features | Combine 6 modalities |
| 7 | ml.alignment | Fused + ADL | Aligned features | Ensure time alignment |
| 8 | combine_datasets | Per-subject data | Combined CSV | Stack all subjects |
| 9 | ml.quality_check | Combined data | QC report | Validate combined data |
| 10 | ml.feature_selection | Combined data | Selected features | Reduce to 50 features |
| 11 | train_multisub_xgboost | Data + features | Model + plots | Train and visualize |

---

Generated: 2026-01-19  
Status: ✅ Complete and Production-Ready
