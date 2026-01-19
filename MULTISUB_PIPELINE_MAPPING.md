# Multi-Subject Pipeline - Complete Script Mapping

## Executive Summary

The multi-subject pipeline consists of **3 entry points** and **11 active modules**. All other files should be reviewed for deletion or archival.

---

## 🎯 Pipeline Flow (What Actually Runs)

```
START: python run_multisub_pipeline.py
        │
        ├─► For each subject (sim_elderly3, sim_healthy3, sim_severe3):
        │   │
        │   └─► SUBPROCESS: python run_pipeline.py
        │       │
        │       ├─► preprocessing/imu.py → preprocess_imu()
        │       ├─► preprocessing/ppg.py → preprocess_ppg()
        │       ├─► preprocessing/eda.py → preprocess_eda()
        │       ├─► preprocessing/rr.py → preprocess_rr()
        │       ├─► windowing/windows.py → create_windows()
        │       ├─► features/manual_features_imu.py → compute_top_imu_features_from_windows()
        │       ├─► ml/targets/run_target_alignment.py → run_alignment()
        │       ├─► ml/run_fusion.py → main()
        │       │   ├─► ml/fusion/fuse_windows.py
        │       │   └─► ml/features/sanitise.py
        │       ├─► windowing/feature_quality_check_any.py (subprocess)
        │       └─► ml/feature_selection_and_qc.py (optional: standalone run)
        │
        ├─► combine_datasets()
        │   └─► Loads: fused_aligned_{window}.csv from each subject
        │
        ├─► ml/quality_check.py → check_data_quality()
        │
        ├─► ml/feature_selection.py → select_features()
        │   └─► ml/feature_selection_and_qc.py → select_and_prune_features()
        │
        └─► Save: multisub_aligned_10.0s.csv, features_selected_pruned.csv

END: python train_multisub_xgboost.py
        │
        ├─► Load: multisub_aligned_10.0s.csv
        ├─► Load: features_selected_pruned.csv (if exists)
        ├─► Train XGBoost model
        └─► Generate 7 plots
```

---

## ✅ ACTIVE SCRIPTS (Keep All)

### Core Entry Points (3)
| File | Purpose | What It Does |
|------|---------|-------------|
| **run_multisub_pipeline.py** | Main orchestrator | Runs 3 subject pipelines, combines datasets, selects features |
| **run_pipeline.py** | Single-subject pipeline | Preprocessing → windowing → features → fusion → alignment |
| **train_multisub_xgboost.py** | Model training | Trains XGBoost on combined multi-subject data, generates 7 plots |

### Preprocessing Modules (4)
| File | Used By | Functions |
|------|---------|-----------|
| **preprocessing/imu.py** | run_pipeline.py | `preprocess_imu()` |
| **preprocessing/ppg.py** | run_pipeline.py | `preprocess_ppg()` |
| **preprocessing/eda.py** | run_pipeline.py | `preprocess_eda()` |
| **preprocessing/rr.py** | run_pipeline.py | `preprocess_rr()` |

### Windowing & Features (3)
| File | Used By | Functions |
|------|---------|-----------|
| **windowing/windows.py** | run_pipeline.py | `create_windows()` |
| **windowing/feature_quality_check_any.py** | run_pipeline.py (subprocess) | QC analysis generation |
| **features/manual_features_imu.py** | run_pipeline.py | `compute_top_imu_features_from_windows()` |

### ML Components (6)
| File | Used By | Functions |
|------|---------|-----------|
| **ml/alignment.py** | run_multisub_pipeline.py | `align_fused_data()`, `save_aligned_data()` |
| **ml/fusion.py** | run_multisub_pipeline.py, run_pipeline.py | `fuse_modalities()`, `save_fused_data()` |
| **ml/quality_check.py** | run_multisub_pipeline.py | `check_data_quality()`, `print_qc_results()` |
| **ml/feature_selection.py** | run_multisub_pipeline.py | `select_features()`, `save_feature_selection_outputs()` |
| **ml/feature_selection_and_qc.py** | ml/feature_selection.py | `select_and_prune_features()`, `perform_pca_analysis()` |
| **ml/run_fusion.py** | run_pipeline.py | `main()` (orchestrates fusion) |

### Supporting Modules (5)
| File | Used By | Purpose |
|------|---------|---------|
| **ml/targets/run_target_alignment.py** | run_pipeline.py | `run_alignment()` - temporal alignment with ADL targets |
| **ml/fusion/fuse_windows.py** | ml/run_fusion.py | Fuses multi-modality windows |
| **ml/features/sanitise.py** | ml/run_fusion.py | Cleans feature columns |
| **ml/scalers/imu_scaler.py** | Used somewhere | IMU scaling utilities |
| **ml/time/ensure_unix.py** | Used somewhere | Time conversion utilities |

---

## ❌ UNUSED SCRIPTS (Delete Candidates)

| File | Status | Reason |
|------|--------|--------|
| **ml/feature_extraction.py** | NEVER CALLED | Placeholder with no real functionality |
| **ml/train_and_save_all.py** | ARCHIVED | Legacy single-subject trainer (obsolete, use run_pipeline.py + train_multisub_xgboost.py instead) |
| **features/vitalpy_ppg.py** | LEGACY | Old PPG feature extractor (replaced by features in run_pipeline.py) |
| **features/eda_features.py** | LEGACY | EDA features (integrated into run_pipeline.py) |
| **features/ppg_features.py** | LEGACY | PPG features (integrated into run_pipeline.py) |
| **features/rr_features.py** | LEGACY | RR features (integrated into run_pipeline.py) |
| **features/tifex.py** | LEGACY | Old feature extraction engine (replaced by manual_features_imu.py) |
| **windowing/feature_check_from_tifey.py** | UNUSED | Old QC script (replaced by feature_quality_check_any.py) |
| **ml/targets/adl_alignment.py** | INTERNAL | Used only by run_target_alignment.py |

---

## 📊 Data Files Generated & Used

### Intermediate Files (Per Subject)
```
{subject}/effort_estimation_output/{subject}/
├── imu_bioz/
│   ├── imu_preprocessed.csv
│   ├── imu_windows_10.0s.csv
│   └── imu_features_10.0s.csv
├── imu_wrist/
│   ├── imu_preprocessed.csv
│   ├── imu_windows_10.0s.csv
│   └── imu_features_10.0s.csv
├── ppg_green/
│   ├── ppg_green_preprocessed.csv
│   ├── ppg_green_windows_10.0s.csv
│   └── ppg_green_features_10.0s.csv
├── ppg_infra/
│   ├── ppg_infra_preprocessed.csv
│   ├── ppg_infra_windows_10.0s.csv
│   └── ppg_infra_features_10.0s.csv
├── ppg_red/
│   ├── ppg_red_preprocessed.csv
│   ├── ppg_red_windows_10.0s.csv
│   └── ppg_red_features_10.0s.csv
├── eda/
│   ├── eda_preprocessed.csv
│   ├── eda_windows_10.0s.csv
│   └── eda_features_10.0s.csv
├── rr/
│   ├── rr_preprocessed.csv
│   ├── rr_windows_10.0s.csv
│   └── rr_features_10.0s.csv
├── fused_10.0s.csv              ← All modalities combined
└── fused_aligned_10.0s.csv      ← With Borg labels
```

### Multi-Subject Combined Files
```
multisub_combined/
├── multisub_aligned_10.0s.csv   ← All subjects combined
├── qc_10.0s/
│   ├── features_selected_pruned.csv  ← Pre-selected features
│   └── [QC analysis files]
└── models/
    ├── xgboost_multisub_10.0s.json
    ├── feature_importance_multisub_10.0s.csv
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

## 🔧 Configuration Files

| File | Used By | Purpose |
|------|---------|---------|
| **config/pipeline.yaml** | run_pipeline.py | Single-subject pipeline config (datasets, preprocessing params) |
| **config/training.yaml** | run_pipeline.py | Training parameters |

---

## 📋 Summary: What to Keep/Delete

### ✅ KEEP (Active Pipeline)
- `run_multisub_pipeline.py`
- `run_pipeline.py`
- `train_multisub_xgboost.py`
- All in `preprocessing/` (imu.py, ppg.py, eda.py, rr.py)
- All in `windowing/` (windows.py, feature_quality_check_any.py)
- `features/manual_features_imu.py`
- All in `ml/` except those listed below
- All in `ml/targets/`, `ml/fusion/`, `ml/features/`, `ml/scalers/`, `ml/time/`

### 🗑️ DELETE (Unused)
- `ml/feature_extraction.py` (never called, placeholder)

### 📦 ARCHIVE/CONSIDER DELETING (Legacy but not in active pipeline)
- `ml/train_and_save_all.py` (old single-subject trainer)
- `features/vitalpy_ppg.py`
- `features/eda_features.py`
- `features/ppg_features.py`
- `features/rr_features.py`
- `features/tifex.py`
- `windowing/feature_check_from_tifey.py`

---

## 🚀 Execution Sequence

### Step 1: Run Multi-Subject Pipeline
```bash
python run_multisub_pipeline.py
```
**Time:** ~15-20 minutes (processes 3 subjects)
**Output:** `multisub_aligned_10.0s.csv`, `features_selected_pruned.csv`

### Step 2: Train Model
```bash
python train_multisub_xgboost.py
```
**Time:** ~5-10 minutes
**Output:** Model JSON, 7 plots, metrics

---

## 📝 Module Dependencies

```
run_multisub_pipeline.py
├─ subprocess: run_pipeline.py (per subject)
│  ├─ preprocessing.imu
│  ├─ preprocessing.ppg
│  ├─ preprocessing.eda
│  ├─ preprocessing.rr
│  ├─ windowing.windows
│  ├─ features.manual_features_imu
│  ├─ ml.targets.run_target_alignment
│  └─ ml.run_fusion
│     ├─ ml.fusion
│     ├─ ml.fusion.fuse_windows
│     └─ ml.features.sanitise
├─ ml.alignment
├─ ml.fusion
├─ ml.quality_check
├─ ml.feature_selection
│  └─ ml.feature_selection_and_qc
└─ pd, yaml, Path

train_multisub_xgboost.py
├─ ml (utilities)
├─ xgb, sklearn
└─ pd, numpy, matplotlib
```

---

## ⚠️ Critical Notes

1. **No file in `preprocessing/` is unused** - all 4 (imu, ppg, eda, rr) are called by run_pipeline.py
2. **All modality-specific feature extractors have been consolidated** into run_pipeline.py - old individual scripts can be deleted
3. **run_pipeline.py is the workhorse** - called once per subject by run_multisub_pipeline.py via subprocess
4. **ml/run_fusion.py is still used** - run_pipeline.py calls it, don't delete
5. **All ml/targets/, ml/fusion/, ml/features/ subdirectories are used** - support ml/alignment and ml/fusion
6. **Configuration is critical** - config/pipeline.yaml defines which files to process and parameters

---

## 🎯 Recommendations

### IMMEDIATE (Safe to Delete Now)
- Delete: `ml/feature_extraction.py` (1 file)

### SOON (Review for Deletion)
- Review: `ml/train_and_save_all.py` (single-subject trainer - may want to keep as backup)
- Review: All legacy feature extractors in `features/` directory

### ARCHIVE (Keep but Don't Use)
- Move old feature extraction scripts to `features/legacy/` folder if deleting

---

Generated: 2026-01-19
Pipeline Status: ✅ Production Ready
