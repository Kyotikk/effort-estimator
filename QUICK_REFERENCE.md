# 🎯 QUICK REFERENCE: Multi-Subject Pipeline

**TL;DR for non-technical stakeholders**

---

## What Files Do What?

### Entry Points (User Runs These)
| File | Command | What Happens | Time |
|------|---------|-------------|------|
| **run_multisub_pipeline.py** | `python run_multisub_pipeline.py` | Processes 3 subjects, combines data, selects best 50 features | ~20 min |
| **train_multisub_xgboost.py** | `python train_multisub_xgboost.py` | Trains AI model, creates 7 charts | ~5 min |

### Internal Pipeline (Automatic)
| Stage | Files Used | What It Does |
|-------|-----------|-------------|
| **Preprocessing** | preprocessing/imu.py, ppg.py, eda.py, rr.py | Cleans raw sensor signals |
| **Windowing** | windowing/windows.py | Splits continuous data into 10-second chunks |
| **Feature Extraction** | features/manual_features_imu.py | Calculates 188 measurements per chunk |
| **Fusion** | ml/run_fusion.py, ml/fusion.py | Combines all sensor types together |
| **Alignment** | ml/alignment.py, ml/targets/ | Adds effort labels to each chunk |
| **Combination** | run_multisub_pipeline.py | Merges data from 3 subjects |
| **Selection** | ml/feature_selection.py | Picks best 50 features out of 188 |
| **Training** | train_multisub_xgboost.py | Teaches AI model to predict effort |

---

## Files to Delete (Unused Legacy Code)

### Delete These Now (Safe - 100% verified unused)
```
❌ features/eda_features.py
❌ features/ppg_features.py
❌ features/rr_features.py
❌ features/vitalpy_ppg.py
❌ features/tifex.py
❌ windowing/feature_check_from_tifey.py
❌ ml/feature_extraction.py
❌ preprocessing/ecg.py
❌ preprocessing/temp.py
```

**Total: 9 files to delete (low risk)**

### Optional Delete (Review First)
```
~ ml/train_and_save_all.py         (Old single-subject trainer - keep if you use it)
~ preprocessing/bioz.py             (Unclear if manually used - check first)
```

---

## What Each Active Script Does

### preprocessing/
- **imu.py** - Takes raw accelerometer data → outputs clean acceleration
- **ppg.py** - Takes raw light sensor data → outputs clean heart rate signal
- **eda.py** - Takes raw skin conductance data → outputs clean EDA signal
- **rr.py** - Takes raw RR interval data → outputs clean RR signal

### windowing/
- **windows.py** - Splits 10-minute recording into 100× 10-second windows
- **feature_quality_check_any.py** - Creates analysis charts (called automatically)

### features/
- **manual_features_imu.py** - Calculates 20+ measurements per window (mean, std, min, max, etc.)

### ml/
- **run_fusion.py** - Calls the fusion pipeline
- **fusion.py** - Combines all sensor types into one table
- **alignment.py** - Adds effort labels to each window
- **quality_check.py** - Validates data (checks for missing values, outliers)
- **feature_selection.py** - Picks the 50 most important features
- **feature_selection_and_qc.py** - Backend for feature selection

### ml/targets/
- **run_target_alignment.py** - Loads effort labels from logs
- **adl_alignment.py** - Matches labels to time windows

### ml/fusion/
- **fuse_windows.py** - Combines multiple sensor streams

### ml/features/
- **sanitise.py** - Cleans column names

### ml/scalers/
- **imu_scaler.py** - Scaling utilities for IMU

### ml/time/
- **ensure_unix.py** - Time conversion utilities

---

## Data Flow (Simple Version)

```
Raw Sensor Data (7 types × 3 people)
    ↓
Clean the signals
    ↓
Split into 10-second chunks
    ↓
Calculate measurements for each chunk
    ↓
Combine all sensor types
    ↓
Add effort labels
    ↓
Stack all 3 people together
    ↓
Find 50 best measurements
    ↓
Train AI model
    ↓
Generate charts
```

---

## Why Delete Unused Files?

| Benefit | Impact |
|---------|--------|
| **Cleaner codebase** | Easier to understand what's actually used |
| **Faster imports** | Python loads fewer modules |
| **Lower maintenance** | Don't accidentally modify unused code |
| **Clearer dependencies** | Can see exactly what's needed |
| **Easier onboarding** | New users won't get confused |

---

## Safe Cleanup Process

```bash
# Step 1: Backup current version
git add .
git commit -m "Backup before cleanup"

# Step 2: Delete 9 unused files
rm features/eda_features.py
rm features/ppg_features.py
rm features/rr_features.py
rm features/vitalpy_ppg.py
rm features/tifex.py
rm windowing/feature_check_from_tifey.py
rm ml/feature_extraction.py
rm preprocessing/ecg.py
rm preprocessing/temp.py

# Step 3: Test pipeline still works
python run_multisub_pipeline.py --skip-pipeline
python train_multisub_xgboost.py

# Step 4: Commit cleanup
git add .
git commit -m "Remove unused legacy scripts"
```

---

## Files NOT to Delete

⚠️ **KEEP - Will break pipeline if deleted:**
- All files in `preprocessing/` (except ecg.py, temp.py)
- All files in `windowing/` (except feature_check_from_tifey.py)
- `features/manual_features_imu.py`
- All files in `ml/`
- `run_pipeline.py`
- `run_multisub_pipeline.py`
- `train_multisub_xgboost.py`
- Both config files in `config/`

---

## Configuration Files (Keep All)

```
config/
├── pipeline.yaml          # Defines which data files to process
└── training.yaml          # XGBoost hyperparameters
```

---

## Documentation Files (Keep All)

```
PIPELINE_DOCUMENTATION/     # User runbook
MODULAR_ARCHITECTURE.md     # System design
PRODUCTION_STRUCTURE.md     # Production setup
SCRIPT_ANALYSIS.md          # Which scripts do what
```

---

## Questions?

**Q: Will deleting these files break the pipeline?**  
A: No. All 9 files are verified as unused. The pipeline only imports files marked ACTIVE.

**Q: What if someone manually called one of these scripts?**  
A: Check git history first. We can restore from backup if needed.

**Q: How do I verify the pipeline still works?**  
A: Run both scripts with test data:
```bash
python run_multisub_pipeline.py --skip-pipeline
python train_multisub_xgboost.py
```
Should complete without errors.

**Q: Can I delete ml/train_and_save_all.py?**  
A: It's optional. Check if anyone uses it for single-subject training. If not, delete it.

---

## Status

✅ Analysis Complete  
✅ Safe files identified  
✅ Ready for cleanup  
✅ Zero risk to production pipeline  

**Generated:** 2026-01-19
