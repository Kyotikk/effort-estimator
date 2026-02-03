# Pipeline Documentation Index

## Quick Navigation

Welcome to the comprehensive Effort Estimation Pipeline documentation for **3 elderly patients**. This folder contains detailed explanations of every stage from raw sensor data to model predictions.

---

## 📊 Current Results Summary

| Window | N Samples | XGBoost r | Ridge r | Best MAE |
|--------|-----------|-----------|---------|----------|
| **5s** | 855 | **0.626** | **0.644** | **1.17** |
| 10s | 424 | 0.548 | 0.567 | 1.30 |
| 30s | 100 | 0.364 | 0.184 | 1.34 |

**Best Configuration:** 5s windows with Ridge regression (r=0.644, MAE=1.17)

---

## 📚 Documentation Files

### 🎯 [00_START_HERE.md](00_START_HERE.md) - QUICK START
**Essential overview and commands (5 min read)**

Contains:
- Quick start commands
- Window size comparison table
- Key file locations
- Troubleshooting guide

**Read this if:** You want to run the pipeline immediately

---

### 📖 [README.md](README.md) - FULL OVERVIEW
**Complete pipeline documentation (15 min read)**

Contains:
- Executive summary with results
- Full pipeline architecture diagram
- All performance metrics
- Feature selection details
- Technical specifications

**Read this if:** You want the complete picture

---

## 🔧 Stage-by-Stage Detailed Docs

### [01_PREPROCESSING.md](01_PREPROCESSING.md)
**Raw sensor data → Clean signals (10 min read)**

Covers:
- IMU acceleration preprocessing (gravity removal, noise filtering)
- PPG wrist sensor preprocessing (3 wavelengths, HPF for weak signals)
- EDA electrodermal preprocessing (stress + conductance)
- RR respiratory rate infrastructure
- Signal quality analysis

**Key processing:**
- All signals resampled to 32 Hz
- PPG Green: No HPF (strong signal)
- PPG Infra/Red: HPF 0.5 Hz (baseline drift removal)

---

### [02_WINDOWING.md](02_WINDOWING.md)
**Continuous signals → Fixed-duration windows (8 min read)**

Covers:
- Windowing algorithm (10% overlap)
- Window metadata (t_start, t_center, t_end)
- Window size comparison (5s vs 10s vs 30s)
- Why 5s windows perform best
- Fusion tolerance requirements

**Key metrics:**
- 5s windows: 855 labeled (best)
- 10s windows: 424 labeled
- 30s windows: 100 labeled

---

### [03_FEATURE_EXTRACTION.md](03_FEATURE_EXTRACTION.md)
**Windowed signals → 270+ numerical features (12 min read)**

Covers:
- IMU features (~90 per sensor)
- PPG features (~44 per wavelength + HRV)
- EDA features (~40 basic + advanced)
- Feature definitions with examples
- tifex engine usage

**Key breakdown:**
- Total raw: 270+ features
- After selection: 48 features (5s)

---

### [04_ALIGNMENT_AND_FUSION.md](04_ALIGNMENT_AND_FUSION.md)
**Features + Borg labels → Training dataset (10 min read)**

Covers:
- ADL timestamp matching with window t_center
- Borg effort rating extraction
- Multi-modality fusion
- Tolerance settings per window size
- Why some windows are dropped

**Key metrics:**
- Labeled windows (5s): 855
- Fusion tolerance: 2s (5s windows), 5s (10s windows)

---

### [05_FEATURE_SELECTION.md](05_FEATURE_SELECTION.md)
**270+ features → 48 selected features (8 min read)**

Covers:
- Correlation-based selection (top 100)
- Redundancy pruning (r > 0.90 threshold)
- PCA quality checks
- Final feature distribution by modality

**Key metrics:**
- Before: 270+ features
- After: 48 features
- Selection method: Correlation + pruning

---

### [06_TRAINING.md](06_TRAINING.md)
**Training XGBoost and Ridge models (10 min read)**

Covers:
- GroupKFold cross-validation (5 folds)
- Activity group creation (prevents data leakage)
- XGBoost vs Ridge comparison
- Hyperparameter choices
- Feature importance analysis

**Key results:**
- XGBoost: r=0.626, MAE=1.22
- Ridge: r=0.644, MAE=1.17

---

### [07_PERFORMANCE_METRICS.md](07_PERFORMANCE_METRICS.md)
**Full results analysis (12 min read)**

Covers:
- Detailed metrics for all window sizes
- Pearson r interpretation
- MAE/RMSE analysis
- Per-subject breakdown
- Feature importance rankings
- Comparison with literature
- Limitations and future work

**Key findings:**
- 5s windows optimal
- Ridge slightly outperforms XGBoost
- r=0.64 competitive with literature

---

### [08_MULTISUB_ROADMAP.md](08_MULTISUB_ROADMAP.md)
**Multi-subject expansion strategy (8 min read)**

Covers:
- Current 3-subject setup
- Leave-one-subject-out validation (future)
- Adding healthy/severe cohorts
- Generalization considerations

---

## 📁 Key Output Files

```
/Users/pascalschlegel/data/interim/elderly_combined/
├── elderly_aligned_5.0s.csv      # Best: 855 samples
├── elderly_aligned_10.0s.csv     # Comparison: 424 samples
├── elderly_aligned_30.0s.csv     # Poor: 100 samples
│
├── qc_5.0s/                       # Feature selection QC
│   ├── features_selected_pruned.csv
│   └── pca_*.csv
│
├── xgboost_results/               # 5s XGBoost (r=0.626)
│   ├── summary.yaml
│   ├── feature_importance.csv
│   └── predictions.csv
│
├── linear_results/                # 5s Ridge (r=0.644)
│   ├── summary.yaml
│   ├── coefficients.csv
│   └── predictions.csv
│
├── xgboost_results_10.0s/         # 10s comparison
├── ridge_results_10.0s/
├── xgboost_results_30.0s/
└── ridge_results_30.0s/
```

---

## 🚀 Quick Commands

### Run Full Pipeline (5s, Recommended)
```bash
cd /Users/pascalschlegel/effort-estimator
python run_elderly_pipeline.py
```

### Run Specific Window Size
```bash
python run_elderly_10s_30s.py --window 5.0   # Best
python run_elderly_10s_30s.py --window 10.0  # Comparison
python run_elderly_10s_30s.py --window 30.0  # Poor
```

### Compare Results
```bash
python compare_results.py
```

---

## 📋 Reading Order

**For quick understanding:**
1. [00_START_HERE.md](00_START_HERE.md) (5 min)
2. [07_PERFORMANCE_METRICS.md](07_PERFORMANCE_METRICS.md) (12 min)

**For complete understanding:**
1. [README.md](README.md) (15 min)
2. [01_PREPROCESSING.md](01_PREPROCESSING.md) through [06_TRAINING.md](06_TRAINING.md) (60 min)
3. [07_PERFORMANCE_METRICS.md](07_PERFORMANCE_METRICS.md) (12 min)

**For methodology only:**
1. [README.md](README.md) (15 min)
2. [05_FEATURE_SELECTION.md](05_FEATURE_SELECTION.md) (8 min)
3. [06_TRAINING.md](06_TRAINING.md) (10 min)

---

## 📊 Subject Information

| Subject | Data Path | Borg Samples (5s) |
|---------|-----------|-------------------|
| sim_elderly3 | parsingsim3 | ~280 |
| sim_elderly4 | parsingsim4 | ~290 |
| sim_elderly5 | parsingsim5 | ~285 |
| **Total** | - | **855** |

---

## 🔬 Technical Details

- **Python:** 3.8+
- **Key Libraries:** xgboost, scikit-learn, pandas, numpy, scipy
- **CV Method:** GroupKFold (5 folds, grouped by activity)
- **Best Model:** Ridge regression (alpha=1.0)
- **Best Window:** 5 seconds (10% overlap)

