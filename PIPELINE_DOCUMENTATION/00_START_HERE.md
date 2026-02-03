# Effort Estimation Pipeline - START HERE

## Overview

**Production-ready multi-subject effort estimation system** predicting Borg effort ratings (0-8) from physiological signals recorded from **3 elderly patients** (sim_elderly3, sim_elderly4, sim_elderly5).

**Current Status:** ✅ Production Ready
- **Best Model (5s windows):** r = 0.626 (XGBoost), r = 0.644 (Ridge)
- **Comparison Model (10s windows):** r = 0.548 (XGBoost), r = 0.567 (Ridge)
- GroupKFold cross-validation across 65 activities (no data leakage)
- 48-51 selected features from 270+ raw features

---

## Quick Start

### Run Elderly Patient Pipeline (Recommended)
```bash
cd /Users/pascalschlegel/effort-estimator

# Run full pipeline for 5s windows (best performance)
python run_elderly_pipeline.py

# Or run specific window sizes
python run_elderly_10s_30s.py --window 5.0   # Best: r=0.626
python run_elderly_10s_30s.py --window 10.0  # Comparison: r=0.548
python run_elderly_10s_30s.py --window 30.0  # Poor: r=0.364
```

**Output:** Results in `/Users/pascalschlegel/data/interim/elderly_combined/`

---

## 📊 Window Size Comparison Results

| Window | N Samples | Features | XGBoost r | XGBoost MAE | Ridge r | Ridge MAE |
|--------|-----------|----------|-----------|-------------|---------|-----------|
| **5s** | 855 | 48 | **0.626** | 1.22 | **0.644** | 1.17 |
| 10s | 424 | 51 | 0.548 | 1.36 | 0.567 | 1.30 |
| 30s | 100 | 20 | 0.364 | 1.34 | 0.184 | 1.69 |

**Key Finding:** 5s windows outperform larger windows due to:
1. More samples (855 vs 424 vs 100) → better statistical power
2. Better temporal resolution → captures rapid effort changes
3. Activities often last <30s → larger windows average across activities

---

## 📂 Documentation Contents

```
PIPELINE_DOCUMENTATION/
├── 00_START_HERE.md              ← You are here
├── INDEX.md                      Navigation guide
├── README.md                     Full technical overview
│
├── 01_PREPROCESSING.md           Raw signals → Clean signals
├── 02_WINDOWING.md              Continuous → 5s/10s windows
├── 03_FEATURE_EXTRACTION.md      Signals → 270+ features
├── 04_ALIGNMENT_AND_FUSION.md    Multi-modality fusion + Borg labels
├── 05_FEATURE_SELECTION.md       270+ → 48 features (correlation pruning)
├── 06_TRAINING.md                XGBoost/Ridge training with GroupKFold CV
├── 07_PERFORMANCE_METRICS.md     Full results analysis
└── 08_MULTISUB_ROADMAP.md        Multi-subject strategy
```

---

## Current Performance (5s Windows - Best)

| Metric | XGBoost | Ridge |
|--------|---------|-------|
| **Pearson r** | 0.626 | 0.644 |
| **MAE** | 1.22 | 1.17 |
| **RMSE** | 1.52 | 1.48 |
| **N Samples** | 855 | 855 |
| **N Features** | 48 | 48 |

**Interpretation:** Model explains ~40% of variance (r²=0.39-0.41) in Borg ratings with ±1.2 Borg point average error across 3 elderly subjects using GroupKFold CV.

---

## Data Pipeline

```
Raw Signals (PPG Green/Infra/Red, EDA, IMU, RR)
    ↓
Preprocessing (cleaning, resampling to 32 Hz)
    ↓
Windowing (5s windows, 10% overlap → step 4.5s)
    ↓
Feature Extraction (270+ features per window)
    ↓
Alignment (temporal sync with ADL/Borg labels)
    ↓
Fusion (combined across modalities, tolerance 2s)
    ↓
Feature Selection (270+ → 48 via correlation pruning)
    ↓
GroupKFold CV Training (5 folds, grouped by activity)
    ↓
Model + Metrics
```

---

## 3 Elderly Subjects

| Subject | Data Source | N Samples | N Labeled (Borg) |
|---------|-------------|-----------|------------------|
| sim_elderly3 | parsingsim3 | ~350 | ~280 |
| sim_elderly4 | parsingsim4 | ~300 | ~290 |
| sim_elderly5 | parsingsim5 | ~300 | ~285 |
| **Combined** | All 3 | 855 | 855 |

---

## Key Files

### Main Scripts
- **run_elderly_pipeline.py** - Full pipeline for 3 elderly patients
- **run_elderly_10s_30s.py** - Run specific window sizes with proper tolerances
- **compare_results.py** - Compare performance across window sizes
- **run_pipeline.py** - Generic single-subject pipeline

### Modules
- **preprocessing/** - Signal cleaning (PPG, EDA, IMU, RR)
- **features/** - Feature extraction (tifex.py engine + HRV)
- **windowing/** - Window creation and management
- **ml/feature_selection_and_qc.py** - Feature selection + PCA QC
- **ml/fusion/** - Multi-modality fusion logic
- **ml/targets/** - Temporal alignment with Borg labels

---

## Feature Distribution (5s Windows)

After correlation pruning (threshold 0.90):

| Modality | Raw Features | Selected |
|----------|--------------|----------|
| PPG (3 colors) | ~130 | 19 |
| EDA | ~40 | 8 |
| IMU | ~90 | 19 |
| HRV | ~30 | 2 |
| **Total** | ~290 | 48 |

**Top Features by Importance:**
1. `ppg_green_range` - PPG amplitude range
2. `ppg_green_p95` - PPG 95th percentile
3. `acc_x_dyn__cardinality` - IMU unique values
4. `eda_stress_skin_max` - EDA stress indicator
5. `ppg_green_trim_mean_10` - Trimmed mean

---

## Model Configuration

### XGBoost
```python
XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
)
```

### Ridge Regression
```python
Ridge(alpha=1.0)  # L2 regularization
StandardScaler()  # Features standardized before training
```

### Cross-Validation
```python
GroupKFold(n_splits=5)
# Groups = activity IDs (subject + Borg transitions)
# Prevents data leakage between related windows
```

---

## Output Structure

```
/Users/pascalschlegel/data/interim/elderly_combined/
├── elderly_aligned_5.0s.csv       # Combined 5s features (855 samples)
├── elderly_aligned_10.0s.csv      # Combined 10s features (424 samples)
├── elderly_aligned_30.0s.csv      # Combined 30s features (100 samples)
│
├── qc_5.0s/                        # 5s feature selection QC
│   ├── features_selected_pruned.csv
│   ├── pca_variance_explained.csv
│   └── pca_loadings.csv
│
├── xgboost_results/                # 5s XGBoost results
│   ├── summary.yaml
│   ├── feature_importance.csv
│   └── predictions.csv
│
├── linear_results/                 # 5s Ridge results
│   ├── summary.yaml
│   ├── coefficients.csv
│   └── predictions.csv
│
├── xgboost_results_10.0s/          # 10s XGBoost results
├── ridge_results_10.0s/            # 10s Ridge results
├── xgboost_results_30.0s/          # 30s XGBoost results
└── ridge_results_30.0s/            # 30s Ridge results
```

---

## Why 5s Windows Work Best

### Statistical Power
- More samples: 855 (5s) vs 424 (10s) vs 100 (30s)
- More activities covered: 65 (5s) vs 61 (10s) vs 40 (30s)

### Temporal Resolution
- Activities often last 10-30 seconds
- 5s windows capture transitions within activities
- 30s windows average across multiple effort levels

### HRV Validity
- Task Force 1996 recommends 5 min for traditional HRV
- Ultra-short (<1 min) acceptable with 10+ beats
- 5s at 60 BPM ≈ 5 beats (borderline but functional)
- 10s at 60 BPM ≈ 10 beats (minimum recommended)

### Fusion Tolerance
- 5s windows: 2s tolerance (handles 4s sensor offset)
- 10s windows: 5s tolerance (needed for proper fusion)
- 30s windows: 15s tolerance (still lost some data)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Import error | Check all dependencies in requirements.txt |
| 0 labeled samples | Check ADL file format and time alignment |
| Poor 10s results | Ensure tolerance_s is 5.0, not 2.0 |
| Missing features CSV | Run full pipeline, not just training |
| GroupKFold error | Need at least 5 unique activities |

---

## Technical Specs

- **Python:** 3.8+
- **Key Libraries:** XGBoost, scikit-learn, pandas, numpy, scipy
- **Training Time:** ~2-5 minutes per window size
- **Memory:** ~500MB for full pipeline
- **Data Format:** CSV-based, no database required

---

## Next Steps

1. ✅ **5s pipeline complete** - Use as primary model
2. ✅ **10s comparison** - Shows 5s is better
3. 📊 **Review metrics** - Check summary.yaml files
4. 🔧 **Customize** - Adjust window overlap, feature selection
5. 📈 **Add subjects** - Extend to healthy/severe patients

---

## References

- **Borg Scale:** Borg, G. (1982). Psychophysical bases of perceived exertion.
- **HRV Guidelines:** Task Force (1996). Heart rate variability standards.
- **XGBoost:** Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System.

