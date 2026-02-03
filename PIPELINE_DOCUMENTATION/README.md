# Effort Estimation Pipeline - Complete Documentation

## Executive Summary

This pipeline predicts **Borg effort ratings (0-8 scale)** from multi-modal physiological sensor data using XGBoost and Ridge regression. The multi-subject model combines data from **3 elderly patients** using **PPG (3 wavelengths), IMU accelerometer, EDA electrodermal activity, and RR intervals**.

**Best Results (5s windows):**
- **XGBoost:** r = 0.626, MAE = 1.22 Borg points
- **Ridge:** r = 0.644, MAE = 1.17 Borg points

**Current Status:** ✅ Production-ready with GroupKFold cross-validation
**Data:** 855 labeled windows (5s duration) from 3 elderly subjects
**Model Features:** 48 selected from 270+ raw (correlation pruning approach)
**Validation:** GroupKFold CV (5 folds, grouped by activity ID)

---

## 📊 Window Size Comparison

| Window | N Samples | Features | XGBoost r | XGBoost MAE | Ridge r | Ridge MAE |
|--------|-----------|----------|-----------|-------------|---------|-----------|
| **5s** | 855 | 48 | **0.626** | 1.22 | **0.644** | 1.17 |
| 10s | 424 | 51 | 0.548 | 1.36 | 0.567 | 1.30 |
| 30s | 100 | 20 | 0.364 | 1.34 | 0.184 | 1.69 |

**Conclusion:** 5s windows provide optimal balance of temporal resolution and sample size.

---

## 📊 Pipeline Architecture Overview

```
RAW SENSOR DATA (3 Elderly Subjects: sim_elderly3, sim_elderly4, sim_elderly5)
    ↓
1. PREPROCESSING (clean & normalize signals)
    ├── IMU Bioz (chest accelerometer) → 32 Hz
    ├── IMU Wrist (wrist accelerometer) → 32 Hz
    ├── PPG Green (wrist) → 32 Hz [strongest signal]
    ├── PPG Infrared (wrist) → 32 Hz [medium signal, HPF 0.5 Hz]
    ├── PPG Red (wrist) → 32 Hz [weak signal, HPF 0.5 Hz]
    ├── EDA (electrodermal) → 32 Hz
    └── RR (respiratory intervals) → ~1 Hz
    ↓
2. WINDOWING (segment into 5s windows with 10% overlap)
    └── Creates windows with metadata (t_center, t_start, t_end)
    ↓
3. FEATURE EXTRACTION (compute statistical & signal features)
    ├── IMU Bioz → ~45 features (acceleration dynamics)
    ├── IMU Wrist → ~45 features (acceleration dynamics)
    ├── PPG Green → ~44 features (signal stats + HRV)
    ├── PPG Infra → ~44 features (signal stats + HRV)
    ├── PPG Red → ~44 features (signal stats + HRV)
    ├── EDA Basic → ~26 features (tonic, conductance)
    └── EDA Advanced → ~21 features (SCL, SCR, phasic)
    ↓
4. ALIGNMENT & LABELING (match windows with Borg effort labels)
    └── Align t_center with ADL activity timestamps
    ↓
5. FUSION (combine all modality features into single matrix)
    ├── Tolerance: 2s (5s windows), 5s (10s windows)
    └── Output: 270+ features per window
    ↓
6. FEATURE SELECTION (reduce dimensionality)
    ├── Method: Top 100 by correlation → correlation pruning (0.90 threshold)
    └── Output: 48 features (5s) / 51 features (10s)
    ↓
7. SCALING (for Ridge only)
    └── StandardScaler (zero mean, unit variance)
    ↓
8. TRAINING (GroupKFold Cross-Validation)
    ├── XGBoost: n_estimators=100, max_depth=4, learning_rate=0.1
    ├── Ridge: alpha=1.0 (L2 regularization)
    ├── CV: 5 folds grouped by activity (subject + Borg transitions)
    └── Prevents data leakage between correlated windows
    ↓
OUTPUT: Predictions + Metrics + Feature Importance
```

---

## 🔄 Execution Flow

### Quick Start (Elderly Patients - Recommended)
```bash
cd /Users/pascalschlegel/effort-estimator

# Run full pipeline for all 3 elderly subjects
python run_elderly_pipeline.py

# Or run specific window sizes
python run_elderly_10s_30s.py --window 5.0   # Best
python run_elderly_10s_30s.py --window 10.0  # Comparison
```

### Key Directories
- **Input:** `/Users/pascalschlegel/data/interim/`
  - `parsingsim3/sim_elderly3/` 
  - `parsingsim4/sim_elderly4/`
  - `parsingsim5/sim_elderly5/`
- **Output:** `/Users/pascalschlegel/data/interim/elderly_combined/`
  - `elderly_aligned_5.0s.csv` - Combined features
  - `qc_5.0s/` - Feature selection QC
  - `xgboost_results/` - Model metrics
  - `linear_results/` - Ridge metrics

---

## 📊 Model Performance Details

### 5s Windows (Primary - Best Performance)

**XGBoost Results:**
```
Pearson r:     0.626 (p = 2.8e-90)
RMSE:          1.52 Borg points
MAE:           1.22 Borg points
N Samples:     855
N Activities:  65
N Features:    48
CV Method:     GroupKFold (5 folds)
```

**Ridge Regression Results:**
```
Pearson r:     0.644 (p = 1.4e-97)
RMSE:          1.48 Borg points
MAE:           1.17 Borg points
N Samples:     855
N Activities:  65
N Features:    48
CV Method:     GroupKFold (5 folds)
```

### 10s Windows (Comparison)

**XGBoost Results:**
```
Pearson r:     0.548 (p = 1.2e-34)
RMSE:          1.65 Borg points
MAE:           1.36 Borg points
N Samples:     424
N Activities:  61
N Features:    51
```

**Ridge Regression Results:**
```
Pearson r:     0.567 (p = 2.0e-37)
RMSE:          1.68 Borg points
MAE:           1.30 Borg points
N Samples:     424
N Activities:  61
N Features:    51
```

---

## Feature Selection Results

### 5s Windows

**Raw Features:** 270+ (IMU 90 + PPG 130 + EDA 50)
**Selected Features:** 48
**Selection Method:** Top 100 by correlation → correlation pruning (0.90 threshold)

**Distribution by Modality:**
| Modality | Features | % |
|----------|----------|---|
| PPG (all colors) | 19 | 40% |
| IMU (bioz + wrist) | 19 | 40% |
| EDA | 8 | 17% |
| HRV | 2 | 4% |

**Top 10 Features by XGBoost Importance:**
1. `ppg_green_range` - 0.1824
2. `ppg_green_p95` - 0.0949
3. `acc_x_dyn__cardinality` - 0.0642
4. `eda_stress_skin_max` - 0.0456
5. `ppg_green_trim_mean_10` - 0.0398
6. `acc_y_dyn__harmonic_mean_of_abs` - 0.0374
7. `ppg_infra_p95` - 0.0321
8. `eda_phasic_energy` - 0.0298
9. `acc_z_dyn__lower_complete_moment` - 0.0287
10. `ppg_green_ddx_kurtosis` - 0.0256

**Top 10 Features by Ridge |Coefficient|:**
1. `ppg_green_p95` - 0.8487
2. `acc_x_dyn__cardinality` - 0.6033
3. `ppg_red_signal_energy` - 0.3918
4. `ppg_infra_n_peaks` - 0.3738
5. `acc_x_dyn__quantile_0.9` - 0.3500
6. `ppg_infra_shape_factor` - 0.3451
7. `eda_cc_min` - 0.3200
8. `eda_stress_skin_max` - 0.3120
9. `acc_y_dyn__harmonic_mean_of_abs` - 0.2254
10. `ppg_infra_ddx_kurtosis` - 0.2251

---

## 📋 Detailed Stage-by-Stage Breakdown

### [1️⃣ PREPROCESSING](01_PREPROCESSING.md)

**What it does:** Loads raw sensor data, applies noise reduction, resamples to target frequency

**Modalities:**

| Modality | Source | Target FS | Cleaning | Special Processing |
|----------|--------|-----------|----------|-------------------|
| **IMU Bioz** | Chest | 32 Hz | Noise filter (5 Hz), gravity removal | Dynamic acceleration |
| **IMU Wrist** | Wrist | 32 Hz | Noise filter (5 Hz), gravity removal | Dynamic acceleration |
| **PPG Green** | Wrist | 32 Hz | Butterworth filter | NO HPF (strong signal) |
| **PPG Infra** | Wrist | 32 Hz | Butterworth filter | HPF 0.5 Hz |
| **PPG Red** | Wrist | 32 Hz | Butterworth filter | HPF 0.5 Hz |
| **EDA** | Bioz | 32 Hz | Butterworth filter | Stress skin + CC signals |
| **RR** | Bioz | ~1 Hz | Basic validation | Respiratory intervals |

### [2️⃣ WINDOWING](02_WINDOWING.md)

**Parameters:**
- Window length: 5s (primary), 10s (comparison)
- Overlap: 10% (step = 4.5s for 5s windows)
- Output: Windows with t_start, t_center, t_end metadata

### [3️⃣ FEATURE EXTRACTION](03_FEATURE_EXTRACTION.md)

**Feature Types:**
- **Statistical:** mean, std, min, max, percentiles, IQR, skew, kurtosis
- **Signal:** zero-crossing rate, energy, cardinality, entropy
- **Derivative:** dx_mean, dx_std, ddx_mean, ddx_std
- **HRV:** SDNN, RMSSD, pNN50, LF/HF ratio (for PPG)
- **EDA-specific:** SCL, SCR count, phasic energy, tonic slope

### [4️⃣ ALIGNMENT & FUSION](04_ALIGNMENT_AND_FUSION.md)

**Alignment:**
- Match window t_center with ADL activity timestamps
- Extract Borg effort rating from ADL labels
- Filter to windows within ADL recording time

**Fusion:**
- Combine features across modalities by t_center
- Tolerance: 2s (5s windows), 5s (10s windows)
- Handle missing modalities (NaN for missing)

### [5️⃣ FEATURE SELECTION](05_FEATURE_SELECTION.md)

**Method:**
1. Select top 100 features by correlation with Borg
2. Within each modality, prune redundant features (r > 0.90)
3. Keep feature with highest target correlation from each cluster

**Result:** 270+ → 48 features (5s) / 51 features (10s)

### [6️⃣ TRAINING](06_TRAINING.md)

**GroupKFold Cross-Validation:**
- 5 folds, grouped by activity ID
- Activity ID = subject + Borg level transitions
- Prevents data leakage between correlated windows

**Models:**
- XGBoost: Tree-based ensemble, handles non-linear relationships
- Ridge: Linear with L2 regularization, interpretable coefficients

### [7️⃣ PERFORMANCE METRICS](07_PERFORMANCE_METRICS.md)

**Primary Metrics:**
- Pearson r: Correlation between predicted and actual Borg
- MAE: Mean absolute error in Borg points
- RMSE: Root mean square error in Borg points

---

## 3 Elderly Subjects

| Subject | Data Path | Windows (5s) | Labeled |
|---------|-----------|--------------|---------|
| sim_elderly3 | parsingsim3 | ~350 | ~280 |
| sim_elderly4 | parsingsim4 | ~300 | ~290 |
| sim_elderly5 | parsingsim5 | ~300 | ~285 |
| **Combined** | - | 855 | 855 |

---

## Technical Specifications

### Hardware Requirements
- **Memory:** ~500MB for full pipeline
- **CPU:** Any modern processor
- **Storage:** ~100MB for output files

### Software Requirements
- **Python:** 3.8+
- **Key Libraries:**
  - xgboost >= 1.5
  - scikit-learn >= 1.0
  - pandas >= 1.3
  - numpy >= 1.20
  - scipy >= 1.7
  - pyyaml >= 5.0

### Timing
- Full pipeline (3 subjects, 5s windows): ~5 minutes
- Single window size run: ~2 minutes
- Training only: ~30 seconds

---

## Limitations & Future Work

### Current Limitations
1. **Elderly patients only** - Need validation on healthy/severe
2. **Short recordings** - ~25 minutes per subject
3. **Borg 0-8 range** - Not full 0-20 scale
4. **HRV validity** - 5s borderline for ultra-short HRV

### Future Improvements
1. Add healthy and severe patient cohorts
2. Implement leave-one-subject-out CV
3. Test longer window sizes with more data
4. Add real-time inference mode
5. Deploy as REST API

---

## References

1. **Borg Scale:** Borg, G. (1982). Psychophysical bases of perceived exertion. Medicine & Science in Sports & Exercise.
2. **HRV Guidelines:** Task Force (1996). Heart rate variability: Standards of measurement. Circulation.
3. **XGBoost:** Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
4. **GroupKFold:** Scikit-learn documentation on grouped cross-validation.
