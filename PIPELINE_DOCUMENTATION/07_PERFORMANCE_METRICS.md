# Stage 7: Performance Metrics & Evaluation

## Overview

This document provides comprehensive analysis of model performance across window sizes, including statistical interpretation and comparison metrics.

---

## Window Size Comparison Summary

| Window | N Samples | N Activities | Features | XGBoost r | XGBoost MAE | Ridge r | Ridge MAE |
|--------|-----------|--------------|----------|-----------|-------------|---------|-----------|
| **5s** | 855 | 65 | 48 | **0.626** | 1.22 | **0.644** | 1.17 |
| 10s | 424 | 61 | 51 | 0.548 | 1.36 | 0.567 | 1.30 |
| 30s | 100 | 40 | 20 | 0.364 | 1.34 | 0.184 | 1.69 |

**Key Finding:** 5s windows achieve best performance across all metrics.

---

## 5s Windows - Primary Results (Best)

### XGBoost Performance

```
Model:           XGBoost Regressor
CV Method:       GroupKFold (5 folds)
N Samples:       855
N Activities:    65 (used as groups)
N Features:      48

Metrics:
  Pearson r:     0.626
  p-value:       2.8e-90 (highly significant)
  R²:            0.392 (39.2% variance explained)
  RMSE:          1.52 Borg points
  MAE:           1.22 Borg points
```

### Ridge Regression Performance

```
Model:           Ridge Regression (alpha=1.0)
CV Method:       GroupKFold (5 folds)
N Samples:       855
N Activities:    65 (used as groups)
N Features:      48 (standardized)

Metrics:
  Pearson r:     0.644
  p-value:       1.4e-97 (highly significant)
  R²:            0.415 (41.5% variance explained)
  RMSE:          1.48 Borg points
  MAE:           1.17 Borg points
```

---

## 10s Windows - Comparison Results

### XGBoost Performance

```
Model:           XGBoost Regressor
CV Method:       GroupKFold (5 folds)
N Samples:       424
N Activities:    61 (used as groups)
N Features:      51

Metrics:
  Pearson r:     0.548
  p-value:       1.21e-34 (highly significant)
  R²:            0.300 (30.0% variance explained)
  RMSE:          1.65 Borg points
  MAE:           1.36 Borg points
```

### Ridge Regression Performance

```
Model:           Ridge Regression (alpha=1.0)
CV Method:       GroupKFold (5 folds)
N Samples:       424
N Activities:    61 (used as groups)
N Features:      51 (standardized)

Metrics:
  Pearson r:     0.567
  p-value:       2.01e-37 (highly significant)
  R²:            0.321 (32.1% variance explained)
  RMSE:          1.68 Borg points
  MAE:           1.30 Borg points
```

---

## 30s Windows - Poor Performance

### XGBoost Performance

```
Model:           XGBoost Regressor
CV Method:       GroupKFold (5 folds)
N Samples:       100
N Activities:    40 (used as groups)
N Features:      20

Metrics:
  Pearson r:     0.364
  p-value:       2.01e-04 (significant)
  R²:            0.132 (13.2% variance explained)
  RMSE:          1.68 Borg points
  MAE:           1.34 Borg points
```

### Ridge Regression Performance

```
Model:           Ridge Regression (alpha=1.0)
CV Method:       GroupKFold (5 folds)
N Samples:       100
N Activities:    40 (used as groups)
N Features:      20 (standardized)

Metrics:
  Pearson r:     0.184
  p-value:       0.067 (NOT significant at α=0.05)
  R²:            0.034 (3.4% variance explained)
  RMSE:          2.11 Borg points
  MAE:           1.69 Borg points
```

**Note:** 30s windows lost sim_elderly5 labels due to alignment issues (0 labeled samples).

---

## Metric Interpretation

### Pearson Correlation (r)

**Definition:**
$$r = \frac{\sum(y_{true} - \bar{y}_{true})(y_{pred} - \bar{y}_{pred})}{\sqrt{\sum(y_{true} - \bar{y}_{true})^2 \sum(y_{pred} - \bar{y}_{pred})^2}}$$

**Interpretation for effort estimation:**
| r Value | Interpretation |
|---------|----------------|
| 0.80+ | Excellent |
| 0.60-0.80 | Good |
| 0.40-0.60 | Moderate |
| 0.20-0.40 | Weak |
| <0.20 | Very weak |

**Our results:**
- 5s: r = 0.626-0.644 → **Good**
- 10s: r = 0.548-0.567 → **Moderate**
- 30s: r = 0.184-0.364 → **Weak to Very Weak**

### Mean Absolute Error (MAE)

**Definition:**
$$MAE = \frac{1}{n}\sum|y_{true} - y_{pred}|$$

**Interpretation:**
- MAE = 1.17-1.22 (5s) means average prediction is within ~1.2 Borg points
- Borg scale 0-8, so error is ~15% of scale range
- Clinically: Can distinguish "rest" from "moderate" but may confuse adjacent levels

### Root Mean Square Error (RMSE)

**Definition:**
$$RMSE = \sqrt{\frac{1}{n}\sum(y_{true} - y_{pred})^2}$$

**Interpretation:**
- RMSE > MAE indicates some predictions have larger errors
- RMSE = 1.48-1.52 (5s) vs MAE = 1.17-1.22 → moderate outliers
- Ratio RMSE/MAE ≈ 1.26 (reasonable, < 1.4 is good)

---

## Why 5s Outperforms Larger Windows

### 1. Sample Size Effect

```
Window Size    N Samples    Statistical Power
5s             855          High
10s            424          Medium
30s            100          Low
```

With 855 samples, 5s windows have ~2x more data for learning patterns.

### 2. Activity Duration

```
Typical ADL Activity Duration: 10-60 seconds
5s window:  Captures 2-12 windows per activity
10s window: Captures 1-6 windows per activity
30s window: Captures 0-2 windows per activity
```

Longer windows average across activity transitions, losing signal.

### 3. Effort Dynamics

```
Effort changes can occur within seconds:
- Standing up: 2-3 second effort spike
- Walking: Sustained moderate effort
- Sitting down: Brief high effort then rest

5s captures these dynamics; 30s averages them out.
```

### 4. HRV Validity

```
Task Force 1996 Recommendations:
- Standard HRV: 5 minutes minimum
- Ultra-short HRV: 10+ beats required

At 60 BPM:
- 5s window: ~5 beats (borderline)
- 10s window: ~10 beats (minimum valid)
- 30s window: ~30 beats (reliable)

Trade-off: 30s has better HRV but fewer samples.
Our results suggest: Sample size > HRV precision.
```

---

## Cross-Validation Details

### GroupKFold Strategy

**Problem with standard KFold:**
- Adjacent windows are correlated (overlap)
- Same activity appears in train and test
- Model learns temporal patterns, not effort-feature relationships
- Results in over-optimistic performance

**GroupKFold Solution:**
- Group windows by activity ID
- Activity ID = subject + Borg level transitions
- All windows from one activity in same fold
- Prevents data leakage

**Group Distribution (5s):**
```
Subject        N Activities
sim_elderly3   ~22
sim_elderly4   ~21
sim_elderly5   ~22
Total          65
```

### Why p-values Are Extremely Low

With N=855 samples, even moderate correlations yield tiny p-values:

```python
from scipy.stats import pearsonr
import numpy as np

# r=0.626 with n=855
r, p = 0.626, 2.8e-90

# t-statistic: t = r * sqrt(n-2) / sqrt(1-r²)
t = 0.626 * np.sqrt(855-2) / np.sqrt(1-0.626**2)
# t ≈ 23.7

# With df=853, this is highly significant
```

**Interpretation:**
- p < 0.001 confirms correlation is non-zero
- Does NOT mean correlation is "strong" in practical sense
- Large N inflates statistical significance
- Focus on r magnitude and MAE for practical interpretation

---

## Per-Subject Breakdown

### 5s Windows

| Subject | N Labeled | Contribution |
|---------|-----------|--------------|
| sim_elderly3 | ~280 | 33% |
| sim_elderly4 | ~290 | 34% |
| sim_elderly5 | ~285 | 33% |

### 10s Windows

| Subject | N Labeled | Contribution |
|---------|-----------|--------------|
| sim_elderly3 | 143 | 34% |
| sim_elderly4 | 151 | 36% |
| sim_elderly5 | 130 | 31% |

### 30s Windows

| Subject | N Labeled | Contribution |
|---------|-----------|--------------|
| sim_elderly3 | 49 | 49% |
| sim_elderly4 | 51 | 51% |
| sim_elderly5 | 0 | 0% ⚠️ |

**Note:** sim_elderly5 lost all labels at 30s due to ADL-window alignment issues.

---

## Feature Importance Analysis

### Top 10 by XGBoost (5s)

| Rank | Feature | Importance | Modality |
|------|---------|------------|----------|
| 1 | ppg_green_range | 0.1824 | PPG |
| 2 | ppg_green_p95 | 0.0949 | PPG |
| 3 | acc_x_dyn__cardinality | 0.0642 | IMU |
| 4 | eda_stress_skin_max | 0.0456 | EDA |
| 5 | ppg_green_trim_mean_10 | 0.0398 | PPG |
| 6 | acc_y_dyn__harmonic_mean_of_abs | 0.0374 | IMU |
| 7 | ppg_infra_p95 | 0.0321 | PPG |
| 8 | eda_phasic_energy | 0.0298 | EDA |
| 9 | acc_z_dyn__lower_complete_moment | 0.0287 | IMU |
| 10 | ppg_green_ddx_kurtosis | 0.0256 | PPG |

**Interpretation:**
- PPG dominates (especially green channel)
- IMU and EDA provide complementary information
- Top 10 features account for ~60% of importance

### Top 10 by Ridge |Coefficient| (5s)

| Rank | Feature | |Coefficient| | Direction |
|------|---------|--------------|-----------|
| 1 | ppg_green_p95 | 0.8487 | Negative |
| 2 | acc_x_dyn__cardinality | 0.6033 | Positive |
| 3 | ppg_red_signal_energy | 0.3918 | Positive |
| 4 | ppg_infra_n_peaks | 0.3738 | Positive |
| 5 | acc_x_dyn__quantile_0.9 | 0.3500 | Negative |
| 6 | ppg_infra_shape_factor | 0.3451 | Positive |
| 7 | eda_cc_min | 0.3200 | Positive |
| 8 | eda_stress_skin_max | 0.3120 | Negative |
| 9 | acc_y_dyn__harmonic_mean_of_abs | 0.2254 | Negative |
| 10 | ppg_infra_ddx_kurtosis | 0.2251 | Negative |

**Interpretation:**
- ppg_green_p95 negative: Higher PPG amplitude = lower effort (resting state)
- acc_x_dyn__cardinality positive: More movement variety = higher effort
- eda_stress_skin_max negative: Counter-intuitive, may reflect data artifacts

---

## Comparison with Literature

### Effort Estimation from Wearables

| Study | Modalities | r / R² | Notes |
|-------|------------|--------|-------|
| Our work (5s) | PPG+EDA+IMU | r=0.64 | 3 elderly, GroupKFold CV |
| Typical HRV-only | HRV | r~0.50 | Single modality |
| IMU-only | Accelerometer | r~0.45 | Physical activity bias |
| Multi-modal SOTA | Various | r=0.65-0.75 | Lab conditions |

**Assessment:** Our results (r=0.64) are competitive with literature, especially considering:
- Real-world ADL data (not lab)
- Elderly patients (noisier signals)
- Strict GroupKFold CV (no leakage)

---

## Limitations

### 1. Correlation ≠ Causation
- Features correlate with Borg but may not cause it
- Confounding variables (time of day, activity type) not controlled

### 2. Limited Generalization
- 3 elderly subjects from same study
- May not generalize to healthy/young
- Need leave-one-subject-out validation

### 3. Borg Scale Subjectivity
- Self-reported effort varies by person
- Same activity may feel different to different people
- 0-8 scale (not full 0-20)

### 4. Short Recordings
- ~25 minutes per subject
- Limited activity variety
- May not capture all effort levels

---

## Conclusions

### Primary Findings

1. **5s windows optimal:** r=0.626-0.644, MAE=1.17-1.22
2. **10s provides comparison:** r=0.548-0.567, MAE=1.30-1.36
3. **30s unsuitable:** Low sample size, lost subject data

### Model Selection

- **Use Ridge** for interpretability (coefficient signs, linear relationships)
- **Use XGBoost** for deployment (slightly better for non-linear patterns)
- Both perform similarly; Ridge slightly better on this dataset

### Practical Application

- Suitable for effort monitoring in elderly ADLs
- MAE ~1.2 Borg points: Can distinguish major effort levels
- Not suitable for fine-grained effort discrimination (e.g., Borg 5 vs 6)

---

## Output Files

Results saved to `/Users/pascalschlegel/data/interim/elderly_combined/`:

```
xgboost_results/
├── summary.yaml          # Metrics (r, MAE, RMSE)
├── feature_importance.csv # Feature rankings
└── predictions.csv        # y_true, y_pred per sample

linear_results/
├── summary.yaml          # Metrics
├── coefficients.csv      # Feature coefficients
└── predictions.csv       # Predictions

xgboost_results_10.0s/    # 10s window results
ridge_results_10.0s/

xgboost_results_30.0s/    # 30s window results
ridge_results_30.0s/
```
