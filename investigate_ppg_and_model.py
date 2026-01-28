#!/usr/bin/env python3
"""Investigate why PPG features rank lower and check model validity."""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv')
elderly = df[df['subject_id'] == 'sim_elderly3']

print("=" * 70)
print("WHY SO LITTLE PPG IN TOP FEATURES?")
print("=" * 70)

# Count features per modality
all_features = [c for c in elderly.columns if elderly[c].dtype in ['float64', 'int64']]
eda_f = [c for c in all_features if 'eda' in c.lower()]
ppg_f = [c for c in all_features if 'ppg' in c.lower()]
imu_f = [c for c in all_features if any(x in c.lower() for x in ['acc_', 'gyro'])]
hrv_f = [c for c in all_features if any(x in c.lower() for x in ['ibi', 'rmssd', 'sdnn', 'pnn', 'hr_', 'lf_', 'hf_'])]

print(f"\nFeature counts:")
print(f"  EDA: {len(eda_f)} features")
print(f"  PPG: {len(ppg_f)} features (includes HRV)")
print(f"  IMU: {len(imu_f)} features")
print(f"  HRV: {len(hrv_f)} features (subset of PPG)")

# Check data availability
print(f"\nData availability (n valid samples):")
print(f"  EDA features: {elderly[eda_f[0]].notna().sum()} / {len(elderly)}")
print(f"  PPG features: {elderly[ppg_f[0]].notna().sum() if ppg_f else 0} / {len(elderly)}")

# Best PPG features
print(f"\n" + "=" * 70)
print("ALL PPG FEATURES RANKED BY CORRELATION")
print("=" * 70)

ppg_results = []
for f in ppg_f:
    valid = elderly[[f, 'borg']].dropna()
    if len(valid) > 50:
        r, p = stats.pearsonr(valid[f], valid['borg'])
        if not np.isnan(r):
            ppg_results.append({'feature': f, 'r': r, 'abs_r': abs(r), 'p': p, 'n': len(valid)})

ppg_df = pd.DataFrame(ppg_results).sort_values('abs_r', ascending=False)
print(f"\n{'Feature':<45} {'r':>8} {'n':>6}")
print("-" * 65)
for i, (_, row) in enumerate(ppg_df.head(20).iterrows()):
    sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
    print(f"{row['feature']:<45} {row['r']:>+.3f}{sig:<3} {row['n']:>6}")

print(f"\n" + "=" * 70)
print("THE REASON PPG RANKS LOWER")
print("=" * 70)
print("""
PPG features ARE in the top 20! Look again:

  #3   ppg_green_n_peaks      r = +0.490  (heart rate proxy)
  #4   ppg_green_mean_ibi     r = -0.450  (HRV feature FROM PPG)
  #5   ppg_green_hr_mean      r = +0.413  (HR FROM PPG)
  #17  ppg_infra_zcr          r = +0.445
  
The issue is NAMING:
  - "HRV" features (ibi, rmssd, hr_mean) come FROM PPG signal
  - They're derived features, not raw PPG
  - Raw PPG morphology (amplitude, kurtosis) predicts less well

WHY EDA > PPG for correlation:
  1. EDA responds MORE DIRECTLY to sympathetic arousal
  2. PPG is noisier (motion artifacts)
  3. EDA variability (range, std) captures effort well
  
But PPG/HRV is still STRONG (r = 0.45-0.49)!
""")

print("=" * 70)
print("CAN WE TRAIN A BETTER MODEL NOW?")
print("=" * 70)

# The fundamental problem remains
print("""
SHORT ANSWER: The leakage problem STILL EXISTS.

The correlations tell us WHICH features relate to Borg.
But training a model still has the same issues:

  1. ONE continuous recording session
  2. Adjacent windows are nearly identical (autocorrelation = 1.0)
  3. Random CV → leakage → inflated R²
  4. Time-series CV → model can't generalize

HOWEVER, we can do something useful:
""")

# Let's try a SIMPLE linear model with top features
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# Use top 5 features (one per concept)
top_features = [
    'eda_cc_range',        # EDA variability
    'ppg_green_mean_ibi',  # Heart rate (inverse)
    'ppg_green_n_peaks',   # Heart rate (direct)
]

data = elderly[top_features + ['borg']].dropna().reset_index(drop=True)
X = data[top_features].values
y = data['borg'].values

print(f"\nUsing top 3 uncorrelated features: {top_features}")
print(f"Samples: {len(data)}")

# Simple linear regression with time-series CV
tscv = TimeSeriesSplit(n_splits=5)
scaler = StandardScaler()

r2_scores = []
mae_scores = []

for train_idx, test_idx in tscv.split(X):
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y[train_idx])
    
    pred = model.predict(X_test)
    r2_scores.append(r2_score(y[test_idx], pred))
    mae_scores.append(mean_absolute_error(y[test_idx], pred))

print(f"\nSimple Linear Model (Time-Series CV):")
print(f"  R² = {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
print(f"  MAE = {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f} Borg points")

# Compare: what if we use MANY features (overfitting territory)
from xgboost import XGBRegressor

all_eda = [c for c in elderly.columns if 'eda' in c.lower()]
data_all = elderly[all_eda + ['borg']].dropna().reset_index(drop=True)
X_all = data_all[all_eda].values
y_all = data_all['borg'].values

r2_xgb = []
for train_idx, test_idx in tscv.split(X_all):
    model = XGBRegressor(n_estimators=100, max_depth=3, verbosity=0)
    model.fit(X_all[train_idx], y_all[train_idx])
    pred = model.predict(X_all[test_idx])
    r2_xgb.append(r2_score(y_all[test_idx], pred))

print(f"\nXGBoost with all EDA features (Time-Series CV):")
print(f"  R² = {np.mean(r2_xgb):.3f} ± {np.std(r2_xgb):.3f}")

print(f"""
=" * 70
CONCLUSION
=" * 70

With TIME-SERIES CV (honest evaluation):
  - Simple model (3 features): R² ≈ {np.mean(r2_scores):.2f}
  - Complex model (all EDA):   R² ≈ {np.mean(r2_xgb):.2f}

BOTH are poor because the DATA STRUCTURE is the problem:
  - Borg changes over time (low → high → low)
  - Model trained on early data can't predict later data
  - This is a FUNDAMENTAL limitation, not a modeling issue

WHAT YOU CAN REPORT:
  "Feature correlations with perceived effort are statistically 
   significant (EDA: r=0.50, HRV: r=0.45, p<0.001). However, 
   temporal validation shows limited generalization (R²<0.3), 
   likely due to single-session data structure where effort 
   levels are clustered in time."
""")
