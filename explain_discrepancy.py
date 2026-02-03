#!/usr/bin/env python3
"""
UNDERSTAND THE DISCREPANCY: Why 0.67 vs 0.29?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from ml.feature_selection import select_features_consistent

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Load data
all_dfs = []
for i in [1,2,3,4,5]:
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = f'elderly{i}'
        all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True).dropna(subset=['borg'])

# Get IMU features (best performing before)
imu_features = [c for c in df_all.columns if ('acc' in c.lower() or 'gyro' in c.lower()) 
                and df_all[c].dtype in ['float64', 'int64']]

print("="*80)
print("WHY THE DISCREPANCY? Let's trace through carefully")
print("="*80)

# The r=0.67 came from THIS evaluation method:
def run_loso_POOLED(df, features, cal_frac=0.2):
    """
    LOSO with calibration - POOLED correlation at the end.
    This is what gave r=0.67 before.
    """
    subjects = sorted(df['subject'].unique())
    valid_features = [f for f in features if f in df.columns and df[f].notna().mean() > 0.5]
    
    ALL_PREDS = []  # Collect ALL predictions
    ALL_TRUE = []   # Collect ALL true values
    per_sub = {}
    
    for test_sub in subjects:
        train_df = df[df['subject'] != test_sub]
        test_df = df[df['subject'] == test_sub]
        
        n_test = len(test_df)
        n_cal = max(5, int(n_test * cal_frac))
        idx = np.random.permutation(n_test)
        cal_df = test_df.iloc[idx[:n_cal]]
        eval_df = test_df.iloc[idx[n_cal:]]
        
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X_train = scaler.fit_transform(imputer.fit_transform(train_df[valid_features]))
        X_cal = scaler.transform(imputer.transform(cal_df[valid_features]))
        X_eval = scaler.transform(imputer.transform(eval_df[valid_features]))
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, train_df['borg'].values)
        
        # Calibrate
        preds_cal = model.predict(X_cal)
        calibrator = LinearRegression()
        calibrator.fit(preds_cal.reshape(-1, 1), cal_df['borg'].values)
        
        # Predict on eval
        preds = calibrator.predict(model.predict(X_eval).reshape(-1, 1))
        
        ALL_PREDS.extend(preds)
        ALL_TRUE.extend(eval_df['borg'].values)
        
        # Per-subject correlation
        r_sub, _ = pearsonr(preds, eval_df['borg'].values)
        per_sub[test_sub] = r_sub
    
    # POOLED correlation across ALL subjects
    r_pooled, _ = pearsonr(ALL_PREDS, ALL_TRUE)
    
    return r_pooled, per_sub, len(ALL_PREDS)

# Run with IMU (what gave 0.67 before)
print("\n--- IMU ONLY (what gave r=0.67 before) ---")
r_pooled, per_sub, n = run_loso_POOLED(df_all, imu_features, cal_frac=0.2)

print(f"POOLED r = {r_pooled:.3f} (n={n})")
print("\nPer-subject r:")
for sub, r in per_sub.items():
    print(f"  {sub}: r = {r:.3f}")
avg_per_sub = np.mean(list(per_sub.values()))
print(f"\nAverage of per-subject r: {avg_per_sub:.3f}")

print(f"""
================================================================================
THE DISCREPANCY EXPLAINED
================================================================================

POOLED r = {r_pooled:.3f}  ← This is what we reported as "0.67"
Avg per-subject r = {avg_per_sub:.3f}  ← This is the REAL generalization

WHY IS POOLED r HIGHER?
─────────────────────────
The calibration step fits a linear correction for EACH subject:
  pred_calibrated = a * pred_raw + b

This means:
1. Each subject's predictions are shifted/scaled to match their Borg range
2. When you pool all calibrated predictions, they all "line up"
3. But the PER-SUBJECT pattern (within each subject) is still weak

Example:
- elderly1: Borg range [0, 5], model predicts [2.5, 3.5] → calibrated to [0, 5]
- elderly2: Borg range [1, 6], model predicts [2.5, 3.5] → calibrated to [1, 6]

After calibration, predictions span [0, 6] and correlate well with true [0, 6]
BUT within each subject, the pattern might still be wrong!

THE HONEST METRIC:
─────────────────────
Average per-subject r = {avg_per_sub:.3f}

This means: For a NEW person, expect r ≈ {avg_per_sub:.2f} (with calibration)
Without calibration: r ≈ 0.29

================================================================================
""")

# Now let's also check: what about consistent features?
print("\n--- CONSISTENT FEATURES ---")
consistent_features, _ = select_features_consistent(df_all, verbose=False)
r_pooled_c, per_sub_c, n_c = run_loso_POOLED(df_all, consistent_features, cal_frac=0.2)

print(f"POOLED r = {r_pooled_c:.3f}")
print("Per-subject r:")
for sub, r in per_sub_c.items():
    print(f"  {sub}: r = {r:.3f}")
avg_c = np.mean(list(per_sub_c.values()))
print(f"Average per-subject r: {avg_c:.3f}")

print(f"""
================================================================================
FINAL COMPARISON
================================================================================

                          │ POOLED r │ Avg per-sub r │ Interpretation
─────────────────────────────────────────────────────────────────────────────────
IMU features (60)         │   {r_pooled:.3f}   │     {avg_per_sub:.3f}       │ Pooled misleading
Consistent features (7)   │   {r_pooled_c:.3f}   │     {avg_c:.3f}       │ More honest
─────────────────────────────────────────────────────────────────────────────────

WHAT TO REPORT IN THESIS:
1. Pooled r = {r_pooled:.2f} (with calibration) - looks good but misleading
2. Per-subject r = {avg_per_sub:.2f} - this is the real generalization
3. Without calibration: r ≈ 0.29 - true zero-shot performance

The 0.67 number is REAL but it's the pooled metric after calibration.
The more honest number for "can we predict a new person's effort" is {avg_per_sub:.2f}.
""")
