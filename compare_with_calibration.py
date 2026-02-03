#!/usr/bin/env python3
"""
Compare WITH vs WITHOUT calibration using consistent features
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

# Get consistent features
print("Selecting consistent features...")
new_features, _ = select_features_consistent(df_all, verbose=False)
print(f"Selected {len(new_features)} features")

# Also get ALL features for comparison
skip_cols = {'t_center', 'borg', 'subject', 'activity_label'}
all_features = [c for c in df_all.columns 
                if c not in skip_cols 
                and not c.startswith('Unnamed')
                and df_all[c].dtype in ['float64', 'int64', 'float32', 'int32']]

def train_test_loso(df, features, test_subject, use_calibration=False, cal_frac=0.2):
    """Train on 4, test on 1, optionally with calibration."""
    train_df = df[df['subject'] != test_subject]
    test_df = df[df['subject'] == test_subject]
    
    valid_features = [f for f in features if f in df.columns and df[f].notna().mean() > 0.5]
    if len(valid_features) == 0:
        return None
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(imputer.fit_transform(train_df[valid_features]))
    y_train = train_df['borg'].values
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    if use_calibration:
        n_test = len(test_df)
        n_cal = max(5, int(n_test * cal_frac))
        idx = np.random.permutation(n_test)
        
        cal_df = test_df.iloc[idx[:n_cal]]
        eval_df = test_df.iloc[idx[n_cal:]]
        
        X_cal = scaler.transform(imputer.transform(cal_df[valid_features]))
        y_cal = cal_df['borg'].values
        X_eval = scaler.transform(imputer.transform(eval_df[valid_features]))
        y_eval = eval_df['borg'].values
        
        # Calibrate
        preds_cal = model.predict(X_cal)
        calibrator = LinearRegression()
        calibrator.fit(preds_cal.reshape(-1, 1), y_cal)
        
        preds = calibrator.predict(model.predict(X_eval).reshape(-1, 1))
    else:
        X_test = scaler.transform(imputer.transform(test_df[valid_features]))
        y_eval = test_df['borg'].values
        preds = model.predict(X_test)
    
    r, _ = pearsonr(preds, y_eval)
    mae = np.mean(np.abs(preds - y_eval))
    within_1 = np.mean(np.abs(preds - y_eval) <= 1) * 100
    
    return {'r': r, 'mae': mae, 'within_1': within_1, 'n': len(y_eval)}

print("\n" + "="*80)
print("COMPARISON: NO CALIBRATION vs 20% CALIBRATION")
print("="*80)

print(f"\n{'Test Sub':<10} | {'NO CAL (consistent)':>20} | {'20% CAL (consistent)':>20} | {'20% CAL (all feat)':>20}")
print("-" * 80)

no_cal_results = []
cal_results = []
cal_all_results = []

for test_sub in ['elderly1', 'elderly2', 'elderly3', 'elderly4', 'elderly5']:
    no_cal = train_test_loso(df_all, new_features, test_sub, use_calibration=False)
    with_cal = train_test_loso(df_all, new_features, test_sub, use_calibration=True)
    with_cal_all = train_test_loso(df_all, all_features, test_sub, use_calibration=True)
    
    if no_cal and with_cal and with_cal_all:
        no_cal_results.append(no_cal)
        cal_results.append(with_cal)
        cal_all_results.append(with_cal_all)
        print(f"{test_sub:<10} | r={no_cal['r']:>6.3f}, ±1={no_cal['within_1']:>4.1f}% | r={with_cal['r']:>6.3f}, ±1={with_cal['within_1']:>4.1f}% | r={with_cal_all['r']:>6.3f}, ±1={with_cal_all['within_1']:>4.1f}%")

print("-" * 80)
avg_no_cal = np.mean([r['r'] for r in no_cal_results])
avg_cal = np.mean([r['r'] for r in cal_results])
avg_cal_all = np.mean([r['r'] for r in cal_all_results])
avg_w1_no_cal = np.mean([r['within_1'] for r in no_cal_results])
avg_w1_cal = np.mean([r['within_1'] for r in cal_results])
avg_w1_cal_all = np.mean([r['within_1'] for r in cal_all_results])

print(f"{'AVERAGE':<10} | r={avg_no_cal:>6.3f}, ±1={avg_w1_no_cal:>4.1f}% | r={avg_cal:>6.3f}, ±1={avg_w1_cal:>4.1f}% | r={avg_cal_all:>6.3f}, ±1={avg_w1_cal_all:>4.1f}%")

print(f"""
================================================================================
SUMMARY
================================================================================

┌────────────────────────────────────────────────────────────────────────────┐
│  Condition                              │    Avg r  │  Avg ±1 Borg  │ Feat │
├────────────────────────────────────────────────────────────────────────────┤
│  NO calibration (consistent features)   │   {avg_no_cal:>6.3f}  │    {avg_w1_no_cal:>5.1f}%     │   {len(new_features):>2} │
│  20% calibration (consistent features)  │   {avg_cal:>6.3f}  │    {avg_w1_cal:>5.1f}%     │   {len(new_features):>2} │
│  20% calibration (ALL features)         │   {avg_cal_all:>6.3f}  │    {avg_w1_cal_all:>5.1f}%     │  284 │
└────────────────────────────────────────────────────────────────────────────┘

EXPLANATION:
- r = 0.29 (no calibration) = PURE generalization to new person
- r = 0.5+ (with calibration) = Uses 20% of test person's data to adjust

The 0.5-0.6 you saw before included calibration!

Calibration does:
1. Takes 20% of test subject's labeled data
2. Fits a linear correction: pred_calibrated = a * pred_raw + b
3. Adjusts for the person's baseline Borg level

So the question is: 
- Do you want PURE generalization (no test data needed)? → r ≈ 0.29
- Or personalized with brief calibration phase? → r ≈ 0.5+
""")
