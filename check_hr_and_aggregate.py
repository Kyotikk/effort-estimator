#!/usr/bin/env python3
"""
Check HR features and aggregate to activity level
Compare window-level vs activity-level results
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from pathlib import Path

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined/elderly_aligned_5.0s.csv').dropna(subset=['borg'])

print("="*70)
print("1. CHECK FOR HR FEATURES")
print("="*70)

# Find HR-related columns
hr_cols = [c for c in df.columns if 'hr' in c.lower() or 'heart' in c.lower() or 'rr' in c.lower() or 'ibi' in c.lower()]
print(f"\nHR-related columns found: {len(hr_cols)}")
for c in hr_cols[:20]:
    if df[c].notna().sum() > 0:
        r, _ = pearsonr(df[c].fillna(0), df['borg'])
        print(f"  {c[:50]:<50} r={r:+.3f}")

# Check PPG features that might be HR proxies
ppg_hr_cols = [c for c in df.columns if 'peak' in c.lower() or 'ibi' in c.lower() or 'rate' in c.lower()]
print(f"\nPPG peak/rate features: {len(ppg_hr_cols)}")
for c in ppg_hr_cols[:10]:
    if df[c].notna().sum() > 100:
        r, _ = pearsonr(df[c].fillna(0), df['borg'])
        print(f"  {c[:50]:<50} r={r:+.3f}")

print("\n" + "="*70)
print("2. CHECK ACTIVITY LABELS")
print("="*70)

# Check if we have activity information
if 'activity_id' in df.columns:
    print(f"\nActivity IDs: {df['activity_id'].nunique()} unique")
    print(f"Activity distribution:")
    print(df.groupby(['subject', 'activity_id']).size().unstack(fill_value=0))
else:
    print("\n⚠️ No 'activity_id' column found")
    print(f"Available columns with 'act': {[c for c in df.columns if 'act' in c.lower()]}")

print("\n" + "="*70)
print("3. AGGREGATE TO ACTIVITY LEVEL")
print("="*70)

# Load selected features
feat_cols = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined/qc_5.0s/features_selected_pruned.csv', header=None)[0].tolist()
feat_cols = [c for c in feat_cols if c in df.columns]

# Add activity grouping
if 'activity_id' in df.columns:
    group_col = 'activity_id'
elif 'activity' in df.columns:
    group_col = 'activity'
else:
    # Create activity groups based on consecutive Borg values
    df['activity_group'] = (df['borg'].diff().abs() > 0.1).cumsum()
    group_col = 'activity_group'
    print(f"Created activity_group from Borg transitions: {df[group_col].nunique()} groups")

# Aggregate to activity level
print(f"\nAggregating from windows to activities...")
agg_dict = {feat: 'mean' for feat in feat_cols}
agg_dict['borg'] = 'mean'
agg_dict['subject'] = 'first'

df_activity = df.groupby([group_col]).agg(agg_dict).reset_index()
print(f"Window-level samples: {len(df)}")
print(f"Activity-level samples: {len(df_activity)}")

print("\n" + "="*70)
print("4. COMPARE WINDOW vs ACTIVITY LEVEL RESULTS")
print("="*70)

X_win = df[feat_cols].values
y_win = df['borg'].values
subj_win = df['subject'].values

X_act = df_activity[feat_cols].values
y_act = df_activity['borg'].values
subj_act = df_activity['subject'].values

def run_loso_calibrated(X, y, subjects, name):
    """LOSO with linear calibration"""
    all_p, all_t = [], []
    for s in np.unique(subjects):
        m = subjects != s
        sc = StandardScaler()
        mdl = Ridge(alpha=1.0)
        mdl.fit(sc.fit_transform(X[m]), y[m])
        y_raw = mdl.predict(sc.transform(X[~m]))
        y_te = y[~m]
        
        # Linear calibration on first 20%
        n_cal = max(5, int(0.20 * len(y_te)))
        if len(y_te) > n_cal:
            cal = LinearRegression()
            cal.fit(y_raw[:n_cal].reshape(-1,1), y_te[:n_cal])
            y_cal = cal.predict(y_raw.reshape(-1,1))
            all_p.extend(y_cal[n_cal:])
            all_t.extend(y_te[n_cal:])
    
    if len(all_t) > 10:
        r = pearsonr(all_t, all_p)[0]
        mae = np.mean(np.abs(np.array(all_t) - np.array(all_p)))
        within1 = np.mean(np.abs(np.array(all_t) - np.array(all_p)) <= 1) * 100
        print(f"{name:<30} r={r:.3f}, MAE={mae:.2f}, ±1 Borg={within1:.1f}%, n={len(all_t)}")
        return r
    else:
        print(f"{name:<30} Not enough samples")
        return 0

print("\nWith Linear Calibration (20%):")
r_win = run_loso_calibrated(X_win, y_win, subj_win, "Window-level (5s)")
r_act = run_loso_calibrated(X_act, y_act, subj_act, "Activity-level (aggregated)")

print("\n" + "="*70)
print("5. SIMPLE HR-BASED FORMULA (like the plot)")
print("="*70)

# Try to find HR or IBI features
hr_feature = None
for candidate in ['ppg_green_mean_ibi', 'ppg_green_hr', 'hr_mean', 'ppg_green_n_peaks']:
    if candidate in df.columns and df[candidate].notna().sum() > 100:
        hr_feature = candidate
        break

if hr_feature:
    print(f"\nUsing HR proxy: {hr_feature}")
    
    # Calculate HR delta per activity (relative to baseline)
    df['hr_proxy'] = df[hr_feature]
    
    # Per subject baseline (resting)
    for subj in df['subject'].unique():
        subj_mask = df['subject'] == subj
        # Use lowest Borg as baseline
        baseline_mask = subj_mask & (df['borg'] <= df.loc[subj_mask, 'borg'].quantile(0.1))
        if baseline_mask.sum() > 0:
            baseline = df.loc[baseline_mask, hr_feature].mean()
            df.loc[subj_mask, 'hr_delta'] = df.loc[subj_mask, hr_feature] - baseline
    
    if 'hr_delta' in df.columns:
        r_hr, _ = pearsonr(df['hr_delta'].fillna(0), df['borg'])
        print(f"HR_delta correlation with Borg: r={r_hr:.3f}")
        
        # Activity-level
        df_act2 = df.groupby(group_col).agg({'hr_delta': 'mean', 'borg': 'mean', 'subject': 'first'}).dropna()
        if len(df_act2) > 10:
            r_hr_act, _ = pearsonr(df_act2['hr_delta'], df_act2['borg'])
            print(f"HR_delta at activity level: r={r_hr_act:.3f}")
else:
    print("\n⚠️ No HR/IBI features found in data")
    print("Available PPG features:")
    ppg_feats = [c for c in df.columns if 'ppg' in c.lower()][:10]
    for f in ppg_feats:
        print(f"  {f}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
Window-level (5s):      r ≈ {r_win:.2f}
Activity-level:         r ≈ {r_act:.2f}

The r=0.84 plot you showed likely used:
1. Activity-level aggregation (not windows)
2. Actual HR from ECG or reliable PPG peak detection
3. Simple formula: HR_delta × √duration
4. Possibly single subject or lab conditions

To improve your results:
1. ✓ We can aggregate to activity level
2. ? Need reliable HR extraction from PPG
3. ? Need activity duration information
""")
