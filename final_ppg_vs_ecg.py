#!/usr/bin/env python3
"""
FINAL COMPARISON: PPG vs ECG HR FEATURES
========================================
Proper comparison on the SAME subset of data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("FINAL COMPARISON: PPG vs ECG HR ON SAME DATA")
print("="*80)

# =============================================================================
# LOAD AND MERGE DATA
# =============================================================================

# Load scientific data
all_dfs = []
for i in [1,2,3,4,5]:
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = f'elderly{i}'
        all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True).dropna(subset=['borg'])

# Load ECG HR data
tli_df = pd.read_csv("/Users/pascalschlegel/effort-estimator/output/tli_all_subjects.csv")
tli_df = tli_df.dropna(subset=['hr_delta', 'borg'])

# Merge ECG HR to windows
subject_map = {
    'elderly1': 'sim_elderly1',
    'elderly2': 'sim_elderly2', 
    'elderly3': 'sim_elderly3',
    'elderly4': 'sim_elderly4',
    'elderly5': 'sim_elderly5',
}

df_all['hr_delta_ecg'] = np.nan
df_all['hr_mean_ecg'] = np.nan  
df_all['hr_load_ecg'] = np.nan

for sub_short, sub_long in subject_map.items():
    sub_windows = df_all[df_all['subject'] == sub_short]
    sub_activities = tli_df[tli_df['subject'] == sub_long]
    
    for idx in sub_windows.index:
        window_time = df_all.loc[idx, 't_center']
        
        for _, act in sub_activities.iterrows():
            if act['t_start'] <= window_time <= act['t_end']:
                df_all.loc[idx, 'hr_delta_ecg'] = act['hr_delta']
                df_all.loc[idx, 'hr_mean_ecg'] = act.get('hr_mean', np.nan)
                df_all.loc[idx, 'hr_load_ecg'] = act.get('hr_load', np.nan)
                break

# Filter to windows WITH ECG data
df_ecg = df_all[df_all['hr_delta_ecg'].notna()].copy()
subjects_ecg = df_ecg['subject'].unique()

print(f"Total windows: {len(df_all)}")
print(f"Windows with ECG: {len(df_ecg)} ({100*len(df_ecg)/len(df_all):.1f}%)")
print(f"Subjects with ECG: {list(subjects_ecg)}")

# Show per-subject
for sub in sorted(subjects_ecg):
    n = len(df_ecg[df_ecg['subject'] == sub])
    print(f"  {sub}: {n} windows")

# =============================================================================
# DEFINE FEATURE SETS
# =============================================================================

# PPG features
exclude = ['t_center', 'borg', 'subject', 'Unnamed', 'hr_delta_ecg', 'hr_mean_ecg', 'hr_load_ecg', 'activity_label']
ppg_features_all = [c for c in df_ecg.columns if not any(x in c for x in exclude)]

# Select top 20 PPG features by correlation (on ECG subset)
correlations = []
for col in ppg_features_all:
    valid = df_ecg[[col, 'borg']].dropna()
    if len(valid) > 10:
        r, _ = pearsonr(valid[col], valid['borg'])
        correlations.append((col, abs(r)))
correlations.sort(key=lambda x: x[1], reverse=True)
top_20_ppg = [c[0] for c in correlations[:20]]

# ECG features
ecg_features = ['hr_delta_ecg', 'hr_load_ecg']

print(f"\nTop 20 PPG features by correlation:")
for feat, r in correlations[:10]:
    print(f"  {feat}: r = {r:.3f}")

# =============================================================================
# RUN LOSO EVALUATION
# =============================================================================

def run_loso_calibrated(df, features, subjects, cal_frac=0.2, name=""):
    """LOSO with per-subject calibration."""
    
    all_preds = []
    all_true = []
    per_sub = {}
    
    for test_sub in subjects:
        train = df[df['subject'] != test_sub]
        test = df[df['subject'] == test_sub]
        
        if len(train) < 20 or len(test) < 10:
            continue
        
        # Remove NaN rows
        train = train.dropna(subset=features + ['borg'])
        test = test.dropna(subset=features + ['borg'])
        
        if len(train) < 20 or len(test) < 10:
            continue
        
        # Split test into cal/eval
        n = len(test)
        n_cal = max(5, int(n * cal_frac))
        
        idx = np.arange(n)
        np.random.shuffle(idx)
        cal_idx = idx[:n_cal]
        eval_idx = idx[n_cal:]
        
        test_cal = test.iloc[cal_idx]
        test_eval = test.iloc[eval_idx]
        
        X_train = np.nan_to_num(train[features].values)
        y_train = train['borg'].values
        X_cal = np.nan_to_num(test_cal[features].values)
        y_cal = test_cal['borg'].values
        X_eval = np.nan_to_num(test_eval[features].values)
        y_eval = test_eval['borg'].values
        
        # Scale and train
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_cal_s = scaler.transform(X_cal)
        X_eval_s = scaler.transform(X_eval)
        
        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)
        
        # Calibrate
        preds_cal = model.predict(X_cal_s)
        lr = LinearRegression()
        lr.fit(preds_cal.reshape(-1, 1), y_cal)
        
        # Evaluate
        preds_raw = model.predict(X_eval_s)
        preds = lr.predict(preds_raw.reshape(-1, 1))
        
        all_preds.extend(preds)
        all_true.extend(y_eval)
        
        if len(preds) > 2:
            r_sub, _ = pearsonr(preds, y_eval)
            per_sub[test_sub] = {'r': r_sub, 'n': len(y_eval)}
    
    if len(all_preds) < 10:
        return None
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    r, _ = pearsonr(all_preds, all_true)
    mae = np.mean(np.abs(all_preds - all_true))
    within_1 = np.mean(np.abs(all_preds - all_true) <= 1) * 100
    
    return {'name': name, 'r': r, 'mae': mae, 'within_1': within_1, 'n': len(all_preds), 'per_sub': per_sub}

print("\n" + "="*60)
print("LOSO + 20% PER-SUBJECT CALIBRATION")
print("="*60)

# Model 1: PPG only
print("\n--- Model 1: PPG Features Only (20 features) ---")
result1 = run_loso_calibrated(df_ecg, top_20_ppg, subjects_ecg, name="PPG only")
if result1:
    print(f"  OVERALL: r = {result1['r']:.3f}, MAE = {result1['mae']:.2f}, ±1 Borg = {result1['within_1']:.1f}%")
    for sub, s in result1['per_sub'].items():
        print(f"    {sub}: r = {s['r']:.3f}, n = {s['n']}")

# Model 2: ECG only  
print("\n--- Model 2: ECG HR Only (2 features) ---")
result2 = run_loso_calibrated(df_ecg, ecg_features, subjects_ecg, name="ECG only")
if result2:
    print(f"  OVERALL: r = {result2['r']:.3f}, MAE = {result2['mae']:.2f}, ±1 Borg = {result2['within_1']:.1f}%")
    for sub, s in result2['per_sub'].items():
        print(f"    {sub}: r = {s['r']:.3f}, n = {s['n']}")

# Model 3: PPG + ECG combined
print("\n--- Model 3: PPG + ECG Combined (22 features) ---")
combined = top_20_ppg + ecg_features
result3 = run_loso_calibrated(df_ecg, combined, subjects_ecg, name="PPG+ECG")
if result3:
    print(f"  OVERALL: r = {result3['r']:.3f}, MAE = {result3['mae']:.2f}, ±1 Borg = {result3['within_1']:.1f}%")
    for sub, s in result3['per_sub'].items():
        print(f"    {sub}: r = {s['r']:.3f}, n = {s['n']}")

# Model 4: Top 5 PPG + ECG
print("\n--- Model 4: Top 5 PPG + ECG (7 features) ---")
top_5_ppg = [c[0] for c in correlations[:5]]
small_combined = top_5_ppg + ecg_features
result4 = run_loso_calibrated(df_ecg, small_combined, subjects_ecg, name="Top5 PPG+ECG")
if result4:
    print(f"  OVERALL: r = {result4['r']:.3f}, MAE = {result4['mae']:.2f}, ±1 Borg = {result4['within_1']:.1f}%")
    for sub, s in result4['per_sub'].items():
        print(f"    {sub}: r = {s['r']:.3f}, n = {s['n']}")

# =============================================================================
# NOW: RUN ON ALL 5 SUBJECTS (PPG only, no ECG dependency)
# =============================================================================

print("\n" + "="*60)
print("ALL 5 SUBJECTS (PPG only - no ECG dependency)")
print("="*60)

# Recompute correlations on full data
correlations_full = []
for col in ppg_features_all:
    if col in df_all.columns:
        valid = df_all[[col, 'borg']].dropna()
        if len(valid) > 10:
            r, _ = pearsonr(valid[col], valid['borg'])
            correlations_full.append((col, abs(r)))
correlations_full.sort(key=lambda x: x[1], reverse=True)
top_20_ppg_full = [c[0] for c in correlations_full[:20]]

print("\n--- All 5 Subjects: PPG Features Only (20 features) ---")
result_full = run_loso_calibrated(df_all, top_20_ppg_full, df_all['subject'].unique(), name="PPG only (all)")
if result_full:
    print(f"  OVERALL: r = {result_full['r']:.3f}, MAE = {result_full['mae']:.2f}, ±1 Borg = {result_full['within_1']:.1f}%")
    for sub, s in result_full['per_sub'].items():
        print(f"    {sub}: r = {s['r']:.3f}, n = {s['n']}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY TABLE")
print("="*80)

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│               LOSO + 20% Per-Subject Calibration Results                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ ON ECG SUBSET (4 subjects, {len(df_ecg)} windows):                                    │
│   Model                    │ Features │ r     │ MAE  │ ±1 Borg            │
│   ─────────────────────────────────────────────────────────────────────────│
│   PPG only                 │   20     │ {result1['r'] if result1 else np.nan:.3f} │ {result1['mae'] if result1 else np.nan:.2f} │ {result1['within_1'] if result1 else np.nan:.1f}%             │
│   ECG HR only              │    2     │ {result2['r'] if result2 else np.nan:.3f} │ {result2['mae'] if result2 else np.nan:.2f} │ {result2['within_1'] if result2 else np.nan:.1f}%             │
│   PPG + ECG combined       │   22     │ {result3['r'] if result3 else np.nan:.3f} │ {result3['mae'] if result3 else np.nan:.2f} │ {result3['within_1'] if result3 else np.nan:.1f}%             │
│   Top 5 PPG + ECG          │    7     │ {result4['r'] if result4 else np.nan:.3f} │ {result4['mae'] if result4 else np.nan:.2f} │ {result4['within_1'] if result4 else np.nan:.1f}%             │
│                                                                             │
│ ON FULL DATA (5 subjects, {len(df_all)} windows):                                   │
│   PPG only                 │   20     │ {result_full['r'] if result_full else np.nan:.3f} │ {result_full['mae'] if result_full else np.nan:.2f} │ {result_full['within_1'] if result_full else np.nan:.1f}%             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

KEY INSIGHT:
• ECG HR alone (r = {result2['r'] if result2 else np.nan:.2f}) outperforms PPG features (r = {result1['r'] if result1 else np.nan:.2f})
• Combined PPG + ECG (r = {result3['r'] if result3 else np.nan:.2f}) may give best results
• Clean HR signal is most predictive of subjective effort
""")
