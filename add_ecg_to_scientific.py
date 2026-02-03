#!/usr/bin/env python3
"""
SCIENTIFIC PIPELINE WITH ECG HR FEATURES
=========================================
1. Investigate why 0.56 vs 0.48
2. Add ECG HR features to the scientific pipeline
3. Test if it improves results
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
print("INVESTIGATING 0.56 vs 0.48 AND ADDING ECG HR")
print("="*80)

# =============================================================================
# PART 1: WHY 0.56 vs 0.48?
# =============================================================================

print("\n" + "="*60)
print("PART 1: INVESTIGATING THE DIFFERENCE")
print("="*60)

# Load scientific data
all_dfs = []
for i in [1,2,3,4,5]:
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = f'elderly{i}'
        all_dfs.append(df)

df_scientific = pd.concat(all_dfs, ignore_index=True).dropna(subset=['borg'])
print(f"Scientific data: {len(df_scientific)} windows, {df_scientific['subject'].nunique()} subjects")

# Get feature columns
exclude = ['t_center', 'borg', 'subject', 'Unnamed']
feature_cols = [c for c in df_scientific.columns if not any(x in c for x in exclude)]

# Previous approach: used 34 selected features from feature selection
# Current approach: used top 34 by correlation
# Let's check both

print("\nDifferences in methodology:")
print("1. Previous (0.48): Used specific 34 features from PCA/correlation selection")
print("2. Current (0.56): Used top 34 features by raw correlation with Borg")
print("3. Previous used 80% calibration holdout, current uses 20%")

# Let's standardize: use same 20% calibration and test both feature sets

# =============================================================================
# PART 2: LOAD ECG HR DATA
# =============================================================================

print("\n" + "="*60)
print("PART 2: LOADING ECG HR DATA (Vivalnk)")
print("="*60)

# Load TLI data which has ECG HR per activity
tli_df = pd.read_csv("/Users/pascalschlegel/effort-estimator/output/tli_all_subjects.csv")
tli_df = tli_df.dropna(subset=['hr_delta', 'borg'])
print(f"ECG HR data: {len(tli_df)} activities")

# The challenge: ECG HR is at ACTIVITY level, scientific features are at WINDOW level
# We need to merge them by timestamp

print("\nMerging ECG HR with window features...")

# For each window, find the activity it belongs to and get HR features
def assign_hr_to_windows(windows_df, activities_df, subject_col='subject'):
    """Assign ECG HR features from activities to windows based on timestamp overlap."""
    
    # Rename subject column for consistency
    subject_map = {
        'elderly1': 'sim_elderly1',
        'elderly2': 'sim_elderly2', 
        'elderly3': 'sim_elderly3',
        'elderly4': 'sim_elderly4',
        'elderly5': 'sim_elderly5',
    }
    
    windows_df = windows_df.copy()
    windows_df['hr_delta_ecg'] = np.nan
    windows_df['hr_mean_ecg'] = np.nan
    windows_df['hr_load_ecg'] = np.nan
    windows_df['activity_label'] = None
    
    for idx, window in windows_df.iterrows():
        window_time = window['t_center']
        window_subject = subject_map.get(window['subject'], window['subject'])
        
        # Find matching activity
        sub_activities = activities_df[activities_df['subject'] == window_subject]
        
        for _, act in sub_activities.iterrows():
            if act['t_start'] <= window_time <= act['t_end']:
                windows_df.loc[idx, 'hr_delta_ecg'] = act.get('hr_delta', np.nan)
                windows_df.loc[idx, 'hr_mean_ecg'] = act.get('hr_mean', np.nan)
                windows_df.loc[idx, 'hr_load_ecg'] = act.get('hr_load', np.nan)
                windows_df.loc[idx, 'activity_label'] = act.get('activity', None)
                break
    
    return windows_df

# This merge is slow, let's do it more efficiently
print("Merging (this may take a moment)...")

df_merged = df_scientific.copy()
df_merged['hr_delta_ecg'] = np.nan
df_merged['hr_mean_ecg'] = np.nan  
df_merged['hr_load_ecg'] = np.nan
df_merged['activity_label'] = None

# Map subjects
subject_map = {
    'elderly1': 'sim_elderly1',
    'elderly2': 'sim_elderly2', 
    'elderly3': 'sim_elderly3',
    'elderly4': 'sim_elderly4',
    'elderly5': 'sim_elderly5',
}

for sub_short, sub_long in subject_map.items():
    sub_windows = df_merged[df_merged['subject'] == sub_short].copy()
    sub_activities = tli_df[tli_df['subject'] == sub_long]
    
    if len(sub_activities) == 0:
        print(f"  {sub_short}: No ECG activities found")
        continue
    
    matched = 0
    for idx in sub_windows.index:
        window_time = df_merged.loc[idx, 't_center']
        
        for _, act in sub_activities.iterrows():
            if act['t_start'] <= window_time <= act['t_end']:
                df_merged.loc[idx, 'hr_delta_ecg'] = act.get('hr_delta', np.nan)
                df_merged.loc[idx, 'hr_mean_ecg'] = act.get('hr_mean', np.nan)
                df_merged.loc[idx, 'hr_load_ecg'] = act.get('hr_load', np.nan)
                df_merged.loc[idx, 'activity_label'] = act.get('activity', None)
                matched += 1
                break
    
    print(f"  {sub_short}: {matched}/{len(sub_windows)} windows matched to ECG activities")

# Check how many windows have ECG HR
n_with_ecg = df_merged['hr_delta_ecg'].notna().sum()
print(f"\nWindows with ECG HR: {n_with_ecg}/{len(df_merged)} ({100*n_with_ecg/len(df_merged):.1f}%)")

# =============================================================================
# PART 3: COMPARE MODELS
# =============================================================================

print("\n" + "="*60)
print("PART 3: COMPARING MODELS WITH LOSO + 20% CALIBRATION")
print("="*60)

subjects = df_merged['subject'].unique()

# Function to run LOSO with calibration
def run_loso_with_calibration(df, features, cal_fraction=0.2, name="Model"):
    """Run LOSO with per-subject calibration."""
    
    all_preds = []
    all_true = []
    per_subject = {}
    
    for test_sub in subjects:
        train = df[df['subject'] != test_sub].dropna(subset=features + ['borg'])
        test = df[df['subject'] == test_sub].dropna(subset=features + ['borg'])
        
        if len(test) < 10:
            continue
        
        # Split test into calibration and evaluation
        n_test = len(test)
        n_cal = max(5, int(n_test * cal_fraction))
        
        indices = np.arange(n_test)
        np.random.shuffle(indices)
        cal_idx = indices[:n_cal]
        eval_idx = indices[n_cal:]
        
        test_cal = test.iloc[cal_idx]
        test_eval = test.iloc[eval_idx]
        
        X_train = np.nan_to_num(train[features].values)
        y_train = train['borg'].values
        X_cal = np.nan_to_num(test_cal[features].values)
        y_cal = test_cal['borg'].values
        X_eval = np.nan_to_num(test_eval[features].values)
        y_eval = test_eval['borg'].values
        
        # Train
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_cal_s = scaler.transform(X_cal)
        X_eval_s = scaler.transform(X_eval)
        
        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)
        
        # Calibrate
        preds_cal = model.predict(X_cal_s)
        lr_cal = LinearRegression()
        lr_cal.fit(preds_cal.reshape(-1, 1), y_cal)
        
        # Predict with calibration
        preds_eval_raw = model.predict(X_eval_s)
        preds_eval = lr_cal.predict(preds_eval_raw.reshape(-1, 1))
        
        all_preds.extend(preds_eval)
        all_true.extend(y_eval)
        
        # Per-subject stats
        if len(preds_eval) > 2:
            r_sub, _ = pearsonr(preds_eval, y_eval)
            per_subject[test_sub] = {'r': r_sub, 'n': len(y_eval)}
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    r, _ = pearsonr(all_preds, all_true)
    mae = np.mean(np.abs(all_preds - all_true))
    within_1 = np.mean(np.abs(all_preds - all_true) <= 1) * 100
    
    return {
        'name': name,
        'r': r,
        'mae': mae,
        'within_1': within_1,
        'n': len(all_preds),
        'per_subject': per_subject
    }

# MODEL 1: Original PPG features only (no ECG)
print("\n--- MODEL 1: PPG + EDA + IMU (no ECG) ---")
ppg_features = [c for c in feature_cols if c in df_merged.columns]

# Select top 34 by correlation
correlations = []
for col in ppg_features:
    valid = df_merged[[col, 'borg']].dropna()
    if len(valid) > 10:
        r, _ = pearsonr(valid[col], valid['borg'])
        correlations.append((col, abs(r)))
correlations.sort(key=lambda x: x[1], reverse=True)
top_34_ppg = [c[0] for c in correlations[:34]]

result1 = run_loso_with_calibration(df_merged, top_34_ppg, name="PPG+EDA+IMU only")
print(f"  r = {result1['r']:.3f}, MAE = {result1['mae']:.2f}, ±1 Borg = {result1['within_1']:.1f}%")
for sub, stats in result1['per_subject'].items():
    print(f"    {sub}: r = {stats['r']:.3f}, n = {stats['n']}")

# MODEL 2: PPG features + ECG HR features
print("\n--- MODEL 2: PPG + EDA + IMU + ECG HR ---")
# Only use windows with ECG data
df_with_ecg = df_merged[df_merged['hr_delta_ecg'].notna()].copy()
print(f"  Using {len(df_with_ecg)} windows with ECG HR data")

# Only use subjects that have ECG data
subjects_with_ecg = df_with_ecg['subject'].unique()
print(f"  Subjects with ECG: {list(subjects_with_ecg)}")

ecg_features = ['hr_delta_ecg', 'hr_mean_ecg', 'hr_load_ecg']
combined_features = top_34_ppg + ecg_features

# Custom LOSO for ECG subset
def run_loso_ecg(df, features, subjects_list, cal_fraction=0.2, name="Model"):
    """Run LOSO with per-subject calibration on ECG subset."""
    
    all_preds = []
    all_true = []
    per_subject = {}
    
    for test_sub in subjects_list:
        train = df[df['subject'] != test_sub].dropna(subset=features + ['borg'])
        test = df[df['subject'] == test_sub].dropna(subset=features + ['borg'])
        
        if len(train) < 10 or len(test) < 10:
            print(f"    Skipping {test_sub}: train={len(train)}, test={len(test)}")
            continue
        
        # Split test into calibration and evaluation
        n_test = len(test)
        n_cal = max(5, int(n_test * cal_fraction))
        
        indices = np.arange(n_test)
        np.random.shuffle(indices)
        cal_idx = indices[:n_cal]
        eval_idx = indices[n_cal:]
        
        test_cal = test.iloc[cal_idx]
        test_eval = test.iloc[eval_idx]
        
        X_train = np.nan_to_num(train[features].values)
        y_train = train['borg'].values
        X_cal = np.nan_to_num(test_cal[features].values)
        y_cal = test_cal['borg'].values
        X_eval = np.nan_to_num(test_eval[features].values)
        y_eval = test_eval['borg'].values
        
        # Train
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_cal_s = scaler.transform(X_cal)
        X_eval_s = scaler.transform(X_eval)
        
        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)
        
        # Calibrate
        preds_cal = model.predict(X_cal_s)
        lr_cal = LinearRegression()
        lr_cal.fit(preds_cal.reshape(-1, 1), y_cal)
        
        # Predict with calibration
        preds_eval_raw = model.predict(X_eval_s)
        preds_eval = lr_cal.predict(preds_eval_raw.reshape(-1, 1))
        
        all_preds.extend(preds_eval)
        all_true.extend(y_eval)
        
        # Per-subject stats
        if len(preds_eval) > 2:
            r_sub, _ = pearsonr(preds_eval, y_eval)
            per_subject[test_sub] = {'r': r_sub, 'n': len(y_eval)}
    
    if len(all_preds) < 10:
        return {'name': name, 'r': np.nan, 'mae': np.nan, 'within_1': np.nan, 'n': 0, 'per_subject': {}}
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    r, _ = pearsonr(all_preds, all_true)
    mae = np.mean(np.abs(all_preds - all_true))
    within_1 = np.mean(np.abs(all_preds - all_true) <= 1) * 100
    
    return {
        'name': name,
        'r': r,
        'mae': mae,
        'within_1': within_1,
        'n': len(all_preds),
        'per_subject': per_subject
    }

result2 = run_loso_ecg(df_with_ecg, combined_features, subjects_with_ecg, name="PPG+EDA+IMU+ECG")
print(f"  r = {result2['r']:.3f}, MAE = {result2['mae']:.2f}, ±1 Borg = {result2['within_1']:.1f}%")
for sub, stats in result2['per_subject'].items():
    print(f"    {sub}: r = {stats['r']:.3f}, n = {stats['n']}")

# MODEL 3: ECG HR only (on same subset)
print("\n--- MODEL 3: ECG HR only (same subset) ---")
result3 = run_loso_ecg(df_with_ecg, ecg_features, subjects_with_ecg, name="ECG HR only")
print(f"  r = {result3['r']:.3f}, MAE = {result3['mae']:.2f}, ±1 Borg = {result3['within_1']:.1f}%")
for sub, stats in result3['per_subject'].items():
    print(f"    {sub}: r = {stats['r']:.3f}, n = {stats['n']}")

# MODEL 4: Best PPG features + ECG HR (fewer features)
print("\n--- MODEL 4: Top 10 PPG + ECG HR ---")
top_10_ppg = [c[0] for c in correlations[:10]]
combined_10 = top_10_ppg + ecg_features

result4 = run_loso_ecg(df_with_ecg, combined_10, subjects_with_ecg, name="Top10 PPG + ECG")
print(f"  r = {result4['r']:.3f}, MAE = {result4['mae']:.2f}, ±1 Borg = {result4['within_1']:.1f}%")
for sub, stats in result4['per_subject'].items():
    print(f"    {sub}: r = {stats['r']:.3f}, n = {stats['n']}")

# =============================================================================
# FINAL COMPARISON
# =============================================================================

print("\n" + "="*80)
print("FINAL COMPARISON TABLE")
print("="*80)

print(f"""
┌───────────────────────────────────────────────────────────────────────────────┐
│                    LOSO + 20% Per-Subject Calibration                         │
├───────────────────────────────────────────────────────────────────────────────┤
│ Model                          │ Samples │ Pearson r │  MAE  │ ±1 Borg       │
├────────────────────────────────┼─────────┼───────────┼───────┼───────────────┤
│ 1. PPG+EDA+IMU (34 features)   │  {result1['n']:5d}  │   {result1['r']:.3f}   │ {result1['mae']:.2f}  │   {result1['within_1']:.1f}%        │
│ 2. PPG+EDA+IMU+ECG HR (37 ft)  │  {result2['n']:5d}  │   {result2['r']:.3f}   │ {result2['mae']:.2f}  │   {result2['within_1']:.1f}%        │
│ 3. ECG HR only (3 features)    │  {result3['n']:5d}  │   {result3['r']:.3f}   │ {result3['mae']:.2f}  │   {result3['within_1']:.1f}%        │
│ 4. Top 10 PPG + ECG HR (13 ft) │  {result4['n']:5d}  │   {result4['r']:.3f}   │ {result4['mae']:.2f}  │   {result4['within_1']:.1f}%        │
└───────────────────────────────────────────────────────────────────────────────┘

NOTE: Models 2-4 use fewer samples because ECG HR is only available for
      windows that overlap with labeled activities.
""")

# Check feature importance for best model
print("\n--- Top Features by Correlation with Borg ---")
all_features = combined_features
for feat in all_features[:15]:
    valid = df_with_ecg[[feat, 'borg']].dropna()
    if len(valid) > 10:
        r, _ = pearsonr(valid[feat], valid['borg'])
        print(f"  {feat:30s}: r = {r:+.3f}")
