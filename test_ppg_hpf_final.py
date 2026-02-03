#!/usr/bin/env python3
"""
Test PPG with HPF - Fixed time alignment using t_unix
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.insert(0, '/Users/pascalschlegel/effort-estimator')
from features.ppg_features import _basic_features
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TESTING PPG GREEN: HPF vs NO HPF")
print("=" * 70)

def extract_features_with_unix_time(ppg_df, fused_df, prefix, fs=32.0, win_sec=5.0):
    """Extract features aligning on unix time"""
    half_win_samples = int(win_sec * fs / 2)
    
    # Get unix times
    t_ppg = ppg_df['t_unix'].values
    ppg_vals = ppg_df['value'].values
    
    rows = []
    for _, row in fused_df.iterrows():
        if pd.isna(row['borg']):
            continue
            
        t_center = row['t_center']  # This is unix time
        
        # Find closest PPG sample
        center_idx = np.argmin(np.abs(t_ppg - t_center))
        start_idx = max(0, center_idx - half_win_samples)
        end_idx = min(len(ppg_vals), center_idx + half_win_samples)
        
        if end_idx - start_idx < half_win_samples:
            continue
        
        window = ppg_vals[start_idx:end_idx]
        feats = _basic_features(window, prefix)
        feats['borg'] = row['borg']
        rows.append(feats)
    
    return pd.DataFrame(rows)


results_hpf = []
results_old = []

for i in range(1, 6):
    subj = f'P{i}'
    base = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}')
    
    green_hpf_path = base / 'ppg_green' / 'ppg_green_preprocessed_hpf.csv'
    green_old_path = base / 'ppg_green' / 'ppg_green_preprocessed.csv'
    fused_path = base / 'fused_aligned_5.0s.csv'
    
    if not all(p.exists() for p in [green_hpf_path, green_old_path, fused_path]):
        print(f"{subj}: missing files")
        continue
    
    green_hpf = pd.read_csv(green_hpf_path)
    green_old = pd.read_csv(green_old_path)
    fused = pd.read_csv(fused_path)
    
    # Extract features
    df_hpf = extract_features_with_unix_time(green_hpf, fused, 'hpf_')
    df_old = extract_features_with_unix_time(green_old, fused, 'old_')
    
    df_hpf['subject'] = subj
    df_old['subject'] = subj
    
    results_hpf.append(df_hpf)
    results_old.append(df_old)
    print(f"{subj}: {len(df_hpf)} windows extracted")

# Combine
df_hpf = pd.concat(results_hpf, ignore_index=True)
df_old = pd.concat(results_old, ignore_index=True)

hpf_cols = [c for c in df_hpf.columns if c.startswith('hpf_')]
old_cols = [c for c in df_old.columns if c.startswith('old_')]

print(f"\nTotal: {len(df_hpf)} windows, {len(hpf_cols)} features")

# LOSO for HPF
print("\n" + "=" * 70)
print("LOSO: PPG GREEN WITH HPF (0.5 Hz)")
print("=" * 70)

subjects = sorted(df_hpf['subject'].unique())
r_hpf_list = []

for test_subj in subjects:
    train = df_hpf[df_hpf['subject'] != test_subj]
    test = df_hpf[df_hpf['subject'] == test_subj]
    
    X_train = train[hpf_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_train = train['borg'].values
    X_test = test[hpf_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_test = test['borg'].values
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    r, _ = pearsonr(y_test, y_pred)
    r_hpf_list.append(r)
    print(f"  {test_subj}: r = {r:.2f}")

mean_r_hpf = np.mean(r_hpf_list)
print(f"\n  Mean r = {mean_r_hpf:.2f}")

# LOSO for OLD
print("\n" + "=" * 70)
print("LOSO: PPG GREEN WITHOUT HPF (original)")
print("=" * 70)

r_old_list = []

for test_subj in subjects:
    train = df_old[df_old['subject'] != test_subj]
    test = df_old[df_old['subject'] == test_subj]
    
    X_train = train[old_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_train = train['borg'].values
    X_test = test[old_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_test = test['borg'].values
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    r, _ = pearsonr(y_test, y_pred)
    r_old_list.append(r)
    print(f"  {test_subj}: r = {r:.2f}")

mean_r_old = np.mean(r_old_list)
print(f"\n  Mean r = {mean_r_old:.2f}")

# Summary
print("\n" + "=" * 70)
print("FINAL COMPARISON: GREEN PPG ONLY")
print("=" * 70)
print(f"WITHOUT HPF: LOSO r = {mean_r_old:.2f}")
print(f"WITH HPF:    LOSO r = {mean_r_hpf:.2f}")
print(f"Change:      Δr = {mean_r_hpf - mean_r_old:+.2f}")
print(f"\nReference - IMU: LOSO r = 0.52")
