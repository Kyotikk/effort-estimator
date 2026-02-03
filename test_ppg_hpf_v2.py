#!/usr/bin/env python3
"""
Re-extract PPG features using HPF-filtered green, then test LOSO.
Fixed version - uses start_idx/end_idx from fused data directly.
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
print("RE-EXTRACTING PPG GREEN FEATURES WITH HPF")
print("=" * 70)

# Extract new features for each subject
all_dfs = []

for i in range(1, 6):
    subj = f'P{i}'
    base = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}')
    
    # Load HPF green signal
    green_hpf_path = base / 'ppg_green' / 'ppg_green_preprocessed_hpf.csv'
    if not green_hpf_path.exists():
        print(f"{subj}: missing green_hpf, skipping")
        continue
    
    green_df = pd.read_csv(green_hpf_path)
    green_vals = green_df['value'].values
    
    # Load fused data to get window indices and borg
    fused_path = base / 'fused_aligned_5.0s.csv'
    if not fused_path.exists():
        print(f"{subj}: missing fused, skipping")
        continue
    
    fused = pd.read_csv(fused_path)
    fused = fused.dropna(subset=['borg'])
    
    # Extract features for each window using the indices from fused data
    # We need to map window times to PPG sample indices
    # Using t_center and assuming 32Hz sampling
    
    fs = 32.0
    win_sec = 5.0
    half_win = int(win_sec * fs / 2)
    
    rows = []
    for _, row in fused.iterrows():
        t_center = row['t_center']
        borg = row['borg']
        
        # Find corresponding PPG samples
        # t_center is relative time, green_df has t_sec
        t_ppg = green_df['t_sec'].values
        center_idx = np.argmin(np.abs(t_ppg - t_center))
        
        start_idx = max(0, center_idx - half_win)
        end_idx = min(len(green_vals), center_idx + half_win)
        
        if end_idx - start_idx < half_win:  # Skip if not enough samples
            continue
        
        window_data = green_vals[start_idx:end_idx]
        feats = _basic_features(window_data, prefix='ppg_green_hpf_')
        feats['borg'] = borg
        feats['subject'] = subj
        rows.append(feats)
    
    df_subj = pd.DataFrame(rows)
    print(f"{subj}: {len(df_subj)} windows with HPF green features")
    all_dfs.append(df_subj)

df = pd.concat(all_dfs, ignore_index=True)
df = df.dropna(subset=['borg'])

# Get HPF feature columns
hpf_cols = [c for c in df.columns if 'ppg_green_hpf_' in c]
print(f"\nTotal: {len(df)} windows, {len(hpf_cols)} HPF green features")

# Run LOSO
print("\n" + "=" * 70)
print("LOSO WITH HPF GREEN FEATURES ONLY")
print("=" * 70)

subjects = sorted(df['subject'].unique())
results_hpf = []

for test_subj in subjects:
    train = df[df['subject'] != test_subj]
    test = df[df['subject'] == test_subj]
    
    X_train = train[hpf_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_train = train['borg'].values
    X_test = test[hpf_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_test = test['borg'].values
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    r, _ = pearsonr(y_test, y_pred)
    results_hpf.append({'subject': test_subj, 'r': r})
    print(f"  {test_subj}: r = {r:.2f}")

mean_r_hpf = np.mean([r['r'] for r in results_hpf])
print(f"\n  Mean r = {mean_r_hpf:.2f}")

# Now compare with OLD green (without HPF) - same procedure
print("\n" + "=" * 70)
print("LOSO WITH ORIGINAL GREEN FEATURES (NO HPF) - SAME PROCEDURE")
print("=" * 70)

all_dfs_old = []
for i in range(1, 6):
    subj = f'P{i}'
    base = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}')
    
    # Load ORIGINAL green signal (no HPF)
    green_path = base / 'ppg_green' / 'ppg_green_preprocessed.csv'
    if not green_path.exists():
        continue
    
    green_df = pd.read_csv(green_path)
    green_vals = green_df['value'].values
    
    fused_path = base / 'fused_aligned_5.0s.csv'
    fused = pd.read_csv(fused_path)
    fused = fused.dropna(subset=['borg'])
    
    fs = 32.0
    win_sec = 5.0
    half_win = int(win_sec * fs / 2)
    
    rows = []
    for _, row in fused.iterrows():
        t_center = row['t_center']
        borg = row['borg']
        t_ppg = green_df['t_sec'].values
        center_idx = np.argmin(np.abs(t_ppg - t_center))
        start_idx = max(0, center_idx - half_win)
        end_idx = min(len(green_vals), center_idx + half_win)
        if end_idx - start_idx < half_win:
            continue
        window_data = green_vals[start_idx:end_idx]
        feats = _basic_features(window_data, prefix='ppg_green_old_')
        feats['borg'] = borg
        feats['subject'] = subj
        rows.append(feats)
    
    df_subj = pd.DataFrame(rows)
    all_dfs_old.append(df_subj)

df_old = pd.concat(all_dfs_old, ignore_index=True)
old_cols = [c for c in df_old.columns if 'ppg_green_old_' in c]

results_old = []
for test_subj in sorted(df_old['subject'].unique()):
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
    results_old.append({'subject': test_subj, 'r': r})
    print(f"  {test_subj}: r = {r:.2f}")

mean_r_old = np.mean([r['r'] for r in results_old])
print(f"\n  Mean r = {mean_r_old:.2f}")

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)
print(f"PPG Green WITHOUT HPF: LOSO r = {mean_r_old:.2f}")
print(f"PPG Green WITH HPF:    LOSO r = {mean_r_hpf:.2f}")
print(f"Improvement:           Δr = {mean_r_hpf - mean_r_old:+.2f}")
print(f"\nIMU (reference):       LOSO r = 0.52")
