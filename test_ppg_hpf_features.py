#!/usr/bin/env python3
"""
Re-extract PPG features using HPF-filtered green, then test LOSO.
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

def extract_ppg_features_for_windows(ppg_csv, windows_csv, prefix):
    """Extract PPG features for each window"""
    sig = pd.read_csv(ppg_csv)
    win = pd.read_csv(windows_csv)
    
    x = sig['value'].values
    
    rows = []
    for _, w in win.iterrows():
        s = int(w['start_idx'])
        e = int(w['end_idx'])
        feats = _basic_features(x[s:e], prefix=prefix)
        feats['t_center'] = w['t_center']
        rows.append(feats)
    
    return pd.DataFrame(rows)

# Extract new features for each subject
all_features = []

for i in range(1, 6):
    subj = f'P{i}'
    base = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}')
    
    # Load HPF green
    green_hpf_path = base / 'ppg_green' / 'ppg_green_preprocessed_hpf.csv'
    windows_path = base / 'imu_bioz' / 'imu_windows_5.0s.csv'  # Windows are in imu_bioz folder
    
    if not green_hpf_path.exists():
        print(f"{subj}: missing green_hpf, skipping")
        continue
    if not windows_path.exists():
        print(f"{subj}: missing windows, skipping")
        continue
    
    # Extract features
    green_feats = extract_ppg_features_for_windows(str(green_hpf_path), str(windows_path), 'ppg_green_hpf_')
    
    # Also load existing fused data for borg labels
    fused_path = base / 'fused_aligned_5.0s.csv'
    fused = pd.read_csv(fused_path)
    
    # Merge on t_center
    merged = pd.merge(
        fused[['t_center', 'borg', 'subject'] if 'subject' in fused.columns else ['t_center', 'borg']],
        green_feats,
        on='t_center',
        how='inner'
    )
    merged['subject'] = subj
    
    print(f"{subj}: {len(merged)} windows with new HPF green features")
    all_features.append(merged)

df = pd.concat(all_features, ignore_index=True)
df = df.dropna(subset=['borg'])

# Get new HPF feature columns
hpf_cols = [c for c in df.columns if 'ppg_green_hpf_' in c]
print(f"\nTotal: {len(df)} windows, {len(hpf_cols)} HPF green features")

# Run LOSO
print("\n" + "=" * 70)
print("LOSO WITH HPF GREEN FEATURES")
print("=" * 70)

subjects = sorted(df['subject'].unique())
results = []

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
    results.append({'subject': test_subj, 'r': r})
    print(f"  {test_subj}: r = {r:.2f}")

mean_r = np.mean([r['r'] for r in results])
print(f"\n  Mean r = {mean_r:.2f}")

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"PPG Green WITHOUT HPF: r = 0.28 (old)")
print(f"PPG Green WITH HPF:    r = {mean_r:.2f} (new)")
print(f"IMU (reference):       r = 0.52")
