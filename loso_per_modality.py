#!/usr/bin/env python3
"""LOSO evaluation per modality."""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor

# Load all subjects
dfs = []
for i in range(1, 6):
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = i
        dfs.append(df)
        print(f"P{i}: {len(df)} windows")

df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal windows: {len(df)}")

# Define modality groups
imu_cols = [c for c in df.columns if 'acc_' in c and '_r' not in c]
ppg_cols = [c for c in df.columns if 'ppg_' in c]
eda_cols = [c for c in df.columns if 'eda_' in c]

print(f"\nFeature counts:")
print(f"  IMU: {len(imu_cols)}")
print(f"  PPG: {len(ppg_cols)}")
print(f"  EDA: {len(eda_cols)}")

def loso_evaluate(feature_cols, name):
    """Run LOSO cross-validation for a set of features."""
    results = []
    for test_subj in sorted(df['subject'].unique()):
        train_df = df[df['subject'] != test_subj].dropna(subset=['borg'])
        test_df = df[df['subject'] == test_subj].dropna(subset=['borg'])
        
        valid_cols = [c for c in feature_cols if c in train_df.columns]
        X_train = train_df[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_df['borg'].values
        X_test = test_df[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_test = test_df['borg'].values
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        if len(y_test) > 2 and np.std(y_test) > 0:
            r, _ = pearsonr(y_test, y_pred)
            results.append(r)
            print(f"    P{test_subj}: r = {r:.3f}")
    
    mean_r = np.mean(results)
    return mean_r, results

print("\n" + "="*50)
print("LOSO PER MODALITY (Random Forest)")
print("="*50)

print("\nIMU features:")
r_imu, _ = loso_evaluate(imu_cols, "IMU")
print(f"  → Mean LOSO r = {r_imu:.3f}")

print("\nPPG features:")
r_ppg, _ = loso_evaluate(ppg_cols, "PPG")
print(f"  → Mean LOSO r = {r_ppg:.3f}")

print("\nEDA features:")
r_eda, _ = loso_evaluate(eda_cols, "EDA")
print(f"  → Mean LOSO r = {r_eda:.3f}")

print("\n" + "="*50)
print("FINAL SUMMARY")
print("="*50)
print(f"""
| Modality | Features | LOSO r |
|----------|----------|--------|
| IMU      | {len(imu_cols):>3}      | {r_imu:.2f}   |
| PPG      | {len(ppg_cols):>3}      | {r_ppg:.2f}   |
| EDA      | {len(eda_cols):>3}      | {r_eda:.2f}   |
""")
