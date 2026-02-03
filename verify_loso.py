#!/usr/bin/env python3
"""Quick verification: What is PPG LOSO actually?"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr

paths = [
    '/Users/pascalschlegel/data/interim/parsingsim1/sim_elderly1/effort_estimation_output/elderly_sim_elderly1/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim2/sim_elderly2/effort_estimation_output/elderly_sim_elderly2/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim4/sim_elderly4/effort_estimation_output/elderly_sim_elderly4/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim5/sim_elderly5/effort_estimation_output/elderly_sim_elderly5/fused_aligned_5.0s.csv',
]

print("Loading data...")
dfs = []
for i, p in enumerate(paths, 1):
    df = pd.read_csv(p)
    df['subject'] = f'P{i}'
    dfs.append(df)
    print(f"  P{i}: {len(df)} rows")
combined = pd.concat(dfs).dropna(subset=['borg'])
print(f"Total with Borg: {len(combined)}")

# Get PPG columns
ppg_cols = [c for c in combined.columns if any(x in c.lower() for x in ['ppg', 'hr', 'ibi', 'rmssd', 'sdnn', 'pnn'])]
ppg_cols = [c for c in ppg_cols if combined[c].notna().mean() > 0.3 and combined[c].std() > 1e-10]
print(f"\nPPG features: {len(ppg_cols)}")

# Get IMU columns
imu_cols = [c for c in combined.columns if 'acc' in c.lower() or 'gyro' in c.lower()]
imu_cols = [c for c in imu_cols if combined[c].notna().mean() > 0.3 and combined[c].std() > 1e-10]
print(f"IMU features: {len(imu_cols)}")

print("\n" + "="*60)
print("LOSO EVALUATION")
print("="*60)

def run_loso(df, feature_cols, name):
    subjects = df['subject'].unique()
    per_subj_r = []
    
    for test_subj in subjects:
        train = df[df['subject'] != test_subj].dropna(subset=feature_cols + ['borg'])
        test = df[df['subject'] == test_subj].dropna(subset=feature_cols + ['borg'])
        
        if len(train) < 20 or len(test) < 5:
            continue
            
        X_train, y_train = train[feature_cols].values, train['borg'].values
        X_test, y_test = test[feature_cols].values, test['borg'].values
        
        imp = SimpleImputer(strategy='median')
        scl = StandardScaler()
        X_train = scl.fit_transform(imp.fit_transform(X_train))
        X_test = scl.transform(imp.transform(X_test))
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        r, _ = pearsonr(y_test, y_pred)
        per_subj_r.append(r)
        print(f"  {test_subj}: r = {r:.3f}")
    
    mean_r = np.mean(per_subj_r)
    print(f"  MEAN: r = {mean_r:.3f}")
    return mean_r

print(f"\nIMU ({len(imu_cols)} features):")
imu_r = run_loso(combined, imu_cols, "IMU")

print(f"\nPPG ({len(ppg_cols)} features):")
ppg_r = run_loso(combined, ppg_cols, "PPG")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"IMU LOSO mean r: {imu_r:.3f}")
print(f"PPG LOSO mean r: {ppg_r:.3f}")
