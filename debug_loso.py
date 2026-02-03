#!/usr/bin/env python3
"""Debug LOSO calculation."""

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
        print(f'P{i}: {len(df)} windows, {df["borg"].notna().sum()} with Borg')

df_all = pd.concat(dfs, ignore_index=True)
imu_cols = [c for c in df_all.columns if 'acc_' in c and '_r' not in c]
print(f'\nTotal: {len(df_all)} windows, {len(imu_cols)} IMU features')

# LOSO
per_subj_r = []
all_y_true = []
all_y_pred = []

for test_subj in sorted(df_all['subject'].unique()):
    train_df = df_all[df_all['subject'] != test_subj].dropna(subset=['borg'])
    test_df = df_all[df_all['subject'] == test_subj].dropna(subset=['borg'])
    
    X_train = train_df[imu_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_df['borg'].values
    X_test = test_df[imu_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test_df['borg'].values
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    if len(y_test) > 2 and np.std(y_test) > 0:
        r, _ = pearsonr(y_test, y_pred)
        per_subj_r.append(r)
        print(f'P{test_subj}: n={len(y_test)}, r={r:.3f}')
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

print(f'\nMean of per-subject r: {np.mean(per_subj_r):.3f}')
print(f'Combined pooled r: {pearsonr(all_y_true, all_y_pred)[0]:.3f}')
