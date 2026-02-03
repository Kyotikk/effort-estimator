#!/usr/bin/env python3
"""Test HR load features only for effort estimation"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Load all data
all_dfs = []
for i in [1,2,3,4,5]:
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = f'elderly{i}'
        all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True).dropna(subset=['borg'])

# Define feature sets
hr_features = [c for c in df_all.columns if 'hr' in c.lower() and 'ppg' in c.lower()]
hr_peak_features = [c for c in df_all.columns if 'ppg' in c.lower() and ('n_peaks' in c.lower() or 'cross_rate' in c.lower())]
imu_features = [c for c in df_all.columns if 'acc' in c.lower()]

print('='*70)
print('HR LOAD FEATURES TEST')
print('='*70)
print(f'HR features: {len(hr_features)}')
print(f'HR + peak features: {len(hr_features) + len(hr_peak_features)}')
print(f'IMU features (for comparison): {len(imu_features)}')

def evaluate_random_cal(df, features, model, cal_frac=0.2):
    subjects = sorted(df['subject'].unique())
    valid_features = [f for f in features if f in df.columns 
                      and df[f].notna().mean() > 0.5 
                      and df[f].std() > 1e-10]
    
    if len(valid_features) == 0:
        return float('nan'), {}
    
    per_subject = {}
    for test_sub in subjects:
        train_df = df[df['subject'] != test_sub]
        test_df = df[df['subject'] == test_sub]
        
        n_test = len(test_df)
        n_cal = max(5, int(n_test * cal_frac))
        idx = np.random.permutation(n_test)
        cal_idx = idx[:n_cal]
        eval_idx = idx[n_cal:]
        
        if len(eval_idx) < 5:
            continue
        
        X_train = train_df[valid_features].values
        y_train = train_df['borg'].values
        X_test = test_df[valid_features].values
        y_test = test_df['borg'].values
        
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_train = scaler.fit_transform(imputer.fit_transform(X_train))
        X_test = scaler.transform(imputer.transform(X_test))
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        cal_offset = y_test[cal_idx].mean() - y_pred[cal_idx].mean()
        y_pred_cal = y_pred + cal_offset
        
        r, _ = pearsonr(y_pred_cal[eval_idx], y_test[eval_idx])
        per_subject[test_sub] = r
    
    return np.mean(list(per_subject.values())), per_subject

print()
print(f"{'Config':<35} | {'Per-sub r':>10} | Per-subject breakdown")
print('-'*90)

configs = [
    ('HR only (18 features) + Ridge', hr_features, Ridge(alpha=1.0)),
    ('HR only + RF', hr_features, RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1)),
    ('HR + peaks (27) + Ridge', hr_features + hr_peak_features, Ridge(alpha=1.0)),
    ('HR + peaks + RF', hr_features + hr_peak_features, RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1)),
    ('IMU only (60) + RF', imu_features, RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)),
    ('HR + IMU combined + RF', hr_features + imu_features, RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)),
]

for name, features, model in configs:
    avg_r, per_sub = evaluate_random_cal(df_all, features, model, cal_frac=0.2)
    if per_sub:
        breakdown = ' '.join([f"{s[-1]}:{r:.2f}" for s, r in per_sub.items()])
        print(f"{name:<35} | {avg_r:>10.3f} | {breakdown}")
    else:
        print(f"{name:<35} | {'N/A':>10} | No valid features")

print()
print('='*70)
print('HR FEATURE CORRELATIONS WITH BORG (pooled)')
print('='*70)

for f in hr_features[:6]:  # Just green HR
    valid = df_all[f].notna()
    if valid.sum() > 100:
        r, _ = pearsonr(df_all.loc[valid, f], df_all.loc[valid, 'borg'])
        print(f'  {f}: r = {r:.3f}')

print()
print('='*70)
print('CONCLUSION')
print('='*70)
