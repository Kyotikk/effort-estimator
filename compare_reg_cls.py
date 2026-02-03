#!/usr/bin/env python3
"""Compare regression vs classification for LOSO"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr, spearmanr

paths = [
    '/Users/pascalschlegel/data/interim/parsingsim1/sim_elderly1/effort_estimation_output/elderly_sim_elderly1/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim2/sim_elderly2/effort_estimation_output/elderly_sim_elderly2/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim4/sim_elderly4/effort_estimation_output/elderly_sim_elderly4/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim5/sim_elderly5/effort_estimation_output/elderly_sim_elderly5/fused_aligned_5.0s.csv',
]

def borg_to_4class(b):
    if b <= 1: return 0
    elif b <= 3: return 1
    elif b <= 5: return 2
    else: return 3

dfs = []
for i, p in enumerate(paths, 1):
    df = pd.read_csv(p)
    df['subject'] = f'P{i}'
    dfs.append(df)
combined = pd.concat(dfs).dropna(subset=['borg'])

imu_cols = [c for c in combined.columns if 'acc' in c.lower() or 'gyro' in c.lower()]
imu_cols = [c for c in imu_cols if combined[c].notna().mean() > 0.3 and combined[c].std() > 1e-10]

combined['cat'] = combined['borg'].apply(borg_to_4class)

print('LOSO - Comparing Regression vs Classification')
print('=' * 60)

subjects = combined['subject'].unique()
reg_r, cls_acc, cls_adj, spearman_r = [], [], [], []

for test_subj in subjects:
    train = combined[combined['subject'] != test_subj].dropna(subset=imu_cols + ['borg'])
    test = combined[combined['subject'] == test_subj].dropna(subset=imu_cols + ['borg'])
    
    X_train, y_train_reg = train[imu_cols].values, train['borg'].values
    X_test, y_test_reg = test[imu_cols].values, test['borg'].values
    y_train_cls, y_test_cls = train['cat'].values, test['cat'].values
    
    imp = SimpleImputer(strategy='median')
    scl = StandardScaler()
    X_train_s = scl.fit_transform(imp.fit_transform(X_train))
    X_test_s = scl.transform(imp.transform(X_test))
    
    # Regression
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train_s, y_train_reg)
    y_pred_reg = rf_reg.predict(X_test_s)
    r, _ = pearsonr(y_test_reg, y_pred_reg)
    reg_r.append(r)
    
    # Classification
    rf_cls = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf_cls.fit(X_train_s, y_train_cls)
    y_pred_cls = rf_cls.predict(X_test_s)
    acc = accuracy_score(y_test_cls, y_pred_cls)
    adj = np.mean(np.abs(y_test_cls - y_pred_cls) <= 1)
    cls_acc.append(acc)
    cls_adj.append(adj)
    
    # Spearman on classification (ordinal correlation)
    rho, _ = spearmanr(y_test_cls, y_pred_cls)
    spearman_r.append(rho)
    
    print(f'{test_subj}: Reg r={r:.2f} | Cls acc={acc*100:.0f}%, ±1={adj*100:.0f}%, Spearman={rho:.2f}')

print('-' * 60)
print(f'MEAN: Reg r={np.mean(reg_r):.2f} | Cls acc={np.mean(cls_acc)*100:.0f}%, ±1={np.mean(cls_adj)*100:.0f}%, Spearman={np.mean(spearman_r):.2f}')
print()
print(f'Regression r=0.56 is the right metric!')
print(f'Classification accuracy 32% sounds worse but ±1 category = {np.mean(cls_adj)*100:.0f}%')
