#!/usr/bin/env python3
"""Quick check: within-subject accuracy vs across-subject"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

print('WITHIN-SUBJECT 4-class (train/test split within each person)')
print('=' * 60)
within_accs = []
for i, p in enumerate(paths, 1):
    df = pd.read_csv(p).dropna(subset=['borg'])
    imu_cols = [c for c in df.columns if 'acc' in c.lower() or 'gyro' in c.lower()]
    imu_cols = [c for c in imu_cols if df[c].notna().mean() > 0.3 and df[c].std() > 1e-10]
    
    df['cat'] = df['borg'].apply(borg_to_4class)
    X, y = df[imu_cols].values, df['cat'].values
    
    if len(np.unique(y)) < 2:
        print(f'P{i}: skipped (only 1 class)')
        continue
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    imp = SimpleImputer(strategy='median')
    scl = StandardScaler()
    X_train = scl.fit_transform(imp.fit_transform(X_train))
    X_test = scl.transform(imp.transform(X_test))
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    within_accs.append(acc)
    print(f'P{i}: {acc*100:.1f}% ({len(df)} samples)')

print(f'\nMean within-subject: {np.mean(within_accs)*100:.1f}%')
print(f'LOSO across-subject: ~32%')
print(f'Random chance: 25%')
