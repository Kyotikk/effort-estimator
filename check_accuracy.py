#!/usr/bin/env python3
"""Check prediction accuracy within ±1 Borg"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load the available data with Borg labels
df = pd.read_csv('output/tli_all_subjects.csv')

# Features available
feature_cols = ['rms_acc_mag', 'rms_jerk', 'imu_load']
df_clean = df.dropna(subset=['borg'] + feature_cols).copy()

subjects = sorted(df_clean['subject'].unique())

# LOSO predictions
all_preds = []
for test_subj in subjects:
    train = df_clean[df_clean['subject'] != test_subj]
    test = df_clean[df_clean['subject'] == test_subj].copy()
    
    X_train = train[feature_cols].values
    y_train = train['borg'].values
    X_test = test[feature_cols].values
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    test['predicted'] = rf.predict(X_test)
    all_preds.append(test)

results = pd.concat(all_preds)
results['error'] = np.abs(results['borg'] - results['predicted'])

print('=== PREDICTION ACCURACY (LOSO - IMU features) ===')
print()
print(f'Total activity bouts: {len(results)}')
print()

for threshold in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    within = (results['error'] <= threshold).sum()
    pct = 100 * within / len(results)
    print(f'Within ±{threshold:.1f} Borg: {within}/{len(results)} = {pct:.1f}%')

print()
print(f'Mean Absolute Error: {results["error"].mean():.2f} Borg points')
print(f'Median Error: {results["error"].median():.2f} Borg points')
