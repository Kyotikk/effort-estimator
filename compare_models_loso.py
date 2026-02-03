#!/usr/bin/env python3
"""Compare Ridge vs XGBoost on LOSO CV."""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb
from pathlib import Path

DATA_PATH = Path('/Users/pascalschlegel/data/interim/elderly_combined')

# Load data
df = pd.read_csv(DATA_PATH / 'elderly_aligned_5.0s.csv').dropna(subset=['borg'])
features_df = pd.read_csv(DATA_PATH / 'qc_5.0s/features_selected_pruned.csv', header=None)
feature_cols = [f for f in features_df[0].tolist() if f in df.columns]

X = df[feature_cols].values
y = df['borg'].values
subjects = df['subject'].values

print('='*60)
print('LOSO CV: Ridge vs XGBoost Comparison')
print('='*60)

all_results = {}

for model_name in ['Ridge', 'XGBoost']:
    print(f'\n{model_name}:')
    results = []
    
    for test_subj in np.unique(subjects):
        train_mask = subjects != test_subj
        test_mask = subjects == test_subj
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        if model_name == 'Ridge':
            model = Ridge(alpha=1.0)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
        else:
            model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
        
        r, _ = pearsonr(y_test, y_pred)
        mae = np.abs(y_test - y_pred).mean()
        results.append({'subj': test_subj, 'r': r, 'mae': mae})
        print(f'  {test_subj}: r={r:.3f}, MAE={mae:.2f}')
    
    avg_r = np.mean([r['r'] for r in results])
    avg_mae = np.mean([r['mae'] for r in results])
    print(f'  --> AVERAGE: r={avg_r:.3f}, MAE={avg_mae:.2f}')
    all_results[model_name] = {'r': avg_r, 'mae': avg_mae}

print()
print('='*60)
print('SUMMARY')
print('='*60)
print(f"Ridge LOSO:   r={all_results['Ridge']['r']:.3f}, MAE={all_results['Ridge']['mae']:.2f}")
print(f"XGBoost LOSO: r={all_results['XGBoost']['r']:.3f}, MAE={all_results['XGBoost']['mae']:.2f}")
print()
print('CONCLUSION: Both models perform similarly poorly on LOSO')
print('The problem is DATA (only 3 subjects), not MODEL choice')
