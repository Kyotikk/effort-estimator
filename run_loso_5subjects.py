#!/usr/bin/env python3
"""
Combine all 5 elderly subjects and run LOSO evaluation.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb
from pathlib import Path

# Collect all 5 subjects at 5.0s window
SUBJECTS = [
    ('sim_elderly1', '/Users/pascalschlegel/data/interim/parsingsim1/sim_elderly1/effort_estimation_output/elderly_sim_elderly1/fused_aligned_5.0s.csv'),
    ('sim_elderly2', '/Users/pascalschlegel/data/interim/parsingsim2/sim_elderly2/effort_estimation_output/elderly_sim_elderly2/fused_aligned_5.0s.csv'),
    ('sim_elderly3', '/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3/fused_aligned_5.0s.csv'),
    ('sim_elderly4', '/Users/pascalschlegel/data/interim/parsingsim4/sim_elderly4/effort_estimation_output/elderly_sim_elderly4/fused_aligned_5.0s.csv'),
    ('sim_elderly5', '/Users/pascalschlegel/data/interim/parsingsim5/sim_elderly5/effort_estimation_output/elderly_sim_elderly5/fused_aligned_5.0s.csv'),
]

OUTPUT_DIR = Path('/Users/pascalschlegel/data/interim/elderly_combined_5subj')
OUTPUT_DIR.mkdir(exist_ok=True)

print('='*70)
print('COMBINING ALL 5 ELDERLY SUBJECTS (5s windows)')
print('='*70)

all_dfs = []
for subj, path in SUBJECTS:
    try:
        df = pd.read_csv(path)
        df['subject'] = subj
        labeled = df['borg'].notna().sum()
        borg_min = df['borg'].min() if labeled > 0 else 0
        borg_max = df['borg'].max() if labeled > 0 else 0
        print(f'{subj}: {len(df)} samples, {labeled} labeled, Borg range: {borg_min:.1f}-{borg_max:.1f}')
        all_dfs.append(df)
    except Exception as e:
        print(f'{subj}: ERROR - {e}')

combined = pd.concat(all_dfs, ignore_index=True)
combined = combined.dropna(subset=['borg'])
print(f'\nCombined: {len(combined)} labeled samples from {combined["subject"].nunique()} subjects')

# Save combined dataset
combined.to_csv(OUTPUT_DIR / 'all_5_elderly_5s.csv', index=False)
print(f'Saved to: {OUTPUT_DIR}/all_5_elderly_5s.csv')

# Get feature columns
meta_cols = ['t_center', 'subject', 'label', 'borg', 'activity', 'modality', 
             'valid', 'n_samples', 'win_sec', 'valid_r', 'n_samples_r', 'win_sec_r']
feature_cols = [c for c in combined.columns if c not in meta_cols and not c.endswith('_r')]

# Clean features - drop columns with NaN or constant
X_df = combined[feature_cols].copy()
X_df = X_df.dropna(axis=1)
X_df = X_df.loc[:, X_df.std() > 1e-6]
feature_cols_clean = X_df.columns.tolist()

print(f'\nFeatures after cleaning: {len(feature_cols_clean)}')

X = X_df.values
y = combined['borg'].values
subjects = combined['subject'].values

# =============================================================================
# LOSO CROSS-VALIDATION
# =============================================================================
print('\n' + '='*70)
print('LOSO CROSS-VALIDATION (Leave-One-Subject-Out)')
print('='*70)

unique_subjects = np.unique(subjects)
print(f'Subjects: {unique_subjects}')

results = []

for test_subj in unique_subjects:
    train_mask = subjects != test_subj
    test_mask = subjects == test_subj
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    if len(X_test) < 10:
        print(f'  {test_subj}: skipped (only {len(X_test)} samples)')
        continue
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_s, y_train)
    y_pred_ridge = ridge.predict(X_test_s)
    r_ridge, _ = pearsonr(y_test, y_pred_ridge)
    mae_ridge = np.abs(y_test - y_pred_ridge).mean()
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
    xgb_model.fit(X_train_s, y_train)
    y_pred_xgb = xgb_model.predict(X_test_s)
    r_xgb, _ = pearsonr(y_test, y_pred_xgb)
    mae_xgb = np.abs(y_test - y_pred_xgb).mean()
    
    results.append({
        'subject': test_subj,
        'n_train': len(y_train),
        'n_test': len(y_test),
        'borg_mean': y_test.mean(),
        'borg_std': y_test.std(),
        'ridge_r': r_ridge,
        'ridge_mae': mae_ridge,
        'xgb_r': r_xgb,
        'xgb_mae': mae_xgb,
    })
    
    print(f'  {test_subj}: Ridge r={r_ridge:.3f} MAE={mae_ridge:.2f} | XGB r={r_xgb:.3f} MAE={mae_xgb:.2f} (n={len(y_test)}, train={len(y_train)})')

# Summary
print('\n' + '='*70)
print('LOSO SUMMARY (5 SUBJECTS)')
print('='*70)

results_df = pd.DataFrame(results)

print('\nPer-subject results:')
print(results_df.to_string(index=False))

print(f'\n{"="*70}')
print('AVERAGE LOSO METRICS:')
print(f'{"="*70}')
print(f'  Ridge:   r = {results_df["ridge_r"].mean():.3f} (+/- {results_df["ridge_r"].std():.3f}), MAE = {results_df["ridge_mae"].mean():.2f}')
print(f'  XGBoost: r = {results_df["xgb_r"].mean():.3f} (+/- {results_df["xgb_r"].std():.3f}), MAE = {results_df["xgb_mae"].mean():.2f}')

# Save results
results_df.to_csv(OUTPUT_DIR / 'loso_results_5subjects.csv', index=False)
print(f'\nSaved: {OUTPUT_DIR}/loso_results_5subjects.csv')
