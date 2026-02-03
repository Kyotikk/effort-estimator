#!/usr/bin/env python3
"""Compare results across window sizes."""

import yaml
from pathlib import Path

base = Path('/Users/pascalschlegel/data/interim/elderly_combined')

# Load all results
results = {}
for ws in [5.0, 10.0, 30.0]:
    xgb_path = base / f'xgboost_results_{ws}s' / 'summary.yaml'
    ridge_path = base / f'ridge_results_{ws}s' / 'summary.yaml'
    
    # Handle old naming for 5s
    if ws == 5.0:
        xgb_path = base / 'xgboost_results' / 'summary.yaml'
        ridge_path = base / 'linear_results' / 'summary.yaml'
    
    if xgb_path.exists() and ridge_path.exists():
        with open(xgb_path) as f:
            xgb = yaml.safe_load(f)
        with open(ridge_path) as f:
            ridge = yaml.safe_load(f)
        results[ws] = {'xgboost': xgb, 'ridge': ridge}

print('='*80)
print('WINDOW SIZE COMPARISON - EFFORT ESTIMATION (3 ELDERLY PATIENTS)')
print('='*80)
print()
print(f"{'Window':<10} {'N':<8} {'Feats':<8} {'XGB r':<10} {'XGB MAE':<10} {'Ridge r':<10} {'Ridge MAE':<10}")
print('-'*80)

for ws in sorted(results.keys()):
    xgb = results[ws]['xgboost']
    ridge = results[ws]['ridge']
    print(f"{ws:.0f}s{'':<7} {xgb['n_samples']:<8} {xgb['n_features']:<8} {xgb['pearson_r']:<10.3f} {xgb['mae']:<10.2f} {ridge['pearson_r']:<10.3f} {ridge['mae']:<10.2f}")

print()
print('INTERPRETATION:')
print('-'*80)
print('5s windows have best performance (more samples, better temporal resolution)')
print('As window size increases: fewer samples, signal averaging reduces discrimination')
print('30s lost sim_elderly5 labels due to Borg/window alignment issues')
print()
print('CONCLUSION: 5s windows are optimal for this dataset/task')
