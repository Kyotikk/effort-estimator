#!/usr/bin/env python3
"""Get the best features for predicting Borg effort."""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv')
elderly = df[df['subject_id'] == 'sim_elderly3']

# Get all numeric feature columns
exclude = ['borg', 'subject_id', 'window_id', 'start_idx', 'end_idx', 'valid', 
           't_start', 't_center', 't_end', 'n_samples', 'win_sec']
features = [c for c in elderly.columns if c not in exclude and elderly[c].dtype in ['float64', 'int64']]

# Calculate correlations
results = []
for f in features:
    valid = elderly[[f, 'borg']].dropna()
    if len(valid) > 50:
        r, p = stats.pearsonr(valid[f], valid['borg'])
        if not np.isnan(r):
            if 'eda' in f.lower():
                mod = 'EDA'
            elif any(x in f.lower() for x in ['ibi', 'rmssd', 'sdnn', 'pnn', 'hr_', 'lf_', 'hf_']):
                mod = 'HRV'
            elif any(x in f.lower() for x in ['acc_', 'gyro']):
                mod = 'IMU'
            else:
                mod = 'PPG'
            results.append({'feature': f, 'r': r, 'abs_r': abs(r), 'p': p, 'mod': mod})

df_r = pd.DataFrame(results).sort_values('abs_r', ascending=False)

print('=' * 75)
print('TOP 20 FEATURES FOR PREDICTING BORG (Elderly)')
print('=' * 75)
print(f"{'Rank':<5} {'Feature':<45} {'r':>7} {'Modality':>8}")
print('-' * 75)
for i, (_, row) in enumerate(df_r.head(20).iterrows()):
    sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
    print(f"{i+1:<5} {row['feature']:<45} {row['r']:>+.3f}{sig:<3} {row['mod']:>8}")

print()
print('=' * 75)
print('BEST BY MODALITY')
print('=' * 75)
for mod in ['EDA', 'HRV', 'IMU', 'PPG']:
    best = df_r[df_r['mod']==mod].head(3)
    print(f"\n{mod}:")
    for _, row in best.iterrows():
        print(f"  {row['feature']:<42} r = {row['r']:>+.3f}")

print()
print('=' * 75)
print('SUMMARY: USE THESE FEATURES')
print('=' * 75)
print("""
TOP PREDICTORS (in order):

1. EDA features (best single predictors, r ~ 0.50):
   - eda_cc_range, eda_scl_std, eda_phasic_max
   - Interpretation: Skin conductance variability increases with effort

2. PPG/HR features (r ~ 0.45-0.49):
   - ppg_green_n_peaks (heart rate proxy)
   - Interpretation: More peaks = faster heart rate = more effort

3. HRV features (r ~ 0.41-0.45):
   - ppg_green_mean_ibi (negative correlation)
   - ppg_green_hr_mean (positive correlation)  
   - Interpretation: Shorter IBI / faster HR = more effort

4. IMU features (r ~ 0.30):
   - acc_x_dyn__max (movement intensity)
   - Interpretation: More movement = more effort (but weaker than physio)
""")
