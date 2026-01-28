#!/usr/bin/env python3
"""Analyze RMSSD and check if it should be included."""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv')
elderly = df[df['subject_id'] == 'sim_elderly3']

print('RMSSD CORRELATION ANALYSIS')
print('=' * 60)
print()
print('Yes, NEGATIVE correlation is GOOD - inverse relationship:')
print('  - Higher effort -> lower RMSSD (less HRV variability)')
print('  - This is physiologically correct!')
print()

# Compare correlations
features = ['ppg_green_mean_ibi', 'ppg_green_rmssd', 'ppg_green_sdnn', 'ppg_green_hr_mean']
print(f"{'Feature':<25} {'r':<10} {'Interpretation':<30}")
print('-' * 65)

for f in features:
    valid = elderly[[f, 'borg']].dropna()
    if len(valid) > 30:
        r, p = stats.pearsonr(valid[f], valid['borg'])
        if 'ibi' in f:
            interp = 'Higher effort -> shorter IBI'
        elif 'rmssd' in f:
            interp = 'Higher effort -> less HRV'
        elif 'sdnn' in f:
            interp = 'Higher effort -> less variability'
        else:
            interp = 'Higher effort -> faster HR'
        print(f'{f:<25} {r:>6.3f}    {interp}')

print()
print('SHOULD YOU INCLUDE RMSSD?')
print('=' * 60)
print('|r| = 0.25 is weak-moderate correlation')
print('Compare: mean_ibi |r| = 0.45 (stronger)')
print()
print('VERDICT: Include RMSSD if:')
print('  1. It adds independent information (not just correlated with IBI)')
print('  2. XGBoost feature importance shows it helps')
print()

# Check if RMSSD adds info beyond IBI
valid = elderly[['ppg_green_mean_ibi', 'ppg_green_rmssd', 'borg']].dropna()
ibi_rmssd_corr = stats.pearsonr(valid['ppg_green_mean_ibi'], valid['ppg_green_rmssd'])[0]
print(f'IBI-RMSSD correlation: r = {ibi_rmssd_corr:.3f}')
if abs(ibi_rmssd_corr) > 0.7:
    print('-> IBI and RMSSD are highly correlated (redundant)')
else:
    print('-> RMSSD provides additional information beyond IBI')

print()
print('=' * 60)
print('BOTTOM LINE')
print('=' * 60)
print('''
1. r = -0.25 for RMSSD is VALID (negative = inverse relationship)
2. mean_ibi (r = -0.45) is BETTER predictor than RMSSD (r = -0.25)
3. Include RMSSD anyway - XGBoost can use both!
4. But... the R² = 0.89 is FAKE due to temporal leakage :(
''')
