#!/usr/bin/env python3
"""Analyze if PPG amplitude features are valid or capture subject differences"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

df = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined/elderly_aligned_5.0s.csv')
df = df.dropna(subset=['borg'])

# Check PPG amplitude features
amp_features = ['ppg_green_max', 'ppg_green_rms', 'ppg_green_mean', 'ppg_red_max', 'ppg_infra_max']
amp_features = [f for f in amp_features if f in df.columns]

print('='*70)
print('PPG AMPLITUDE FEATURE ANALYSIS')
print('Is "signal power" a valid feature or just inter-subject noise?')
print('='*70)

for feat in amp_features:
    print(f'\n{feat}:')
    
    # Overall correlation with Borg
    r_overall, _ = pearsonr(df[feat].fillna(0), df['borg'])
    print(f'  Overall correlation with Borg: r = {r_overall:.3f}')
    
    # Per-subject correlation (does it track effort WITHIN each person?)
    within_subj_r = []
    for subj in df['subject'].unique():
        subj_data = df[df['subject'] == subj]
        if len(subj_data) > 10:
            r, _ = pearsonr(subj_data[feat].fillna(0), subj_data['borg'])
            within_subj_r.append(r)
            print(f'    P{subj[-1]}: r = {r:.3f}')
    
    print(f'  Mean within-subject r: {np.mean(within_subj_r):.3f}')
    
    # Check inter-subject variance vs intra-subject variance
    between_subj_var = df.groupby('subject')[feat].mean().var()
    within_subj_var = df.groupby('subject')[feat].var().mean()
    ratio = between_subj_var / within_subj_var if within_subj_var > 0 else 999
    
    print(f'  Between-subject variance: {between_subj_var:.2e}')
    print(f'  Within-subject variance:  {within_subj_var:.2e}')
    print(f'  Ratio (between/within):   {ratio:.2f}')
    
    if ratio > 5:
        print(f'  ⚠️  HIGH inter-subject variance - may capture SUBJECT not EFFORT')
    elif ratio < 0.5:
        print(f'  ✓  LOW inter-subject variance - likely captures EFFORT')
    else:
        print(f'  ~  MIXED - captures both subject and effort variance')

print('\n' + '='*70)
print('INTERPRETATION GUIDE:')
print('='*70)
print('''
If Ratio >> 1:  Feature primarily captures SUBJECT DIFFERENCES
                → Problematic for cross-subject models (Method 1 fails)
                → Works fine with calibration (Method 3) or within-subject (Method 4)

If Ratio << 1:  Feature primarily captures EFFORT CHANGES
                → Good for all methods
                → Generalizes across subjects

PROBLEM with raw PPG amplitude (max, mean, rms):
  - Skin tone affects light absorption
  - Sensor placement varies between sessions
  - Blood vessel depth differs between people
  
BETTER PPG features:
  - Normalized features (range, iqr, p90-p10)
  - Shape features (kurtosis, crest_factor)
  - Derivative features (dx_*, ddx_*)
  - Rate features (zcr, n_peaks)
''')

# Compare with better PPG features
print('='*70)
print('COMPARISON: Amplitude vs Normalized/Shape Features')
print('='*70)

good_features = ['ppg_green_range', 'ppg_green_iqr', 'ppg_green_p90_p10', 
                 'ppg_green_crest_factor', 'ppg_green_zcr', 'ppg_green_n_peaks']
good_features = [f for f in good_features if f in df.columns]

for feat in good_features:
    between_var = df.groupby('subject')[feat].mean().var()
    within_var = df.groupby('subject')[feat].var().mean()
    ratio = between_var / within_var if within_var > 0 else 999
    
    within_r = []
    for subj in df['subject'].unique():
        subj_data = df[df['subject'] == subj]
        if len(subj_data) > 10:
            r, _ = pearsonr(subj_data[feat].fillna(0), subj_data['borg'])
            within_r.append(r)
    
    status = "✓" if ratio < 2 else "⚠️" if ratio < 5 else "❌"
    print(f'{status} {feat:30} ratio={ratio:.2f}, mean within-r={np.mean(within_r):.3f}')
