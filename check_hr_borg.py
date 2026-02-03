#!/usr/bin/env python3
"""Quick check: Does HR actually correlate with Borg?"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

paths = [
    '/Users/pascalschlegel/data/interim/parsingsim1/sim_elderly1/effort_estimation_output/elderly_sim_elderly1/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim2/sim_elderly2/effort_estimation_output/elderly_sim_elderly2/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim4/sim_elderly4/effort_estimation_output/elderly_sim_elderly4/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim5/sim_elderly5/effort_estimation_output/elderly_sim_elderly5/fused_aligned_5.0s.csv',
]

print('HR vs Borg correlation per subject:')
print('='*60)
all_dfs = []
for i, p in enumerate(paths, 1):
    try:
        df = pd.read_csv(p).dropna(subset=['borg', 'ppg_green_hr_mean'])
        df['subject'] = f'P{i}'
        all_dfs.append(df)
        if len(df) > 5:
            r, _ = pearsonr(df['ppg_green_hr_mean'], df['borg'])
            hr_min = df['ppg_green_hr_mean'].min()
            hr_max = df['ppg_green_hr_mean'].max()
            borg_min = df['borg'].min()
            borg_max = df['borg'].max()
            print(f'P{i}: r = {r:.3f}  (n={len(df)}, HR: {hr_min:.0f}-{hr_max:.0f} bpm, Borg: {borg_min:.1f}-{borg_max:.1f})')
    except Exception as e:
        print(f'P{i}: Error - {e}')

# Pooled
if all_dfs:
    combined = pd.concat(all_dfs)
    r_pooled, _ = pearsonr(combined['ppg_green_hr_mean'], combined['borg'])
    print(f'\nPooled: r = {r_pooled:.3f} (n={len(combined)})')
    
    # The problem: different baselines
    print('\n' + '='*60)
    print('THE PROBLEM: Different HR baselines per patient')
    print('='*60)
    for subj in combined['subject'].unique():
        sub_df = combined[combined['subject'] == subj]
        hr_at_low = sub_df[sub_df['borg'] <= 2]['ppg_green_hr_mean'].mean()
        hr_at_high = sub_df[sub_df['borg'] >= 4]['ppg_green_hr_mean'].mean()
        print(f'{subj}: HR at Borg≤2 = {hr_at_low:.0f} bpm, HR at Borg≥4 = {hr_at_high:.0f} bpm')
