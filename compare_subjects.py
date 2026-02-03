#!/usr/bin/env python3
"""Compare correlations across all 3 subjects."""

import pandas as pd
import numpy as np
from pathlib import Path

subjects = ['sim_elderly3', 'sim_healthy3', 'sim_severe3']
base = Path('/Users/pascalschlegel/data/interim/parsingsim3')

print('CORRELATION WITH BORG ACROSS ALL 3 SUBJECTS')
print('='*60)
print('r_time = correlation with time (spurious if high)')

for subj in subjects:
    fused = base / subj / 'effort_estimation_output' / f'parsingsim3_{subj}' / 'fused_aligned_10.0s.csv'
    if not fused.exists():
        print(f'\n{subj}: NO FUSED DATA')
        continue
    
    df = pd.read_csv(fused)
    df = df.dropna(subset=['borg'])
for feat in ['hr_load_sqrt', 'trimp_banister', 'imu_load_sqrt', 'mad_load']:
    r_e = stats.pearsonr(elderly[feat], elderly['borg'])[0]
    r_h = stats.pearsonr(healthy[feat], healthy['borg'])[0]
    print(f'  {feat:24s} r={r_e:+.3f}         r={r_h:+.3f}')

# Combined score optimization
print(f'\n' + '='*70)
print('OPTIMAL COMBINED WEIGHTS')
print('='*70)

for df, name in [(elderly, 'sim_elderly3'), (healthy, 'sim_healthy3')]:
    z_hr = (df['hr_load_sqrt'] - df['hr_load_sqrt'].mean()) / df['hr_load_sqrt'].std()
    z_imu = (df['mad_load'] - df['mad_load'].mean()) / df['mad_load'].std()
    
    best_r = 0
    best_w = 0
    for w in np.arange(0.5, 1.0, 0.05):
        combined = w * z_hr + (1-w) * z_imu
        r = stats.pearsonr(combined, df['borg'])[0]
        if r > best_r:
            best_r = r
            best_w = w
    
    print(f'{name}: w_HR={best_w:.2f}, w_IMU={1-best_w:.2f} → r={best_r:.3f}')

# Test with fixed 0.8/0.2 weights
print(f'\n' + '='*70)
print('FIXED WEIGHTS (0.8 HR / 0.2 IMU)')
print('='*70)

for df, name in [(elderly, 'sim_elderly3'), (healthy, 'sim_healthy3')]:
    z_hr = (df['hr_load_sqrt'] - df['hr_load_sqrt'].mean()) / df['hr_load_sqrt'].std()
    z_imu = (df['mad_load'] - df['mad_load'].mean()) / df['mad_load'].std()
    combined = 0.8 * z_hr + 0.2 * z_imu
    r = stats.pearsonr(combined, df['borg'])[0]
    print(f'{name}: r={r:.3f}')
