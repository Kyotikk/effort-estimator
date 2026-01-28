#!/usr/bin/env python3
"""Analyze feature correlations with Borg."""
import pandas as pd
import numpy as np

df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv')
df_labeled = df.dropna(subset=['borg'])
print(f'Samples: {len(df_labeled)}')

# Get all numeric columns
skip = ['t_center', 'window_id', 'borg', 'subject', 'activity', 'start_idx', 'end_idx', 't_start', 't_end', 'modality']
feat_cols = [c for c in df_labeled.columns if c not in skip and df_labeled[c].dtype in ['float64', 'float32', 'int64', 'int32']]
print(f'Total features: {len(feat_cols)}')

# Calculate correlations
correlations = []
for col in feat_cols:
    if df_labeled[col].notna().all():  # Skip if has NaN
        r = np.corrcoef(df_labeled[col], df_labeled['borg'])[0, 1]
        if not np.isnan(r):
            correlations.append((col, abs(r), r))

# Sort by absolute correlation
correlations.sort(key=lambda x: x[1], reverse=True)

print('\nTop 30 features by correlation:')
for i, (col, abs_r, r) in enumerate(correlations[:30], 1):
    hrv_flag = ' [HRV]' if 'rmssd' in col or 'hr_mean' in col or 'mean_ibi' in col or 'sdnn' in col else ''
    print(f'{i:2}. {col}: r = {r:+.3f}{hrv_flag}')

print('\nHRV features ranking:')
hrv_ranks = []
for i, (col, abs_r, r) in enumerate(correlations):
    if 'rmssd' in col or 'hr_mean' in col or 'mean_ibi' in col or 'sdnn' in col:
        hrv_ranks.append((i+1, col, r))

for rank, col, r in hrv_ranks:
    print(f'  Rank {rank}: {col}: r = {r:+.3f}')

# Count features above correlation threshold
n_above_02 = sum(1 for _, abs_r, _ in correlations if abs_r > 0.2)
n_above_028 = sum(1 for _, abs_r, _ in correlations if abs_r > 0.28)
print(f'\nFeatures with |r| > 0.2: {n_above_02}')
print(f'Features with |r| > 0.28 (HRV level): {n_above_028}')
