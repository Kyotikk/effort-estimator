#!/usr/bin/env python3
"""Check for data leakage and overfitting causes."""

import pandas as pd
import numpy as np

# Load the aligned data
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv')
df_labeled = df.dropna(subset=['borg'])

print('=== DATA LEAKAGE CHECK ===')
print(f'Total labeled samples: {len(df_labeled)}')
print(f'Unique Borg values: {sorted(df_labeled["borg"].unique())}')
print(f'\nBorg value counts:')
print(df_labeled['borg'].value_counts().sort_index())

# Check for time overlap
print(f'\n=== TIME OVERLAP CHECK ===')
print(f't_center range: {df_labeled["t_center"].min():.0f} - {df_labeled["t_center"].max():.0f}')
print(f'Window length: 10s, overlap: 70%')

# Check consecutive windows
t_sorted = df_labeled['t_center'].sort_values()
t_diff = t_sorted.diff().dropna()
print(f'Min time gap between windows: {t_diff.min():.2f}s')
print(f'Median time gap: {t_diff.median():.2f}s')

# Count overlapping windows
overlap_count = (t_diff < 10).sum()
print(f'Windows with <10s gap (OVERLAPPING): {overlap_count}/{len(t_diff)} ({100*overlap_count/len(t_diff):.1f}%)')

# Check activity labels
if 'activity' in df_labeled.columns:
    print(f'\n=== ACTIVITY CHECK ===')
    print(f'Unique activities: {df_labeled["activity"].nunique()}')
    print(f'\nWindows per activity:')
    for act in df_labeled['activity'].unique():
        sub = df_labeled[df_labeled['activity'] == act]
        n = len(sub)
        borg = sub['borg'].iloc[0]
        print(f'  {act}: {n} windows, borg={borg}')

print('\n=== LEAKAGE DIAGNOSIS ===')
print('PROBLEM: 70% overlap means adjacent windows share 70% of their data!')
print('  - Random train/test split puts OVERLAPPING windows in both sets')
print('  - Model "memorizes" the overlapping signal patterns')
print('  - This is TEMPORAL LEAKAGE')
print('\nSOLUTION: Split by ACTIVITY BOUT, not by random window')
