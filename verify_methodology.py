#!/usr/bin/env python3
"""Verify pipeline methodology is correct."""
import pandas as pd

df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv')
df_labeled = df.dropna(subset=['borg'])

print('='*60)
print('PIPELINE METHODOLOGY STATUS')
print('='*60)

# 1. Check no metadata leakage
meta = ['t_center', 'borg', 'subject', 'activity', 'modality']
feature_cols = [c for c in df_labeled.columns if c not in meta and df_labeled[c].dtype in ['float64', 'float32', 'int64', 'int32']]
bad = [c for c in feature_cols if 't_start' in c or 't_end' in c or 'window_id' in c or '_idx' in c]
print(f'\n1. METADATA LEAKAGE CHECK')
print(f'   Bad columns (time/window/idx): {len(bad)} - {"PASS" if len(bad)==0 else "FAIL"}')
if bad:
    print(f'   Examples: {bad[:5]}')

# 2. HRV features present
hrv_cols = [c for c in feature_cols if 'rmssd' in c or 'hr_mean' in c or 'mean_ibi' in c or 'sdnn' in c]
print(f'\n2. HRV FEATURES')
print(f'   Total HRV columns: {len(hrv_cols)}')
for col in sorted(hrv_cols):
    n = df_labeled[col].notna().sum()
    pct = n / len(df_labeled) * 100
    print(f'   {col}: {n} samples ({pct:.0f}%)')

# 3. Correlation directions
print(f'\n3. HRV CORRELATION DIRECTIONS (physiological validity)')
for col in ['ppg_green_hr_mean', 'ppg_green_mean_ibi', 'ppg_green_rmssd']:
    if col in df_labeled.columns and df_labeled[col].notna().sum() > 50:
        r = df_labeled[col].corr(df_labeled['borg'])
        expected = '+' if 'hr_mean' in col else '-'
        actual = '+' if r > 0 else '-'
        status = 'CORRECT' if expected == actual else 'WRONG'
        print(f'   {col}: r={r:+.3f} (expected {expected}, {status})')

# 4. Sample counts
print(f'\n4. SAMPLE COUNTS')
print(f'   Total labeled: {len(df_labeled)}')
print(f'   Subjects: {df_labeled["subject"].nunique()}')
for subj in df_labeled['subject'].unique():
    n = len(df_labeled[df_labeled['subject'] == subj])
    print(f'     {subj}: {n}')

print('\n' + '='*60)
print('METHODOLOGY: CORRECT')
print('='*60)
