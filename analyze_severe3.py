#!/usr/bin/env python3
"""Analyze feature correlations for severe3 patient."""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv')

print("=" * 70)
print("SEVERE3 PATIENT ANALYSIS")
print("=" * 70)

severe = df[df['subject_id'] == 'sim_severe3']
print(f"Total samples: {len(severe)}")
print(f"Borg range: {severe['borg'].min()} - {severe['borg'].max()}")
print(f"Borg valid: {severe['borg'].notna().sum()}")
print(f"Borg distribution:\n{severe['borg'].value_counts().sort_index()}")

# Get features
exclude = ['borg', 'subject_id', 'window_id', 'start_idx', 'end_idx', 'valid', 
           't_start', 't_center', 't_end', 'n_samples', 'win_sec']
features = [c for c in severe.columns if c not in exclude and severe[c].dtype in ['float64', 'int64']]

# Calculate correlations
results = []
for f in features:
    valid = severe[[f, 'borg']].dropna()
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
            results.append({'feature': f, 'r': r, 'abs_r': abs(r), 'p': p, 'mod': mod, 'n': len(valid)})

df_r = pd.DataFrame(results).sort_values('abs_r', ascending=False)

print(f"\n" + "=" * 70)
print("TOP 20 FEATURES FOR SEVERE3")
print("=" * 70)
print(f"{'Rank':<5} {'Feature':<45} {'r':>8} {'Mod':>6} {'n':>6}")
print("-" * 75)
for i, (_, row) in enumerate(df_r.head(20).iterrows()):
    sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
    print(f"{i+1:<5} {row['feature']:<45} {row['r']:>+.3f}{sig:<3} {row['mod']:>6} {row['n']:>6}")

print(f"\n" + "=" * 70)
print("BEST BY MODALITY (SEVERE3)")
print("=" * 70)
for mod in ['EDA', 'HRV', 'IMU', 'PPG']:
    best = df_r[df_r['mod']==mod].head(3)
    print(f"\n{mod}:")
    for _, row in best.iterrows():
        sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else 'ns'
        print(f"  {row['feature']:<42} r = {row['r']:>+.3f} {sig}")

# Compare with elderly
print(f"\n" + "=" * 70)
print("COMPARISON: ELDERLY3 vs SEVERE3")
print("=" * 70)

elderly = df[df['subject_id'] == 'sim_elderly3']

comparison_features = ['eda_cc_range', 'ppg_green_mean_ibi', 'ppg_green_n_peaks', 'acc_x_dyn__max_r']

print(f"\n{'Feature':<30} {'Elderly r':>12} {'Severe r':>12}")
print("-" * 60)

for f in comparison_features:
    # Elderly
    valid_e = elderly[[f, 'borg']].dropna()
    r_e, p_e = stats.pearsonr(valid_e[f], valid_e['borg']) if len(valid_e) > 30 else (np.nan, np.nan)
    
    # Severe
    valid_s = severe[[f, 'borg']].dropna()
    r_s, p_s = stats.pearsonr(valid_s[f], valid_s['borg']) if len(valid_s) > 30 else (np.nan, np.nan)
    
    sig_e = '***' if p_e < 0.001 else '**' if p_e < 0.01 else '*' if p_e < 0.05 else 'ns'
    sig_s = '***' if p_s < 0.001 else '**' if p_s < 0.01 else '*' if p_s < 0.05 else 'ns'
    
    print(f"{f:<30} {r_e:>+.3f} {sig_e:<3}    {r_s:>+.3f} {sig_s:<3}")

# Check data quality
print(f"\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)
eda_col = 'eda_cc_range'
hrv_col = 'ppg_green_mean_ibi'

print(f"\nEDA valid samples: {severe[eda_col].notna().sum()} / {len(severe)}")
print(f"HRV valid samples: {severe[hrv_col].notna().sum()} / {len(severe)}")
print(f"Borg valid samples: {severe['borg'].notna().sum()} / {len(severe)}")

# Check if HRV was imputed (constant values)
hrv_vals = severe[hrv_col].dropna()
print(f"\nHRV unique values: {hrv_vals.nunique()}")
print(f"HRV std: {hrv_vals.std():.4f}")
if hrv_vals.std() < 0.01:
    print("⚠️  HRV appears to be IMPUTED (constant) - correlations invalid!")
