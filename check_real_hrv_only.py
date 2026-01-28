#!/usr/bin/env python3
"""Check HRV correlation using only REAL (non-imputed) data."""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv')

severe = df[df['subject_id'] == 'sim_severe3'].copy()
elderly = df[df['subject_id'] == 'sim_elderly3'].copy()

print("=" * 70)
print("HRV CORRELATION WITH ONLY REAL DATA (no imputation)")
print("=" * 70)

# Find the median value (imputed values are at median)
hrv_col = 'ppg_green_mean_ibi'

# Severe patient
s_median = severe[hrv_col].median()
print(f"\nSevere patient median IBI: {s_median:.1f}")

# Filter OUT rows where HRV equals median (those are imputed)
severe_real = severe[severe[hrv_col] != s_median].copy()
severe_all = severe.copy()

print(f"\nSEVERE PATIENT:")
print(f"  Total windows: {len(severe_all)}")
print(f"  Windows with REAL HRV: {len(severe_real)} ({len(severe_real)/len(severe_all)*100:.1f}%)")

# Check correlation with real data only
valid_all = severe_all[[hrv_col, 'borg']].dropna()
valid_real = severe_real[[hrv_col, 'borg']].dropna()

if len(valid_all) > 30:
    r_all, p_all = stats.pearsonr(valid_all[hrv_col], valid_all['borg'])
    print(f"\n  With ALL data (including imputed):")
    print(f"    n = {len(valid_all)}, r = {r_all:.3f}, p = {p_all:.4f}")

if len(valid_real) > 30:
    r_real, p_real = stats.pearsonr(valid_real[hrv_col], valid_real['borg'])
    print(f"\n  With REAL data only (no imputed):")
    print(f"    n = {len(valid_real)}, r = {r_real:.3f}, p = {p_real:.4f}")
    
    if p_real < 0.05:
        print(f"    ✅ SIGNIFICANT! HRV works for severe too!")
    else:
        print(f"    ❌ Still not significant")
else:
    print(f"\n  Not enough real data (n={len(valid_real)})")

# Also check elderly for comparison
print(f"\nELDERLY PATIENT (reference):")
e_median = elderly[hrv_col].median()
elderly_real = elderly[elderly[hrv_col] != e_median].copy()

valid_e_all = elderly[[hrv_col, 'borg']].dropna()
valid_e_real = elderly_real[[hrv_col, 'borg']].dropna()

print(f"  Windows with REAL HRV: {len(elderly_real)} ({len(elderly_real)/len(elderly)*100:.1f}%)")

if len(valid_e_real) > 30:
    r_e, p_e = stats.pearsonr(valid_e_real[hrv_col], valid_e_real['borg'])
    print(f"  With REAL data: n = {len(valid_e_real)}, r = {r_e:.3f}, p = {p_e:.4f}")

# Check other HRV features too
print(f"\n" + "=" * 70)
print("ALL HRV FEATURES - SEVERE PATIENT (REAL DATA ONLY)")
print("=" * 70)

hrv_features = ['ppg_green_mean_ibi', 'ppg_green_rmssd', 'ppg_green_sdnn', 
                'ppg_green_hr_mean', 'ppg_green_hr_std', 'ppg_green_n_peaks']

print(f"\n{'Feature':<30} {'n_real':>8} {'r':>8} {'p':>10} {'Sig?':>6}")
print("-" * 70)

for f in hrv_features:
    if f not in severe.columns:
        continue
    
    f_median = severe[f].median()
    severe_real_f = severe[severe[f] != f_median]
    valid = severe_real_f[[f, 'borg']].dropna()
    
    if len(valid) > 30:
        r, p = stats.pearsonr(valid[f], valid['borg'])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{f:<30} {len(valid):>8} {r:>+8.3f} {p:>10.4f} {sig:>6}")
    else:
        print(f"{f:<30} {len(valid):>8}    (too few samples)")

print(f"\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
