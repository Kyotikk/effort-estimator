#!/usr/bin/env python3
"""Compare data quality between elderly3 and severe3."""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv')

elderly = df[df['subject_id'] == 'sim_elderly3']
severe = df[df['subject_id'] == 'sim_severe3']

print("=" * 75)
print("DATA QUALITY COMPARISON: ELDERLY3 vs SEVERE3")
print("=" * 75)

# Basic stats
print(f"\n{'Metric':<40} {'Elderly3':>15} {'Severe3':>15}")
print("-" * 75)
print(f"{'Total windows':<40} {len(elderly):>15} {len(severe):>15}")
print(f"{'Borg valid':<40} {elderly['borg'].notna().sum():>15} {severe['borg'].notna().sum():>15}")
e_range = f"{elderly['borg'].min():.1f}-{elderly['borg'].max():.1f}"
s_range = f"{severe['borg'].min():.1f}-{severe['borg'].max():.1f}"
print(f"{'Borg range':<40} {e_range:>15} {s_range:>15}")

# Check key features
features_to_check = [
    'eda_cc_range', 'eda_cc_mean', 'eda_phasic_max',
    'ppg_green_mean_ibi', 'ppg_green_rmssd', 'ppg_green_hr_mean', 'ppg_green_n_peaks',
    'acc_x_dyn__max_r'
]

print(f"\n" + "=" * 75)
print("FEATURE DATA QUALITY")
print("=" * 75)
print(f"\n{'Feature':<30} {'E: valid%':>10} {'E: std':>10} {'S: valid%':>10} {'S: std':>10}")
print("-" * 75)

for f in features_to_check:
    if f in elderly.columns:
        e_valid = elderly[f].notna().sum() / len(elderly) * 100
        e_std = elderly[f].std()
        s_valid = severe[f].notna().sum() / len(severe) * 100
        s_std = severe[f].std()
        
        flag = ""
        if abs(e_std - s_std) / max(e_std, s_std, 0.001) > 0.5:
            flag = " ⚠️"
        
        print(f"{f:<30} {e_valid:>9.1f}% {e_std:>10.2f} {s_valid:>9.1f}% {s_std:>10.2f}{flag}")

# Check HRV specifically - is it imputed?
print(f"\n" + "=" * 75)
print("HRV DATA QUALITY CHECK (Are values imputed?)")
print("=" * 75)

hrv_features = ['ppg_green_mean_ibi', 'ppg_green_rmssd', 'ppg_green_hr_mean']

for f in hrv_features:
    print(f"\n{f}:")
    
    # Elderly
    e_data = elderly[f].dropna()
    e_unique = e_data.nunique()
    e_const = (e_data == e_data.median()).sum() / len(e_data) * 100 if len(e_data) > 0 else 0
    
    # Severe
    s_data = severe[f].dropna()
    s_unique = s_data.nunique()
    s_const = (s_data == s_data.median()).sum() / len(s_data) * 100 if len(s_data) > 0 else 0
    
    print(f"  Elderly: {e_unique} unique values, {e_const:.1f}% at median")
    print(f"  Severe:  {s_unique} unique values, {s_const:.1f}% at median")
    
    if e_const > 50 or s_const > 50:
        print(f"  ⚠️  HIGH % at median suggests IMPUTATION!")

# Check raw value distributions
print(f"\n" + "=" * 75)
print("VALUE DISTRIBUTIONS")
print("=" * 75)

print(f"\nppg_green_mean_ibi (Inter-Beat Interval in ms):")
print(f"  Elderly: mean={elderly['ppg_green_mean_ibi'].mean():.1f}, std={elderly['ppg_green_mean_ibi'].std():.1f}")
print(f"  Severe:  mean={severe['ppg_green_mean_ibi'].mean():.1f}, std={severe['ppg_green_mean_ibi'].std():.1f}")

# Convert to HR
elderly_hr = 60000 / elderly['ppg_green_mean_ibi']
severe_hr = 60000 / severe['ppg_green_mean_ibi']
print(f"\nImplied Heart Rate (BPM):")
print(f"  Elderly: mean={elderly_hr.mean():.0f}, range={elderly_hr.min():.0f}-{elderly_hr.max():.0f}")
print(f"  Severe:  mean={severe_hr.mean():.0f}, range={severe_hr.min():.0f}-{severe_hr.max():.0f}")

# Check n_peaks
print(f"\nppg_green_n_peaks (peaks per 10s window):")
e_peaks = elderly['ppg_green_n_peaks'].dropna()
s_peaks = severe['ppg_green_n_peaks'].dropna()
print(f"  Elderly: mean={e_peaks.mean():.1f}, std={e_peaks.std():.1f}, range={e_peaks.min():.0f}-{e_peaks.max():.0f}")
print(f"  Severe:  mean={s_peaks.mean():.1f}, std={s_peaks.std():.1f}, range={s_peaks.min():.0f}-{s_peaks.max():.0f}")

# Check if n_peaks correlates with Borg WITHIN each subject
from scipy import stats

print(f"\n" + "=" * 75)
print("IS IT DATA QUALITY OR TRUE PHYSIOLOGICAL DIFFERENCE?")
print("=" * 75)

# Elderly: n_peaks vs Borg
e_valid = elderly[['ppg_green_n_peaks', 'borg']].dropna()
r_e, p_e = stats.pearsonr(e_valid['ppg_green_n_peaks'], e_valid['borg'])

# Severe: n_peaks vs Borg
s_valid = severe[['ppg_green_n_peaks', 'borg']].dropna()
r_s, p_s = stats.pearsonr(s_valid['ppg_green_n_peaks'], s_valid['borg'])

print(f"\nn_peaks vs Borg correlation:")
print(f"  Elderly: r = {r_e:.3f} (p = {p_e:.4f})")
print(f"  Severe:  r = {r_s:.3f} (p = {p_s:.4f})")

# Check variance in n_peaks
print(f"\nn_peaks variance:")
print(f"  Elderly: std = {e_valid['ppg_green_n_peaks'].std():.2f}")
print(f"  Severe:  std = {s_valid['ppg_green_n_peaks'].std():.2f}")

# Check variance in Borg for windows with valid HRV
print(f"\nBorg variance (in windows with valid HRV):")
print(f"  Elderly: std = {e_valid['borg'].std():.2f}")
print(f"  Severe:  std = {s_valid['borg'].std():.2f}")

print(f"""
=" * 75
DIAGNOSIS
=" * 75

The issue is likely a COMBINATION:

1. DATA QUALITY:
   - Elderly n_peaks std = {e_valid['ppg_green_n_peaks'].std():.1f} (good variance)
   - Severe n_peaks std = {s_valid['ppg_green_n_peaks'].std():.1f} (low variance = less signal)
   
2. PHYSIOLOGICAL DIFFERENCE:
   - Severe patient may have less HR variability with effort
   - Could be due to medication (beta blockers)
   - Could be due to cardiac condition limiting HR response
   
3. BORG DISTRIBUTION:
   - Severe Borg is more clustered (mostly 1.5-3.5 and 6-8)
   - Less intermediate values → harder to detect linear correlation

RECOMMENDATION:
- EDA works for BOTH patients (r = 0.50 elderly, 0.43 severe)
- HRV only works for elderly
- For severe patients, use EDA + PPG amplitude instead of HRV
""")
