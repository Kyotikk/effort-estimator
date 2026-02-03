#!/usr/bin/env python3
"""
Fix TLI: Use INTENSITY (not accumulated load × duration)

The original TLI multiplied intensity by duration, which makes Resting
look like more work than short bursts. Borg measures perceived INTENSITY,
not total accumulated work.
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load results
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/tli_results.csv')

print("=== PROBLEM: Duration-weighted load ===")
print("Resting (63s) has higher TLI than Stand to Sit (3s) because load = intensity × duration")
print()

# Show the problem
for act in ['Resting', 'Stand to Sit', 'Stand', 'Level Walking']:
    rows = df[df['activity']==act]
    if len(rows) > 0:
        row = rows.iloc[0]
        print(f"  {act:20s}: duration={row.duration_s:6.1f}s, TLI={row.TLI:+.2f}, borg={row.borg}")

print("\n" + "="*60)
print("=== FIX: Use intensity metrics directly ===")
print()

# INTENSITY metrics (no duration weighting)
# HR intensity = heart rate elevation above baseline
hr_intensity = df['hr_delta']

# IMU intensity = RMS acceleration (movement vigor)  
imu_intensity = df['rms_acc_mag']

# Z-score normalize
z_hr = (hr_intensity - hr_intensity.mean()) / hr_intensity.std()
z_imu = (imu_intensity - imu_intensity.mean()) / imu_intensity.std()

# Combined Effort Intensity Index (EII)
df['EII'] = 0.5 * z_hr + 0.5 * z_imu

# Correlations with Borg
print("Correlations with Borg CR10:")
r_hr, p_hr = stats.pearsonr(hr_intensity.fillna(0), df['borg'])
r_imu, p_imu = stats.pearsonr(imu_intensity.fillna(0), df['borg'])
r_eii, p_eii = stats.pearsonr(df['EII'].fillna(0), df['borg'])
r_old, p_old = stats.pearsonr(df['TLI'].fillna(0), df['borg'])

print(f"  HR delta (intensity):  r={r_hr:+.3f} (p={p_hr:.4f})")
print(f"  IMU RMS (intensity):   r={r_imu:+.3f} (p={p_imu:.4f})")
print(f"  EII (combined):        r={r_eii:+.3f} (p={p_eii:.4f})")
print(f"  Old TLI (load-based):  r={r_old:+.3f} (p={p_old:.4f})")

print("\n" + "="*60)
print("=== NOW MAKES SENSE ===")
print()
print("Activity                    HR_delta  IMU_RMS    EII   Borg")
print("-"*60)

# Sort by EII
df_sorted = df.sort_values('EII', ascending=False)
for _, row in df_sorted.head(10).iterrows():
    print(f"{row.activity:25s} {row.hr_delta:8.1f} {row.rms_acc_mag:8.3f} {row.EII:+6.2f}  {row.borg:4.1f}")

print("...")
for _, row in df_sorted.tail(5).iterrows():
    print(f"{row.activity:25s} {row.hr_delta:8.1f} {row.rms_acc_mag:8.3f} {row.EII:+6.2f}  {row.borg:4.1f}")

# Save corrected results
df_sorted.to_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/tli_intensity_fixed.csv', index=False)
print("\nSaved intensity-based results to tli_intensity_fixed.csv")
