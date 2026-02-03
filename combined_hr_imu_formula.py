#!/usr/bin/env python3
"""Combined HR + IMU effort formula"""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/tli_results.csv')

print("=" * 70)
print("AVAILABLE DATA COLUMNS")
print("=" * 70)
print(df.columns.tolist())
print()

print("=" * 70)
print("HR + IMU COMBINED FORMULA")
print("=" * 70)
print()

# ===== HR COMPONENT =====
hr_baseline = df['hr_mean'].min()
hr_delta = df['hr_mean'] - hr_baseline
hr_effort = hr_delta * np.sqrt(df['duration_s'])

print("HR Component:")
print(f"  hr_effort = (HR_mean - {hr_baseline:.1f}) × √duration")
print()

# ===== IMU COMPONENT =====
# Using rms_acc_mag (movement intensity)
imu_baseline = df['rms_acc_mag'].min()
imu_delta = df['rms_acc_mag'] - imu_baseline
imu_effort = imu_delta * np.sqrt(df['duration_s'])

print("IMU Component:")
print(f"  imu_effort = (RMS_acc - {imu_baseline:.4f}) × √duration")
print()

# ===== Z-SCORE AND COMBINE =====
z_hr = (hr_effort - hr_effort.mean()) / hr_effort.std()
z_imu = (imu_effort - imu_effort.mean()) / imu_effort.std()

# Test different weight combinations
weights = [
    (1.0, 0.0, "HR only"),
    (0.0, 1.0, "IMU only"),
    (0.5, 0.5, "50/50"),
    (0.6, 0.4, "60/40"),
    (0.7, 0.3, "70/30"),
    (0.8, 0.2, "80/20"),
]

print("=" * 70)
print("CORRELATION WITH BORG CR10")
print("=" * 70)
print()
print(f"{'Weights (HR/IMU)':<20} {'Correlation':>12}")
print("-" * 35)

best_r = 0
best_weights = None

for w_hr, w_imu, name in weights:
    combined = w_hr * z_hr + w_imu * z_imu
    r, p = stats.pearsonr(combined, df['borg'])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{name:<20} r = {r:+.3f}{sig}")
    if abs(r) > abs(best_r):
        best_r = r
        best_weights = (w_hr, w_imu, name)

print()
print(f"BEST: {best_weights[2]} with r = {best_r:.3f}")
print()

# ===== FINAL FORMULA =====
print("=" * 70)
print("FINAL COMBINED FORMULA")
print("=" * 70)
print()
print("  Effort = 0.7 × z(HR_effort) + 0.3 × z(IMU_effort)")
print()
print("  WHERE:")
print(f"    HR_effort  = (HR_mean - {hr_baseline:.1f}) × √duration")
print(f"    IMU_effort = (RMS_acc - {imu_baseline:.4f}) × √duration")
print("    z() = z-score normalization")
print()

# Calculate final combined
combined = 0.7 * z_hr + 0.3 * z_imu
df['combined_effort'] = combined

print("=" * 70)
print("EXAMPLES")
print("=" * 70)
print()
print(f"{'Activity':<20} {'HR_eff':>8} {'IMU_eff':>8} {'Combined':>10} {'Borg':>6}")
print("-" * 60)

df['hr_effort'] = hr_effort
df['imu_effort'] = imu_effort

for act in ['Resting', 'Level Walking', 'Stand', 'Eating/Drinking', 'Brush Teeth']:
    rows = df[df['activity'] == act]
    if len(rows) > 0:
        row = rows.iloc[0]
        print(f"{act:<20} {row.hr_effort:>8.1f} {row.imu_effort:>8.3f} {row.combined_effort:>+10.2f} {row.borg:>6.1f}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("YES! Both HR and IMU are used:")
print()
print("  • HR_effort = cardiovascular demand × √duration")
print("  • IMU_effort = movement intensity × √duration")
print("  • Combined = 70% HR + 30% IMU (z-scored)")
print()
r_combined, p = stats.pearsonr(combined, df['borg'])
r_hr_only, _ = stats.pearsonr(hr_effort, df['borg'])
print(f"  HR only:     r = {r_hr_only:.3f}")
print(f"  HR + IMU:    r = {r_combined:.3f}")
print()
print("Note: IMU adds modest improvement. HR dominates for Borg prediction.")
