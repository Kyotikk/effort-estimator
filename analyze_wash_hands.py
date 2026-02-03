#!/usr/bin/env python3
"""Analyze why Wash Hands is an outlier"""

import pandas as pd
import numpy as np

df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/tli_results.csv')

# Calculate effort
hr_baseline = df['hr_mean'].min()
hr_effort = (df['hr_mean'] - hr_baseline) * np.sqrt(df['duration_s'])
imu_baseline = df['rms_acc_mag'].min()
imu_effort = (df['rms_acc_mag'] - imu_baseline) * np.sqrt(df['duration_s'])

z_hr = (hr_effort - hr_effort.mean()) / hr_effort.std()
z_imu = (imu_effort - imu_effort.mean()) / imu_effort.std()
combined = 0.8 * z_hr + 0.2 * z_imu

df['combined'] = combined
df['hr_effort'] = hr_effort

print("=" * 70)
print("WASH HANDS ANALYSIS - WHY IS IT OFF?")
print("=" * 70)
print()

wh = df[df['activity']=='Wash Hands'].iloc[0]
print(f"Activity: Wash Hands")
print(f"  Duration:    {wh.duration_s:.1f}s (fairly long)")
print(f"  HR mean:     {wh.hr_mean:.1f} BPM")
print(f"  HR delta:    {wh.hr_mean - hr_baseline:.1f} BPM above baseline (ELEVATED!)")
print(f"  Combined:    {wh.combined:+.2f} (ranks HIGH)")
print(f"  Borg:        {wh.borg} (rated EASY)")
print()

print("=" * 70)
print("THE PROBLEM")
print("=" * 70)
print()
print("  Model predicts: HIGH effort (elevated HR + long duration)")
print("  Actual Borg:    LOW (1.0 - very easy)")
print()
print("  MISMATCH: HR says hard, but participant says easy!")
print()

print("=" * 70)
print("COMPARE TO SIMILAR ACTIVITIES")
print("=" * 70)
print()
print(f"{'Activity':<25} {'Dur':>6} {'HR':>6} {'Effort':>8} {'Borg':>6} {'Note'}")
print("-" * 70)

for _, row in df.sort_values('combined', ascending=False).head(12).iterrows():
    note = ""
    if row.activity == 'Wash Hands':
        note = "<<< OUTLIER: High HR, Low Borg"
    elif row.borg <= 2 and row.combined > 0:
        note = "similar issue?"
    print(f"{row.activity:<25} {row.duration_s:>6.0f} {row.hr_mean:>6.1f} {row.combined:>+8.2f} {row.borg:>6.1f}  {note}")

print()
print("=" * 70)
print("LIKELY EXPLANATION: HR RECOVERY LAG")
print("=" * 70)
print()
print("Timeline:")
print("  1. Participant does hard activity (walking, standing)")
print("  2. HR goes up to ~90+ BPM")
print("  3. Participant starts washing hands (easy task)")
print("  4. HR is STILL elevated (takes 1-2 min to recover)")
print("  5. We measure HR during wash = elevated")
print("  6. But participant feels it's easy (Borg = 1)")
print()
print("The HR reflects PREVIOUS effort, not current task!")
print()

# Check what came before Wash Hands
print("=" * 70)
print("WHAT CAME BEFORE WASH HANDS?")
print("=" * 70)
print()

# Find index of Wash Hands
wh_idx = df[df['activity']=='Wash Hands'].index[0]
print(f"Activity sequence around Wash Hands (index {wh_idx}):")
print()

for i in range(max(0, wh_idx-3), min(len(df), wh_idx+2)):
    row = df.iloc[i]
    marker = ">>> " if row.activity == 'Wash Hands' else "    "
    print(f"{marker}{row.activity:<25} HR={row.hr_mean:.1f}, Borg={row.borg}")

print()
print("=" * 70)
print("SOLUTION OPTIONS")
print("=" * 70)
print()
print("1. Use HR CHANGE from start to end of activity (not mean)")
print("2. Use HR relative to previous activity")  
print("3. Account for HR recovery time constant")
print("4. Accept that some activities have HR lag (model limitation)")
print()
print("This is a known problem in HR-based effort estimation!")
