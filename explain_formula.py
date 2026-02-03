#!/usr/bin/env python3
"""Explain the winning effort formula"""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/tli_results.csv')

print("=" * 70)
print("THE WINNING FORMULA: r = 0.835")
print("=" * 70)
print()
print("  Effort = HR_delta × √(duration)")
print()
print("WHERE:")
hr_baseline = df['hr_mean'].min()
print(f"  • HR_baseline = min(HR_mean across all activities) = {hr_baseline:.1f} BPM")
print(f"  • HR_delta = HR_mean - HR_baseline  (how much HR elevated)")
print(f"  • duration = activity duration in seconds")
print(f"  • √duration = square root of duration (sublinear scaling)")
print()

print("=" * 70)
print("CALCULATION EXAMPLES")
print("=" * 70)
print()

# Calculate
df['hr_delta_calc'] = df['hr_mean'] - hr_baseline
df['sqrt_dur'] = np.sqrt(df['duration_s'])
df['effort'] = df['hr_delta_calc'] * df['sqrt_dur']

print(f"{'Activity':<20} {'HR_mean':>7} {'HR_Δ':>6} {'Dur(s)':>7} {'√dur':>6} {'EFFORT':>8} {'Borg':>5}")
print("-" * 65)

examples = ['Resting', 'Level Walking', 'Stand', 'Stand to Sit', 'Eating/Drinking', 'Brush Teeth']
for act in examples:
    rows = df[df['activity'] == act]
    if len(rows) > 0:
        row = rows.iloc[0]
        print(f"{act:<20} {row.hr_mean:>7.1f} {row.hr_delta_calc:>6.1f} {row.duration_s:>7.1f} {row.sqrt_dur:>6.2f} {row.effort:>8.1f} {row.borg:>5.1f}")

print()
print("=" * 70)
print("WHAT'S ALREADY IN THE DATA vs. THE NEW FORMULA")
print("=" * 70)
print()
print("EXISTING hr_load (from tli_results.csv):")
print("  hr_load = HR_delta × duration     (LINEAR duration)")
print()
print("NEW winning formula:")
print("  effort  = HR_delta × √duration    (SQRT duration)")
print()
print("THE KEY DIFFERENCE:")
print()

# Compare
row_rest = df[df['activity']=='Resting'].iloc[0]
row_walk = df[df['activity']=='Level Walking'].iloc[0]
row_sit = df[df['activity']=='Stand to Sit'].iloc[0]

print(f"  Activity           Duration   HR_delta   hr_load(×d)   effort(×√d)   Borg")
print(f"  {'-'*70}")
print(f"  Resting            {row_rest.duration_s:>6.0f}s    {row_rest.hr_delta_calc:>6.1f}     {row_rest.hr_load:>8.1f}      {row_rest.effort:>8.1f}     {row_rest.borg}")
print(f"  Level Walking      {row_walk.duration_s:>6.0f}s    {row_walk.hr_delta_calc:>6.1f}    {row_walk.hr_load:>8.1f}      {row_walk.effort:>8.1f}     {row_walk.borg}")
print(f"  Stand to Sit       {row_sit.duration_s:>6.0f}s    {row_sit.hr_delta_calc:>6.1f}      {row_sit.hr_load:>8.1f}       {row_sit.effort:>8.1f}     {row_sit.borg}")

print()
print("INTERPRETATION:")
print("  • With LINEAR duration: Resting (63s) accumulates 182 load units")
print("  • With SQRT duration:   Resting (63s) only gets 23 effort units")
print("  • Why? Because: 2.9 BPM × √63 = 2.9 × 7.9 = 23")
print("  •               vs.     2.9 BPM × 63  = 2.9 × 63 = 182")
print()
print("The SQRT makes duration matter LESS for low-intensity activities!")
print()

# Verify correlations
r_old, _ = stats.pearsonr(df['hr_load'], df['borg'])
r_new, _ = stats.pearsonr(df['effort'], df['borg'])

print("=" * 70)
print("CORRELATION COMPARISON")
print("=" * 70)
print()
print(f"  hr_load (HR_delta × duration):    r = {r_old:.3f}")
print(f"  effort  (HR_delta × √duration):   r = {r_new:.3f}  ← WINNER")
print()
print("=" * 70)
print("SUMMARY: WHAT THE FORMULA DOES")
print("=" * 70)
print("""
1. Takes HR_delta (heart rate elevation above resting baseline)
   → This measures INTENSITY (how hard is the cardiovascular system working)

2. Multiplies by √(duration) instead of duration
   → This makes duration effect SUBLINEAR
   → 4× longer activity = only 2× more effort (not 4×)
   → Captures: "standing for 3 min is harder than 30 sec, but not 6× harder"

3. Result: LOW intensity activities stay LOW even with long duration
   → Resting for 1 hour ≠ lots of effort
   → Walking for 1 min ≈ more effort than resting for 10 min

This matches Stevens' Power Law from psychophysics research!
""")
