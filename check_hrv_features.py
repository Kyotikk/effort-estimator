#!/usr/bin/env python3
"""Check where HRV features went and their correlations."""

import pandas as pd
from pathlib import Path
from scipy import stats

print("=" * 70)
print("INVESTIGATING MISSING HRV FEATURES")
print("=" * 70)

# Check if HRV features exist in source
hrv_file = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/ppg_green/ppg_green_hrv_aligned_10.0s.csv")

if hrv_file.exists():
    hrv_df = pd.read_csv(hrv_file)
    labeled = hrv_df.dropna(subset=['borg'])
    
    print(f"\n✓ HRV aligned file exists with {len(labeled)} labeled samples")
    
    # Check HRV correlations
    hrv_cols = ['ppg_green_rmssd', 'ppg_green_mean_ibi', 'ppg_green_hr_mean', 
                'ppg_green_sdnn', 'ppg_green_pnn50', 'ppg_green_n_peaks']
    
    print("\n📊 HRV Feature Correlations (elderly3 only):")
    print("-" * 50)
    for col in hrv_cols:
        if col in labeled.columns:
            valid = labeled[[col, 'borg']].dropna()
            if len(valid) > 20:
                r, p = stats.pearsonr(valid[col], valid['borg'])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {col:<30} r = {r:+.3f} {sig}")
else:
    print("✗ HRV file not found!")

# Check combined data
print("\n" + "=" * 70)
print("CHECKING COMBINED DATA")
print("=" * 70)

combined = pd.read_csv("/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv")
cols = combined.columns.tolist()

hrv_in_combined = [c for c in cols if 'rmssd' in c.lower() or 'ibi' in c.lower() or 'hr_mean' in c.lower() or 'sdnn' in c.lower()]
print(f"\nHRV features in combined data: {hrv_in_combined if hrv_in_combined else 'NONE!'}")

# Explanation
print("\n" + "=" * 70)
print("🔴 PROBLEM FOUND!")  
print("=" * 70)
print("""
The HRV features (rmssd, mean_ibi, hr_mean, sdnn, pnn50) are NOT in 
the combined data because:

1. HRV features are in separate files:
   - ppg_green_features_10.0s.csv      ← raw PPG signal stats
   - ppg_green_hrv_features_10.0s.csv  ← HRV metrics ✓

2. The fusion config only includes ppg_green_features:
   
   modalities:
     ppg_green: .../ppg_green_features_{window_length}.csv
   
   It does NOT include:
     ppg_green_hrv: .../ppg_green_hrv_features_{window_length}.csv  ← MISSING!

FIX: Add HRV modalities to config/pipeline.yaml fusion section.
""")

# Explain the other questions
print("\n" + "=" * 70)
print("WHAT IS ppg_green_trim_mean_10?")
print("=" * 70)
print("""
trim_mean_10 = 10% trimmed mean of raw PPG signal amplitude
  - Removes extreme 10% values (top 10% + bottom 10%)
  - Computes mean of remaining 80%
  - Robust central tendency measure

Why it correlates (r = -0.50) with effort:
  - PPG amplitude reflects blood volume in tissue
  - During effort: sympathetic activation → vasoconstriction
  - Vasoconstriction → reduced peripheral blood volume
  - Result: lower PPG amplitude = lower trim_mean
  - Higher Borg → Lower trim_mean (negative correlation)
""")

print("\n" + "=" * 70)
print("WHY acc_*_dyn FEATURES?")
print("=" * 70)
print("""
acc_*_dyn = DYNAMIC component of accelerometer
  - Raw accel = static (gravity ~9.8 m/s²) + dynamic (movement)
  - "dyn" = highpass filtered to isolate movement only

Feature names like acc_z_dyn__sum_of_absolute_changes:
  - These are tsfresh-generated statistical features
  - Capture movement patterns, activity intensity
  - Correlations are weak (r ~ 0.17-0.20)

Why IMU features are weaker than EDA/PPG:
  - Movement ≠ perceived effort
  - You can move fast with low effort (walking downhill)
  - Or move slow with high effort (carrying heavy load)
  - Physiological signals (EDA, HR) reflect true metabolic effort
""")
