#!/usr/bin/env python3
"""Re-run alignment, combine, and train with HRV features."""
import pandas as pd
from pathlib import Path
import yaml

from ml.targets.run_target_alignment import run_alignment
from ml.features.sanitise import sanitise_features

base = Path('/Users/pascalschlegel/data/interim/parsingsim3')
subjects = ['sim_elderly3', 'sim_healthy3', 'sim_severe3']
win_sec = 10.0

print("=" * 70)
print("STEP 1: ALIGNMENT")
print("=" * 70)

for subj in subjects:
    print(f"\n▶ {subj}")
    subj_dir = base / subj / "effort_estimation_output" / f"parsingsim3_{subj}"
    
    # Paths
    fused_path = subj_dir / f"fused_features_{win_sec:.1f}s.csv"
    imu_win_path = subj_dir / "imu_bioz" / f"imu_windows_{win_sec:.1f}s.csv"
    aligned_path = subj_dir / f"fused_aligned_{win_sec:.1f}s.csv"
    
    # ADL path - find any ADL file
    adl_candidates = list((base / subj / "scai_app").glob("ADLs*.csv"))
    if not adl_candidates:
        print(f"  ⚠ No ADL file found!")
        continue
    adl_path = adl_candidates[0]  # Take first match
    print(f"  Using ADL: {adl_path.name}")
    
    if fused_path.exists():
        print(f"  Aligning {fused_path.name}...")
        run_alignment(
            features_path=str(fused_path),
            windows_path=str(imu_win_path),
            adl_path=str(adl_path),
            out_path=str(aligned_path),
        )
        
        # Check HRV columns
        df = pd.read_csv(aligned_path)
        hrv_cols = [c for c in df.columns if 'rmssd' in c.lower() or 'hr_mean' in c.lower()]
        print(f"  ✓ {len(df)} aligned samples, HRV cols: {hrv_cols[:4]}...")
    else:
        print(f"  ⚠ Missing: {fused_path}")

print("\n" + "=" * 70)
print("STEP 2: COMBINE SUBJECTS")
print("=" * 70)

dfs = []
for subj in subjects:
    aligned_path = base / subj / "effort_estimation_output" / f"parsingsim3_{subj}" / f"fused_aligned_{win_sec:.1f}s.csv"
    if aligned_path.exists():
        df = pd.read_csv(aligned_path)
        df['subject'] = subj
        dfs.append(df)
        print(f"  {subj}: {len(df)} samples")

combined = pd.concat(dfs, ignore_index=True)
print(f"\n✓ Combined: {len(combined)} total samples")

# HRV check
hrv_cols = [c for c in combined.columns if 'rmssd' in c.lower() or 'hr_mean' in c.lower() or 'mean_ibi' in c.lower() or 'sdnn' in c.lower()]
print(f"✓ HRV features: {hrv_cols}")

# Save
out_dir = base / 'multisub_combined'
out_dir.mkdir(exist_ok=True)
combined.to_csv(out_dir / f'multisub_aligned_{win_sec:.1f}s.csv', index=False)
print(f"\n✓ Saved to {out_dir / f'multisub_aligned_{win_sec:.1f}s.csv'}")

print("\n" + "=" * 70)
print("STEP 3: CORRELATION ANALYSIS")
print("=" * 70)

# Filter to labeled samples
df_labeled = combined.dropna(subset=['borg_effort'])
print(f"Labeled samples: {len(df_labeled)}")

# Get HRV correlations with Borg
target = 'borg_effort'
correlations = []
for col in hrv_cols:
    if col in df_labeled.columns:
        r = df_labeled[col].corr(df_labeled[target])
        correlations.append((col, r))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)
print(f"\nHRV correlations with {target}:")
for col, r in correlations:
    print(f"  {col}: r = {r:+.3f}")

print("\n" + "=" * 70)
print("STEP 4: FEATURE COUNTS")
print("=" * 70)

# Count feature types
all_cols = [c for c in combined.columns if c not in ['t_center', 'window_id', 'borg_effort', 'subject', 'activity']]
imu_cols = [c for c in all_cols if c.startswith('acc_') or c.startswith('gyr_')]
ppg_cols = [c for c in all_cols if c.startswith('ppg_')]
eda_cols = [c for c in all_cols if c.startswith('eda_')]
other_cols = [c for c in all_cols if c not in imu_cols + ppg_cols + eda_cols]

print(f"Total features: {len(all_cols)}")
print(f"  IMU features: {len(imu_cols)}")
print(f"  PPG features: {len(ppg_cols)}")
print(f"  EDA features: {len(eda_cols)}")
print(f"  Other: {len(other_cols)}")

print("\nDone!")
