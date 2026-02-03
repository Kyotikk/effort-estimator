#!/usr/bin/env python3
"""Show what the pipeline actually does vs what I manually did"""
import pandas as pd
import os

print("="*70)
print("HONEST ANSWER: What actually happened")
print("="*70)

# What the pipeline extracts
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_aligned_10.0s.csv')

meta = ['t_center', 'borg', 'activity', 'activity_id', 'subject_id', 'valid', 'n_samples', 'win_sec', 'modality']
features = [c for c in df.columns if c not in meta and not c.startswith('Unnamed')]

print("\n1. PIPELINE EXTRACTS ALL THESE FEATURES:")
print(f"   Total: {len(features)}")
print(f"   - PPG: {len([f for f in features if f.startswith('ppg_')])}")
print(f"   - ACC: {len([f for f in features if f.startswith('acc_')])}")
print(f"   - EDA: {len([f for f in features if f.startswith('eda_')])}")

print("\n2. PIPELINE'S FEATURE SELECTION (from ml/feature_selection.py):")
print("   Uses Mutual Information (MI) to select top features")
print("   BUT: MI picks EDA features that correlate with TIME, not Borg!")

print("\n3. WHAT I DID MANUALLY IN THIS CHAT:")
print("   - Looked at correlation of each feature with Borg")
print("   - Picked 2-3 best from each modality")
print("   - This is NOT what the pipeline does!")

print("\n4. THE 7 FEATURES I MANUALLY SELECTED:")
manual = ['ppg_green_hr_mean', 'ppg_green_hr_max', 
          'acc_x_dyn__quantile_0.6', 'acc_z_dyn__sum_of_absolute_changes', 'acc_y_dyn__sample_entropy',
          'eda_phasic_max', 'eda_cc_range']
for f in manual:
    print(f"   - {f}")

print("\n" + "="*70)
print("BOTTOM LINE:")
print("="*70)
print("""
- Pipeline extracts 280+ features (correct)
- Pipeline uses MI for selection (problematic - picks time-correlated junk)
- I manually picked 7 features by correlation with Borg (ad-hoc, not in pipeline)
- To do this properly: need to fix feature selection in pipeline to use
  correlation-based selection instead of MI
""")
