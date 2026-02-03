#!/usr/bin/env python3
"""
Test activity-level formulas with LOSO (train on 4, test on 1).
Compare: Screenshot was ONE subject pooled → r = 0.82
         Now: Activity-level LOSO → ???
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

print("="*70)
print("ACTIVITY-LEVEL LOSO TEST")
print("Screenshot was based on ONE subject (sim_elderly3) only!")
print("="*70)

# Load activity-level data for all subjects
activity_dfs = []
for i in range(1, 6):
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/effort_features_full.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = i
        activity_dfs.append(df)
        print(f"P{i}: {len(df)} activities")
    else:
        # Check alternative path
        alt_path = Path(f'/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly{i}/effort_estimation_output/effort_features_full.csv')
        if alt_path.exists():
            df = pd.read_csv(alt_path)
            df['subject'] = i
            activity_dfs.append(df)
            print(f"P{i}: {len(df)} activities (alt path)")

if not activity_dfs:
    print("\nActivity-level files not found for all subjects.")
    print("Checking what's available...")
    # List available files
    import os
    for i in range(1, 6):
        base = f'/Users/pascalschlegel/data/interim/parsingsim{i}'
        if os.path.exists(base):
            print(f"\n{base}:")
            for root, dirs, files in os.walk(base):
                for f in files:
                    if 'effort' in f.lower() or 'feature' in f.lower():
                        print(f"  {os.path.join(root, f)}")
else:
    df_all = pd.concat(activity_dfs, ignore_index=True)
    print(f"\nTotal activities: {len(df_all)}")
    
    # Check what columns exist
    print(f"\nAvailable features:")
    hr_feats = [c for c in df_all.columns if 'hr' in c.lower()]
    imu_feats = [c for c in df_all.columns if 'imu' in c.lower() or 'mad' in c.lower() or 'acc' in c.lower()]
    print(f"  HR features: {hr_feats[:10]}")
    print(f"  IMU features: {imu_feats[:10]}")
    
    # Test the formula from screenshot
    if 'hr_load_sqrt' in df_all.columns:
        print("\n" + "="*70)
        print("POOLED CORRELATION (like screenshot)")
        print("="*70)
        
        r_hr = pearsonr(df_all['hr_load_sqrt'].fillna(0), df_all['borg'])[0]
        print(f"hr_load_sqrt (pooled): r = {r_hr:.3f}")
        
        if 'mad_load' in df_all.columns:
            z_hr = (df_all['hr_load_sqrt'] - df_all['hr_load_sqrt'].mean()) / (df_all['hr_load_sqrt'].std() + 1e-8)
            z_imu = (df_all['mad_load'] - df_all['mad_load'].mean()) / (df_all['mad_load'].std() + 1e-8)
            combined = 0.8 * z_hr + 0.2 * z_imu
            r_comb = pearsonr(combined.fillna(0), df_all['borg'])[0]
            print(f"Combined 0.8×HR + 0.2×MAD (pooled): r = {r_comb:.3f}")
        
        # Now LOSO
        print("\n" + "="*70)
        print("LOSO CORRELATION (honest evaluation)")
        print("="*70)
        
        results_hr = []
        results_comb = []
        
        for test_subj in df_all['subject'].unique():
            train = df_all[df_all['subject'] != test_subj]
            test = df_all[df_all['subject'] == test_subj].dropna(subset=['borg', 'hr_load_sqrt'])
            
            if len(test) > 2:
                # Just correlation on test set (no ML, just formula)
                r_hr_test = pearsonr(test['hr_load_sqrt'].fillna(0), test['borg'])[0]
                results_hr.append(r_hr_test)
                print(f"  P{test_subj}: hr_load_sqrt r = {r_hr_test:.3f} (n={len(test)})")
                
                if 'mad_load' in test.columns:
                    # Use training stats for z-scoring
                    z_hr_test = (test['hr_load_sqrt'] - train['hr_load_sqrt'].mean()) / (train['hr_load_sqrt'].std() + 1e-8)
                    z_imu_test = (test['mad_load'] - train['mad_load'].mean()) / (train['mad_load'].std() + 1e-8)
                    combined_test = 0.8 * z_hr_test + 0.2 * z_imu_test
                    r_comb_test = pearsonr(combined_test.fillna(0), test['borg'])[0]
                    results_comb.append(r_comb_test)
        
        print(f"\n  Mean LOSO r (hr_load_sqrt): {np.mean(results_hr):.3f}")
        if results_comb:
            print(f"  Mean LOSO r (combined): {np.mean(results_comb):.3f}")
        
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        print(f"""
| Evaluation     | HR formula | Combined |
|----------------|------------|----------|
| Pooled (5 subj)| {r_hr:.2f}       | {r_comb:.2f}     |
| LOSO           | {np.mean(results_hr):.2f}       | {np.mean(results_comb) if results_comb else 'N/A':.2f}     |

Screenshot (1 subject, sim_elderly3): r = 0.82
This is WITHIN one subject, not cross-subject!
""")
