#!/usr/bin/env python3
"""Check why HRV is missing for healthy and severe subjects"""

import pandas as pd
from pathlib import Path

# Check HRV files for each subject
for sub in ['sim_elderly3', 'sim_healthy3', 'sim_severe3']:
    print(f'\n{"="*60}')
    print(f'{sub}')
    print('='*60)
    
    hrv_path = f'/Users/pascalschlegel/data/interim/parsingsim3/{sub}/effort_estimation_output/parsingsim3_{sub}/ppg_green/ppg_green_hrv_features_10.0s.csv'
    
    if Path(hrv_path).exists():
        df = pd.read_csv(hrv_path)
        print(f'HRV file exists: {len(df)} rows')
        
        # Check for actual HRV values
        hrv_cols = ['ppg_green_mean_ibi', 'ppg_green_rmssd', 'ppg_green_hr_mean', 'ppg_green_n_peaks']
        for col in hrv_cols:
            if col in df.columns:
                valid = df[col].notna().sum()
                total = len(df)
                pct = 100 * valid / total
                print(f'  {col}: {valid}/{total} valid ({pct:.1f}%)')
                if valid > 0:
                    print(f'    Range: {df[col].min():.1f} - {df[col].max():.1f}')
    else:
        print(f'HRV file NOT FOUND')
        
    # Check the combined file
    combined = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv')
    sub_combined = combined[combined['subject_id'] == sub]
    
    print(f'\nIn combined dataset:')
    for col in ['ppg_green_mean_ibi', 'ppg_green_rmssd', 'ppg_green_hr_mean']:
        if col in sub_combined.columns:
            valid = sub_combined[col].notna().sum()
            total = len(sub_combined)
            print(f'  {col}: {valid}/{total} valid ({100*valid/total:.1f}%)')
