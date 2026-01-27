#!/usr/bin/env python3
"""
Verify time alignment between ADL activities and sensor signals.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os

base = '/Users/pascalschlegel/data/interim'
adl_offset = -8.3 * 3600  # The offset we're using

datasets = [
    ('parsingsim3', 'sim_elderly3'),
    ('parsingsim3', 'sim_healthy3'),
    ('parsingsim3', 'sim_severe3'),
    ('parsingsim4', 'sim_elderly4'),
    ('parsingsim4', 'sim_healthy4'),
    ('parsingsim4', 'sim_severe4'),
    ('parsingsim5', 'sim_elderly5'),
    ('parsingsim5', 'sim_healthy5'),
    ('parsingsim5', 'sim_severe5'),
]

print("TIME ALIGNMENT VERIFICATION")
print("=" * 80)
print(f"Using ADL offset: {adl_offset/3600:.1f} hours")
print("=" * 80)

for sim, cond in datasets:
    print(f"\n{sim}/{cond}")
    print("-" * 40)
    
    path = f'{base}/{sim}/{cond}'
    
    # Find PPG file
    ppg_files = glob.glob(f'{path}/corsano_wrist_ppg2_green_6/2025-*.csv.gz')
    if not ppg_files:
        ppg_files = glob.glob(f'{path}/corsano_wrist_ppg2_green_6/2025-*.csv')
    
    # Find EDA file
    eda_files = glob.glob(f'{path}/corsano_bioz_emography/2025-*.csv.gz')
    if not eda_files:
        eda_files = glob.glob(f'{path}/corsano_bioz_emography/2025-*.csv')
    
    # Find ADL file
    adl_files = glob.glob(f'{path}/scai_app/ADLs*.csv.gz')
    if not adl_files:
        adl_files = glob.glob(f'{path}/scai_app/ADLs*.csv')
    
    if not ppg_files or not adl_files:
        print("  Missing files!")
        continue
    
    # Load full PPG to get actual range
    ppg = pd.read_csv(ppg_files[0])
    ppg_start, ppg_end = ppg['time'].min(), ppg['time'].max()
    ppg_duration = (ppg_end - ppg_start) / 60
    
    # Load EDA
    if eda_files:
        eda = pd.read_csv(eda_files[0])
        eda_start, eda_end = eda['time'].min(), eda['time'].max()
    else:
        eda_start, eda_end = None, None
    
    # Load ADL
    adl = pd.read_csv(adl_files[0])
    adl_start_raw, adl_end_raw = adl['time'].min(), adl['time'].max()
    adl_start = adl_start_raw + adl_offset
    adl_end = adl_end_raw + adl_offset
    adl_duration = (adl_end - adl_start) / 60
    
    # Print times
    print(f"  PPG:           {datetime.fromtimestamp(ppg_start).strftime('%Y-%m-%d %H:%M:%S')} - "
          f"{datetime.fromtimestamp(ppg_end).strftime('%H:%M:%S')} ({ppg_duration:.1f} min)")
    
    if eda_start:
        eda_dur = (eda_end - eda_start) / 60
        print(f"  EDA:           {datetime.fromtimestamp(eda_start).strftime('%Y-%m-%d %H:%M:%S')} - "
              f"{datetime.fromtimestamp(eda_end).strftime('%H:%M:%S')} ({eda_dur:.1f} min)")
    
    print(f"  ADL (raw):     {datetime.fromtimestamp(adl_start_raw).strftime('%Y-%m-%d %H:%M:%S')} - "
          f"{datetime.fromtimestamp(adl_end_raw).strftime('%H:%M:%S')}")
    print(f"  ADL (shifted): {datetime.fromtimestamp(adl_start).strftime('%Y-%m-%d %H:%M:%S')} - "
          f"{datetime.fromtimestamp(adl_end).strftime('%H:%M:%S')} ({adl_duration:.1f} min)")
    
    # Check overlap with PPG
    overlap_start = max(ppg_start, adl_start)
    overlap_end = min(ppg_end, adl_end)
    
    if overlap_start < overlap_end:
        overlap_min = (overlap_end - overlap_start) / 60
        coverage = overlap_min / adl_duration * 100
        print(f"  ✓ OVERLAP: {overlap_min:.1f} min ({coverage:.0f}% of ADL covered)")
    else:
        gap_min = (overlap_start - overlap_end) / 60
        print(f"  ✗ NO OVERLAP - gap of {gap_min:.1f} minutes")
        
        # Show where times actually are
        if adl_end < ppg_start:
            print(f"    ADL ends {(ppg_start - adl_end)/60:.1f} min BEFORE PPG starts")
        else:
            print(f"    ADL starts {(adl_start - ppg_end)/60:.1f} min AFTER PPG ends")

# Now check if the actual extracted features make sense
print("\n" + "=" * 80)
print("CHECKING EXTRACTED DATA QUALITY")
print("=" * 80)

df = pd.read_csv('output/adl_effort_features.csv')

for patient in df['patient'].unique():
    pdata = df[df['patient'] == patient]
    n_total = len(pdata)
    n_hrv = pdata['hrv_mean_hr'].notna().sum()
    n_eda = pdata['eda_tonic_mean'].notna().sum()
    
    print(f"\n{patient}:")
    print(f"  Activities: {n_total}")
    print(f"  With HRV: {n_hrv}/{n_total} ({100*n_hrv/n_total:.0f}%)")
    print(f"  With EDA: {n_eda}/{n_total} ({100*n_eda/n_total:.0f}%)")
    
    # Check if HR values are physiological
    hr_vals = pdata['hrv_mean_hr'].dropna()
    if len(hr_vals) > 0:
        print(f"  HR range: {hr_vals.min():.0f} - {hr_vals.max():.0f} bpm (mean: {hr_vals.mean():.0f})")
        if hr_vals.min() < 40 or hr_vals.max() > 180:
            print(f"    ⚠️ WARNING: HR values outside normal range!")
