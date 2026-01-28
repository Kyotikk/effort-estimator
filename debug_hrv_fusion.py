#!/usr/bin/env python3
"""
Debug HRV fusion - why are HRV features being dropped?
"""

import pandas as pd
from pathlib import Path
from ml.fusion.fuse_windows import fuse_feature_tables

DATA_ROOT = "/Users/pascalschlegel/data/interim/parsingsim3"

def debug_hrv_fusion():
    """Debug why HRV features are disappearing during fusion."""
    
    sub = 'sim_severe3'
    out_dir = f'{DATA_ROOT}/{sub}/effort_estimation_output/parsingsim3_{sub}'
    
    # Load PPG green and HRV
    ppg_df = pd.read_csv(f'{out_dir}/ppg_green/ppg_green_features_10.0s.csv')
    hrv_df = pd.read_csv(f'{out_dir}/ppg_green/ppg_green_hrv_features_10.0s.csv')
    
    print("=" * 70)
    print("PPG GREEN FEATURES")
    print("=" * 70)
    print(f"Columns ({len(ppg_df.columns)}): {list(ppg_df.columns)[:15]}...")
    print(f"t_center range: {ppg_df['t_center'].min():.1f} - {ppg_df['t_center'].max():.1f}")
    
    print("\n" + "=" * 70)
    print("PPG GREEN HRV FEATURES")
    print("=" * 70)
    print(f"Columns ({len(hrv_df.columns)}): {list(hrv_df.columns)}")
    print(f"t_center range: {hrv_df['t_center'].min():.1f} - {hrv_df['t_center'].max():.1f}")
    
    # Check for HRV features
    hrv_cols = [c for c in hrv_df.columns if any(x in c.lower() for x in ['rmssd', 'sdnn', 'hr_mean', 'pnn', 'lf_hf'])]
    print(f"\nHRV feature columns: {hrv_cols}")
    
    print("\n" + "=" * 70)
    print("ATTEMPTING FUSION OF PPG + HRV")
    print("=" * 70)
    
    # Try simple fusion of just these two
    fused = fuse_feature_tables(
        tables=[ppg_df, hrv_df],
        join_col="t_center",
        tolerance_sec=2.0,
    )
    
    print(f"Fused shape: {fused.shape}")
    print(f"Fused columns: {list(fused.columns)}")
    
    # Check HRV in fused
    hrv_in_fused = [c for c in fused.columns if any(x in c.lower() for x in ['rmssd', 'sdnn', 'hr_mean', 'pnn', 'lf_hf'])]
    print(f"\nHRV features in fused: {hrv_in_fused}")
    
    # The issue might be column overlap - check what columns are in both
    common = set(ppg_df.columns) & set(hrv_df.columns)
    print(f"\nColumns in BOTH dataframes: {common}")
    
    # PPG has n_peaks and peak_quality too!
    print("\nChecking for column conflicts...")
    for col in ppg_df.columns:
        if col in hrv_df.columns and col not in ['window_id', 'start_idx', 'end_idx', 't_start', 't_center', 't_end', 'modality']:
            print(f"  CONFLICT: {col}")


if __name__ == "__main__":
    debug_hrv_fusion()
