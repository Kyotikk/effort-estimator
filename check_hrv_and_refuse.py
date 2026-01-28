#!/usr/bin/env python3
"""
Check HRV files and re-run fusion to include HRV features.
Fixed: Don't drop HRV columns with NaN - impute them instead.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ml.fusion.fuse_windows import fuse_feature_tables

DATA_ROOT = "/Users/pascalschlegel/data/interim/parsingsim3"
SUBJECTS = ["sim_elderly3", "sim_healthy3", "sim_severe3"]
WINDOW_LENGTH = "10.0s"


def sanitise_features_keep_hrv(df, nan_threshold=0.5):
    """
    Sanitize features but preserve HRV columns with NaN imputation.
    
    - Drop bool columns
    - Drop non-numeric columns  
    - Handle duplicate columns
    - For numeric columns with NaN: impute with median if <threshold, else drop
    - ALWAYS keep HRV columns (impute even if >50% NaN)
    """
    dropped = []
    out = df.copy()
    
    # Remove duplicate columns first (keep first occurrence)
    out = out.loc[:, ~out.columns.duplicated()]
    
    # Drop bool
    bool_cols = out.select_dtypes(include=["bool"]).columns.tolist()
    out = out.drop(columns=bool_cols)
    dropped.extend(bool_cols)
    
    # Drop non-numeric
    non_numeric = out.select_dtypes(exclude=[np.number]).columns.tolist()
    out = out.drop(columns=non_numeric)
    dropped.extend(non_numeric)
    
    # HRV columns we want to preserve
    hrv_patterns = ['mean_ibi', 'rmssd', 'sdnn', 'hr_mean', 'hr_std', 'hr_min', 'hr_max', 
                    'pnn50', 'pnn20', 'lf_power', 'hf_power', 'lf_hf', 'sdsd', 'cv_ibi']
    
    def is_hrv_col(col):
        return any(p in col.lower() for p in hrv_patterns)
    
    # Handle NaN - don't blindly drop, impute or drop based on threshold
    cols_to_drop = []
    for col in out.columns:
        nan_frac = out[col].isna().mean()
        
        # ALWAYS keep HRV columns - impute even if high NaN
        if is_hrv_col(col):
            if nan_frac > 0:
                median_val = out[col].median()
                if pd.isna(median_val):
                    median_val = 0  # If all NaN, use 0
                out[col] = out[col].fillna(median_val)
            continue  # Never drop HRV
            
        if nan_frac > nan_threshold:
            # Too many NaNs - drop
            cols_to_drop.append(col)
        elif nan_frac > 0:
            # Some NaNs - impute with median
            median_val = out[col].median()
            if pd.isna(median_val):
                median_val = 0  # Fallback
            out[col] = out[col].fillna(median_val)
    
    out = out.drop(columns=cols_to_drop)
    dropped.extend(cols_to_drop)
    
    return out, sorted(set(dropped))


def check_files(subject):
    """Check all fusion input files for a subject."""
    out_dir = f'{DATA_ROOT}/{subject}/effort_estimation_output/parsingsim3_{subject}'
    
    files = {
        'imu_bioz': f'{out_dir}/imu_bioz/imu_features_{WINDOW_LENGTH}.csv',
        'imu_wrist': f'{out_dir}/imu_wrist/imu_features_{WINDOW_LENGTH}.csv',
        'ppg_green': f'{out_dir}/ppg_green/ppg_green_features_{WINDOW_LENGTH}.csv',
        'ppg_green_hrv': f'{out_dir}/ppg_green/ppg_green_hrv_features_{WINDOW_LENGTH}.csv',
        'ppg_infra': f'{out_dir}/ppg_infra/ppg_infra_features_{WINDOW_LENGTH}.csv',
        'ppg_infra_hrv': f'{out_dir}/ppg_infra/ppg_infra_hrv_features_{WINDOW_LENGTH}.csv',
        'ppg_red': f'{out_dir}/ppg_red/ppg_red_features_{WINDOW_LENGTH}.csv',
        'ppg_red_hrv': f'{out_dir}/ppg_red/ppg_red_hrv_features_{WINDOW_LENGTH}.csv',
        'eda': f'{out_dir}/eda/eda_features_{WINDOW_LENGTH}.csv',
        'eda_advanced': f'{out_dir}/eda/eda_advanced_features_{WINDOW_LENGTH}.csv',
    }
    
    # Also load the aligned file with borg labels
    aligned_path = f'{out_dir}/fused_aligned_{WINDOW_LENGTH}.csv'
    
    print(f"\n{subject}:")
    available = {}
    for name, path in files.items():
        exists = Path(path).exists()
        if exists:
            df = pd.read_csv(path)
            print(f"  ✓ {name}: {len(df)} rows, {len(df.columns)} cols")
            available[name] = df
        else:
            print(f"  ✗ {name}: MISSING")
    
    # Load targets separately
    if Path(aligned_path).exists():
        aligned_df = pd.read_csv(aligned_path)
        if 'borg' in aligned_df.columns:
            # Extract just borg and t_center for joining
            available['_targets'] = aligned_df[['t_center', 'borg']].copy()
            n_labeled = aligned_df['borg'].notna().sum()
            print(f"  ✓ targets: {n_labeled} labeled windows")
    
    return available


def fuse_with_hrv(subject, files_dict):
    """Fuse all available modalities including HRV."""
    
    # Separate targets from features
    targets_df = files_dict.pop('_targets', None)
    
    print(f"\n  Fusing {len(files_dict)} modalities...")
    
    tables = list(files_dict.values())
    
    fused = fuse_feature_tables(
        tables=tables,
        join_col="t_center",
        tolerance_sec=2.0,
    )
    
    # Sanitize with NaN imputation instead of dropping
    fused, dropped = sanitise_features_keep_hrv(fused, nan_threshold=0.5)
    print(f"  Fused: {len(fused)} rows, {len(fused.columns)} cols (dropped {len(dropped)} invalid)")
    
    # Check HRV features
    hrv_cols = [c for c in fused.columns if any(x in c.lower() for x in ['rmssd', 'sdnn', 'hr_mean', 'pnn', 'lf_hf'])]
    print(f"  HRV features: {len(hrv_cols)}")
    if hrv_cols:
        print(f"    Examples: {hrv_cols[:5]}")
    
    # Add targets (borg labels)
    if targets_df is not None:
        # Drop rows with NaN t_center before merging
        targets_clean = targets_df.dropna(subset=['t_center']).copy()
        fused_clean = fused.dropna(subset=['t_center']).copy()
        
        # Merge on t_center
        fused = pd.merge_asof(
            fused_clean.sort_values('t_center'),
            targets_clean.sort_values('t_center'),
            on='t_center',
            tolerance=2.0,
            direction='nearest'
        )
        n_labeled = fused['borg'].notna().sum()
        print(f"  Borg labels: {n_labeled} windows")
    
    # Add subject ID
    fused['subject_id'] = subject
    
    return fused


def main():
    print("=" * 70)
    print("CHECKING HRV FILES AND RE-FUSING WITH HRV FEATURES")
    print("=" * 70)
    
    all_fused = []
    
    for subject in SUBJECTS:
        files = check_files(subject)
        
        if files:
            fused = fuse_with_hrv(subject, files)
            # Reset index to avoid concat issues
            fused = fused.reset_index(drop=True)
            all_fused.append(fused)
    
    # Combine all subjects
    if all_fused:
        print("\n" + "=" * 70)
        print("COMBINING ALL SUBJECTS")
        print("=" * 70)
        
        # Get union of all columns
        all_cols = set()
        for df in all_fused:
            all_cols.update(df.columns)
        
        # Align all dataframes to have same columns
        aligned = []
        for df in all_fused:
            for col in all_cols:
                if col not in df.columns:
                    df[col] = np.nan
            aligned.append(df)
        
        combined = pd.concat(aligned, ignore_index=True)
        print(f"Combined: {len(combined)} rows, {len(combined.columns)} cols")
        
        # Check HRV in combined
        hrv_cols = [c for c in combined.columns if any(x in c.lower() for x in ['rmssd', 'sdnn', 'hr_mean', 'pnn', 'lf_hf'])]
        print(f"HRV features in combined: {len(hrv_cols)}")
        if hrv_cols:
            print(f"  {hrv_cols[:10]}...")
        
        # Save
        out_dir = Path(DATA_ROOT) / "multisub_combined"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = out_dir / f"multisub_aligned_{WINDOW_LENGTH}.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nSaved to: {out_path}")
        print(f"Total: {len(combined)} samples, {len(combined.columns)} features")


if __name__ == "__main__":
    main()
