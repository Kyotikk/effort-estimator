#!/usr/bin/env python3
"""
Minimal Training Load Index (TLI) - Computed from RAW data.

This script loads raw HR and IMU data from interim folder and computes
TLI per activity WITHOUT using any pre-computed features.

Components:
- IMU: RMS acceleration, RMS jerk, active time
- HR: Baseline-relative HR delta × duration

To revert to previous pipeline: git checkout v1.0-methodology-complete
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import gzip
from typing import Tuple, Optional


# =============================================================================
# Data Loading Functions
# =============================================================================

def parse_adl_timestamp(ts_str: str) -> float:
    """Parse ADL timestamp string to Unix timestamp.
    
    Note: ADL timestamps are in local display time but sensor data
    is in a different timezone. Apply 8-hour correction.
    """
    # Format: 04-12-2025-17-46-22-089 (DD-MM-YYYY-HH-MM-SS-mmm)
    try:
        parts = ts_str.split('-')
        if len(parts) >= 6:
            day, month, year, hour, minute, second = parts[:6]
            ms = int(parts[6]) if len(parts) > 6 else 0
            dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), ms * 1000)
            # Apply timezone correction: subtract 8 hours (28800 seconds)
            # This aligns ADL times with sensor timestamps
            return dt.timestamp() + (ms / 1000.0) - 28800
    except:
        pass
    return np.nan


def load_adl_labels(subject_dir: Path) -> pd.DataFrame:
    """
    Load ADL labels with start/end times and Borg ratings.
    
    Returns DataFrame with columns:
    - activity: activity name
    - t_start: start timestamp (unix)
    - t_end: end timestamp (unix)
    - duration_s: duration in seconds
    - borg: Borg CR10 rating
    """
    adl_path = subject_dir / 'scai_app' / 'ADLs_1.csv'
    
    if not adl_path.exists():
        raise FileNotFoundError(f"ADL labels not found at {adl_path}")
    
    # Read raw file, skip header rows
    with open(adl_path, 'r') as f:
        lines = f.readlines()
    
    # Find data start (after header)
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('Time,ADLs,Effort'):
            data_start = i + 1
            break
    
    # Parse activities
    activities = []
    current_activity = None
    current_start = None
    
    for line in lines[data_start:]:
        parts = line.strip().split(',')
        if len(parts) < 2:
            continue
        
        timestamp_str = parts[0]
        adl_label = parts[1]
        effort = parts[2] if len(parts) > 2 else ''
        
        if 'Start' in adl_label:
            # Start of activity
            activity_name = adl_label.replace(' Start', '')
            current_activity = activity_name
            current_start = parse_adl_timestamp(timestamp_str)
        
        elif 'End' in adl_label and current_activity is not None:
            # End of activity
            t_end = parse_adl_timestamp(timestamp_str)
            
            # Parse effort (Borg rating)
            borg = np.nan
            if effort:
                try:
                    borg = float(effort)
                except:
                    pass
            
            if not np.isnan(current_start) and not np.isnan(t_end):
                activities.append({
                    'activity': current_activity,
                    't_start': current_start,
                    't_end': t_end,
                    'duration_s': t_end - current_start,
                    'borg': borg
                })
            
            current_activity = None
            current_start = None
    
    return pd.DataFrame(activities)


def load_hr_data(subject_dir: Path) -> pd.DataFrame:
    """Load raw HR data from vivalnk sensor."""
    hr_dir = subject_dir / 'vivalnk_vv330_heart_rate'
    
    dfs = []
    for f in sorted(hr_dir.glob('*.csv.gz')):
        with gzip.open(f, 'rt') as gz:
            df = pd.read_csv(gz)
            dfs.append(df)
    
    if not dfs:
        raise FileNotFoundError(f"No HR data found in {hr_dir}")
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Filter invalid HR values (negative values are error codes)
    df = df[df['hr'] > 0].copy()
    
    return df


def load_acc_data(subject_dir: Path) -> pd.DataFrame:
    """Load raw acceleration data from vivalnk sensor."""
    acc_dir = subject_dir / 'vivalnk_vv330_acceleration'
    
    dfs = []
    for f in sorted(acc_dir.glob('*.csv.gz')):
        with gzip.open(f, 'rt') as gz:
            df = pd.read_csv(gz)
            dfs.append(df)
    
    if not dfs:
        raise FileNotFoundError(f"No acceleration data found in {acc_dir}")
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Convert from raw units to g (assuming LSB = 1/4096 g typical for accelerometers)
    # Adjust scale factor based on actual sensor specs
    scale = 1.0 / 4096.0  # Convert to g
    df['x_g'] = df['x'] * scale
    df['y_g'] = df['y'] * scale
    df['z_g'] = df['z'] * scale
    
    # Compute magnitude
    df['mag_g'] = np.sqrt(df['x_g']**2 + df['y_g']**2 + df['z_g']**2)
    
    return df


# =============================================================================
# Feature Computation Functions
# =============================================================================

def compute_hr_features_per_activity(
    adl_df: pd.DataFrame,
    hr_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute HR features for each activity window.
    
    Returns DataFrame with added columns:
    - hr_mean: mean HR during activity
    - hr_min: min HR during activity
    - hr_max: max HR during activity
    - hr_samples: number of HR samples
    """
    results = []
    
    for _, row in adl_df.iterrows():
        t_start = row['t_start']
        t_end = row['t_end']
        
        # Get HR samples within activity window
        mask = (hr_df['time'] >= t_start) & (hr_df['time'] <= t_end)
        hr_window = hr_df.loc[mask, 'hr']
        
        if len(hr_window) > 0:
            hr_mean = hr_window.mean()
            hr_min = hr_window.min()
            hr_max = hr_window.max()
            hr_samples = len(hr_window)
        else:
            hr_mean = np.nan
            hr_min = np.nan
            hr_max = np.nan
            hr_samples = 0
        
        results.append({
            **row.to_dict(),
            'hr_mean': hr_mean,
            'hr_min': hr_min,
            'hr_max': hr_max,
            'hr_samples': hr_samples,
        })
    
    return pd.DataFrame(results)


def compute_imu_features_per_activity(
    df: pd.DataFrame,
    acc_df: pd.DataFrame,
    activity_threshold_g: float = 0.2,
) -> pd.DataFrame:
    """
    Compute IMU features for each activity window.
    
    Returns DataFrame with added columns:
    - rms_acc_mag: RMS of acceleration magnitude
    - rms_jerk: RMS of jerk (derivative of magnitude)
    - active_time_s: time where acc_mag > threshold
    - acc_samples: number of acceleration samples
    """
    results = []
    
    for _, row in df.iterrows():
        t_start = row['t_start']
        t_end = row['t_end']
        
        # Get acceleration samples within activity window
        mask = (acc_df['time'] >= t_start) & (acc_df['time'] <= t_end)
        acc_window = acc_df.loc[mask]
        
        if len(acc_window) > 1:
            mag = acc_window['mag_g'].values
            time = acc_window['time'].values
            
            # RMS of magnitude
            rms_acc = np.sqrt(np.mean(mag**2))
            
            # Jerk (derivative of magnitude)
            dt = np.diff(time)
            dt[dt == 0] = 0.001  # Avoid division by zero
            jerk = np.diff(mag) / dt
            rms_jerk = np.sqrt(np.mean(jerk**2))
            
            # Active time (magnitude above threshold)
            # Note: subtract ~1g for gravity to get dynamic component
            dynamic_mag = np.abs(mag - 1.0)  # Approximate dynamic acceleration
            active_samples = np.sum(dynamic_mag > activity_threshold_g)
            
            # Estimate sampling rate from data
            mean_dt = np.mean(dt)
            fs = 1.0 / mean_dt if mean_dt > 0 else 5.0  # ~5 Hz typical
            active_time = active_samples / fs
            
            acc_samples = len(acc_window)
        else:
            rms_acc = np.nan
            rms_jerk = np.nan
            active_time = 0.0
            acc_samples = len(acc_window)
        
        results.append({
            **row.to_dict(),
            'rms_acc_mag': rms_acc,
            'rms_jerk': rms_jerk,
            'active_time_s': active_time,
            'acc_samples': acc_samples,
        })
    
    return pd.DataFrame(results)


# =============================================================================
# TLI Computation
# =============================================================================

def compute_tli(
    df: pd.DataFrame,
    weight_hr: float = 0.5,
    weight_imu: float = 0.5,
) -> pd.DataFrame:
    """
    Compute Training Load Index from HR and IMU features.
    
    Steps:
    1. Compute HR baseline (minimum observed HR)
    2. Compute HR delta and HR load
    3. Compute IMU intensity and IMU load
    4. Z-score both loads
    5. Combine with weights
    """
    df = df.copy()
    
    # --- HR Load ---
    # Baseline = minimum HR (approximation of resting)
    hr_baseline = df['hr_mean'].min()
    df['hr_baseline'] = hr_baseline
    
    # HR delta (clamped to non-negative)
    df['hr_delta'] = np.maximum(0, df['hr_mean'] - hr_baseline)
    
    # HR load = delta × duration
    df['hr_load'] = df['hr_delta'] * df['duration_s']
    
    # --- IMU Load ---
    # IMU intensity = RMS_acc + RMS_jerk
    df['imu_intensity'] = df['rms_acc_mag'].fillna(0) + df['rms_jerk'].fillna(0)
    
    # IMU load = active_time × intensity
    # Use duration as fallback if active_time is 0
    effective_time = df['active_time_s'].copy()
    effective_time[effective_time < 1] = df.loc[effective_time < 1, 'duration_s']
    df['imu_load'] = effective_time * df['imu_intensity']
    
    # --- Z-score normalization ---
    def zscore(x):
        std = x.std()
        if std == 0 or np.isnan(std):
            return x * 0
        return (x - x.mean()) / std
    
    df['z_hr_load'] = zscore(df['hr_load'].fillna(0))
    df['z_imu_load'] = zscore(df['imu_load'].fillna(0))
    
    # --- TLI ---
    df['TLI'] = weight_hr * df['z_hr_load'] + weight_imu * df['z_imu_load']
    
    return df


# =============================================================================
# Main Pipeline
# =============================================================================

def run_tli_pipeline(
    subject_dir: Path,
    weight_hr: float = 0.5,
    weight_imu: float = 0.5,
) -> pd.DataFrame:
    """
    Run complete TLI pipeline for a subject.
    
    Parameters
    ----------
    subject_dir : Path
        Path to subject's interim data directory
    weight_hr : float
        Weight for HR component (default 0.5)
    weight_imu : float
        Weight for IMU component (default 0.5)
    
    Returns
    -------
    pd.DataFrame
        TLI results per activity
    """
    print(f"Loading data from: {subject_dir}")
    
    # Load raw data
    print("  Loading ADL labels...")
    adl_df = load_adl_labels(subject_dir)
    print(f"    Found {len(adl_df)} activities")
    
    print("  Loading HR data...")
    hr_df = load_hr_data(subject_dir)
    print(f"    Found {len(hr_df)} HR samples")
    
    print("  Loading acceleration data...")
    acc_df = load_acc_data(subject_dir)
    print(f"    Found {len(acc_df)} acceleration samples")
    
    # Compute features
    print("  Computing HR features per activity...")
    df = compute_hr_features_per_activity(adl_df, hr_df)
    
    print("  Computing IMU features per activity...")
    df = compute_imu_features_per_activity(df, acc_df)
    
    # Compute TLI
    print("  Computing TLI...")
    df = compute_tli(df, weight_hr=weight_hr, weight_imu=weight_imu)
    
    return df


def print_results(df: pd.DataFrame):
    """Print summary of TLI results."""
    print("\n" + "=" * 70)
    print("TLI RESULTS SUMMARY")
    print("=" * 70)
    
    # Basic stats
    print(f"\nActivities: {len(df)}")
    print(f"Activities with Borg: {df['borg'].notna().sum()}")
    print(f"Activities with HR: {df['hr_mean'].notna().sum()}")
    print(f"Activities with IMU: {df['rms_acc_mag'].notna().sum()}")
    
    # TLI stats
    print("\n--- TLI Statistics ---")
    print(f"  TLI range: [{df['TLI'].min():.2f}, {df['TLI'].max():.2f}]")
    print(f"  TLI mean: {df['TLI'].mean():.2f}")
    print(f"  TLI std: {df['TLI'].std():.2f}")
    
    # HR stats
    print("\n--- HR Statistics ---")
    print(f"  HR baseline: {df['hr_baseline'].iloc[0]:.1f} BPM")
    print(f"  HR mean range: [{df['hr_mean'].min():.1f}, {df['hr_mean'].max():.1f}] BPM")
    print(f"  HR load range: [{df['hr_load'].min():.1f}, {df['hr_load'].max():.1f}]")
    
    # IMU stats
    print("\n--- IMU Statistics ---")
    print(f"  RMS acc range: [{df['rms_acc_mag'].min():.3f}, {df['rms_acc_mag'].max():.3f}] g")
    print(f"  RMS jerk range: [{df['rms_jerk'].min():.3f}, {df['rms_jerk'].max():.3f}] g/s")
    print(f"  IMU load range: [{df['imu_load'].min():.1f}, {df['imu_load'].max():.1f}]")
    
    # Correlation with Borg
    df_valid = df.dropna(subset=['borg', 'TLI'])
    if len(df_valid) > 2:
        from scipy.stats import pearsonr, spearmanr
        
        print("\n--- Correlation with Borg CR10 ---")
        
        for col in ['hr_load', 'imu_load', 'TLI']:
            r_p, p_p = pearsonr(df_valid[col], df_valid['borg'])
            r_s, p_s = spearmanr(df_valid[col], df_valid['borg'])
            sig = '***' if p_p < 0.001 else '**' if p_p < 0.01 else '*' if p_p < 0.05 else ''
            print(f"  {col:12s}: r={r_p:+.3f} (p={p_p:.4f}){sig}, ρ={r_s:+.3f}")
    
    # Top/bottom activities by TLI
    print("\n--- Top 5 Activities by TLI ---")
    top5 = df.nlargest(5, 'TLI')[['activity', 'duration_s', 'hr_mean', 'TLI', 'borg']]
    for _, row in top5.iterrows():
        borg_str = f"Borg={row['borg']:.1f}" if pd.notna(row['borg']) else "Borg=N/A"
        print(f"  {row['activity']:30s} TLI={row['TLI']:+.2f}  HR={row['hr_mean']:.0f}  {borg_str}")
    
    print("\n--- Bottom 5 Activities by TLI ---")
    bot5 = df.nsmallest(5, 'TLI')[['activity', 'duration_s', 'hr_mean', 'TLI', 'borg']]
    for _, row in bot5.iterrows():
        borg_str = f"Borg={row['borg']:.1f}" if pd.notna(row['borg']) else "Borg=N/A"
        print(f"  {row['activity']:30s} TLI={row['TLI']:+.2f}  HR={row['hr_mean']:.0f}  {borg_str}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute TLI from raw data')
    parser.add_argument('--subject', default='sim_elderly3', help='Subject ID')
    parser.add_argument('--data-dir', default='/Users/pascalschlegel/data/interim/parsingsim3',
                       help='Base data directory')
    parser.add_argument('--weight-hr', type=float, default=0.5, help='Weight for HR')
    parser.add_argument('--weight-imu', type=float, default=0.5, help='Weight for IMU')
    parser.add_argument('--output', default=None, help='Output CSV path')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Training Load Index (TLI) - From RAW Data")
    print("=" * 70)
    print(f"\nTo revert to previous pipeline: git checkout v1.0-methodology-complete\n")
    
    subject_dir = Path(args.data_dir) / args.subject
    
    if not subject_dir.exists():
        print(f"ERROR: Subject directory not found: {subject_dir}")
        exit(1)
    
    # Run pipeline
    df = run_tli_pipeline(
        subject_dir,
        weight_hr=args.weight_hr,
        weight_imu=args.weight_imu,
    )
    
    # Print results
    print_results(df)
    
    # Save output
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\n✓ Results saved to: {args.output}")
    else:
        out_path = subject_dir / 'effort_estimation_output' / 'tli_results.csv'
        out_path.parent.mkdir(exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\n✓ Results saved to: {out_path}")
