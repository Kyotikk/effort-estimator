"""
Minimal Training Load Index (TLI) for ADLs.

Research-backed approach using ONLY HR + IMU data with improved intensity metrics.
No legacy features, no RMSSD, no HRmax/Karvonen.

Components:
- IMU: RMS acceleration, RMS jerk, active time
- HR: Baseline-relative HR delta × duration

Reference commit: v1.0-methodology-complete
To revert: git checkout v1.0-methodology-complete
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def compute_imu_features(
    df: pd.DataFrame,
    acc_mag_col: str = 'acc_mag_mean',
    duration_col: str = 'duration_s',
    raw_acc_cols: Optional[Tuple[str, str, str]] = None,
    activity_threshold_g: float = 0.1,
    fs: float = 100.0,
) -> pd.DataFrame:
    """
    Compute IMU intensity features per activity.
    
    If raw accelerometer columns are provided, computes proper RMS and jerk.
    Otherwise falls back to using pre-computed mean values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with one row per activity
    acc_mag_col : str
        Column containing mean acceleration magnitude (fallback)
    duration_col : str
        Column containing activity duration in seconds
    raw_acc_cols : tuple of 3 str, optional
        Column names for raw acc_x, acc_y, acc_z arrays (if available)
    activity_threshold_g : float
        Threshold in g for detecting "active" time (default 0.1g)
    fs : float
        Sampling frequency of raw IMU data (default 100 Hz)
    
    Returns
    -------
    pd.DataFrame
        Copy of input with added columns:
        - rms_acc_mag
        - rms_jerk
        - active_time_s
        - imu_intensity_raw
        - imu_load
    """
    df = df.copy()
    
    # Check if we have raw accelerometer data
    has_raw = raw_acc_cols is not None and all(c in df.columns for c in raw_acc_cols)
    
    if has_raw:
        # Compute from raw signals
        acc_x, acc_y, acc_z = raw_acc_cols
        
        rms_acc_list = []
        rms_jerk_list = []
        active_time_list = []
        
        for idx, row in df.iterrows():
            x = np.array(row[acc_x])
            y = np.array(row[acc_y])
            z = np.array(row[acc_z])
            
            # Magnitude
            mag = np.sqrt(x**2 + y**2 + z**2)
            
            # RMS of magnitude
            rms_acc = np.sqrt(np.mean(mag**2))
            
            # Jerk (derivative of magnitude)
            jerk = np.diff(mag) * fs  # multiply by fs to get proper units
            rms_jerk = np.sqrt(np.mean(jerk**2)) if len(jerk) > 0 else 0.0
            
            # Active time (where magnitude > threshold)
            active_samples = np.sum(mag > activity_threshold_g)
            active_time = active_samples / fs
            
            rms_acc_list.append(rms_acc)
            rms_jerk_list.append(rms_jerk)
            active_time_list.append(active_time)
        
        df['rms_acc_mag'] = rms_acc_list
        df['rms_jerk'] = rms_jerk_list
        df['active_time_s'] = active_time_list
        
    else:
        # Fallback: use pre-computed features
        # RMS ≈ mean for positive values (approximation)
        if acc_mag_col in df.columns:
            df['rms_acc_mag'] = df[acc_mag_col].fillna(0)
        else:
            # Try to compute from individual axis means
            acc_cols = [c for c in df.columns if c.startswith('acc_') and '_mean' in c and 'dyn' not in c]
            if len(acc_cols) >= 3:
                # Approximate magnitude from axis means (rough estimate)
                x = df.get('acc_x_mean', 0)
                y = df.get('acc_y_mean', 0)
                z = df.get('acc_z_mean', 0)
                df['rms_acc_mag'] = np.sqrt(x**2 + y**2 + z**2)
            else:
                df['rms_acc_mag'] = 1.0  # Default if no data
        
        # Cannot compute jerk without raw data
        df['rms_jerk'] = 0.0
        
        # Active time approximated as full duration (conservative)
        if duration_col in df.columns:
            df['active_time_s'] = df[duration_col].fillna(0)
        else:
            df['active_time_s'] = 10.0  # Default window size
    
    # IMU intensity = RMS_acc + RMS_jerk
    df['imu_intensity_raw'] = df['rms_acc_mag'] + df['rms_jerk']
    
    # IMU load = active_time × intensity
    df['imu_load'] = df['active_time_s'] * df['imu_intensity_raw']
    
    return df


def compute_hr_load(
    df: pd.DataFrame,
    hr_col: str = 'hr_mean',
    duration_col: str = 'duration_s',
    patient_col: str = 'subject',
    baseline_method: str = 'min',
) -> pd.DataFrame:
    """
    Compute HR-based internal load relative to baseline.
    
    Baseline is estimated as the minimum HR observed for the patient
    (approximation when explicit rest windows aren't available).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with HR and duration per activity
    hr_col : str
        Column containing mean HR for the activity
    duration_col : str
        Column containing activity duration in seconds
    patient_col : str
        Column identifying the patient/subject
    baseline_method : str
        How to estimate baseline: 'min' (minimum HR) or 'p10' (10th percentile)
    
    Returns
    -------
    pd.DataFrame
        Copy of input with added columns:
        - hr_baseline
        - hr_delta
        - hr_load
    """
    df = df.copy()
    
    # Estimate baseline HR per patient
    if baseline_method == 'min':
        baseline = df.groupby(patient_col)[hr_col].transform('min')
    elif baseline_method == 'p10':
        baseline = df.groupby(patient_col)[hr_col].transform(lambda x: x.quantile(0.10))
    else:
        raise ValueError(f"Unknown baseline_method: {baseline_method}")
    
    df['hr_baseline'] = baseline
    
    # HR delta (clamped to non-negative)
    df['hr_delta'] = np.maximum(0, df[hr_col] - df['hr_baseline'])
    
    # HR load = delta × duration
    duration = df[duration_col] if duration_col in df.columns else 10.0
    df['hr_load'] = df['hr_delta'] * duration
    
    return df


def zscore_loads(
    df: pd.DataFrame,
    patient_col: str = 'subject',
    hr_load_col: str = 'hr_load',
    imu_load_col: str = 'imu_load',
) -> pd.DataFrame:
    """
    Z-score normalize loads per patient.
    
    This aligns scales so HR and IMU loads have equal footing before weighting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with hr_load and imu_load columns
    patient_col : str
        Column identifying the patient/subject
    hr_load_col : str
        Column containing HR load values
    imu_load_col : str
        Column containing IMU load values
    
    Returns
    -------
    pd.DataFrame
        Copy of input with added columns:
        - z_hr_load
        - z_imu_load
    """
    df = df.copy()
    
    def zscore(x):
        """Z-score with handling for constant values."""
        std = x.std()
        if std == 0 or np.isnan(std):
            return x * 0  # Return zeros if no variance
        return (x - x.mean()) / std
    
    df['z_hr_load'] = df.groupby(patient_col)[hr_load_col].transform(zscore)
    df['z_imu_load'] = df.groupby(patient_col)[imu_load_col].transform(zscore)
    
    # Fill any remaining NaN with 0
    df['z_hr_load'] = df['z_hr_load'].fillna(0)
    df['z_imu_load'] = df['z_imu_load'].fillna(0)
    
    return df


def compute_tli_minimal(
    df: pd.DataFrame,
    weight_hr: float = 0.5,
    weight_imu: float = 0.5,
    hr_col: str = 'hr_mean',
    acc_mag_col: str = 'acc_mag_mean',
    duration_col: str = 'duration_s',
    patient_col: str = 'subject',
    activity_col: str = 'activity',
) -> pd.DataFrame:
    """
    Compute the minimal Training Load Index (TLI) using HR + IMU.
    
    TLI_min = weight_hr * z_HR_load + weight_imu * z_IMU_load
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with HR, IMU, duration per activity
    weight_hr : float
        Weight for HR component (default 0.5)
    weight_imu : float
        Weight for IMU component (default 0.5)
    hr_col : str
        Column containing mean HR
    acc_mag_col : str
        Column containing mean acceleration magnitude
    duration_col : str
        Column containing activity duration in seconds
    patient_col : str
        Column identifying the patient/subject
    activity_col : str
        Column identifying the activity
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - patient_id, activity_id, duration_s
        - hr_mean, hr_baseline, hr_delta, hr_load, z_hr_load
        - rms_acc_mag, imu_intensity_raw, imu_load, z_imu_load
        - TLI_min
    """
    # Step 1: Compute IMU features
    df = compute_imu_features(df, acc_mag_col=acc_mag_col, duration_col=duration_col)
    
    # Step 2: Compute HR load
    df = compute_hr_load(df, hr_col=hr_col, duration_col=duration_col, patient_col=patient_col)
    
    # Step 3: Z-score normalize per patient
    df = zscore_loads(df, patient_col=patient_col)
    
    # Step 4: Compute TLI
    df['TLI_min'] = weight_hr * df['z_hr_load'] + weight_imu * df['z_imu_load']
    
    # Select output columns
    output_cols = [
        patient_col, activity_col, duration_col,
        hr_col, 'hr_baseline', 'hr_delta', 'hr_load', 'z_hr_load',
        'rms_acc_mag', 'rms_jerk', 'active_time_s', 'imu_intensity_raw', 'imu_load', 'z_imu_load',
        'TLI_min'
    ]
    
    # Only include columns that exist
    output_cols = [c for c in output_cols if c in df.columns]
    
    return df[output_cols]


# ============================================================================
# Convenience functions for loading data from parsingsim3
# ============================================================================

def load_elderly_data(data_dir: str = '/Users/pascalschlegel/data/interim/parsingsim3') -> pd.DataFrame:
    """
    Load and prepare data for sim_elderly3 subject.
    
    Returns DataFrame with:
    - Activity-level aggregated features
    - HR mean per activity
    - IMU acceleration magnitude per activity
    - Duration per activity
    - Borg labels
    """
    import os
    
    # Try combined multisub data first
    combined_path = os.path.join(data_dir, 'multisub_combined/multisub_aligned_10.0s.csv')
    aligned_path = os.path.join(data_dir, 'sim_elderly3/effort_estimation_output/aligned_features_10.0s.csv')
    
    if os.path.exists(combined_path):
        df = pd.read_csv(combined_path)
        # Filter to elderly only
        if 'subject' in df.columns:
            df = df[df['subject'] == 'sim_elderly3'].copy()
        print(f"Loaded from combined data: {len(df)} windows")
    elif os.path.exists(aligned_path):
        df = pd.read_csv(aligned_path)
        if 'subject' not in df.columns:
            df['subject'] = 'sim_elderly3'
    else:
        raise FileNotFoundError(f"No data found at {combined_path} or {aligned_path}")
    
    # Ensure we have the needed columns
    # Look for HR column
    hr_cols = [c for c in df.columns if 'hr_mean' in c.lower()]
    if hr_cols:
        df['hr_mean'] = df[hr_cols[0]]
    
    # Look for acceleration magnitude
    acc_mag_cols = [c for c in df.columns if 'acc_mag' in c.lower() and 'mean' in c.lower()]
    if acc_mag_cols:
        df['acc_mag_mean'] = df[acc_mag_cols[0]]
    else:
        # Compute from individual axes if available
        if all(c in df.columns for c in ['acc_x_mean', 'acc_y_mean', 'acc_z_mean']):
            df['acc_mag_mean'] = np.sqrt(
                df['acc_x_mean']**2 + df['acc_y_mean']**2 + df['acc_z_mean']**2
            )
    
    # Duration (assume 10s windows if not specified)
    if 'duration_s' not in df.columns:
        df['duration_s'] = 10.0
    
    return df


def run_tli_analysis(
    df: pd.DataFrame = None,
    weight_hr: float = 0.5,
    weight_imu: float = 0.5,
) -> pd.DataFrame:
    """
    Run complete TLI analysis on data.
    
    Parameters
    ----------
    df : pd.DataFrame, optional
        Input data. If None, loads elderly data automatically.
    weight_hr : float
        Weight for HR component
    weight_imu : float
        Weight for IMU component
    
    Returns
    -------
    pd.DataFrame
        TLI results with all intermediate values
    """
    if df is None:
        df = load_elderly_data()
    
    # Filter to labeled data only
    if 'borg' in df.columns:
        df = df.dropna(subset=['borg'])
    
    # Compute TLI
    result = compute_tli_minimal(
        df,
        weight_hr=weight_hr,
        weight_imu=weight_imu,
    )
    
    # Add Borg back if available
    if 'borg' in df.columns:
        result['borg'] = df['borg'].values
    
    return result


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Training Load Index (TLI) - Minimal Implementation")
    print("=" * 60)
    print()
    print("To revert to previous pipeline:")
    print("  git checkout v1.0-methodology-complete")
    print()
    
    # Try to load and analyze data
    try:
        df = load_elderly_data()
        print(f"Loaded {len(df)} windows for sim_elderly3")
        
        # Check available columns
        print(f"\nAvailable columns: {len(df.columns)}")
        hr_cols = [c for c in df.columns if 'hr' in c.lower()]
        print(f"HR-related columns: {hr_cols[:5]}...")
        acc_cols = [c for c in df.columns if 'acc' in c.lower()][:5]
        print(f"ACC-related columns: {acc_cols}...")
        
        # Run TLI
        result = run_tli_analysis(df)
        print(f"\nTLI computed for {len(result)} activities")
        
        # Show summary
        print("\n" + "-" * 40)
        print("TLI Summary Statistics:")
        print("-" * 40)
        for col in ['hr_load', 'z_hr_load', 'imu_load', 'z_imu_load', 'TLI_min']:
            if col in result.columns:
                print(f"  {col}: mean={result[col].mean():.2f}, std={result[col].std():.2f}")
        
        # Correlation with Borg
        if 'borg' in result.columns:
            from scipy.stats import pearsonr, spearmanr
            
            print("\n" + "-" * 40)
            print("Correlation with Borg CR10:")
            print("-" * 40)
            
            for col in ['hr_load', 'imu_load', 'TLI_min']:
                if col in result.columns:
                    mask = ~(result[col].isna() | result['borg'].isna())
                    if mask.sum() > 2:
                        r_p, p_p = pearsonr(result.loc[mask, col], result.loc[mask, 'borg'])
                        r_s, p_s = spearmanr(result.loc[mask, col], result.loc[mask, 'borg'])
                        print(f"  {col}: Pearson r={r_p:.3f} (p={p_p:.3f}), Spearman r={r_s:.3f}")
        
        print("\n✓ TLI analysis complete!")
        
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
        print("\nTo use this module, ensure aligned features exist at:")
        print("  /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/aligned_features_10.0s.csv")
