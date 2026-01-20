"""
Module 3: Define Effort Bouts

Effort bouts are time intervals of physical activity.
Prefer ADL intervals from logs; fallback to IMU intensity thresholding.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def parse_adl_intervals(adl_path: Path) -> pd.DataFrame:
    """
    Parse ADL (Activities of Daily Living) intervals from CSV.
    
    Expected columns: t_start, t_end, task_name (or activity)
    Optional: borg, intensity
    
    Args:
        adl_path: Path to ADL CSV
        
    Returns:
        adl_df: DataFrame with columns [t_start, t_end, task_name, ...]
    """
    if not Path(adl_path).exists():
        logger.warning(f"ADL file not found: {adl_path}")
        return pd.DataFrame()
    
    try:
        adl_df = pd.read_csv(adl_path)
    except Exception as e:
        logger.error(f"Failed to parse ADL file: {e}")
        return pd.DataFrame()
    
    # Normalize column names
    rename_map = {
        'activity': 'task_name',
        'Activity': 'task_name',
        'task': 'task_name',
        'Task': 'task_name',
    }
    adl_df.rename(columns=rename_map, inplace=True)
    
    # Ensure required columns
    required = ['t_start', 't_end']
    missing = [c for c in required if c not in adl_df.columns]
    if missing:
        logger.error(f"ADL missing columns: {missing}")
        return pd.DataFrame()
    
    adl_df['task_name'] = adl_df.get('task_name', 'unknown')
    
    logger.info(f"Parsed {len(adl_df)} ADL intervals from {adl_path}")
    return adl_df[['t_start', 't_end', 'task_name']]


def detect_imu_effort_bouts(
    imu_features_df: pd.DataFrame,
    intensity_col: str = "acc_mag_rms",
    threshold_percentile: float = 60.0,
    min_duration_sec: float = 20.0,
    merge_gap_sec: float = 5.0,
    time_col: str = "t_center",
) -> pd.DataFrame:
    """
    Detect effort bouts from IMU intensity thresholding.
    
    Args:
        imu_features_df: DataFrame with features per window, columns include time_col and intensity_col
        intensity_col: Name of intensity feature to threshold
        threshold_percentile: Use this percentile as activity threshold
        min_duration_sec: Minimum bout duration (seconds)
        merge_gap_sec: Merge bouts separated by < this gap (seconds)
        time_col: Timestamp column name
        
    Returns:
        bouts_df: DataFrame with columns [bout_id, t_start, t_end, task_name]
    """
    if imu_features_df.empty or intensity_col not in imu_features_df.columns:
        logger.warning("No IMU features for bout detection")
        return pd.DataFrame(columns=['bout_id', 't_start', 't_end', 'task_name'])
    
    # Compute threshold
    intensity = imu_features_df[intensity_col].dropna()
    threshold = np.percentile(intensity, threshold_percentile)
    
    logger.info(f"IMU intensity threshold (p{threshold_percentile}): {threshold:.4f}")
    
    # Mark active windows
    imu_features_df = imu_features_df.copy()
    imu_features_df['active'] = imu_features_df[intensity_col] >= threshold
    
    # Convert to time sequence
    active_times = imu_features_df[imu_features_df['active']][time_col].values
    
    if len(active_times) == 0:
        logger.warning("No active windows detected")
        return pd.DataFrame(columns=['bout_id', 't_start', 't_end', 'task_name'])
    
    # Detect contiguous active regions
    time_diffs = np.diff(active_times)
    gap_indices = np.where(time_diffs > merge_gap_sec)[0]
    
    # Split into bouts
    bouts = []
    bout_id = 0
    start_idx = 0
    
    for gap_idx in gap_indices:
        end_idx = gap_idx + 1
        bout_times = active_times[start_idx:end_idx]
        bout_duration = bout_times[-1] - bout_times[0]
        
        if bout_duration >= min_duration_sec:
            bouts.append({
                'bout_id': bout_id,
                't_start': bout_times[0],
                't_end': bout_times[-1],
                'task_name': 'imu_detected',
            })
            bout_id += 1
        
        start_idx = end_idx
    
    # Last bout
    bout_times = active_times[start_idx:]
    if len(bout_times) > 0:
        bout_duration = bout_times[-1] - bout_times[0]
        if bout_duration >= min_duration_sec:
            bouts.append({
                'bout_id': bout_id,
                't_start': bout_times[0],
                't_end': bout_times[-1],
                'task_name': 'imu_detected',
            })
    
    bouts_df = pd.DataFrame(bouts)
    logger.info(f"Detected {len(bouts_df)} IMU-based effort bouts")
    
    return bouts_df


def get_effort_bouts(
    adl_path: Optional[Path] = None,
    imu_features_df: Optional[pd.DataFrame] = None,
    **imu_kwargs
) -> pd.DataFrame:
    """
    Get effort bouts: prefer ADL intervals, fallback to IMU thresholding.
    
    Args:
        adl_path: Path to ADL CSV (preferred)
        imu_features_df: IMU features dataframe (fallback)
        **imu_kwargs: Arguments for detect_imu_effort_bouts
        
    Returns:
        bouts_df: Effort bouts
    """
    # Try ADL first
    if adl_path:
        bouts_df = parse_adl_intervals(adl_path)
        if not bouts_df.empty:
            bouts_df['bout_id'] = range(len(bouts_df))
            return bouts_df
    
    # Fallback to IMU
    if imu_features_df is not None and not imu_features_df.empty:
        logger.info("No ADL data; using IMU thresholding")
        return detect_imu_effort_bouts(imu_features_df, **imu_kwargs)
    
    logger.error("No ADL or IMU data for bout detection")
    return pd.DataFrame(columns=['bout_id', 't_start', 't_end', 'task_name'])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    import sys
    if len(sys.argv) > 1:
        adl_path = sys.argv[1]
        bouts = parse_adl_intervals(adl_path)
        print(bouts)
