"""
Module 5: Extract Features X During Effort Bouts

Aggregate IMU, EDA, PPG features computed during effort interval.
Output: one row per bout with features + targets.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


def aggregate_windowed_features(
    features_df: pd.DataFrame,
    bouts_df: pd.DataFrame,
    time_col: str = "t_center",
    method: str = "mean",
    prefix: str = "",
) -> pd.DataFrame:
    """
    Aggregate per-window features over effort bout intervals.
    
    Args:
        features_df: Per-window features [t_start, t_center, t_end, feat_1, feat_2, ...]
        bouts_df: Effort bouts [bout_id, t_start, t_end, ...]
        time_col: Timestamp column ("t_center" or "t_start")
        method: "mean", "median", "std", "min", "max"
        prefix: Add prefix to output column names (e.g., "imu_")
        
    Returns:
        bout_features_df: One row per bout, aggregated features
    """
    if features_df.empty or bouts_df.empty:
        logger.warning(f"Empty features or bouts data")
        return pd.DataFrame()
    
    # Feature columns (exclude time/id columns)
    exclude_cols = {'t_start', 't_center', 't_end', 'start_idx', 'end_idx', 
                    'window_id', 'modality', 'bout_id'}
    feat_cols = [c for c in features_df.columns if c not in exclude_cols]
    
    bout_features = []
    
    for idx, bout in bouts_df.iterrows():
        bout_id = bout['bout_id']
        bout_start = bout['t_start']
        bout_end = bout['t_end']
        
        # Select windows whose centers fall within bout
        mask = (features_df[time_col] >= bout_start) & (features_df[time_col] < bout_end)
        bout_windows = features_df[mask]
        
        if bout_windows.empty:
            logger.warning(f"  Bout {bout_id}: no windows in interval [{bout_start:.1f}, {bout_end:.1f}]")
            continue
        
        # Aggregate features
        agg_features = {'bout_id': bout_id}
        
        for feat in feat_cols:
            if feat in bout_windows.columns:
                values = bout_windows[feat].dropna()
                
                if len(values) == 0:
                    agg_features[f"{prefix}{feat}"] = np.nan
                elif method == "mean":
                    agg_features[f"{prefix}{feat}"] = values.mean()
                elif method == "median":
                    agg_features[f"{prefix}{feat}"] = values.median()
                elif method == "std":
                    agg_features[f"{prefix}{feat}"] = values.std()
                elif method == "min":
                    agg_features[f"{prefix}{feat}"] = values.min()
                elif method == "max":
                    agg_features[f"{prefix}{feat}"] = values.max()
        
        bout_features.append(agg_features)
    
    bout_features_df = pd.DataFrame(bout_features)
    logger.info(f"Aggregated features for {len(bout_features_df)}/{len(bouts_df)} bouts")
    
    return bout_features_df


def extract_hr_from_ibi(
    ibi_df: pd.DataFrame,
    bout_t_start: float,
    bout_t_end: float,
) -> dict:
    """
    Extract HR-related features during effort bout from IBI data.
    
    Args:
        ibi_df: IBI timeseries [t, ibi_sec]
        bout_t_start, bout_t_end: Bout interval (seconds)
        
    Returns:
        features: {'hr_mean', 'hr_std', 'rmssd_during_effort', ...}
    """
    if ibi_df.empty:
        return {}
    
    # IBIs during effort
    mask = (ibi_df['t'] >= bout_t_start) & (ibi_df['t'] < bout_t_end)
    effort_ibi = ibi_df.loc[mask, 'ibi_sec'].values
    
    features = {}
    
    if len(effort_ibi) >= 2:
        # Heart rate from mean IBI
        hr_mean = 60.0 / effort_ibi.mean()
        features['hr_mean'] = hr_mean
        features['hr_std'] = 60.0 * effort_ibi.std() / (effort_ibi.mean() ** 2)
        
        # RMSSD during effort
        diffs = np.diff(effort_ibi)
        features['rmssd_during_effort'] = np.sqrt(np.mean(diffs ** 2))
    
    return features


def build_model_table(
    bouts_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    imu_features_df: Optional[pd.DataFrame] = None,
    eda_features_df: Optional[pd.DataFrame] = None,
    ibi_df: Optional[pd.DataFrame] = None,
    session_id: str = "unknown",
) -> pd.DataFrame:
    """
    Build model table: one row per bout with features X and labels y.
    
    Args:
        bouts_df: Effort bouts
        labels_df: HRV recovery labels
        imu_features_df: Per-window IMU features
        eda_features_df: Per-window EDA features
        ibi_df: IBI timeseries (for HR extraction)
        session_id: Session identifier
        
    Returns:
        model_table_df: Comprehensive feature + label table
    """
    # Start with labels
    model_table = labels_df.copy()
    model_table['session_id'] = session_id
    
    # Aggregate IMU features per bout
    if imu_features_df is not None and not imu_features_df.empty:
        imu_agg = aggregate_windowed_features(
            imu_features_df,
            bouts_df,
            prefix="imu_",
            method="mean"
        )
        if not imu_agg.empty:
            model_table = model_table.merge(imu_agg, on='bout_id', how='left')
    
    # Aggregate EDA features per bout
    if eda_features_df is not None and not eda_features_df.empty:
        eda_agg = aggregate_windowed_features(
            eda_features_df,
            bouts_df,
            prefix="eda_",
            method="mean"
        )
        if not eda_agg.empty:
            model_table = model_table.merge(eda_agg, on='bout_id', how='left')
    
    # Extract HR features from IBI
    if ibi_df is not None and not ibi_df.empty:
        hr_features_list = []
        for idx, bout in bouts_df.iterrows():
            hr_feat = extract_hr_from_ibi(ibi_df, bout['t_start'], bout['t_end'])
            hr_feat['bout_id'] = bout['bout_id']
            hr_features_list.append(hr_feat)
        
        if hr_features_list:
            hr_features_df = pd.DataFrame(hr_features_list)
            model_table = model_table.merge(hr_features_df, on='bout_id', how='left')
    
    logger.info(
        f"Built model table: {len(model_table)} rows × {len(model_table.columns)} columns"
    )
    logger.info(f"  Labeled rows (QC pass): {model_table['qc_ok'].sum()}")
    
    return model_table


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    import sys
    if len(sys.argv) > 3:
        labels_path = sys.argv[1]
        imu_path = sys.argv[2]
        bouts_path = sys.argv[3]
        
        labels_df = pd.read_csv(labels_path)
        imu_df = pd.read_csv(imu_path)
        bouts_df = pd.read_csv(bouts_path)
        
        model_table = build_model_table(
            bouts_df,
            labels_df,
            imu_features_df=imu_df,
            session_id="demo"
        )
        print(model_table.head())
        print(f"Shape: {model_table.shape}")
