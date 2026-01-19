"""
Fusion Module
=============
Combines feature DataFrames from different modalities (IMU, PPG, RR, EDA)
into a single fused feature matrix aligned by time windows.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def fuse_modalities(
    modality_dfs: dict,
    on: str = "t_start",
    method: str = "inner",
) -> pd.DataFrame:
    """
    Fuse feature DataFrames from multiple modalities.
    
    Args:
        modality_dfs: Dict mapping modality name -> DataFrame
                      (each must have t_start, t_center, t_end columns)
        on: Column to join on (t_start, t_center, or t_end)
        method: Join method ('inner', 'outer', 'left', 'right')
    
    Returns:
        Fused DataFrame with all modality features
    """
    if not modality_dfs:
        raise ValueError("No modality DataFrames provided")
    
    dfs_list = list(modality_dfs.values())
    names_list = list(modality_dfs.keys())
    
    # Start with first modality
    fused = dfs_list[0].copy()
    fused[f"modality_source"] = names_list[0]
    
    # Merge remaining modalities
    for name, df in zip(names_list[1:], dfs_list[1:]):
        # Rename feature columns to avoid conflicts
        feature_cols = [c for c in df.columns if c not in ["t_start", "t_center", "t_end", "start_idx", "end_idx", "n_samples", "win_sec"]]
        df_renamed = df[["t_start", "t_center", "t_end"] + feature_cols].copy()
        df_renamed = df_renamed.rename(columns={c: f"{name}_{c}" for c in feature_cols})
        
        # Merge on time window
        fused = pd.merge(
            fused,
            df_renamed,
            on=on,
            how=method,
            suffixes=("", "_dup")
        )
        
        logger.info(f"Fused {name}: {fused.shape[0]} rows x {fused.shape[1]} columns")
    
    logger.info(f"Final fused data: {fused.shape[0]} windows x {fused.shape[1]} features")
    
    return fused
