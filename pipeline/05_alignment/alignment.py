"""
Alignment Module
================
Aligns fused features with ground-truth target labels (Borg effort scale).
Matches window time ranges with ADL (Activity of Daily Living) annotations.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def align_with_targets(
    fused_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    time_col: str = "t_start",
) -> pd.DataFrame:
    """
    Align fused features with target effort labels.
    
    Args:
        fused_df: Fused feature DataFrame with t_start, t_center, t_end columns
        targets_df: Target DataFrame with t_start, t_end, effort_label columns
        time_col: Which time column to use for alignment (t_start, t_center, or t_end)
    
    Returns:
        Aligned DataFrame with effort labels added
    """
    if "effort" not in fused_df.columns and "effort_label" not in targets_df.columns:
        logger.warning("No effort target column found")
    
    # Perform time-based alignment
    result = fused_df.copy()
    result["effort"] = np.nan
    
    for idx, row in result.iterrows():
        window_time = row[time_col]
        
        # Find matching target in window time range
        match = targets_df[
            (targets_df["t_start"] <= window_time) &
            (targets_df["t_end"] >= window_time)
        ]
        
        if len(match) > 0:
            result.loc[idx, "effort"] = match.iloc[0].get("effort_label", match.iloc[0].get("effort"))
    
    labeled = result["effort"].notna().sum()
    logger.info(f"Aligned {labeled}/{len(result)} windows with effort labels")
    
    return result
