"""RR (Respiration Rate) Feature Extraction."""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def extract_rr_features(
    rr_df: pd.DataFrame,
    windows_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract RR features for each window.
    
    Args:
        rr_df: Preprocessed RR DataFrame with columns: t_unix, rr_signal
        windows_df: Windows DataFrame with columns: start_idx, end_idx, ...
    
    Returns:
        DataFrame with one row per window + extracted features
    """
    rows = []
    
    for _, win in windows_df.iterrows():
        start = int(win["start_idx"])
        end = int(win["end_idx"])
        
        signal = rr_df.iloc[start:end]["rr_signal"].values
        
        if signal.size == 0:
            continue
        
        features = {
            "rr_mean": float(np.mean(signal)),
            "rr_std": float(np.std(signal)),
            "rr_min": float(np.min(signal)),
            "rr_max": float(np.max(signal)),
        }
        
        # Add window metadata
        for col in windows_df.columns:
            features[col] = win[col]
        
        rows.append(features)
    
    result = pd.DataFrame(rows)
    logger.info(f"Extracted RR features: {result.shape[0]} windows x {result.shape[1]} columns")
    
    return result
