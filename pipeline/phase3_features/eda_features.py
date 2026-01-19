"""EDA (Electrodermal Activity) Feature Extraction."""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def extract_eda_features(
    eda_df: pd.DataFrame,
    windows_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract EDA features for each window.
    
    Args:
        eda_df: Preprocessed EDA DataFrame with columns: t_unix, eda_signal
        windows_df: Windows DataFrame with columns: start_idx, end_idx, ...
    
    Returns:
        DataFrame with one row per window + extracted features
    """
    rows = []
    
    for _, win in windows_df.iterrows():
        start = int(win["start_idx"])
        end = int(win["end_idx"])
        
        signal = eda_df.iloc[start:end]["eda_signal"].values
        
        if signal.size == 0:
            continue
        
        features = {
            "eda_mean": float(np.mean(signal)),
            "eda_std": float(np.std(signal)),
            "eda_min": float(np.min(signal)),
            "eda_max": float(np.max(signal)),
            "eda_range": float(np.max(signal) - np.min(signal)),
        }
        
        # Add window metadata
        for col in windows_df.columns:
            features[col] = win[col]
        
        rows.append(features)
    
    result = pd.DataFrame(rows)
    logger.info(f"Extracted EDA features: {result.shape[0]} windows x {result.shape[1]} columns")
    
    return result
