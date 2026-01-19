"""PPG Feature Extraction."""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def extract_ppg_features(
    ppg_df: pd.DataFrame,
    windows_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract PPG features for each window.
    
    Args:
        ppg_df: Preprocessed PPG DataFrame with columns: t_unix, ppg_signal
        windows_df: Windows DataFrame with columns: start_idx, end_idx, ...
    
    Returns:
        DataFrame with one row per window + extracted features
    """
    rows = []
    
    for _, win in windows_df.iterrows():
        start = int(win["start_idx"])
        end = int(win["end_idx"])
        
        signal = ppg_df.iloc[start:end]["ppg_signal"].values
        
        if signal.size == 0:
            continue
        
        features = {
            "ppg_mean": float(np.mean(signal)),
            "ppg_std": float(np.std(signal)),
            "ppg_min": float(np.min(signal)),
            "ppg_max": float(np.max(signal)),
            "ppg_rms": float(np.sqrt(np.mean(signal ** 2))),
        }
        
        # Add window metadata
        for col in windows_df.columns:
            features[col] = win[col]
        
        rows.append(features)
    
    result = pd.DataFrame(rows)
    logger.info(f"Extracted PPG features: {result.shape[0]} windows x {result.shape[1]} columns")
    
    return result
