"""
Module 2: IBI → Windowed RMSSD time series

Compute RMSSD (root mean square of successive differences) in sliding windows.
Each window includes IBIs with timestamps inside [t_start, t_end].
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def compute_rmssd(ibi_values: np.ndarray) -> float:
    """
    Compute RMSSD: root mean square of successive differences of IBIs.
    
    Args:
        ibi_values: IBI durations (seconds, 1D array)
        
    Returns:
        rmssd: RMSSD in seconds (or NaN if insufficient data)
    """
    if len(ibi_values) < 2:
        return np.nan
    
    # Successive differences
    diffs = np.diff(ibi_values)
    
    # RMSSD
    rmssd = np.sqrt(np.mean(diffs ** 2))
    return rmssd


def compute_rmssd_windows(
    ibi_df: pd.DataFrame,
    window_len_sec: float = 60.0,
    step_sec: float = 10.0,
    min_beats: int = 10,
    t_start_offset: float = 0.0,
) -> pd.DataFrame:
    """
    Compute RMSSD in sliding windows over IBI timeseries.
    
    Args:
        ibi_df: DataFrame with columns [t, ibi_sec]
        window_len_sec: Window length (seconds)
        step_sec: Step between window starts (seconds)
        min_beats: Minimum IBIs required per window; otherwise rmssd = NaN
        t_start_offset: Offset for first window start (default: min IBI time)
        
    Returns:
        rmssd_df: DataFrame with columns [t_start, t_center, t_end, rmssd]
    """
    if ibi_df.empty:
        logger.warning("Empty IBI data")
        return pd.DataFrame(columns=["t_start", "t_center", "t_end", "rmssd"])
    
    ibi_times = ibi_df["t"].values
    ibi_values = ibi_df["ibi_sec"].values
    
    t_min = ibi_times.min()
    t_max = ibi_times.max()
    
    # Set first window start
    if t_start_offset <= 0:
        t_start_offset = t_min
    
    # Create windows
    windows = []
    t_start = t_start_offset
    
    while t_start < t_max:
        t_end = t_start + window_len_sec
        
        # Find IBIs within this window
        mask = (ibi_times >= t_start) & (ibi_times < t_end)
        window_ibi = ibi_values[mask]
        
        if len(window_ibi) >= min_beats:
            rmssd = compute_rmssd(window_ibi)
        else:
            rmssd = np.nan
        
        t_center = (t_start + t_end) / 2.0
        
        windows.append({
            "t_start": t_start,
            "t_center": t_center,
            "t_end": t_end,
            "rmssd": rmssd,
            "n_beats": len(window_ibi),
        })
        
        t_start += step_sec
    
    rmssd_df = pd.DataFrame(windows)
    
    n_valid = rmssd_df["rmssd"].notna().sum()
    logger.info(
        f"Computed {len(rmssd_df)} RMSSD windows ({n_valid} valid, "
        f"{window_len_sec}s window, {step_sec}s step)"
    )
    
    return rmssd_df[["t_start", "t_center", "t_end", "rmssd"]]


def get_rmssd_in_interval(
    rmssd_df: pd.DataFrame,
    t_start: float,
    t_end: float,
    method: str = "mean",
) -> float:
    """
    Aggregate RMSSD values for windows within a time interval.
    
    Args:
        rmssd_df: RMSSD windows dataframe
        t_start, t_end: Time interval (seconds)
        method: "mean", "median", or "last"
        
    Returns:
        agg_rmssd: Aggregated RMSSD value (or NaN)
    """
    if rmssd_df.empty:
        return np.nan
    
    # Windows whose centers fall within interval
    mask = (rmssd_df["t_center"] >= t_start) & (rmssd_df["t_center"] < t_end)
    interval_rmssd = rmssd_df.loc[mask, "rmssd"]
    
    # Keep only valid (non-NaN) values
    interval_rmssd = interval_rmssd.dropna()
    
    if len(interval_rmssd) == 0:
        return np.nan
    
    if method == "mean":
        return interval_rmssd.mean()
    elif method == "median":
        return interval_rmssd.median()
    elif method == "last":
        return interval_rmssd.iloc[-1]
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    if len(sys.argv) > 1:
        ibi_path = sys.argv[1]
        ibi_df = pd.read_csv(ibi_path)
        rmssd_df = compute_rmssd_windows(ibi_df, window_len_sec=60.0, step_sec=10.0)
        print(rmssd_df.head(10))
        print(f"Valid RMSSD windows: {rmssd_df['rmssd'].notna().sum()}")
