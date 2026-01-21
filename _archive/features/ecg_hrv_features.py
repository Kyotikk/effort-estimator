"""ECG-derived HRV features: RMSSD and lnRMSSD windowing."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class RMSSDWindower:
    """
    Compute windowed RMSSD (Root Mean Square of Successive Differences) from RR intervals.
    
    RMSSD is a time-domain HRV measure reflecting parasympathetic activity.
    lnRMSSD is the natural log of RMSSD (normalized, reduces skewness).
    
    Windows are defined by time, not by RR interval count, to align with activity periods.
    """
    
    def __init__(self, window_length_s: float = 60, overlap_frac: float = 0.5, 
                 min_rr_per_window: int = 3, verbose: bool = False):
        """
        Initialize RMSSD windower.
        
        Args:
            window_length_s: window duration (seconds)
            overlap_frac: overlap fraction (0.0-1.0)
            min_rr_per_window: minimum valid RR intervals required to compute RMSSD
            verbose: enable debug logging
        """
        self.window_length_s = window_length_s
        self.overlap_frac = overlap_frac
        self.step_s = window_length_s * (1 - overlap_frac)
        self.min_rr_per_window = min_rr_per_window
        self.verbose = verbose
    
    def compute_rmssd(self, rr_intervals: np.ndarray) -> Tuple[float, int, float]:
        """
        Compute RMSSD from RR intervals.
        
        RMSSD = sqrt(mean((RR[i+1] - RR[i])^2))
        
        Args:
            rr_intervals: array of RR intervals (ms), only valid intervals
            
        Returns:
            (rmssd, n_rr, ln_rmssd)
            where:
            - rmssd is in ms
            - n_rr is number of valid RR intervals
            - ln_rmssd is log(rmssd) or NaN if rmssd=0
        """
        if len(rr_intervals) < 2:
            return np.nan, len(rr_intervals), np.nan
        
        # Successive differences
        diffs = np.diff(rr_intervals)
        
        # RMSSD
        rmssd = np.sqrt(np.mean(diffs ** 2))
        
        # lnRMSSD (natural log)
        ln_rmssd = np.log(rmssd) if rmssd > 0 else np.nan
        
        return rmssd, len(rr_intervals), ln_rmssd
    
    def create_windows(self, t_start: float, t_end: float) -> List[Tuple[float, float, int]]:
        """
        Create non-overlapping or overlapping time windows.
        
        Args:
            t_start: start time (seconds)
            t_end: end time (seconds)
            
        Returns:
            list of (window_start, window_end, window_id)
        """
        windows = []
        window_id = 0
        current_start = t_start
        
        while current_start + self.window_length_s <= t_end:
            window_end = current_start + self.window_length_s
            windows.append((current_start, window_end, window_id))
            window_id += 1
            current_start += self.step_s
        
        return windows
    
    def extract_window_rr(self, rr_df: pd.DataFrame, 
                         window_start: float, window_end: float,
                         use_valid_only: bool = True) -> np.ndarray:
        """
        Extract RR intervals within a time window.
        
        Args:
            rr_df: DataFrame with columns [t_rr, rr_ms, is_valid]
            window_start: window start time (seconds)
            window_end: window end time (seconds)
            use_valid_only: only include intervals marked is_valid=True
            
        Returns:
            array of RR intervals in window (ms)
        """
        mask = (rr_df['t_rr'] >= window_start) & (rr_df['t_rr'] < window_end)
        
        if use_valid_only:
            mask = mask & rr_df['is_valid']
        
        window_rr = rr_df.loc[mask, 'rr_ms'].values
        
        return window_rr
    
    def window_rmssd(self, rr_df: pd.DataFrame, session_id: str = None) -> pd.DataFrame:
        """
        Compute RMSSD for all windows in RR interval series.
        
        Args:
            rr_df: DataFrame with columns [t_rr, rr_ms, is_valid, ...]
            session_id: override session_id from df
            
        Returns:
            DataFrame with columns:
            [session_id, window_id, t_start, t_end, t_center, 
             n_rr_total, n_rr_valid, frac_valid, rmssd, ln_rmssd]
        """
        if len(rr_df) == 0:
            logger.warning("Empty RR DataFrame")
            return pd.DataFrame()
        
        if session_id is None:
            session_id = rr_df['session_id'].iloc[0] if 'session_id' in rr_df.columns else 'unknown'
        
        t_min = rr_df['t_rr'].min()
        t_max = rr_df['t_rr'].max()
        
        logger.info(f"Creating windows: {t_min:.1f}s - {t_max:.1f}s ({t_max - t_min:.1f}s duration)")
        windows = self.create_windows(t_min, t_max)
        logger.info(f"Created {len(windows)} windows (length={self.window_length_s}s, step={self.step_s}s)")
        
        results = []
        for w_start, w_end, w_id in windows:
            # Extract RR intervals
            window_rr = self.extract_window_rr(rr_df, w_start, w_end, use_valid_only=True)
            n_rr_valid = len(window_rr)
            
            # Also count total (including invalid) for quality
            mask_all = (rr_df['t_rr'] >= w_start) & (rr_df['t_rr'] < w_end)
            n_rr_total = mask_all.sum()
            frac_valid = n_rr_valid / n_rr_total if n_rr_total > 0 else 0
            
            # Compute RMSSD if enough valid RR intervals
            if n_rr_valid >= self.min_rr_per_window:
                rmssd, n_rr, ln_rmssd = self.compute_rmssd(window_rr)
            else:
                rmssd, n_rr, ln_rmssd = np.nan, n_rr_valid, np.nan
            
            results.append({
                'session_id': session_id,
                'window_id': w_id,
                't_start': w_start,
                't_end': w_end,
                't_center': (w_start + w_end) / 2,
                'n_rr_total': n_rr_total,
                'n_rr_valid': n_rr_valid,
                'frac_valid': frac_valid,
                'rmssd': rmssd,
                'ln_rmssd': ln_rmssd
            })
        
        df_windows = pd.DataFrame(results)
        logger.info(f"Windowing complete: {len(df_windows)} windows, {df_windows['rmssd'].notna().sum()} with valid RMSSD")
        
        return df_windows


def load_rr_intervals(filepath: str) -> pd.DataFrame:
    """Load RR intervals CSV."""
    df = pd.read_csv(filepath)
    logger.info(f"Loaded RR intervals: {len(df)} intervals")
    return df


def save_rmssd_windows(df_windows: pd.DataFrame, output_path: str):
    """Save windowed RMSSD to CSV."""
    df_windows.to_csv(output_path, index=False)
    logger.info(f"Saved RMSSD windows: {output_path}")
