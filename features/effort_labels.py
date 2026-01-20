"""Effort label construction from RMSSD windows using recovery-based strategy."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EffortLabelGenerator:
    """
    Generate physiological effort labels from windowed RMSSD features.
    
    Strategy: RMSSD Recovery
    - Lower RMSSD = higher effort (reduced vagal tone under stress)
    - Higher RMSSD = lower effort/rest (better vagal tone at rest)
    - Compare each window to baseline to determine effort state
    
    Assumptions:
    - First N windows are baseline (at rest, before effort begins)
    - RMSSD drops significantly during effort periods
    - Recovery periods show RMSSD increasing back toward baseline
    """
    
    def __init__(self, 
                 baseline_window_count: int = 3,
                 recovery_threshold_pct: float = -20,
                 verbose: bool = False):
        """
        Initialize effort label generator.
        
        Args:
            baseline_window_count: Number of initial windows to use as baseline (rest state)
            recovery_threshold_pct: RMSSD drop threshold for "effort" label (%)
                                   -20 means RMSSD must drop >20% below baseline to be effort
            verbose: Enable debug logging
        """
        self.baseline_window_count = baseline_window_count
        self.recovery_threshold_pct = recovery_threshold_pct
        self.verbose = verbose
        self.baseline_rmssd_mean = None
        self.baseline_rmssd_std = None
        
    def compute_baseline(self, df_rmssd: pd.DataFrame) -> Dict:
        """
        Compute baseline RMSSD from initial windows.
        
        Args:
            df_rmssd: RMSSD windows DataFrame
            
        Returns:
            baseline_dict: Contains mean, std, n_windows
        """
        # Take first N windows as baseline
        baseline_windows = df_rmssd.head(self.baseline_window_count)
        
        # Use only valid RMSSD values
        valid_rmssd = baseline_windows['rmssd'].dropna()
        
        if len(valid_rmssd) == 0:
            logger.warning("No valid RMSSD in baseline windows")
            return {
                'mean': np.nan,
                'std': np.nan,
                'n_windows': 0,
                'valid_n': 0
            }
        
        self.baseline_rmssd_mean = valid_rmssd.mean()
        self.baseline_rmssd_std = valid_rmssd.std()
        
        baseline_dict = {
            'mean': self.baseline_rmssd_mean,
            'std': self.baseline_rmssd_std,
            'n_windows': len(baseline_windows),
            'valid_n': len(valid_rmssd)
        }
        
        if self.verbose:
            logger.info(f"Baseline RMSSD: {self.baseline_rmssd_mean:.1f} ± {self.baseline_rmssd_std:.1f} ms "
                       f"(n={len(valid_rmssd)} windows)")
        
        return baseline_dict
    
    def generate_labels(self, df_rmssd: pd.DataFrame, 
                       strategy: str = 'rmssd_recovery') -> pd.DataFrame:
        """
        Generate effort labels for all RMSSD windows.
        
        Args:
            df_rmssd: RMSSD windows DataFrame (from Stage 2)
            strategy: Label strategy
                     - 'rmssd_recovery': RMSSD drop below baseline threshold
                     - 'rmssd_zscore': Z-score based (|z| > 1.5 = effort)
                     
        Returns:
            df_labels: Input dataframe with added columns:
                      - effort_binary: 0=Rest, 1=Effort
                      - effort_confidence: 0-1 confidence score
                      - effort_reason: Text description
                      - rmssd_pct_change: % change from baseline
        """
        df_out = df_rmssd.copy()
        
        # Compute baseline if not already done
        if self.baseline_rmssd_mean is None:
            self.compute_baseline(df_rmssd)
        
        # Initialize output columns
        df_out['effort_binary'] = 0
        df_out['effort_confidence'] = 0.0
        df_out['effort_reason'] = ''
        df_out['rmssd_pct_change'] = np.nan
        
        if np.isnan(self.baseline_rmssd_mean):
            logger.error("Cannot compute labels: baseline is NaN")
            return df_out
        
        # Apply strategy
        if strategy == 'rmssd_recovery':
            df_out = self._label_rmssd_recovery(df_out)
        elif strategy == 'rmssd_zscore':
            df_out = self._label_rmssd_zscore(df_out)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return df_out
    
    def _label_rmssd_recovery(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label effort using RMSSD recovery strategy.
        
        Effort = RMSSD drop > threshold below baseline
        Confidence = magnitude of drop / typical variability
        """
        for idx, row in df.iterrows():
            rmssd = row['rmssd']
            
            # Skip NaN RMSSD
            if pd.isna(rmssd):
                df.at[idx, 'effort_binary'] = 0
                df.at[idx, 'effort_confidence'] = 0.0
                df.at[idx, 'effort_reason'] = 'NaN RMSSD'
                continue
            
            # Compute % change from baseline
            pct_change = (rmssd - self.baseline_rmssd_mean) / self.baseline_rmssd_mean * 100
            df.at[idx, 'rmssd_pct_change'] = pct_change
            
            # Threshold-based labeling
            if pct_change <= self.recovery_threshold_pct:
                # RMSSD dropped below threshold → EFFORT
                df.at[idx, 'effort_binary'] = 1
                
                # Confidence: how far below baseline?
                # Scale: 0-1 range based on std deviations below baseline
                confidence = min(1.0, abs(pct_change) / 40.0)  # 40% drop = full confidence
                df.at[idx, 'effort_confidence'] = confidence
                df.at[idx, 'effort_reason'] = f'RMSSD drop {pct_change:.1f}%'
            else:
                # RMSSD above threshold → REST
                df.at[idx, 'effort_binary'] = 0
                
                # Confidence: how much above threshold?
                confidence = min(1.0, (pct_change - self.recovery_threshold_pct) / 20.0)
                df.at[idx, 'effort_confidence'] = confidence
                df.at[idx, 'effort_reason'] = f'RMSSD stable {pct_change:.1f}%'
        
        return df
    
    def _label_rmssd_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label effort using Z-score strategy.
        
        Effort = Z-score < -1.5 (significantly below baseline)
        Rest = Z-score > -0.5 (near or above baseline)
        """
        for idx, row in df.iterrows():
            rmssd = row['rmssd']
            
            if pd.isna(rmssd):
                df.at[idx, 'effort_binary'] = 0
                df.at[idx, 'effort_confidence'] = 0.0
                df.at[idx, 'effort_reason'] = 'NaN RMSSD'
                continue
            
            # Compute Z-score
            if self.baseline_rmssd_std > 0:
                z_score = (rmssd - self.baseline_rmssd_mean) / self.baseline_rmssd_std
            else:
                z_score = 0
            
            pct_change = (rmssd - self.baseline_rmssd_mean) / self.baseline_rmssd_mean * 100
            df.at[idx, 'rmssd_pct_change'] = pct_change
            
            if z_score < -1.5:
                df.at[idx, 'effort_binary'] = 1
                df.at[idx, 'effort_confidence'] = min(1.0, abs(z_score) / 3.0)
                df.at[idx, 'effort_reason'] = f'Z-score {z_score:.2f}'
            elif z_score > -0.5:
                df.at[idx, 'effort_binary'] = 0
                df.at[idx, 'effort_confidence'] = min(1.0, z_score / 2.0)
                df.at[idx, 'effort_reason'] = f'Z-score {z_score:.2f}'
            else:
                # Ambiguous zone: weak effort signal
                df.at[idx, 'effort_binary'] = 0
                df.at[idx, 'effort_confidence'] = 0.5
                df.at[idx, 'effort_reason'] = f'Ambiguous Z-score {z_score:.2f}'
        
        return df


def load_rmssd_windows(csv_path: str) -> pd.DataFrame:
    """Load RMSSD windows from CSV."""
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded RMSSD windows: {len(df)} windows from {csv_path}")
    return df


def save_effort_labels(df_labels: pd.DataFrame, csv_path: str) -> None:
    """Save effort labels to CSV."""
    df_labels.to_csv(csv_path, index=False)
    logger.info(f"Saved effort labels: {csv_path}")
