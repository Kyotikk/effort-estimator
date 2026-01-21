"""ECG preprocessing: R-peak detection, RR interval extraction, artifact correction."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ECGProcessor:
    """
    Process raw ECG signals to extract clean RR intervals.
    
    Assumptions:
    - ECG is uniformly sampled at known sampling rate (e.g., 250 Hz, 500 Hz)
    - R-peaks are positive deflections in standard ECG leads
    - RR intervals should be in physiological range (300-2000 ms typical for adults)
    """
    
    def __init__(self, sampling_rate: float = 128, verbose: bool = False):
        """
        Initialize ECG processor.
        
        Args:
            sampling_rate: ECG sampling frequency (Hz) [default: 128 Hz typical for Vivalink]
            verbose: Enable debug logging
        """
        self.fs = sampling_rate
        self.verbose = verbose
        
    def detect_r_peaks(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks using bandpass filtering and simple peak finding.
        
        Strategy:
        1. Bandpass filter (5-40 Hz for QRS complex)
        2. Find positive peaks with adaptive threshold
        3. Remove duplicates within minimum distance
        
        Args:
            ecg_signal: 1D array of ECG samples
            
        Returns:
            peak_indices: indices of detected R-peaks
        """
        from scipy import signal
        
        # Design bandpass filter (5-40 Hz typical QRS band)
        nyquist = self.fs / 2.0
        try:
            sos = signal.butter(4, [5/nyquist, 40/nyquist], 'band', output='sos')
            filtered = signal.sosfilt(sos, ecg_signal)
        except Exception as e:
            logger.warning(f"Bandpass filter failed: {e}, using raw signal")
            filtered = ecg_signal
        
        # Find peaks with sensible threshold
        # Use height relative to signal std (not integrated)
        threshold = np.std(filtered) * 1.5
        
        # Minimum distance between peaks: for HR range 40-200 bpm
        # 40 bpm = 1500ms = 1500/1000*fs samples
        # 200 bpm = 300ms = 300/1000*fs samples
        # Use 400ms (6 bpm) as conservative minimum
        min_distance = int(0.4 * self.fs)
        
        peaks, props = signal.find_peaks(filtered, height=threshold, distance=min_distance)
        
        if self.verbose:
            logger.info(f"Detected {len(peaks)} R-peaks from {len(ecg_signal)} samples (threshold={threshold:.3f})")
        
        return peaks
    
    def peaks_to_rr_intervals(self, peak_indices: np.ndarray, time_array: Optional[np.ndarray] = None,
                              session_id: str = None) -> pd.DataFrame:
        """
        Convert R-peak indices to RR intervals (ms).
        
        Args:
            peak_indices: indices of R-peaks
            session_id: identifier for this recording session
            
        Returns:
            df with columns: [session_id, peak_index, t_rr, rr_ms]
            where t_rr is timestamp and rr_ms is RR interval to next peak
        """
        if len(peak_indices) < 2:
            logger.warning(f"Fewer than 2 peaks detected. Cannot compute RR intervals.")
            return pd.DataFrame(columns=['session_id', 'peak_index', 't_rr', 'rr_ms'])
        
        # Time of each peak (in seconds)
        t_peaks = peak_indices / self.fs
        if len(peak_indices) < 2:
            return pd.DataFrame(columns=['session_id', 'peak_index', 't_rr', 'rr_ms', 'is_valid', 'reason'])
        
        if time_array is not None:
            # Use actual timestamps to compute interval and midpoint times
            rr_intervals = np.diff(time_array[peak_indices]) * 1000.0  # ms
            t_rr = (time_array[peak_indices[:-1]] + time_array[peak_indices[1:]]) / 2.0
        else:
            # Fallback: use sample indices (relative seconds)
            rr_intervals = np.diff(peak_indices) / self.fs * 1000.0  # ms
            t_rr = (peak_indices[:-1] + peak_indices[1:]) / 2.0 / self.fs
        
        df = pd.DataFrame({
            'session_id': session_id,
            'peak_index': peak_indices[:-1],
            't_rr': t_rr,
            'rr_ms': rr_intervals,
            'is_valid': True,
            'reason': ''
        })
        return df
    def remove_artifacts(self, rr_df: pd.DataFrame, 
                        min_rr: float = 300, max_rr: float = 2000,
                        iqr_multiplier: float = 1.5) -> Tuple[pd.DataFrame, Dict]:
        """
        Remove ectopic beats and artifacts from RR intervals.
        
        Removal criteria:
        1. Physiological bounds (e.g., 300-2000 ms)
        2. IQR-based outliers (remove intervals > Q3 + k*IQR or < Q1 - k*IQR)
        3. Mark reasons for removal
        
        Args:
            rr_df: RR intervals DataFrame
            min_rr: minimum physiological RR interval (ms)
            max_rr: maximum physiological RR interval (ms)
            iqr_multiplier: outlier threshold for IQR method
            
        Returns:
            (clean_df, quality_dict)
        """
        n_input = len(rr_df)
        rr_df = rr_df.copy()
        rr_df['is_valid'] = True
        rr_df['reason'] = ''
        
        # Rule 1: Physiological bounds
        out_of_bounds = (rr_df['rr_ms'] < min_rr) | (rr_df['rr_ms'] > max_rr)
        rr_df.loc[out_of_bounds, 'is_valid'] = False
        rr_df.loc[out_of_bounds, 'reason'] = rr_df.loc[out_of_bounds].apply(
            lambda row: f"OOB({'low' if row['rr_ms'] < min_rr else 'high'})",
            axis=1
        )
        
        # Rule 2: IQR-based outliers (only on valid intervals so far)
        valid_mask = rr_df['is_valid']
        if valid_mask.sum() > 4:  # Need at least 5 points for IQR
            q1 = rr_df.loc[valid_mask, 'rr_ms'].quantile(0.25)
            q3 = rr_df.loc[valid_mask, 'rr_ms'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            
            is_outlier = (rr_df['rr_ms'] < lower_bound) | (rr_df['rr_ms'] > upper_bound)
            is_outlier = is_outlier & rr_df['is_valid']  # Only mark valid as outliers
            rr_df.loc[is_outlier, 'is_valid'] = False
            rr_df.loc[is_outlier, 'reason'] = 'IQR_outlier'
        
        n_removed = (~rr_df['is_valid']).sum()
        pct_removed = 100 * n_removed / n_input if n_input > 0 else 0
        
        quality = {
            'n_input': n_input,
            'n_valid': rr_df['is_valid'].sum(),
            'n_removed': n_removed,
            'pct_removed': pct_removed,
            'min_rr_kept': rr_df.loc[rr_df['is_valid'], 'rr_ms'].min() if rr_df['is_valid'].any() else np.nan,
            'max_rr_kept': rr_df.loc[rr_df['is_valid'], 'rr_ms'].max() if rr_df['is_valid'].any() else np.nan,
            'mean_rr_kept': rr_df.loc[rr_df['is_valid'], 'rr_ms'].mean() if rr_df['is_valid'].any() else np.nan,
            'std_rr_kept': rr_df.loc[rr_df['is_valid'], 'rr_ms'].std() if rr_df['is_valid'].any() else np.nan,
        }
        
        if self.verbose:
            logger.info(f"Artifact removal: {n_removed}/{n_input} intervals removed ({pct_removed:.1f}%)")
        
        return rr_df, quality
    
    def process_ecg(self, ecg_data: pd.DataFrame, 
                    ecg_column: str = 'ecg_value',
                    time_column: str = 'time',
                    session_id: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete pipeline: ECG → RR intervals with artifact correction.
        
        Args:
            ecg_data: DataFrame with ECG samples (must include ecg_column)
            ecg_column: name of column containing ECG values
            session_id: session identifier
            
        Returns:
            (rr_df, quality_summary)
        """
        logger.info(f"Processing ECG: {len(ecg_data)} samples at {self.fs} Hz")
        time_array = ecg_data[time_column].values if time_column in ecg_data.columns else None
        
        # Detect R-peaks
        peaks = self.detect_r_peaks(ecg_data[ecg_column].values)
        
        # Convert to RR intervals (use absolute timestamps if available)
        rr_df = self.peaks_to_rr_intervals(peaks, time_array=time_array, session_id=session_id)
        
        if len(rr_df) == 0:
            logger.error("No RR intervals extracted")
            return rr_df, {'error': 'no_peaks_detected', 'n_samples': len(ecg_data)}
        
        # Remove artifacts
        rr_clean, quality = self.remove_artifacts(rr_df)
        
        logger.info(f"RR processing complete: {quality['n_valid']} valid intervals")
        
        return rr_clean, quality


def load_ecg_csv(filepath: str, time_column: str = 't', 
                 ecg_column: str = 'ecg_value') -> pd.DataFrame:
    """
    Load ECG CSV with expected schema.
    
    Expected columns: time_column (seconds or unix timestamp), ecg_column (mV or normalized)
    
    Args:
        filepath: path to ECG CSV
        time_column: name of time column
        ecg_column: name of ECG signal column
        
    Returns:
        DataFrame with at least [time_column, ecg_column]
    """
    df = pd.read_csv(filepath)
    
    if time_column not in df.columns:
        raise ValueError(f"Column '{time_column}' not found. Available: {df.columns.tolist()}")
    if ecg_column not in df.columns:
        raise ValueError(f"Column '{ecg_column}' not found. Available: {df.columns.tolist()}")
    
    logger.info(f"Loaded ECG: {len(df)} samples, columns: {df.columns.tolist()}")
    return df


def save_rr_intervals(rr_df: pd.DataFrame, output_path: str):
    """Save RR intervals to CSV."""
    rr_df.to_csv(output_path, index=False)
    logger.info(f"Saved RR intervals: {output_path}")


def save_quality_summary(quality: Dict, output_path: str):
    """Save quality summary as JSON."""
    import json
    with open(output_path, 'w') as f:
        json.dump(quality, f, indent=2, default=str)
    logger.info(f"Saved quality summary: {output_path}")
