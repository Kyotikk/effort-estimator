"""
ECG preprocessing for R-peak detection and RR interval extraction.
Used ONLY for computing effort labels (RMSSD-based), never as input features.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ECGPreprocessor:
    """
    Processes raw ECG signal to extract clean RR intervals.
    """
    
    def __init__(self, sampling_rate: float = 256.0):
        """
        Args:
            sampling_rate: ECG sampling frequency in Hz
        """
        self.sampling_rate = sampling_rate
        
    def bandpass_filter(self, ecg_signal: np.ndarray, lowcut: float = 0.5, 
                       highcut: float = 40.0) -> np.ndarray:
        """
        Apply bandpass filter to remove baseline wander and high-frequency noise.
        
        Args:
            ecg_signal: Raw ECG signal
            lowcut: Lower cutoff frequency (Hz)
            highcut: Upper cutoff frequency (Hz)
            
        Returns:
            Filtered ECG signal
        """
        from scipy.signal import butter, filtfilt
        
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, ecg_signal)
        
        return filtered_signal
    
    def detect_r_peaks(self, ecg_signal: np.ndarray, 
                      min_distance_ms: float = 300.0) -> np.ndarray:
        """
        Detect R-peaks using Pan-Tompkins inspired algorithm.
        
        Args:
            ecg_signal: Filtered ECG signal
            min_distance_ms: Minimum distance between peaks (ms)
            
        Returns:
            Array of R-peak indices
        """
        from scipy.signal import find_peaks
        
        # Differentiate to emphasize QRS slopes
        diff_signal = np.diff(ecg_signal)
        diff_signal = np.concatenate([[0], diff_signal])  # Pad to maintain length
        
        # Square the signal to emphasize peaks
        squared = diff_signal ** 2
        
        # Moving average integration (wider window for better smoothing)
        window_size = int(0.08 * self.sampling_rate)  # 80ms window
        integrated = np.convolve(squared, np.ones(window_size) / window_size, mode='same')
        
        # Find peaks with minimum distance constraint
        min_distance_samples = int((min_distance_ms / 1000.0) * self.sampling_rate)
        
        # Adaptive threshold: use percentile-based approach (more robust)
        threshold = 0.3 * np.percentile(integrated, 99)
        
        peaks, properties = find_peaks(integrated, 
                                      distance=min_distance_samples,
                                      height=threshold,
                                      prominence=threshold * 0.3)
        
        logger.info(f"Detected {len(peaks)} R-peaks (threshold: {threshold:.2f})")
        
        return peaks
    
    def compute_rr_intervals(self, r_peaks: np.ndarray) -> np.ndarray:
        """
        Compute RR intervals from R-peak indices.
        
        Args:
            r_peaks: Array of R-peak sample indices
            
        Returns:
            RR intervals in milliseconds
        """
        if len(r_peaks) < 2:
            return np.array([])
        
        rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000.0  # Convert to ms
        return rr_intervals
    
    def clean_rr_intervals(self, rr_intervals: np.ndarray, 
                          min_rr: float = 300.0, 
                          max_rr: float = 2000.0,
                          max_diff_percent: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove ectopic beats and artifacts from RR intervals.
        
        Args:
            rr_intervals: Raw RR intervals (ms)
            min_rr: Minimum valid RR interval (ms)
            max_rr: Maximum valid RR interval (ms)
            max_diff_percent: Maximum percent change between consecutive RRs
            
        Returns:
            clean_rr: Cleaned RR intervals
            valid_mask: Boolean mask of valid intervals
        """
        if len(rr_intervals) == 0:
            return np.array([]), np.array([])
        
        valid_mask = np.ones(len(rr_intervals), dtype=bool)
        
        # Rule 1: Physiological limits
        valid_mask &= (rr_intervals >= min_rr) & (rr_intervals <= max_rr)
        
        # Rule 2: Successive difference filter
        if len(rr_intervals) > 1:
            rr_diff = np.abs(np.diff(rr_intervals))
            rr_percent_change = (rr_diff / rr_intervals[:-1]) * 100.0
            
            # Mark intervals where change is too large
            outlier_mask = np.zeros(len(rr_intervals), dtype=bool)
            outlier_mask[:-1] |= (rr_percent_change > max_diff_percent)
            outlier_mask[1:] |= (rr_percent_change > max_diff_percent)
            
            valid_mask &= ~outlier_mask
        
        clean_rr = rr_intervals[valid_mask]
        
        logger.info(f"Cleaned RR intervals: {len(clean_rr)}/{len(rr_intervals)} valid "
                   f"({100 * len(clean_rr) / len(rr_intervals):.1f}%)")
        
        return clean_rr, valid_mask
    
    def process_ecg(self, ecg_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete ECG processing pipeline.
        
        Args:
            ecg_signal: Raw ECG signal
            
        Returns:
            r_peaks: R-peak indices
            rr_intervals: Clean RR intervals (ms)
            valid_mask: Boolean mask for valid RR intervals
        """
        # Filter
        filtered_ecg = self.bandpass_filter(ecg_signal)
        
        # Detect R-peaks
        r_peaks = self.detect_r_peaks(filtered_ecg)
        
        # Compute RR intervals
        rr_intervals = self.compute_rr_intervals(r_peaks)
        
        # Clean RR intervals
        clean_rr, valid_mask = self.clean_rr_intervals(rr_intervals)
        
        return r_peaks, clean_rr, valid_mask


def load_vivalnk_ecg(filepath: str) -> Tuple[np.ndarray, float]:
    """
    Load ECG data from VivalNK sensor CSV/gzipped CSV.
    
    Args:
        filepath: Path to data_1.csv or data_1.csv.gz
        
    Returns:
        ecg_signal: ECG amplitude values
        sampling_rate: Detected or assumed sampling rate
    """
    logger.info(f"Loading ECG from {filepath}")
    
    # Read CSV (handles .gz automatically)
    df = pd.read_csv(filepath)
    
    logger.info(f"Loaded {len(df)} samples with columns: {df.columns.tolist()}")
    
    # Identify ECG column (adapt based on actual column name)
    ecg_col = None
    for col in ['ecg', 'ECG', 'value', 'amplitude', 'signal']:
        if col in df.columns:
            ecg_col = col
            break
    
    if ecg_col is None:
        raise ValueError(f"Could not identify ECG column in {filepath}. "
                        f"Available columns: {df.columns.tolist()}")
    
    ecg_signal = df[ecg_col].values
    
    # Estimate sampling rate if timestamp column exists
    sampling_rate = 256.0  # Default for VivalNK
    if 'timestamp' in df.columns or 'time' in df.columns:
        time_col = 'timestamp' if 'timestamp' in df.columns else 'time'
        timestamps = pd.to_datetime(df[time_col])
        time_diffs = timestamps.diff().dt.total_seconds().dropna()
        if len(time_diffs) > 0:
            median_dt = time_diffs.median()
            if median_dt > 0:
                sampling_rate = 1.0 / median_dt
                logger.info(f"Detected sampling rate: {sampling_rate:.2f} Hz")
    
    return ecg_signal, sampling_rate
