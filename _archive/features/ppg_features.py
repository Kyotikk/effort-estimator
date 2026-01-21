"""
PPG feature extraction - HEART RATE LEVEL ONLY (NO HRV).

Computes heart rate-based features from PPG signal that reflect cardiovascular load:
- Mean HR, max HR, min HR, std HR
- HR slope over time
- Signal quality metrics

CRITICAL: NO HRV FEATURES (no RMSSD, SDNN, pNN50, LF/HF, or any RR-based variability).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from scipy.signal import find_peaks, butter, filtfilt
import logging

logger = logging.getLogger(__name__)


class PPGFeatureExtractor:
    """
    Extract heart rate level features from PPG signal.
    
    NO HRV FEATURES - only instantaneous heart rate metrics.
    """
    
    def __init__(self, sampling_rate: float = 64.0):
        """
        Args:
            sampling_rate: PPG sampling frequency (Hz)
        """
        self.sampling_rate = sampling_rate
    
    def bandpass_filter(self, ppg_signal: np.ndarray, 
                       lowcut: float = 0.5, 
                       highcut: float = 8.0) -> np.ndarray:
        """
        Bandpass filter PPG for pulse detection.
        
        Args:
            ppg_signal: Raw PPG signal
            lowcut: Lower cutoff (Hz)
            highcut: Upper cutoff (Hz)
            
        Returns:
            Filtered PPG signal
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = butter(3, [low, high], btype='band')
        filtered = filtfilt(b, a, ppg_signal)
        
        return filtered
    
    def detect_ppg_peaks(self, ppg_signal: np.ndarray) -> np.ndarray:
        """
        Detect PPG pulse peaks for instantaneous HR calculation.
        
        Args:
            ppg_signal: Filtered PPG signal
            
        Returns:
            Array of peak indices
        """
        # Find peaks with minimum distance constraint (max ~200 bpm)
        min_distance = int(self.sampling_rate * 0.3)  # 300ms minimum
        
        # Adaptive threshold
        threshold = 0.3 * (np.max(ppg_signal) - np.min(ppg_signal)) + np.min(ppg_signal)
        
        peaks, _ = find_peaks(ppg_signal, 
                            distance=min_distance,
                            height=threshold)
        
        logger.debug(f"Detected {len(peaks)} PPG peaks")
        return peaks
    
    def compute_instantaneous_hr(self, peak_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute instantaneous heart rate from peak-to-peak intervals.
        
        Args:
            peak_indices: Indices of PPG peaks
            
        Returns:
            hr_values: Instantaneous HR in bpm
            hr_times: Time stamps for each HR value (seconds)
        """
        if len(peak_indices) < 2:
            return np.array([]), np.array([])
        
        # Compute peak-to-peak intervals
        pp_intervals = np.diff(peak_indices) / self.sampling_rate  # seconds
        
        # Convert to HR (beats per minute)
        hr_values = 60.0 / pp_intervals
        
        # Clean physiologically impossible values
        valid_mask = (hr_values >= 30) & (hr_values <= 220)
        hr_values = hr_values[valid_mask]
        
        # Time stamps (at midpoint between peaks)
        hr_times = (peak_indices[:-1][valid_mask] + peak_indices[1:][valid_mask]) / (2 * self.sampling_rate)
        
        return hr_values, hr_times
    
    def extract_features(self, ppg_signal: np.ndarray, 
                        window_start: float = 0.0,
                        window_end: Optional[float] = None) -> Dict[str, float]:
        """
        Extract heart rate level features from PPG segment.
        
        NO HRV FEATURES - only HR statistics and trends.
        
        Args:
            ppg_signal: Raw PPG signal
            window_start: Start time of window (seconds)
            window_end: End time of window (seconds, None = full signal)
            
        Returns:
            Dictionary of features
        """
        if window_end is None:
            window_end = len(ppg_signal) / self.sampling_rate
        
        # Filter PPG
        filtered_ppg = self.bandpass_filter(ppg_signal)
        
        # Detect peaks
        peaks = self.detect_ppg_peaks(filtered_ppg)
        
        # Compute instantaneous HR
        hr_values, hr_times = self.compute_instantaneous_hr(peaks)
        
        features = {}
        
        if len(hr_values) == 0:
            logger.warning(f"No valid HR values in window [{window_start:.1f}, {window_end:.1f}]")
            # Return NaN features
            for key in ['ppg_hr_mean', 'ppg_hr_max', 'ppg_hr_min', 'ppg_hr_std', 
                       'ppg_hr_slope', 'ppg_hr_range', 'ppg_signal_quality']:
                features[key] = np.nan
            features['ppg_n_beats'] = 0
            return features
        
        # HR statistics
        features['ppg_hr_mean'] = np.mean(hr_values)
        features['ppg_hr_max'] = np.max(hr_values)
        features['ppg_hr_min'] = np.min(hr_values)
        features['ppg_hr_std'] = np.std(hr_values)
        features['ppg_hr_range'] = features['ppg_hr_max'] - features['ppg_hr_min']
        features['ppg_n_beats'] = len(hr_values)
        
        # HR trend (slope over time)
        if len(hr_values) >= 3:
            hr_slope = np.polyfit(hr_times, hr_values, 1)[0]
            features['ppg_hr_slope'] = hr_slope  # bpm per second
        else:
            features['ppg_hr_slope'] = 0.0
        
        # Signal quality indicator (based on peak detection consistency)
        expected_n_beats = (window_end - window_start) * (features['ppg_hr_mean'] / 60.0)
        features['ppg_signal_quality'] = len(peaks) / max(expected_n_beats, 1)
        
        logger.debug(f"PPG features: HR {features['ppg_hr_mean']:.1f} bpm "
                    f"(range {features['ppg_hr_min']:.0f}-{features['ppg_hr_max']:.0f}), "
                    f"slope {features['ppg_hr_slope']:.2f} bpm/s")
        
        return features
    
    def extract_features_from_dataframe(self, df: pd.DataFrame,
                                       value_col: str = 'value',
                                       time_col: str = 'timestamp') -> Dict[str, float]:
        """
        Extract features from a DataFrame with PPG data.
        
        Args:
            df: DataFrame with PPG values
            value_col: Name of column containing PPG values
            time_col: Name of column containing timestamps
            
        Returns:
            Dictionary of features
        """
        ppg_signal = df[value_col].values
        
        # Determine time window
        if time_col in df.columns:
            timestamps = pd.to_datetime(df[time_col])
            window_start = 0.0
            window_end = (timestamps.max() - timestamps.min()).total_seconds()
        else:
            window_start = 0.0
            window_end = None
        
        return self.extract_features(ppg_signal, window_start, window_end)


def load_corsano_ppg(filepath: str) -> pd.DataFrame:
    """
    Load PPG data from Corsano sensor CSV/gzipped CSV.
    
    Args:
        filepath: Path to PPG CSV file (e.g., corsano_wrist_ppg2_green_6/2025-12-04.csv.gz)
        
    Returns:
        DataFrame with PPG data
    """
    logger.info(f"Loading PPG from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} PPG samples with columns: {df.columns.tolist()}")
    return df
