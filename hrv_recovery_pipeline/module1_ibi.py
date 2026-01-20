"""
Module 1: PPG → IBI (Inter-beat intervals)

Extract robust inter-beat intervals from green PPG signal.
Detect pulse peaks, compute IBIs, apply validity filters.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, medfilt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def preprocess_ppg_for_peaks(
    signal: np.ndarray,
    fs: float = 128.0,
) -> np.ndarray:
    """
    Preprocess raw PPG signal for robust peak detection.
    
    Steps:
    1. Remove outliers
    2. Detrend to remove DC drift
    3. Bandpass filter (0.5-4 Hz for cardiac range)
    4. Normalize to unit variance
    
    Args:
        signal: Raw PPG signal
        fs: Sampling frequency (Hz)
        
    Returns:
        processed_signal: Ready for peak detection
    """
    if len(signal) < 3 * fs:
        logger.warning("Signal too short for preprocessing")
        return signal
    
    # Step 1: Remove extreme outliers (clipping)
    p5, p95 = np.percentile(signal, [5, 95])
    signal_clipped = np.clip(signal, p5, p95)
    
    # Step 2: Detrend (remove slow baseline wander)
    # Use median filtering to estimate baseline, then subtract
    from scipy.signal import medfilt
    window_size = int(fs * 0.6)  # 0.6s median filter
    if window_size % 2 == 0:
        window_size += 1
    baseline = medfilt(signal_clipped, kernel_size=window_size)
    signal_detrended = signal_clipped - baseline
    
    # Step 3: Bandpass filter (0.5-4 Hz = 30-240 bpm)
    lowcut = 0.5
    highcut = 4.0
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Ensure valid range
    low = np.clip(low, 0.01, 0.98)
    high = np.clip(high, low + 0.01, 0.99)
    
    try:
        b, a = butter(4, [low, high], btype='band')
        signal_filtered = filtfilt(b, a, signal_detrended)
    except Exception as e:
        logger.error(f"Bandpass filtering failed: {e}")
        signal_filtered = signal_detrended
    
    # Step 4: Normalize to unit variance (makes prominence threshold stable)
    std = np.std(signal_filtered)
    if std > 1e-6:
        signal_normalized = signal_filtered / std
    else:
        signal_normalized = signal_filtered
    
    return signal_normalized


def detect_ppg_peaks(
    signal: np.ndarray,
    fs: float = 32.0,
    distance_ms: int = 400,
    prominence_percentile: int = 80,
    preprocess: bool = True,
) -> np.ndarray:
    """
    Detect heartbeat peaks in PPG signal.
    
    Args:
        signal: 1D PPG waveform
        fs: Sampling frequency (Hz)
        distance_ms: Minimum distance between peaks (milliseconds)
        prominence_percentile: Use this percentile of signal range for prominence (ignored if preprocessed)
        preprocess: Apply preprocessing before peak detection (recommended: True)
        
    Returns:
        peak_indices: Indices of detected peaks (1D array)
    """
    if len(signal) < 2 * fs:
        logger.warning(f"Signal too short ({len(signal)} samples), returning empty peaks")
        return np.array([], dtype=int)
    
    # Preprocess: robust detrending + filtering + normalization
    if preprocess:
        signal_processed = preprocess_ppg_for_peaks(signal, fs=fs)
        # After normalization, use fixed prominence (in std units)
        prominence = 0.5  # 0.5 standard deviations
    else:
        # Invert so peaks are high values (typical PPG has peaks as dips)
        signal_processed = -signal
        # Use percentage-based prominence
        signal_range = np.nanmax(signal_processed) - np.nanmin(signal_processed)
        prominence = (signal_range * prominence_percentile / 100.0) if signal_range > 0 else 0.01
    
    # Set distance threshold (samples)
    distance_samples = int(distance_ms * fs / 1000.0)
    
    try:
        peaks, _ = find_peaks(
            signal_processed,
            distance=distance_samples,
            prominence=prominence,
        )
    except Exception as e:
        logger.error(f"Peak detection failed: {e}")
        return np.array([], dtype=int)
    
    return peaks


def compute_ibi_from_peaks(
    peak_times: np.ndarray,
    min_ibi_sec: float = 0.3,
    max_ibi_sec: float = 2.0,
    max_ibi_ratio: float = 1.5,
) -> tuple:
    """
    Convert peak times to inter-beat intervals (IBIs) with validity filtering.
    
    Args:
        peak_times: Timestamps of detected peaks (seconds)
        min_ibi_sec: Minimum plausible IBI (seconds)
        max_ibi_sec: Maximum plausible IBI (seconds)
        max_ibi_ratio: Maximum ratio between consecutive IBIs (reject erratic beats)
        
    Returns:
        ibi_times: Midpoint times of valid IBIs (seconds)
        ibi_values: Valid IBI durations (seconds)
    """
    if len(peak_times) < 2:
        return np.array([]), np.array([])
    
    # Compute intervals between peaks
    peak_diffs = np.diff(peak_times)
    
    # Mask: basic range filter
    valid_mask = (peak_diffs >= min_ibi_sec) & (peak_diffs <= max_ibi_sec)
    
    if not valid_mask.any():
        logger.warning("No valid IBIs after range filtering")
        return np.array([]), np.array([])
    
    # Additional: reject sudden jumps
    if max_ibi_ratio > 0:
        # Ratio test: compare each interval to its neighbors
        ratio_mask = np.ones(len(peak_diffs), dtype=bool)
        for i in range(len(peak_diffs)):
            if i > 0 and valid_mask[i - 1]:
                ratio = peak_diffs[i] / peak_diffs[i - 1]
                if ratio > max_ibi_ratio or ratio < (1.0 / max_ibi_ratio):
                    ratio_mask[i] = False
            if i < len(peak_diffs) - 1 and valid_mask[i + 1]:
                ratio = peak_diffs[i] / peak_diffs[i + 1]
                if ratio > max_ibi_ratio or ratio < (1.0 / max_ibi_ratio):
                    ratio_mask[i] = False
        
        valid_mask = valid_mask & ratio_mask
    
    if not valid_mask.any():
        logger.warning("No valid IBIs after ratio filtering")
        return np.array([]), np.array([])
    
    # Midpoint times for valid IBIs
    ibi_midpoints = (peak_times[:-1] + peak_times[1:]) / 2.0
    ibi_times = ibi_midpoints[valid_mask]
    ibi_values = peak_diffs[valid_mask]
    
    return ibi_times, ibi_values


def extract_ibi_timeseries(
    ppg_df: pd.DataFrame,
    time_col: str = "t_sec",
    value_col: str = "value",
    fs: float = 32.0,
    distance_ms: int = 400,
    prominence_percentile: int = 80,
    min_ibi_sec: float = 0.3,
    max_ibi_sec: float = 2.0,
    max_ibi_ratio: float = 1.5,
) -> pd.DataFrame:
    """
    End-to-end: PPG signal → IBI timeseries.
    
    Args:
        ppg_df: DataFrame with time_col and value_col
        time_col: Name of timestamp column
        value_col: Name of PPG signal column
        fs: Sampling frequency
        distance_ms: Peak detection distance (ms)
        prominence_percentile: Peak detection prominence
        min_ibi_sec, max_ibi_sec, max_ibi_ratio: IBI validity thresholds
        
    Returns:
        ibi_df: DataFrame with columns [t, ibi_sec]
    """
    if ppg_df.empty or value_col not in ppg_df.columns:
        logger.warning("Empty PPG data")
        return pd.DataFrame(columns=["t", "ibi_sec"])
    
    signal = ppg_df[value_col].values
    times = ppg_df[time_col].values
    
    # Detect peaks
    peak_indices = detect_ppg_peaks(
        signal,
        fs=fs,
        distance_ms=distance_ms,
        prominence_percentile=prominence_percentile,
    )
    
    if len(peak_indices) < 2:
        logger.warning(f"Too few peaks detected ({len(peak_indices)})")
        return pd.DataFrame(columns=["t", "ibi_sec"])
    
    peak_times = times[peak_indices]
    
    # Compute IBIs
    ibi_times, ibi_values = compute_ibi_from_peaks(
        peak_times,
        min_ibi_sec=min_ibi_sec,
        max_ibi_sec=max_ibi_sec,
        max_ibi_ratio=max_ibi_ratio,
    )
    
    if len(ibi_times) == 0:
        logger.warning("No valid IBIs extracted")
        return pd.DataFrame(columns=["t", "ibi_sec"])
    
    ibi_df = pd.DataFrame({
        "t": ibi_times,
        "ibi_sec": ibi_values
    })
    
    logger.info(
        f"Extracted {len(ibi_df)} valid IBIs from {len(peak_indices)} peaks "
        f"(HR mean: {60.0 / ibi_values.mean():.1f} bpm)"
    )
    
    return ibi_df


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    if len(sys.argv) > 1:
        ppg_path = sys.argv[1]
        ppg_df = pd.read_csv(ppg_path)
        ibi_df = extract_ibi_timeseries(ppg_df)
        print(ibi_df.head(10))
        print(f"Total IBIs: {len(ibi_df)}")
