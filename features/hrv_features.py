#!/usr/bin/env python3
"""
HRV (Heart Rate Variability) feature extraction from PPG signals.

Implements:
1. PPG preprocessing (bandpass filter, normalization)
2. Systolic peak detection
3. IBI (inter-beat interval) extraction
4. HR metrics: mean, std, min, max, range
5. HRV metrics: RMSSD, SDNN, pNN50, LF/HF ratio
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d


# ============================================================================
# PPG Preprocessing for Peak Detection
# ============================================================================

def bandpass_filter(
    x: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 8.0,
    order: int = 4,
) -> np.ndarray:
    """
    Butterworth bandpass filter for PPG.
    
    - lowcut=0.5 Hz removes baseline drift
    - highcut=8 Hz removes high-frequency noise while preserving cardiac signal
    - Typical HR range: 40-200 bpm = 0.67-3.33 Hz
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Clamp to valid range
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    return signal.sosfiltfilt(sos, x)


def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Z-score normalization."""
    std = np.std(x)
    if std < 1e-10:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


# ============================================================================
# Peak Detection
# ============================================================================

def detect_ppg_peaks(
    x: np.ndarray,
    fs: float,
    min_hr: float = 40.0,
    max_hr: float = 200.0,
) -> np.ndarray:
    """
    Detect systolic peaks in PPG signal.
    
    Uses scipy.signal.find_peaks with physiologically-constrained parameters.
    
    Args:
        x: Filtered/normalized PPG signal
        fs: Sampling frequency (Hz)
        min_hr: Minimum expected heart rate (bpm)
        max_hr: Maximum expected heart rate (bpm)
    
    Returns:
        Array of peak indices
    """
    # Convert HR limits to sample distances
    min_distance = int(fs * 60.0 / max_hr)  # Samples between peaks at max HR
    max_distance = int(fs * 60.0 / min_hr)  # Samples between peaks at min HR
    
    min_distance = max(1, min_distance)
    
    # Find peaks with minimum distance constraint
    # Use prominence to distinguish true peaks from noise
    peaks, properties = signal.find_peaks(
        x,
        distance=min_distance,
        prominence=0.3 * np.std(x),  # Minimum prominence relative to signal variability
        height=None,  # No height constraint
    )
    
    if len(peaks) < 2:
        # Fallback: try with looser constraints
        peaks, _ = signal.find_peaks(
            x,
            distance=min_distance,
            prominence=0.1 * np.std(x),
        )
    
    return peaks


def compute_ibi(peaks: np.ndarray, fs: float) -> np.ndarray:
    """
    Compute Inter-Beat Intervals (IBI) from peak indices.
    
    Args:
        peaks: Array of peak indices
        fs: Sampling frequency
    
    Returns:
        Array of IBI values in milliseconds
    """
    if len(peaks) < 2:
        return np.array([])
    
    # Time differences between consecutive peaks
    ibi_samples = np.diff(peaks)
    ibi_ms = (ibi_samples / fs) * 1000.0  # Convert to milliseconds
    
    return ibi_ms


def clean_ibi(
    ibi: np.ndarray,
    min_ibi: float = 300.0,  # ms (200 bpm)
    max_ibi: float = 1500.0,  # ms (40 bpm)
    max_change: float = 0.25,  # 25% change threshold for ectopic detection
) -> np.ndarray:
    """
    Clean IBI series by removing physiologically implausible values.
    
    Args:
        ibi: Raw IBI values in milliseconds
        min_ibi: Minimum valid IBI (ms)
        max_ibi: Maximum valid IBI (ms)
        max_change: Maximum allowed relative change between consecutive IBIs
    
    Returns:
        Cleaned IBI array (invalid values replaced with NaN)
    """
    if len(ibi) == 0:
        return ibi
    
    ibi_clean = ibi.copy()
    
    # Remove out-of-range values
    mask_range = (ibi_clean >= min_ibi) & (ibi_clean <= max_ibi)
    ibi_clean[~mask_range] = np.nan
    
    # Detect ectopic beats (sudden large changes)
    if len(ibi_clean) > 1:
        # Calculate relative change
        ibi_median = np.nanmedian(ibi_clean)
        if ibi_median > 0:
            relative_diff = np.abs(ibi_clean - ibi_median) / ibi_median
            mask_ectopic = relative_diff > max_change
            ibi_clean[mask_ectopic] = np.nan
    
    return ibi_clean


# ============================================================================
# HRV Feature Extraction
# ============================================================================

def compute_hr_features(ibi_ms: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """
    Compute heart rate features from IBI series.
    
    Args:
        ibi_ms: Inter-beat intervals in milliseconds
        prefix: Prefix for feature names
    
    Returns:
        Dictionary of HR features
    """
    features = {}
    p = prefix
    
    # Filter valid IBIs
    valid_ibi = ibi_ms[np.isfinite(ibi_ms)]
    
    if len(valid_ibi) < 2:
        return {
            f"{p}hr_mean": np.nan,
            f"{p}hr_std": np.nan,
            f"{p}hr_min": np.nan,
            f"{p}hr_max": np.nan,
            f"{p}hr_range": np.nan,
            f"{p}hr_median": np.nan,
        }
    
    # Convert IBI (ms) to HR (bpm): HR = 60000 / IBI
    hr = 60000.0 / valid_ibi
    
    features[f"{p}hr_mean"] = float(np.mean(hr))
    features[f"{p}hr_std"] = float(np.std(hr, ddof=1)) if len(hr) > 1 else np.nan
    features[f"{p}hr_min"] = float(np.min(hr))
    features[f"{p}hr_max"] = float(np.max(hr))
    features[f"{p}hr_range"] = features[f"{p}hr_max"] - features[f"{p}hr_min"]
    features[f"{p}hr_median"] = float(np.median(hr))
    
    return features


def compute_hrv_time_domain(ibi_ms: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """
    Compute time-domain HRV features.
    
    Implements standard HRV metrics per Task Force guidelines (1996).
    
    Args:
        ibi_ms: Inter-beat intervals in milliseconds
        prefix: Prefix for feature names
    
    Returns:
        Dictionary of HRV features
    """
    features = {}
    p = prefix
    
    valid_ibi = ibi_ms[np.isfinite(ibi_ms)]
    
    if len(valid_ibi) < 2:
        return {
            f"{p}rmssd": np.nan,
            f"{p}sdnn": np.nan,
            f"{p}sdsd": np.nan,
            f"{p}pnn50": np.nan,
            f"{p}pnn20": np.nan,
            f"{p}mean_ibi": np.nan,
            f"{p}cv_ibi": np.nan,
        }
    
    # SDNN: Standard deviation of all NN intervals
    features[f"{p}sdnn"] = float(np.std(valid_ibi, ddof=1))
    
    # Mean IBI
    features[f"{p}mean_ibi"] = float(np.mean(valid_ibi))
    
    # Coefficient of variation
    if features[f"{p}mean_ibi"] > 0:
        features[f"{p}cv_ibi"] = features[f"{p}sdnn"] / features[f"{p}mean_ibi"]
    else:
        features[f"{p}cv_ibi"] = np.nan
    
    # Successive differences
    diff_ibi = np.diff(valid_ibi)
    
    if len(diff_ibi) < 1:
        features[f"{p}rmssd"] = np.nan
        features[f"{p}sdsd"] = np.nan
        features[f"{p}pnn50"] = np.nan
        features[f"{p}pnn20"] = np.nan
        return features
    
    # RMSSD: Root mean square of successive differences
    features[f"{p}rmssd"] = float(np.sqrt(np.mean(diff_ibi ** 2)))
    
    # SDSD: Standard deviation of successive differences
    features[f"{p}sdsd"] = float(np.std(diff_ibi, ddof=1)) if len(diff_ibi) > 1 else np.nan
    
    # pNN50: Percentage of successive differences > 50ms
    nn50 = np.sum(np.abs(diff_ibi) > 50)
    features[f"{p}pnn50"] = float(nn50 / len(diff_ibi) * 100)
    
    # pNN20: Percentage of successive differences > 20ms
    nn20 = np.sum(np.abs(diff_ibi) > 20)
    features[f"{p}pnn20"] = float(nn20 / len(diff_ibi) * 100)
    
    return features


def compute_hrv_frequency_domain(
    ibi_ms: np.ndarray,
    peak_times_sec: np.ndarray,
    prefix: str = "",
    fs_interp: float = 4.0,  # Interpolation frequency for spectral analysis
) -> Dict[str, float]:
    """
    Compute frequency-domain HRV features.
    
    Uses Welch's method on interpolated IBI series.
    
    VLF: 0.003-0.04 Hz (not reliable for short windows)
    LF:  0.04-0.15 Hz (sympathetic + parasympathetic)
    HF:  0.15-0.4 Hz (parasympathetic, respiratory sinus arrhythmia)
    
    Args:
        ibi_ms: Inter-beat intervals in milliseconds
        peak_times_sec: Times of peaks in seconds
        prefix: Prefix for feature names
        fs_interp: Interpolation sampling frequency
    
    Returns:
        Dictionary of frequency-domain HRV features
    """
    features = {}
    p = prefix
    
    valid_mask = np.isfinite(ibi_ms)
    valid_ibi = ibi_ms[valid_mask]
    
    # Need at least 2 valid consecutive IBIs and their times
    if len(valid_ibi) < 4:
        return {
            f"{p}lf_power": np.nan,
            f"{p}hf_power": np.nan,
            f"{p}lf_hf_ratio": np.nan,
            f"{p}total_power": np.nan,
            f"{p}lf_norm": np.nan,
            f"{p}hf_norm": np.nan,
        }
    
    # IBI times are at beat n+1 (end of interval)
    # First IBI is between peak 0 and peak 1, recorded at time of peak 1
    if len(peak_times_sec) < 2:
        return {
            f"{p}lf_power": np.nan,
            f"{p}hf_power": np.nan,
            f"{p}lf_hf_ratio": np.nan,
            f"{p}total_power": np.nan,
            f"{p}lf_norm": np.nan,
            f"{p}hf_norm": np.nan,
        }
    
    # Get times for valid IBIs (IBI[i] is at peak_times[i+1])
    ibi_times = peak_times_sec[1:][valid_mask[:len(peak_times_sec)-1]] if len(valid_mask) > 0 else np.array([])
    
    if len(ibi_times) < 4 or len(ibi_times) != len(valid_ibi):
        # Fallback: use cumulative IBI as time
        ibi_times = np.cumsum(valid_ibi) / 1000.0
    
    if len(ibi_times) < 4:
        return {
            f"{p}lf_power": np.nan,
            f"{p}hf_power": np.nan,
            f"{p}lf_hf_ratio": np.nan,
            f"{p}total_power": np.nan,
            f"{p}lf_norm": np.nan,
            f"{p}hf_norm": np.nan,
        }
    
    try:
        # Interpolate IBI to uniform sampling
        duration = ibi_times[-1] - ibi_times[0]
        if duration < 1.0:  # Need at least 1 second of data
            raise ValueError("Insufficient duration")
        
        t_interp = np.arange(ibi_times[0], ibi_times[-1], 1.0 / fs_interp)
        
        if len(t_interp) < 8:
            raise ValueError("Insufficient samples after interpolation")
        
        # Cubic spline interpolation
        f_interp = interp1d(ibi_times, valid_ibi, kind='cubic', fill_value='extrapolate')
        ibi_interp = f_interp(t_interp)
        
        # Detrend
        ibi_interp = signal.detrend(ibi_interp)
        
        # Welch's method for PSD
        nperseg = min(len(ibi_interp), int(fs_interp * 60))  # Up to 60 seconds
        nperseg = max(8, nperseg)  # At least 8 samples
        
        freqs, psd = signal.welch(
            ibi_interp,
            fs=fs_interp,
            nperseg=nperseg,
            noverlap=nperseg // 2,
        )
        
        # Band powers (ms^2)
        vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.4)
        
        # Integrate PSD over bands
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        
        lf_power = np.trapz(psd[lf_mask], dx=df) if np.any(lf_mask) else 0.0
        hf_power = np.trapz(psd[hf_mask], dx=df) if np.any(hf_mask) else 0.0
        total_power = np.trapz(psd, dx=df)
        
        features[f"{p}lf_power"] = float(lf_power)
        features[f"{p}hf_power"] = float(hf_power)
        features[f"{p}total_power"] = float(total_power)
        
        # LF/HF ratio (sympathovagal balance)
        if hf_power > 0:
            features[f"{p}lf_hf_ratio"] = float(lf_power / hf_power)
        else:
            features[f"{p}lf_hf_ratio"] = np.nan
        
        # Normalized units
        lf_hf_sum = lf_power + hf_power
        if lf_hf_sum > 0:
            features[f"{p}lf_norm"] = float(lf_power / lf_hf_sum * 100)
            features[f"{p}hf_norm"] = float(hf_power / lf_hf_sum * 100)
        else:
            features[f"{p}lf_norm"] = np.nan
            features[f"{p}hf_norm"] = np.nan
            
    except Exception:
        features[f"{p}lf_power"] = np.nan
        features[f"{p}hf_power"] = np.nan
        features[f"{p}lf_hf_ratio"] = np.nan
        features[f"{p}total_power"] = np.nan
        features[f"{p}lf_norm"] = np.nan
        features[f"{p}hf_norm"] = np.nan
    
    return features


# ============================================================================
# Main Feature Extraction
# ============================================================================

def extract_hrv_features_window(
    ppg_segment: np.ndarray,
    fs: float,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Extract all HRV features from a single PPG window.
    
    Args:
        ppg_segment: Raw PPG signal segment
        fs: Sampling frequency
        prefix: Prefix for feature names
    
    Returns:
        Dictionary of all HRV features
    """
    features = {}
    
    # Check minimum length (need at least 2 seconds for meaningful HRV)
    if len(ppg_segment) < int(fs * 2):
        # Return NaN features
        nan_features = {
            f"{prefix}hr_mean": np.nan,
            f"{prefix}hr_std": np.nan,
            f"{prefix}hr_min": np.nan,
            f"{prefix}hr_max": np.nan,
            f"{prefix}hr_range": np.nan,
            f"{prefix}hr_median": np.nan,
            f"{prefix}rmssd": np.nan,
            f"{prefix}sdnn": np.nan,
            f"{prefix}sdsd": np.nan,
            f"{prefix}pnn50": np.nan,
            f"{prefix}pnn20": np.nan,
            f"{prefix}mean_ibi": np.nan,
            f"{prefix}cv_ibi": np.nan,
            f"{prefix}lf_power": np.nan,
            f"{prefix}hf_power": np.nan,
            f"{prefix}lf_hf_ratio": np.nan,
            f"{prefix}total_power": np.nan,
            f"{prefix}lf_norm": np.nan,
            f"{prefix}hf_norm": np.nan,
            f"{prefix}n_peaks": 0,
            f"{prefix}peak_quality": np.nan,
        }
        return nan_features
    
    # 1. Preprocess: bandpass filter + normalize
    ppg_filtered = bandpass_filter(ppg_segment, fs)
    ppg_norm = normalize_signal(ppg_filtered)
    
    # 2. Peak detection
    peaks = detect_ppg_peaks(ppg_norm, fs)
    features[f"{prefix}n_peaks"] = len(peaks)
    
    if len(peaks) < 3:
        # Not enough peaks for HRV
        features.update({
            f"{prefix}hr_mean": np.nan,
            f"{prefix}hr_std": np.nan,
            f"{prefix}hr_min": np.nan,
            f"{prefix}hr_max": np.nan,
            f"{prefix}hr_range": np.nan,
            f"{prefix}hr_median": np.nan,
            f"{prefix}rmssd": np.nan,
            f"{prefix}sdnn": np.nan,
            f"{prefix}sdsd": np.nan,
            f"{prefix}pnn50": np.nan,
            f"{prefix}pnn20": np.nan,
            f"{prefix}mean_ibi": np.nan,
            f"{prefix}cv_ibi": np.nan,
            f"{prefix}lf_power": np.nan,
            f"{prefix}hf_power": np.nan,
            f"{prefix}lf_hf_ratio": np.nan,
            f"{prefix}total_power": np.nan,
            f"{prefix}lf_norm": np.nan,
            f"{prefix}hf_norm": np.nan,
            f"{prefix}peak_quality": 0.0,
        })
        return features
    
    # 3. Compute IBI
    ibi_raw = compute_ibi(peaks, fs)
    ibi_clean = clean_ibi(ibi_raw)
    
    # Quality metric: fraction of valid IBIs
    n_valid = np.sum(np.isfinite(ibi_clean))
    features[f"{prefix}peak_quality"] = float(n_valid / len(ibi_raw)) if len(ibi_raw) > 0 else 0.0
    
    # 4. HR features
    features.update(compute_hr_features(ibi_clean, prefix))
    
    # 5. Time-domain HRV
    features.update(compute_hrv_time_domain(ibi_clean, prefix))
    
    # 6. Frequency-domain HRV
    peak_times = peaks / fs
    features.update(compute_hrv_frequency_domain(ibi_clean, peak_times, prefix))
    
    return features


def extract_hrv_features(
    ppg_csv: str,
    windows_csv: str,
    out_path: str,
    time_col: str = "t_sec",
    signal_col: str = "value",
    fs: float = 32.0,
    prefix: str = "hrv_",
) -> pd.DataFrame:
    """
    Extract HRV features for all windows.
    
    Args:
        ppg_csv: Path to preprocessed PPG CSV
        windows_csv: Path to windows CSV
        out_path: Output path for features
        time_col: Name of time column
        signal_col: Name of signal column
        fs: Sampling frequency
        prefix: Prefix for feature names
    
    Returns:
        DataFrame with HRV features
    """
    # Load data
    ppg_df = pd.read_csv(ppg_csv)
    windows_df = pd.read_csv(windows_csv)
    
    if time_col not in ppg_df.columns:
        raise ValueError(f"Missing time column '{time_col}' in PPG file")
    if signal_col not in ppg_df.columns:
        raise ValueError(f"Missing signal column '{signal_col}' in PPG file")
    
    # Get signal
    ppg_signal = ppg_df[signal_col].to_numpy(dtype=float)
    
    # Extract features for each window
    rows = []
    for idx, w in windows_df.iterrows():
        s = int(w["start_idx"])
        e = int(w["end_idx"])
        
        segment = ppg_signal[s:e]
        feats = extract_hrv_features_window(segment, fs, prefix)
        
        # Add window metadata
        feats["window_id"] = int(w.get("window_id", idx))
        feats["start_idx"] = s
        feats["end_idx"] = e
        feats["t_start"] = float(w["t_start"])
        feats["t_center"] = float(w["t_center"])
        feats["t_end"] = float(w["t_end"])
        
        rows.append(feats)
    
    out_df = pd.DataFrame(rows)
    
    # Reorder columns: metadata first
    meta_cols = ["window_id", "start_idx", "end_idx", "t_start", "t_center", "t_end"]
    feat_cols = [c for c in out_df.columns if c not in meta_cols]
    out_df = out_df[meta_cols + feat_cols]
    
    # Save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    
    print(f"✓ HRV features extracted → {out_path} | windows={len(out_df)} | features={len(feat_cols)}")
    
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Extract HRV features from PPG")
    parser.add_argument("--ppg", required=True, help="Path to preprocessed PPG CSV")
    parser.add_argument("--windows", required=True, help="Path to windows CSV")
    parser.add_argument("--out", required=True, help="Output path")
    parser.add_argument("--fs", type=float, default=32.0, help="Sampling frequency")
    parser.add_argument("--time_col", default="t_sec", help="Time column name")
    parser.add_argument("--signal_col", default="value", help="Signal column name")
    parser.add_argument("--prefix", default="hrv_", help="Feature name prefix")
    
    args = parser.parse_args()
    
    extract_hrv_features(
        ppg_csv=args.ppg,
        windows_csv=args.windows,
        out_path=args.out,
        time_col=args.time_col,
        signal_col=args.signal_col,
        fs=args.fs,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
