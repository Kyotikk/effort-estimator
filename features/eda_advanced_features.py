#!/usr/bin/env python3
"""
Advanced EDA (Electrodermal Activity) feature extraction.

Implements:
1. EDA preprocessing (lowpass filter, artifact removal)
2. Tonic/Phasic decomposition (simplified cvxEDA-like approach)
3. SCR (Skin Conductance Response) detection and features
4. SCL (Skin Conductance Level) features
5. Sympathetic arousal metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import median_filter


# ============================================================================
# EDA Preprocessing
# ============================================================================

def lowpass_filter(x: np.ndarray, fs: float, cutoff: float = 5.0, order: int = 4) -> np.ndarray:
    """
    Butterworth lowpass filter for EDA.
    
    EDA is a slow signal (typically < 0.5 Hz for tonic, < 5 Hz for phasic).
    cutoff=5 Hz removes high-frequency noise while preserving all EDA components.
    """
    nyq = 0.5 * fs
    if cutoff >= nyq:
        cutoff = nyq * 0.9
    
    sos = signal.butter(order, cutoff / nyq, btype='low', output='sos')
    return signal.sosfiltfilt(sos, x)


def remove_artifacts(
    x: np.ndarray,
    fs: float,
    threshold: float = 3.0,
    window_sec: float = 1.0,
) -> np.ndarray:
    """
    Remove artifacts from EDA signal using median filtering and outlier detection.
    
    Args:
        x: Raw EDA signal
        fs: Sampling frequency
        threshold: Z-score threshold for outlier detection
        window_sec: Window size for median filter
    """
    # Median filter to remove spikes
    window_samples = int(fs * window_sec)
    if window_samples % 2 == 0:
        window_samples += 1  # Must be odd
    window_samples = max(3, window_samples)
    
    x_median = median_filter(x, size=window_samples)
    
    # Detect outliers (sudden large deviations from median)
    diff = np.abs(x - x_median)
    diff_std = np.std(diff)
    
    if diff_std > 0:
        outlier_mask = diff > threshold * diff_std
        x_clean = x.copy()
        x_clean[outlier_mask] = x_median[outlier_mask]
    else:
        x_clean = x.copy()
    
    return x_clean


def ensure_positive(x: np.ndarray) -> np.ndarray:
    """Ensure EDA values are non-negative (physiological constraint)."""
    return np.maximum(x, 0)


# ============================================================================
# Tonic/Phasic Decomposition
# ============================================================================

def decompose_eda_simple(
    x: np.ndarray,
    fs: float,
    tau: float = 2.0,  # Time constant for tonic (seconds)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple tonic/phasic decomposition using lowpass filtering.
    
    This is a simplified approach compared to cvxEDA, but works well
    for most practical applications.
    
    Tonic (SCL): Baseline skin conductance level - lowpass filtered
    Phasic (SCR): Rapid changes - original minus tonic
    
    Args:
        x: Preprocessed EDA signal
        fs: Sampling frequency
        tau: Time constant for tonic extraction (higher = smoother tonic)
    
    Returns:
        (tonic, phasic) arrays
    """
    # Tonic: very low frequency component (< 0.05 Hz typically)
    # Use lowpass filter with cutoff based on tau
    cutoff = 1.0 / (2 * np.pi * tau)
    cutoff = max(0.01, min(cutoff, 0.5 * fs * 0.9))  # Clamp to valid range
    
    nyq = 0.5 * fs
    sos = signal.butter(2, cutoff / nyq, btype='low', output='sos')
    
    # Pad signal to reduce edge effects
    pad_len = int(fs * 10)  # 10 seconds padding
    x_padded = np.pad(x, pad_len, mode='edge')
    
    tonic_padded = signal.sosfiltfilt(sos, x_padded)
    tonic = tonic_padded[pad_len:-pad_len]
    
    # Phasic: residual after removing tonic
    phasic = x - tonic
    phasic = np.maximum(phasic, 0)  # SCRs are positive deflections
    
    return tonic, phasic


def decompose_eda_highpass(
    x: np.ndarray,
    fs: float,
    cutoff: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alternative decomposition using highpass filter for phasic.
    
    Args:
        x: Preprocessed EDA signal
        fs: Sampling frequency
        cutoff: Highpass cutoff frequency (Hz)
    """
    nyq = 0.5 * fs
    if cutoff >= nyq:
        cutoff = nyq * 0.1
    
    # Highpass for phasic
    sos_hp = signal.butter(2, cutoff / nyq, btype='high', output='sos')
    phasic = signal.sosfiltfilt(sos_hp, x)
    phasic = np.maximum(phasic, 0)
    
    # Tonic is the complement
    tonic = x - phasic
    tonic = np.maximum(tonic, 0)
    
    return tonic, phasic


# ============================================================================
# SCR Detection
# ============================================================================

def detect_scr_peaks(
    phasic: np.ndarray,
    fs: float,
    min_amplitude: float = 0.01,  # Minimum SCR amplitude (µS)
    min_rise_time: float = 0.5,  # Minimum rise time (seconds)
    max_rise_time: float = 5.0,  # Maximum rise time (seconds)
) -> List[Dict]:
    """
    Detect Skin Conductance Responses (SCRs) in phasic component.
    
    An SCR is characterized by:
    - Onset: beginning of rise
    - Peak: maximum amplitude
    - Recovery: return toward baseline
    
    Args:
        phasic: Phasic EDA component
        fs: Sampling frequency
        min_amplitude: Minimum peak amplitude to consider
        min_rise_time: Minimum onset-to-peak time
        max_rise_time: Maximum onset-to-peak time
    
    Returns:
        List of SCR dictionaries with onset, peak, amplitude, rise_time, etc.
    """
    scrs = []
    
    # Find peaks in phasic signal
    min_distance = int(fs * 1.0)  # At least 1 second between peaks
    peaks, properties = signal.find_peaks(
        phasic,
        height=min_amplitude,
        distance=min_distance,
        prominence=min_amplitude * 0.5,
    )
    
    if len(peaks) == 0:
        return scrs
    
    # For each peak, find onset (start of rise)
    for peak_idx in peaks:
        # Search backward for onset
        # Onset: where derivative becomes positive before peak
        search_start = max(0, peak_idx - int(fs * max_rise_time))
        
        # Find where signal starts rising
        onset_idx = search_start
        for i in range(peak_idx - 1, search_start, -1):
            if phasic[i] <= phasic[search_start] or phasic[i] < min_amplitude * 0.1:
                onset_idx = i
                break
        
        # Calculate SCR characteristics
        rise_time = (peak_idx - onset_idx) / fs
        amplitude = phasic[peak_idx] - phasic[onset_idx]
        
        # Check physiological constraints
        if rise_time < min_rise_time or rise_time > max_rise_time:
            continue
        if amplitude < min_amplitude:
            continue
        
        # Find recovery half-time (time to reach 50% of peak after peak)
        recovery_50_idx = peak_idx
        target = phasic[onset_idx] + amplitude * 0.5
        for i in range(peak_idx, min(len(phasic), peak_idx + int(fs * 10))):
            if phasic[i] <= target:
                recovery_50_idx = i
                break
        
        recovery_time = (recovery_50_idx - peak_idx) / fs
        
        scrs.append({
            'onset_idx': onset_idx,
            'peak_idx': peak_idx,
            'onset_time': onset_idx / fs,
            'peak_time': peak_idx / fs,
            'amplitude': amplitude,
            'rise_time': rise_time,
            'recovery_time': recovery_time,
        })
    
    return scrs


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_scl_features(tonic: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """
    Extract Skin Conductance Level (tonic) features.
    """
    p = prefix
    
    if len(tonic) == 0 or not np.any(np.isfinite(tonic)):
        return {
            f"{p}scl_mean": np.nan,
            f"{p}scl_std": np.nan,
            f"{p}scl_min": np.nan,
            f"{p}scl_max": np.nan,
            f"{p}scl_range": np.nan,
            f"{p}scl_slope": np.nan,
            f"{p}scl_auc": np.nan,
        }
    
    valid = tonic[np.isfinite(tonic)]
    
    features = {
        f"{p}scl_mean": float(np.mean(valid)),
        f"{p}scl_std": float(np.std(valid, ddof=0)),
        f"{p}scl_min": float(np.min(valid)),
        f"{p}scl_max": float(np.max(valid)),
        f"{p}scl_range": float(np.max(valid) - np.min(valid)),
    }
    
    # Linear slope (trend)
    if len(valid) > 1:
        t = np.arange(len(valid))
        try:
            slope, _ = np.polyfit(t, valid, 1)
            features[f"{p}scl_slope"] = float(slope)
        except Exception:
            features[f"{p}scl_slope"] = np.nan
    else:
        features[f"{p}scl_slope"] = np.nan
    
    # Area under curve (total tonic activity)
    features[f"{p}scl_auc"] = float(np.trapz(valid))
    
    return features


def extract_scr_features(
    scrs: List[Dict],
    window_duration: float,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Extract Skin Conductance Response (phasic) features.
    """
    p = prefix
    
    if len(scrs) == 0:
        return {
            f"{p}scr_count": 0,
            f"{p}scr_rate": 0.0,
            f"{p}scr_amp_mean": np.nan,
            f"{p}scr_amp_std": np.nan,
            f"{p}scr_amp_max": np.nan,
            f"{p}scr_amp_sum": 0.0,
            f"{p}scr_rise_mean": np.nan,
            f"{p}scr_rise_std": np.nan,
            f"{p}scr_recovery_mean": np.nan,
        }
    
    amplitudes = [s['amplitude'] for s in scrs]
    rise_times = [s['rise_time'] for s in scrs]
    recovery_times = [s['recovery_time'] for s in scrs]
    
    features = {
        f"{p}scr_count": len(scrs),
        f"{p}scr_rate": float(len(scrs) / window_duration) if window_duration > 0 else 0.0,
        f"{p}scr_amp_mean": float(np.mean(amplitudes)),
        f"{p}scr_amp_std": float(np.std(amplitudes, ddof=0)) if len(amplitudes) > 1 else 0.0,
        f"{p}scr_amp_max": float(np.max(amplitudes)),
        f"{p}scr_amp_sum": float(np.sum(amplitudes)),
        f"{p}scr_rise_mean": float(np.mean(rise_times)),
        f"{p}scr_rise_std": float(np.std(rise_times, ddof=0)) if len(rise_times) > 1 else 0.0,
        f"{p}scr_recovery_mean": float(np.mean(recovery_times)),
    }
    
    return features


def extract_phasic_stats(phasic: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """
    Extract statistical features from phasic component.
    """
    p = prefix
    
    if len(phasic) == 0 or not np.any(np.isfinite(phasic)):
        return {
            f"{p}phasic_mean": np.nan,
            f"{p}phasic_std": np.nan,
            f"{p}phasic_max": np.nan,
            f"{p}phasic_auc": np.nan,
            f"{p}phasic_energy": np.nan,
        }
    
    valid = phasic[np.isfinite(phasic)]
    
    return {
        f"{p}phasic_mean": float(np.mean(valid)),
        f"{p}phasic_std": float(np.std(valid, ddof=0)),
        f"{p}phasic_max": float(np.max(valid)),
        f"{p}phasic_auc": float(np.trapz(valid)),
        f"{p}phasic_energy": float(np.sum(valid ** 2)),
    }


def extract_eda_advanced_features_window(
    eda_segment: np.ndarray,
    fs: float,
    prefix: str = "eda_",
) -> Dict[str, float]:
    """
    Extract all advanced EDA features from a single window.
    
    Args:
        eda_segment: Raw EDA signal segment
        fs: Sampling frequency
        prefix: Prefix for feature names
    
    Returns:
        Dictionary of all EDA features
    """
    features = {}
    
    # Check minimum length
    if len(eda_segment) < int(fs * 1):  # At least 1 second
        nan_features = {
            f"{prefix}scl_mean": np.nan,
            f"{prefix}scl_std": np.nan,
            f"{prefix}scl_min": np.nan,
            f"{prefix}scl_max": np.nan,
            f"{prefix}scl_range": np.nan,
            f"{prefix}scl_slope": np.nan,
            f"{prefix}scl_auc": np.nan,
            f"{prefix}scr_count": 0,
            f"{prefix}scr_rate": 0.0,
            f"{prefix}scr_amp_mean": np.nan,
            f"{prefix}scr_amp_std": np.nan,
            f"{prefix}scr_amp_max": np.nan,
            f"{prefix}scr_amp_sum": 0.0,
            f"{prefix}scr_rise_mean": np.nan,
            f"{prefix}scr_rise_std": np.nan,
            f"{prefix}scr_recovery_mean": np.nan,
            f"{prefix}phasic_mean": np.nan,
            f"{prefix}phasic_std": np.nan,
            f"{prefix}phasic_max": np.nan,
            f"{prefix}phasic_auc": np.nan,
            f"{prefix}phasic_energy": np.nan,
        }
        return nan_features
    
    # Preprocess
    eda_clean = remove_artifacts(eda_segment, fs)
    eda_clean = lowpass_filter(eda_clean, fs)
    eda_clean = ensure_positive(eda_clean)
    
    # Decompose into tonic and phasic
    tonic, phasic = decompose_eda_simple(eda_clean, fs)
    
    # Window duration
    duration = len(eda_segment) / fs
    
    # Extract features
    features.update(extract_scl_features(tonic, prefix))
    
    # Detect SCRs
    scrs = detect_scr_peaks(phasic, fs)
    features.update(extract_scr_features(scrs, duration, prefix))
    
    # Phasic statistics
    features.update(extract_phasic_stats(phasic, prefix))
    
    return features


def extract_eda_advanced_features(
    eda_csv: str,
    windows_csv: str,
    out_path: str,
    time_col: str = "t_sec",
    signal_col: str = "eda_cc",
    fs: float = 32.0,
    prefix: str = "eda_",
) -> pd.DataFrame:
    """
    Extract advanced EDA features for all windows.
    
    Args:
        eda_csv: Path to preprocessed EDA CSV
        windows_csv: Path to windows CSV
        out_path: Output path for features
        time_col: Name of time column
        signal_col: Name of signal column (skin conductance)
        fs: Sampling frequency
        prefix: Prefix for feature names
    
    Returns:
        DataFrame with EDA features
    """
    # Load data
    eda_df = pd.read_csv(eda_csv)
    windows_df = pd.read_csv(windows_csv)
    
    if signal_col not in eda_df.columns:
        raise ValueError(f"Missing signal column '{signal_col}' in EDA file. Found: {list(eda_df.columns)}")
    
    # Get signal
    eda_signal = eda_df[signal_col].to_numpy(dtype=float)
    
    # Extract features for each window
    rows = []
    for idx, w in windows_df.iterrows():
        s = int(w["start_idx"])
        e = int(w["end_idx"])
        
        segment = eda_signal[s:e]
        feats = extract_eda_advanced_features_window(segment, fs, prefix)
        
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
    
    print(f"✓ Advanced EDA features extracted → {out_path} | windows={len(out_df)} | features={len(feat_cols)}")
    
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Extract advanced EDA features")
    parser.add_argument("--eda", required=True, help="Path to preprocessed EDA CSV")
    parser.add_argument("--windows", required=True, help="Path to windows CSV")
    parser.add_argument("--out", required=True, help="Output path")
    parser.add_argument("--fs", type=float, default=32.0, help="Sampling frequency")
    parser.add_argument("--time_col", default="t_sec", help="Time column name")
    parser.add_argument("--signal_col", default="eda_cc", help="Signal column name")
    parser.add_argument("--prefix", default="eda_", help="Feature name prefix")
    
    args = parser.parse_args()
    
    extract_eda_advanced_features(
        eda_csv=args.eda,
        windows_csv=args.windows,
        out_path=args.out,
        time_col=args.time_col,
        signal_col=args.signal_col,
        fs=args.fs,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
